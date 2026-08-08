"""Module D - the Decision Engine.

This module contains **no machine learning**.  It is a deterministic, highly
restrictive rule cascade that consumes the four model outputs and decides
whether risk may be put on.  Every rule is evaluated in a fixed order, every
evaluation is recorded, and the *first* rule that rejects short-circuits the
cascade - so the audit log always names one precise cause for a ``NO_TRADE``.

Rule cascade
------------
========  ==========================================================
``R0``    System gate: kill switch, trading flag, portfolio caps.
``R1``    Direction confidence >= threshold, and not a NO_TRADE call.
``R2``    Directional margin over the opposing side is decisive.
``R3``    NO_TRADE probability mass is not itself dominant.
``R4``    Entry model says "now" rather than "wait".
``R5``    Exit geometry is sane and clears the reward/risk floor.
``R6``    Risk model returned non-zero leverage.
``R8``    Regime is not on the blocked list.
``R9``    Model provenance is acceptable for the current trading mode.
========  ==========================================================

Only when every rule passes is a :class:`TradeSignal` constructed - and even
then the Pydantic validator re-checks the barrier geometry before it can leave
this module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Final, Sequence

from config.settings import DecisionSettings, Settings
from core.logger import get_logger
from core.utils import clamp
from module_c_ml.schemas import (
    DecisionResult,
    DecisionVerdict,
    DirectionPrediction,
    EntryPrediction,
    ExitParameters,
    ModelInferenceResult,
    ModelSource,
    RiskAllocation,
    TradeAction,
    TradeSignal,
)

_LOGGER = get_logger(__name__)


class Rule:
    """Stable identifiers for every rule, used verbatim in the audit log."""

    SYSTEM_HALTED: Final[str] = "R0_SYSTEM_HALTED"
    TRADING_DISABLED: Final[str] = "R0_TRADING_DISABLED"
    PORTFOLIO_FULL: Final[str] = "R0_PORTFOLIO_FULL"
    SYMBOL_ALREADY_OPEN: Final[str] = "R0_SYMBOL_ALREADY_OPEN"
    DIRECTION_NO_TRADE: Final[str] = "R1_DIRECTION_IS_NO_TRADE"
    DIRECTION_CONFIDENCE: Final[str] = "R1_DIRECTION_CONFIDENCE_TOO_LOW"
    DIRECTION_MARGIN: Final[str] = "R2_DIRECTIONAL_MARGIN_TOO_THIN"
    NO_TRADE_MASS: Final[str] = "R3_NO_TRADE_MASS_TOO_HIGH"
    ENTRY_REJECTED: Final[str] = "R4_ENTRY_MODEL_SAYS_WAIT"
    REWARD_RISK: Final[str] = "R5_REWARD_RISK_BELOW_FLOOR"
    RISK_ABORT: Final[str] = "R6_RISK_MODEL_ABORT"
    REGIME_BLOCKED: Final[str] = "R8_REGIME_BLOCKED"
    UNTRAINED_MODELS: Final[str] = "R9_UNTRAINED_MODELS_IN_LIVE_MODE"
    SIGNAL_INVALID: Final[str] = "R10_SIGNAL_CONSTRUCTION_FAILED"
    EXECUTE: Final[str] = "ALL_RULES_PASSED"


@dataclass(slots=True)
class DecisionContext:
    """External state the engine must respect but does not own.

    Passing this in (rather than letting the engine query the Risk Guard or the
    executor directly) keeps Module D free of dependencies on Module E, which is
    what makes it unit-testable in isolation.
    """

    risk_guard_state: str = "GREEN"
    trading_enabled: bool = True
    trading_mode: str = "paper"
    open_positions: int = 0
    open_symbols: frozenset[str] = field(default_factory=frozenset)
    equity: float = 0.0
    size_multiplier: float = 1.0


class DecisionEngine:
    """The final arbiter: turns four model opinions into one binding instruction."""

    def __init__(self, settings: Settings) -> None:
        self._settings: Settings = settings
        self._config: DecisionSettings = settings.decision

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def evaluate(
        self,
        inference: ModelInferenceResult,
        context: DecisionContext | None = None,
    ) -> DecisionResult:
        """Run the full rule cascade for one symbol.

        Args:
            inference: The bundled output of the four ML heads.
            context: Live system state (kill switch, open positions, equity).

        Returns:
            A :class:`DecisionResult`.  ``verdict is EXECUTE`` implies ``signal``
            is a fully validated :class:`TradeSignal`; every other verdict names
            the rule that stopped it.
        """
        state: DecisionContext = context or DecisionContext()
        checks: list[dict[str, Any]] = []

        direction: DirectionPrediction = inference.direction
        entry: EntryPrediction = inference.entry
        exit_params: ExitParameters = inference.exit_params
        risk: RiskAllocation = inference.risk

        # --- R0: system-level gates ---------------------------------------
        blocked: DecisionResult | None = self._check_system_gates(inference, state, checks)
        if blocked is not None:
            return blocked

        # --- R1: direction confidence --------------------------------------
        action: TradeAction = direction.action
        self._record(
            checks,
            Rule.DIRECTION_NO_TRADE,
            action is not TradeAction.NO_TRADE,
            f"action={action.value}",
        )
        if action is TradeAction.NO_TRADE:
            return self._reject(
                inference,
                Rule.DIRECTION_NO_TRADE,
                (
                    f"Direction model favours NO_TRADE "
                    f"(p={direction.no_trade_probability:.3f} vs long "
                    f"{direction.long_probability:.3f} / short {direction.short_probability:.3f})"
                ),
                checks,
            )

        threshold: float = self._config.min_direction_confidence
        directional_confidence: float = (
            direction.long_probability
            if action is TradeAction.LONG
            else direction.short_probability
        )
        self._record(
            checks,
            Rule.DIRECTION_CONFIDENCE,
            directional_confidence >= threshold,
            f"confidence={directional_confidence:.4f} threshold={threshold:.4f}",
        )
        if directional_confidence < threshold:
            return self._reject(
                inference,
                Rule.DIRECTION_CONFIDENCE,
                (
                    f"Rejected: direction confidence {directional_confidence:.1%} < "
                    f"{threshold:.0%}"
                ),
                checks,
            )

        # --- R2: margin over the opposing side ------------------------------
        margin: float = direction.directional_margin
        self._record(
            checks,
            Rule.DIRECTION_MARGIN,
            margin >= self._config.min_direction_margin,
            f"margin={margin:.4f} minimum={self._config.min_direction_margin:.4f}",
        )
        if margin < self._config.min_direction_margin:
            return self._reject(
                inference,
                Rule.DIRECTION_MARGIN,
                (
                    f"Rejected: directional edge {margin:.1%} below the required "
                    f"{self._config.min_direction_margin:.1%} - the model is nearly indifferent"
                ),
                checks,
            )

        # --- R3: NO_TRADE mass ----------------------------------------------
        no_trade_mass: float = direction.no_trade_probability
        self._record(
            checks,
            Rule.NO_TRADE_MASS,
            no_trade_mass <= self._config.max_no_trade_probability,
            f"no_trade_mass={no_trade_mass:.4f} cap={self._config.max_no_trade_probability:.4f}",
        )
        if no_trade_mass > self._config.max_no_trade_probability:
            return self._reject(
                inference,
                Rule.NO_TRADE_MASS,
                (
                    f"Rejected: NO_TRADE probability mass {no_trade_mass:.1%} exceeds the "
                    f"{self._config.max_no_trade_probability:.0%} cap"
                ),
                checks,
            )

        # --- R4: entry timing -----------------------------------------------
        self._record(
            checks,
            Rule.ENTRY_REJECTED,
            entry.should_enter,
            f"entry_probability={entry.probability:.4f} ({entry.reason})",
        )
        if not entry.should_enter:
            return self._reject(
                inference,
                Rule.ENTRY_REJECTED,
                (
                    f"Rejected: entry model says wait for the next 5m candle "
                    f"(p={entry.probability:.3f} < {self._config.min_entry_probability:.3f})"
                ),
                checks,
            )

        # --- R5: exit geometry ----------------------------------------------
        reward_risk: float = exit_params.reward_risk_ratio
        self._record(
            checks,
            Rule.REWARD_RISK,
            reward_risk >= self._config.min_reward_risk_ratio,
            f"reward_risk={reward_risk:.3f} floor={self._config.min_reward_risk_ratio:.3f}",
        )
        if reward_risk < self._config.min_reward_risk_ratio:
            return self._reject(
                inference,
                Rule.REWARD_RISK,
                (
                    f"Rejected: reward/risk {reward_risk:.2f} below the "
                    f"{self._config.min_reward_risk_ratio:.2f} floor "
                    f"(TP {exit_params.take_profit_pct:.3%} / SL {exit_params.stop_loss_pct:.3%})"
                ),
                checks,
            )

        # --- R6: risk sizing --------------------------------------------------
        self._record(
            checks,
            Rule.RISK_ABORT,
            not risk.is_abort,
            (
                f"leverage={risk.leverage}x allocation={risk.capital_allocation_pct:.3%} "
                f"score={risk.risk_score:.3f} abort_reason='{risk.abort_reason}'"
            ),
        )
        if risk.is_abort:
            return self._reject(
                inference,
                Rule.RISK_ABORT,
                f"Rejected: risk model returned 0x leverage - {risk.abort_reason or 'sizing floor'}",
                checks,
            )

        # --- R8: regime block-list --------------------------------------------
        regime: int = int(inference.feature_snapshot.get("hmm_regime", -1))
        regime_blocked: bool = regime in self._config.blocked_hmm_regimes
        self._record(
            checks,
            Rule.REGIME_BLOCKED,
            not regime_blocked,
            f"hmm_regime={regime} blocked={list(self._config.blocked_hmm_regimes)}",
        )
        if regime_blocked:
            return self._reject(
                inference,
                Rule.REGIME_BLOCKED,
                f"Rejected: HMM regime {regime} is on the block list",
                checks,
            )

        # --- R9: model provenance ----------------------------------------------
        fallback_in_live: bool = state.trading_mode == "live" and inference.any_fallback
        self._record(
            checks,
            Rule.UNTRAINED_MODELS,
            not fallback_in_live,
            f"any_fallback={inference.any_fallback} mode={state.trading_mode}",
        )
        if fallback_in_live:
            return self._reject(
                inference,
                Rule.UNTRAINED_MODELS,
                (
                    "Rejected: one or more heads fell back to a heuristic; heuristics are "
                    "never allowed to place live orders"
                ),
                checks,
            )

        # --- Signal construction -------------------------------------------------
        signal: TradeSignal | None = self._build_signal(
            inference, action, exit_params, risk, state, directional_confidence, risk.risk_tier
        )
        if signal is None:
            return self._reject(
                inference,
                Rule.SIGNAL_INVALID,
                "Rejected: computed barrier geometry failed validation",
                checks,
            )

        self._record(checks, Rule.EXECUTE, True, f"signal={signal.decision_id}")
        _LOGGER.info(
            "EXECUTE %s %s @ %.6f | %dx | TP %.4f SL %.4f | conf %.1f%%",
            action.value,
            inference.symbol,
            signal.reference_price,
            signal.leverage,
            signal.take_profit,
            signal.stop_loss,
            directional_confidence * 100.0,
        )
        return DecisionResult(
            decision_id=signal.decision_id,
            symbol=inference.symbol,
            verdict=DecisionVerdict.EXECUTE,
            rule_triggered=Rule.EXECUTE,
            reason=(
                f"Trade Executed: {action.value} at {directional_confidence:.1%} confidence, "
                f"{signal.leverage}x leverage, R:R {signal.reward_risk_ratio:.2f}"
            ),
            signal=signal,
            inference=inference,
            checks=checks,
        )

    def evaluate_many(
        self,
        inferences: Sequence[ModelInferenceResult],
        context: DecisionContext | None = None,
    ) -> list[DecisionResult]:
        """Evaluate a batch, respecting the portfolio cap across the batch.

        Symbols are ranked by directional confidence so that when the portfolio
        can only absorb two more positions, they go to the two strongest signals
        rather than to whichever symbol happened to be first alphabetically.
        """
        state: DecisionContext = context or DecisionContext()
        ranked: list[ModelInferenceResult] = sorted(
            inferences, key=lambda item: item.direction.confidence, reverse=True
        )

        results: list[DecisionResult] = []
        open_positions: int = state.open_positions
        open_symbols: set[str] = set(state.open_symbols)

        for inference in ranked:
            local_state = DecisionContext(
                risk_guard_state=state.risk_guard_state,
                trading_enabled=state.trading_enabled,
                trading_mode=state.trading_mode,
                open_positions=open_positions,
                open_symbols=frozenset(open_symbols),
                equity=state.equity,
                size_multiplier=state.size_multiplier,
            )
            result: DecisionResult = self.evaluate(inference, local_state)
            results.append(result)
            if result.is_executable:
                open_positions += 1
                open_symbols.add(inference.symbol)

        return results

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _check_system_gates(
        self,
        inference: ModelInferenceResult,
        state: DecisionContext,
        checks: list[dict[str, Any]],
    ) -> DecisionResult | None:
        """R0 - refuse to even consider a trade when the system says no."""
        halted: bool = state.risk_guard_state.upper() == "RED"
        self._record(checks, Rule.SYSTEM_HALTED, not halted, f"risk_guard={state.risk_guard_state}")
        if halted:
            return self._reject(
                inference,
                Rule.SYSTEM_HALTED,
                "Blocked: Risk Guard is RED - all new risk is forbidden until manual reset",
                checks,
                verdict=DecisionVerdict.BLOCKED,
            )

        self._record(
            checks, Rule.TRADING_DISABLED, state.trading_enabled, f"enabled={state.trading_enabled}"
        )
        if not state.trading_enabled:
            return self._reject(
                inference,
                Rule.TRADING_DISABLED,
                "Blocked: trading is paused by the operator",
                checks,
                verdict=DecisionVerdict.BLOCKED,
            )

        portfolio_full: bool = state.open_positions >= self._config.max_concurrent_positions
        self._record(
            checks,
            Rule.PORTFOLIO_FULL,
            not portfolio_full,
            f"open={state.open_positions} cap={self._config.max_concurrent_positions}",
        )
        if portfolio_full:
            return self._reject(
                inference,
                Rule.PORTFOLIO_FULL,
                (
                    f"Blocked: {state.open_positions} open positions already at the "
                    f"{self._config.max_concurrent_positions} cap"
                ),
                checks,
                verdict=DecisionVerdict.BLOCKED,
            )

        already_open: bool = inference.symbol in state.open_symbols
        self._record(
            checks, Rule.SYMBOL_ALREADY_OPEN, not already_open, f"symbol={inference.symbol}"
        )
        if already_open:
            return self._reject(
                inference,
                Rule.SYMBOL_ALREADY_OPEN,
                f"Blocked: a position is already open on {inference.symbol}",
                checks,
                verdict=DecisionVerdict.BLOCKED,
            )
        return None

    def _build_signal(
        self,
        inference: ModelInferenceResult,
        action: TradeAction,
        exit_params: ExitParameters,
        risk: RiskAllocation,
        state: DecisionContext,
        confidence: float,
        tier: str,
    ) -> TradeSignal | None:
        """Translate percentage geometry into absolute barrier prices.

        Returns ``None`` when the resulting geometry fails the ``TradeSignal``
        validator - a defensive net that keeps a malformed order off the wire.
        """
        price: float = inference.close_price
        take_profit_pct: float = exit_params.take_profit_pct
        stop_loss_pct: float = exit_params.stop_loss_pct
        trailing_pct: float = exit_params.trailing_activation_pct

        if action is TradeAction.LONG:
            take_profit: float = price * (1.0 + take_profit_pct)
            stop_loss: float = price * (1.0 - stop_loss_pct)
            trailing_trigger: float = price * (1.0 + trailing_pct)
        else:
            take_profit = price * (1.0 - take_profit_pct)
            stop_loss = price * (1.0 + stop_loss_pct)
            trailing_trigger = price * (1.0 - trailing_pct)

        # The Risk Guard can throttle sizing without halting the system (YELLOW).
        allocation: float = clamp(
            risk.capital_allocation_pct * max(0.0, state.size_multiplier),
            self._config.min_capital_allocation_pct * 0.1,
            self._config.max_capital_allocation_pct,
        )

        try:
            return TradeSignal(
                symbol=inference.symbol,
                action=action,
                reference_price=price,
                leverage=risk.leverage,
                capital_allocation_pct=allocation,
                take_profit=take_profit,
                stop_loss=stop_loss,
                trailing_trigger=trailing_trigger,
                trailing_distance_pct=exit_params.trailing_distance_pct,
                take_profit_pct=take_profit_pct,
                stop_loss_pct=stop_loss_pct,
                confidence=confidence,
                risk_tier=tier,
                candle_timestamp=inference.timestamp,
                metadata={
                    "direction_probabilities": inference.direction.probabilities,
                    "entry_probability": inference.entry.probability,
                    "risk_score": risk.risk_score,
                    "hmm_regime": inference.feature_snapshot.get("hmm_regime", -1.0),
                    "garch_volatility": inference.feature_snapshot.get("garch_volatility", 0.0),
                    "model_versions": inference.model_versions,
                    "size_multiplier": state.size_multiplier,
                },
            )
        except ValueError as error:
            _LOGGER.error("Signal construction failed for %s: %s", inference.symbol, error)
            return None

    @staticmethod
    def _record(
        checks: list[dict[str, Any]],
        rule: str,
        passed: bool,
        detail: str,
    ) -> None:
        """Append one rule evaluation to the audit trail."""
        checks.append({"rule": rule, "passed": passed, "detail": detail})

    @staticmethod
    def _reject(
        inference: ModelInferenceResult,
        rule: str,
        reason: str,
        checks: list[dict[str, Any]],
        verdict: DecisionVerdict = DecisionVerdict.NO_TRADE,
    ) -> DecisionResult:
        """Build the rejection result and log it at DEBUG level."""
        _LOGGER.debug("%s -> %s (%s)", inference.symbol, verdict.value, rule)
        return DecisionResult(
            symbol=inference.symbol,
            verdict=verdict,
            rule_triggered=rule,
            reason=reason,
            signal=None,
            inference=inference,
            checks=checks,
        )
