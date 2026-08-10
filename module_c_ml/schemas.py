"""Pydantic contracts between the ML subsystem, the Decision Engine and execution.

These schemas are the system's narrow waist.  Module C may only speak to Module D
through :class:`ModelInferenceResult`, and Module D may only speak to Module E
through :class:`TradeSignal`.  Because every field is validated and bounded
(leverage is capped at 10, probabilities must lie in ``[0, 1]``, stop distances
must be positive), an entire class of "the model emitted nonsense and we sent it
to the exchange" bugs is structurally impossible.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator, model_validator

from module_b_features.labeler import LABEL_ORDER, LONG_LABELS, SHORT_LABELS


class TradeAction(str, Enum):
    """The directional instruction a signal carries."""

    LONG = "LONG"
    SHORT = "SHORT"
    NO_TRADE = "NO_TRADE"


class DecisionVerdict(str, Enum):
    """Outcome of one Decision Engine evaluation."""

    EXECUTE = "EXECUTE"
    NO_TRADE = "NO_TRADE"
    BLOCKED = "BLOCKED"


class ModelSource(str, Enum):
    """Where a prediction came from - a fitted booster or the documented fallback."""

    TRAINED = "TRAINED"
    HEURISTIC = "HEURISTIC"
    UNAVAILABLE = "UNAVAILABLE"


def _new_decision_id() -> str:
    """Generate the UUID that ties a decision to its audit row and its trade."""
    return str(uuid.uuid4())


def _utcnow() -> datetime:
    """Timezone-aware UTC now."""
    return datetime.now(tz=timezone.utc)


class DirectionPrediction(BaseModel):
    """Model 1 output: the probability distribution over the direction classes."""

    model_config = ConfigDict(frozen=True)

    probabilities: dict[str, float] = Field(
        description="Probability per class label; keys are a subset of LABEL_ORDER."
    )
    source: ModelSource = Field(default=ModelSource.TRAINED)

    #: Raw two-stage cascade outputs (NOT the multiplied joint probability
    #: in `probabilities`). Lets the Decision Engine gate "is this bar worth
    #: trading" and "which way, how sure" independently, instead of
    #: requiring their product to clear one bar - which silently discards a
    #: confident direction call whenever the gate alone reads under 0.5.
    trade_probability: float = Field(default=0.5, ge=0.0, le=1.0)
    direction_given_trade_probability: float = Field(default=0.5, ge=0.0, le=1.0)

    @field_validator("probabilities")
    @classmethod
    def _validate_distribution(cls, value: dict[str, float]) -> dict[str, float]:
        """Reject unknown classes and renormalise a distribution that drifted."""
        if not value:
            raise ValueError("probabilities must not be empty")
        unknown: set[str] = set(value) - set(LABEL_ORDER)
        if unknown:
            raise ValueError(f"unknown direction classes: {sorted(unknown)}")
        if any(probability < 0.0 for probability in value.values()):
            raise ValueError("probabilities must be non-negative")

        total: float = sum(value.values())
        if total <= 0.0:
            raise ValueError("probabilities must sum to a positive value")
        return {name: probability / total for name, probability in value.items()}

    @computed_field  # type: ignore[prop-decorator]
    @property
    def long_probability(self) -> float:
        """Summed probability mass of every LONG success class."""
        return sum(value for name, value in self.probabilities.items() if name in LONG_LABELS)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def short_probability(self) -> float:
        """Summed probability mass of every SHORT success class."""
        return sum(value for name, value in self.probabilities.items() if name in SHORT_LABELS)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def no_trade_probability(self) -> float:
        """Probability mass assigned to "do nothing"."""
        return max(0.0, 1.0 - self.long_probability - self.short_probability)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def action(self) -> TradeAction:
        """The direction implied by the largest aggregated probability mass."""
        scores: dict[TradeAction, float] = {
            TradeAction.LONG: self.long_probability,
            TradeAction.SHORT: self.short_probability,
            TradeAction.NO_TRADE: self.no_trade_probability,
        }
        return max(scores, key=lambda key: scores[key])

    @computed_field  # type: ignore[prop-decorator]
    @property
    def confidence(self) -> float:
        """Probability mass of the winning aggregated action."""
        return max(self.long_probability, self.short_probability, self.no_trade_probability)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def directional_margin(self) -> float:
        """Edge of the winning direction over the opposing one."""
        return abs(self.long_probability - self.short_probability)


class EntryPrediction(BaseModel):
    """Model 2 output: is *this* candle close the right moment to act?"""

    model_config = ConfigDict(frozen=True)

    probability: float = Field(ge=0.0, le=1.0)
    should_enter: bool
    source: ModelSource = Field(default=ModelSource.TRAINED)
    reason: str = Field(default="")


class ExitParameters(BaseModel):
    """Model 3 output: dynamic trade-management geometry, all as fractions."""

    model_config = ConfigDict(frozen=True)

    take_profit_pct: float = Field(gt=0.0, le=1.0)
    stop_loss_pct: float = Field(gt=0.0, le=1.0)
    trailing_activation_pct: float = Field(ge=0.0, le=1.0)
    trailing_distance_pct: float = Field(gt=0.0, le=1.0)
    source: ModelSource = Field(default=ModelSource.TRAINED)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def reward_risk_ratio(self) -> float:
        """Gross reward-to-risk implied by the barrier geometry."""
        return self.take_profit_pct / self.stop_loss_pct

    @model_validator(mode="after")
    def _validate_geometry(self) -> "ExitParameters":
        """A trailing stop wider than the take profit would never arm."""
        if self.trailing_activation_pct > self.take_profit_pct:
            raise ValueError("trailing activation cannot exceed the take-profit distance")
        return self


class RiskAllocation(BaseModel):
    """Model 4 output: leverage and capital allocation, or an explicit abort."""

    model_config = ConfigDict(frozen=True)

    leverage: int = Field(ge=0, le=10, description="0 means 'do not take this trade'.")
    capital_allocation_pct: float = Field(ge=0.0, le=1.0)
    risk_score: float = Field(ge=0.0, le=1.0)
    risk_tier: str = Field(default="UNKNOWN")
    abort_reason: str = Field(default="")
    source: ModelSource = Field(default=ModelSource.TRAINED)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def is_abort(self) -> bool:
        """``True`` when the risk head vetoed the trade."""
        return self.leverage <= 0 or self.capital_allocation_pct <= 0.0


class ModelInferenceResult(BaseModel):
    """The standardised bundle Module C hands to Module D."""

    model_config = ConfigDict(frozen=True)

    symbol: str
    timestamp: int = Field(description="Open time (ms) of the candle that was scored.")
    close_price: float = Field(gt=0.0)
    direction: DirectionPrediction
    entry: EntryPrediction
    exit_params: ExitParameters
    risk: RiskAllocation
    feature_snapshot: dict[str, float] = Field(default_factory=dict)
    latency_ms: float = Field(default=0.0, ge=0.0)
    model_versions: dict[str, str] = Field(default_factory=dict)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def any_fallback(self) -> bool:
        """``True`` when at least one head fell back to its heuristic."""
        return any(
            component.source is not ModelSource.TRAINED
            for component in (self.direction, self.entry, self.exit_params, self.risk)
        )


class TradeSignal(BaseModel):
    """The single, immutable artefact Module E is allowed to act on."""

    model_config = ConfigDict(frozen=True)

    decision_id: str = Field(default_factory=_new_decision_id)
    created_at: datetime = Field(default_factory=_utcnow)
    symbol: str
    action: TradeAction
    reference_price: float = Field(gt=0.0, description="Close price the decision was made on.")

    leverage: int = Field(ge=1, le=10)
    capital_allocation_pct: float = Field(gt=0.0, le=1.0)

    take_profit: float = Field(gt=0.0, description="Absolute take-profit price.")
    stop_loss: float = Field(gt=0.0, description="Absolute stop-loss price.")
    trailing_trigger: float = Field(gt=0.0, description="Price at which trailing arms.")
    trailing_distance_pct: float = Field(gt=0.0, le=1.0)

    take_profit_pct: float = Field(gt=0.0, le=1.0)
    stop_loss_pct: float = Field(gt=0.0, le=1.0)

    confidence: float = Field(ge=0.0, le=1.0)
    risk_tier: str = Field(default="UNKNOWN")
    candle_timestamp: int = Field(default=0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_price_geometry(self) -> "TradeSignal":
        """Verify the barriers sit on the correct side of the entry price.

        A long whose stop is above its entry, or whose take-profit is below it,
        is a bug that would otherwise be discovered by losing money.
        """
        if self.action is TradeAction.NO_TRADE:
            raise ValueError("a TradeSignal must carry a directional action")

        if self.action is TradeAction.LONG:
            if self.take_profit <= self.reference_price:
                raise ValueError("long take-profit must sit above the entry price")
            if self.stop_loss >= self.reference_price:
                raise ValueError("long stop-loss must sit below the entry price")
            if self.trailing_trigger <= self.reference_price:
                raise ValueError("long trailing trigger must sit above the entry price")
        else:
            if self.take_profit >= self.reference_price:
                raise ValueError("short take-profit must sit below the entry price")
            if self.stop_loss <= self.reference_price:
                raise ValueError("short stop-loss must sit above the entry price")
            if self.trailing_trigger >= self.reference_price:
                raise ValueError("short trailing trigger must sit below the entry price")
        return self

    @computed_field  # type: ignore[prop-decorator]
    @property
    def side(self) -> str:
        """ccxt order side implied by the action."""
        return "buy" if self.action is TradeAction.LONG else "sell"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def reduce_side(self) -> str:
        """ccxt order side that closes the position."""
        return "sell" if self.action is TradeAction.LONG else "buy"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def reward_risk_ratio(self) -> float:
        """Reward-to-risk implied by the absolute barrier prices."""
        risk: float = abs(self.reference_price - self.stop_loss)
        reward: float = abs(self.take_profit - self.reference_price)
        return 0.0 if risk <= 0.0 else reward / risk

    def notional_for(self, equity: float) -> float:
        """Position notional in USDT for a given account equity."""
        return max(0.0, equity) * self.capital_allocation_pct * float(self.leverage)

    def margin_for(self, equity: float) -> float:
        """Initial margin committed for a given account equity."""
        return max(0.0, equity) * self.capital_allocation_pct

    def quantity_for(self, equity: float) -> float:
        """Contract quantity in base units for a given account equity."""
        return self.notional_for(equity) / self.reference_price


class DecisionResult(BaseModel):
    """Everything the Decision Engine produced for one symbol in one cycle.

    A ``NO_TRADE`` verdict is just as important as an ``EXECUTE`` one: the Audit
    Engine persists both, together with the exact rule that fired.
    """

    model_config = ConfigDict(frozen=True)

    decision_id: str = Field(default_factory=_new_decision_id)
    symbol: str
    verdict: DecisionVerdict
    rule_triggered: str
    reason: str
    signal: TradeSignal | None = Field(default=None)
    inference: ModelInferenceResult | None = Field(default=None)
    evaluated_at: datetime = Field(default_factory=_utcnow)
    checks: list[dict[str, Any]] = Field(default_factory=list)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def action(self) -> TradeAction:
        """The action to take (``NO_TRADE`` unless a signal was produced)."""
        return self.signal.action if self.signal is not None else TradeAction.NO_TRADE

    @computed_field  # type: ignore[prop-decorator]
    @property
    def is_executable(self) -> bool:
        """``True`` only for an ``EXECUTE`` verdict carrying a validated signal."""
        return self.verdict is DecisionVerdict.EXECUTE and self.signal is not None
