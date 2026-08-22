#!/usr/bin/env python3
"""Re-verify every audit finding (P1-P30) against the working tree.

The audit in ``docs/BRANCH_AUDIT_archive-features-and-data-recovery.md`` is
evidence-derived: each finding is a statement about a specific commit and a
specific ML diagnostic run.  Both go stale.  Applying a remembered patch to a
finding that has already been fixed - or that now reproduces differently - is
how an audit turns into damage, so this script re-checks the *mechanism* of
every finding directly against the source before anything is changed.

Each check is deliberately narrow: it looks for the exact construct the finding
names, not for a vague pattern.  A check that stops matching is a signal to go
read the code, not to silently skip the fix.

Run from the repository root::

    python scripts/reverify_audit_findings.py                 # source checks
    python scripts/reverify_audit_findings.py --json PATH     # + diagnostic checks

Exit code is 0 when every finding resolved to a definite state (REPRODUCES or
FIXED), 1 when any check could not be evaluated - an unevaluable check is a
gap in the harness, and silently passing it would defeat the point.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

REPO_ROOT = Path(__file__).resolve().parent.parent

REPRODUCES = "REPRODUCES"
FIXED = "FIXED"
UNEVALUABLE = "UNEVALUABLE"


@dataclass
class Finding:
    """One audit finding and the evidence that decides whether it still holds."""

    pid: str
    title: str
    #: Returns (state, detail).  Raising is treated as UNEVALUABLE.
    check: Callable[[], tuple[str, str]]
    #: Only meaningful when a diagnostic JSON is supplied.
    needs_json: bool = False


@dataclass
class Result:
    pid: str
    title: str
    state: str
    detail: str


_source_cache: dict[str, str] = {}


def src(relative: str) -> str:
    """Read a source file from the working tree, cached."""
    if relative not in _source_cache:
        _source_cache[relative] = (REPO_ROOT / relative).read_text(encoding="utf-8")
    return _source_cache[relative]


def exists(relative: str) -> bool:
    return (REPO_ROOT / relative).exists()


def count(relative: str, pattern: str) -> int:
    return len(re.findall(pattern, src(relative)))


def line_of(relative: str, pattern: str) -> int | None:
    """1-indexed line number of the first match, or None."""
    for index, line in enumerate(src(relative).splitlines(), start=1):
        if re.search(pattern, line):
            return index
    return None


def before(relative: str, first: str, second: str) -> bool | None:
    """True when ``first`` appears before ``second`` in the file."""
    a, b = line_of(relative, first), line_of(relative, second)
    if a is None or b is None:
        return None
    return a < b


def verdict(condition: bool, detail: str) -> tuple[str, str]:
    return (REPRODUCES if condition else FIXED, detail)


# ---------------------------------------------------------------------------
# Source-level checks
# ---------------------------------------------------------------------------
ML = "module_c_ml/ml_models.py"
DE = "module_c_ml/decision_engine.py"
MET = "module_c_ml/metrics.py"
BT = "module_e_execution/backtester.py"
MODELS = "module_e_execution/models.py"
PROC = "module_b_features/processor.py"
FEAT = "module_b_features/features.py"
IND = "module_b_features/indicators.py"
LAB = "module_b_features/labeler.py"
DIAG = "module_f_panel/diagnostics.py"
ARCH = "module_a_data/archive_loader.py"
CFG = "config/settings.py"
MAIN = "main.py"


def check_p1() -> tuple[str, str]:
    """Direction metrics computed before _calibrate_cascade replaces self._model."""
    metrics_first = before(ML, r"full_metrics = ml_metrics\.direction_metrics", r"self\._calibrate_cascade\(")
    if metrics_first is None:
        raise LookupError("could not locate both the metrics call and _calibrate_cascade")
    m_line = line_of(ML, r"full_metrics = ml_metrics\.direction_metrics")
    c_line = line_of(ML, r"self\._calibrate_cascade\(")
    guarded = count(ML, r"_assert_metrics_describe_this_model")
    return verdict(
        metrics_first or guarded == 0,
        f"metrics at L{m_line}, calibration at L{c_line}, save-time identity guard={guarded > 0}",
    )


def check_p2() -> tuple[str, str]:
    """_to_dataset drops on labels only, so degenerate rows survive into training."""
    labels_only = count(PROC, r"dropna\(subset=\[\"label\", \"target_risk_score\"\]\)")
    guarded = count(PROC, r"realized_vol_12_is_zero")
    return verdict(
        labels_only > 0 and guarded == 0,
        f"labels-only dropna={labels_only}, zero-vol filter present={guarded > 0}",
    )


def check_p3() -> tuple[str, str]:
    """No nullity gate: nothing refuses a feature that is ~entirely NaN."""
    gate = count(DIAG, r"null_counts_by_feature") if exists(DIAG) else 0
    critical_gate = count(DIAG, r"effectively empty|null_rate.*0\.98|dead feature") if exists(DIAG) else 0
    return verdict(critical_gate == 0, f"null_counts referenced={gate}, nullity gate={critical_gate}")


def check_p4() -> tuple[str, str]:
    """No span guard before isotonic calibration is wired onto a stage."""
    guard = count(ML, r"raw_probability_span|np\.ptp\(")
    return verdict(guard == 0, f"raw-span guard occurrences={guard}")


def check_p5() -> tuple[str, str]:
    """_simulate resolves barriers before filling, so the entry bar is skipped."""
    fill_first = before(BT, r"1\. Fill signals", r"2\. Resolve barriers")
    if fill_first is None:
        raise LookupError("could not locate the numbered simulate phases")
    return verdict(not fill_first, "fill precedes resolve -> the entry bar is tested")


def check_p6() -> tuple[str, str]:
    """Backtest hardcodes a GREEN risk-guard state."""
    hardcoded = count(BT, r'risk_guard_state="GREEN"')
    real_guard = count(BT, r"RiskGuard\(")
    return verdict(
        hardcoded > 0 and real_guard == 0,
        f'hardcoded GREEN={hardcoded}, RiskGuard instantiated={real_guard}',
    )


def check_p7() -> tuple[str, str]:
    """Backtester iterates symbols in dict order instead of using evaluate_many."""
    dict_order = count(BT, r"for symbol, rows in indexed\.items\(\)")
    uses_many = count(BT, r"evaluate_many")
    return verdict(
        dict_order > 0 and uses_many == 0,
        f"dict-order loop={dict_order}, evaluate_many calls={uses_many}",
    )


def check_p8() -> tuple[str, str]:
    """Risk head trains only on realised winners."""
    filt = count(ML, r"usable: pd\.Series = dataset\.direction_target != LabelClass\.NO_TRADE_OR_FAIL\.value")
    return verdict(filt > 0, f"outcome-conditioned training filter occurrences={filt}")


def check_p9() -> tuple[str, str]:
    """Recency half-life is short relative to the training window."""
    match = re.search(r"recency_half_life_days: float = Field\(default=([0-9.]+)", src(CFG))
    train = re.search(r"train_months: float = Field\(default=([0-9.]+)", src(CFG))
    if not match or not train:
        raise LookupError("could not read recency_half_life_days / train_months defaults")
    half_life, months = float(match.group(1)), float(train.group(1))
    span_days = months * 30.0
    return verdict(
        half_life > 0 and half_life < span_days / 4.0,
        f"half_life={half_life}d vs train span~{span_days:.0f}d (quarter={span_days / 4:.0f}d)",
    )


def check_p10() -> tuple[str, str]:
    """Production calibrator is fit on the whole validation block."""
    whole_block = count(ML, r"_fit_production_calibrator\(\s*gate_estimator,\s*validation_features")
    return verdict(whole_block > 0, f"calibrator fit on full validation block={whole_block}")


def check_p11() -> tuple[str, str]:
    """_select_recommended_threshold uses `floor` as a fallback, never as a bound."""
    body = re.search(
        r"def _select_recommended_threshold\(.*?\n(.*?)\n    def ", src(ML), re.S
    )
    if not body:
        raise LookupError("could not isolate _select_recommended_threshold")
    text = body.group(1)
    bounded = "configured_floor)" in text and "max(" in text
    two_sided = count(MET, r"def direction_confidence_sweep")
    return verdict(
        not (bounded and two_sided),
        f"return bounded by the configured floor={bounded}, two-sided sweep={two_sided > 0}",
    )


def check_p12() -> tuple[str, str]:
    """oos_fraction is asserted as a literal rather than measured."""
    computed = count(MAIN, r"bars_overlapping_train_or_validation")
    pinned = count(MAIN, r"start_ms=split\.test_start_ms")
    return verdict(
        not (computed and pinned),
        f"oos overlap measured={computed > 0}, replay window pinned={pinned > 0}",
    )


def check_p13() -> tuple[str, str]:
    """Sortino divides by the count of negative periods, not all periods."""
    wrong = count(BT, r"downside: np\.ndarray = returns\[returns < 0\.0\]")
    return verdict(wrong > 0, f"downside-only denominator occurrences={wrong}")


def check_p14() -> tuple[str, str]:
    """Position.opened_at defaults to wall clock and the backtester never overrides it."""
    wallclock = count(MODELS, r"opened_at: datetime = field\(default_factory=_utcnow\)")
    overridden = count(BT, r"opened_at")
    return verdict(
        wallclock > 0 and overridden == 0,
        f"wall-clock default={wallclock}, backtester sets opened_at={overridden}",
    )


def check_p15() -> tuple[str, str]:
    """Trailing target is a fixed multiple of the TP target."""
    # The labeler defining trailing as 0.5*tp is the *definition*, not the bug.
    # The defect was training a separate regressor to rediscover it, and
    # reporting that regressor's R2 as an independent measurement.
    regressor = count(ML, r'_TARGETS: Final\[tuple\[str, \.\.\.\]\] = \("target_tp_pct", "target_sl_pct", "target_trailing_pct"\)')
    derived = count(ML, r"_TRAILING_TP_FRACTION")
    return verdict(
        regressor > 0 or derived == 0,
        f"trailing regressor trained={regressor > 0}, derived directly={derived > 0}",
    )


def check_p16() -> tuple[str, str]:
    """Calibration adopted on a bare strict inequality, with no effect-size floor."""
    bare = count(MET, r"improved: bool = calibrated_logloss < raw_logloss")
    bare += count(ML, r"improved: bool = calibrated_logloss < raw_logloss")
    return verdict(bare > 0, f"unguarded strict-inequality adoptions={bare}")


def check_p17() -> tuple[str, str]:
    """biggest_data_problem reads only symbol_exclusions_total."""
    if not exists(DIAG):
        raise LookupError("diagnostics.py absent")
    block = re.search(r'"biggest_data_problem": \((.*?)\),\n', src(DIAG), re.S)
    if not block:
        raise LookupError("could not isolate biggest_data_problem")
    text = block.group(1)
    reads_only_exclusions = "symbol_exclusions_total" in text and "null" not in text and "drift" not in text
    return verdict(reads_only_exclusions, f"reads only symbol_exclusions_total={reads_only_exclusions}")


def check_p18() -> tuple[str, str]:
    """Every AVAILABLE return path of the walk-forward summary says 'none measured'."""
    body = re.search(
        r"def _walk_forward_problem_summary\(.*?\n(.*?)\n\n", src(DIAG), re.S
    )
    if not body:
        raise LookupError("could not isolate _walk_forward_problem_summary")
    text = body.group(1)
    returns = re.findall(r"return f?\"([^\"]*)\"", text)
    available = [r for r in returns if "not available" not in r]
    all_none = bool(available) and all(r.startswith("none measured") for r in available)
    return verdict(all_none, f"AVAILABLE-path returns={available!r}")


def check_p19() -> tuple[str, str]:
    """Relaxed settings constructed fresh, discarding operator configuration."""
    fresh = count(MAIN, r"_RELAXED_DECISION_SETTINGS: Final\[DecisionSettings\] = DecisionSettings\(")
    derived = count(MAIN, r"decision\.model_copy\(update=")
    return verdict(fresh > 0 and derived == 0, f"fresh construction={fresh}, derived={derived}")


def check_p20() -> tuple[str, str]:
    """Only the first failing rule is recorded."""
    first_only = count(BT, r"rejection_breakdown\[decision\.rule_triggered\]")
    independent = count(BT, r"rejection_breakdown_independent|for check in decision\.checks")
    return verdict(
        first_only > 0 and independent == 0,
        f"first-match recording={first_only}, independent breakdown={independent}",
    )


def check_p21() -> tuple[str, str]:
    """R4 message prints the configured floor rather than the applied cutoff."""
    prints_config = count(DE, r"\{self\._config\.min_entry_probability:\.3f\}")
    carries = count("module_c_ml/schemas.py", r"threshold")
    return verdict(
        prints_config > 0,
        f"R4 prints config value={prints_config}, EntryPrediction carries threshold={carries}",
    )


def check_p22() -> tuple[str, str]:
    """_fill hardcodes the trailing trigger at half the take-profit."""
    hardcoded = count(BT, r"signal\.take_profit_pct \* 0\.5")
    return verdict(hardcoded > 0, f"hardcoded 0.5*TP trailing trigger occurrences={hardcoded}")


def check_p23() -> tuple[str, str]:
    """Training keeps rows that inference and the backtester drop."""
    train_labels_only = count(PROC, r"dropna\(subset=\[\"label\", \"target_risk_score\"\]\)")
    infer_required = count(BT, r"subset=list\(REQUIRED_FEATURE_COLUMNS\)")
    return verdict(
        train_labels_only > 0 and infer_required > 0,
        f"training labels-only={train_labels_only}, inference REQUIRED gate={infer_required}",
    )


def check_p24() -> tuple[str, str]:
    """Confidence sweep carries no degeneracy flag."""
    flag = count(MET, r"is_degenerate|distinct_predicted_classes")
    return verdict(flag == 0, f"degeneracy flag occurrences={flag}")


def check_p25() -> tuple[str, str]:
    """DX is computed without requiring both DIs to be present, and NaN is filled with 0."""
    fills = count(IND, r"directional_index\.fillna\(0\.0\)")
    both_present = count(IND, r"both_present|_MIN_DI")
    return verdict(
        fills > 0 and both_present == 0,
        f"fillna(0.0) into smoother={fills}, both-DI guard={both_present}",
    )


def check_p26() -> tuple[str, str]:
    """Archive loader uses a total timeout and materialises the whole CSV."""
    total_timeout = count(ARCH, r"ClientTimeout\(total=self\._settings\.data\.archive_timeout_seconds\)")
    slurp = count(ARCH, r"raw: bytes = handle\.read\(\)")
    return verdict(
        total_timeout > 0 or slurp > 0,
        f"total-timeout={total_timeout}, whole-file read={slurp}",
    )


def check_p27() -> tuple[str, str]:
    """class_distribution has no split awareness."""
    signature = re.search(r"def class_distribution\(self([^)]*)\)", src(PROC))
    if not signature:
        raise LookupError("could not locate class_distribution")
    takes_index = "index" in signature.group(1)
    return verdict(not takes_index, f"class_distribution signature=({signature.group(1).strip()})")


def check_p28() -> tuple[str, str]:
    """Risk metrics measured on unclipped estimator output."""
    clipped = count(ML, r"np\.clip\(raw_predictions, 0\.0, 1\.0\)")
    rate = count(ML, r"clipped_prediction_rate")
    return verdict(
        not (clipped and rate),
        f"risk metrics clipped to the documented domain={clipped > 0}, clip rate reported={rate > 0}",
    )


def check_p29() -> tuple[str, str]:
    """Duplicates counted but never dropped."""
    counted = count(PROC, r"duplicate_feature_rows: int = int\(usable\.duplicated")
    dropped = count(PROC, r"drop_duplicates")
    return verdict(counted > 0 and dropped == 0, f"counted={counted}, dropped={dropped}")


def check_p30() -> tuple[str, str]:
    """Four dead-code items: risk veto, dead config, missing snapshot key, liquidation slippage."""
    items = []
    # (a) The veto is unreachable through evaluate() because R1b tests the same
    # quantity first. Keeping it is a defensible choice for callers that bypass
    # the engine - but only if that is written down, so the next reader does not
    # spend an afternoon working out why it never appears in the logs.
    if count(ML, r"if direction_confidence < decision\.min_direction_given_trade_confidence"):
        if not count(ML, r"Defence in depth, not a live gate"):
            items.append("unreachable-risk-veto-undocumented")
    # (b) A configurable nothing reads.
    consumers = 0
    for rel in (DE, BT, "module_e_execution/executor.py", "module_e_execution/paper_trader.py"):
        if exists(rel):
            consumers += count(rel, r"max_positions_per_symbol")
    if consumers == 0 and count(CFG, r"max_positions_per_symbol"):
        items.append("dead-max_positions_per_symbol")
    # (c) R5's stop-vs-labelled-ATR telemetry needs atr_pct in the snapshot.
    if not count(BT, r'"atr_pct"'):
        items.append("missing-atr_pct-in-snapshot")
    # (d) Liquidation is the worst fill, not the best.
    if count(BT, r"if reason is CloseReason\.LIQUIDATION:\n            exit_price: float = raw_price"):
        items.append("liquidation-exempt-from-slippage")
    return verdict(bool(items), f"remaining items={items or 'none'}")


# ---------------------------------------------------------------------------
# Diagnostic-JSON checks (numbers, not code)
# ---------------------------------------------------------------------------
def json_checks(report: dict) -> list[Result]:
    """Re-measure the numeric evidence the audit quoted."""
    out: list[Result] = []

    def add(pid: str, title: str, condition: bool, detail: str) -> None:
        out.append(Result(pid, title, REPRODUCES if condition else FIXED, detail))

    dataset = report.get("dataset", {})
    total = dataset.get("valid_samples", 0)
    nulls = dataset.get("null_counts_by_feature", {})
    dead = sorted(name for name, n in nulls.items() if total and n / total > 0.98)
    add("P3", "features ~entirely null", bool(dead), f"{len(dead)} dead: {', '.join(dead) or 'none'}")

    drift = (report.get("features", {}).get("drift", {}) or {}).get("most_drifted_features", [])
    worst = drift[0] if drift else None
    add(
        "P2",
        "train/validation feature drift",
        bool(worst) and worst.get("mean_shift_in_train_std", 0) > 1.0,
        f"worst={worst.get('feature')} at {worst.get('mean_shift_in_train_std'):.2f} train-sigma"
        if worst
        else "no drift block",
    )

    direction = report.get("direction", {})
    predicted = (direction.get("metrics", {}) or {}).get("predicted_class_distribution", {})
    zero_classes = [k for k, v in predicted.items() if v == 0]
    add(
        "P1",
        "a class is never predicted",
        bool(zero_classes),
        f"never predicted: {', '.join(zero_classes) or 'none'}",
    )

    sweep = direction.get("direction_threshold_sweep", []) or []
    at_half = [r for r in sweep if abs(r.get("threshold", 0) - 0.5) < 1e-9]
    add(
        "P4",
        "stage-2 output never crosses 0.5",
        bool(at_half) and at_half[0].get("signals", 1) == 0,
        f"signals at t=0.50: {at_half[0].get('signals') if at_half else 'n/a'}",
    )

    folds = (report.get("walk_forward", {}) or {}).get("folds", [])
    accs = [f.get("accuracy") for f in folds if f.get("accuracy") is not None]
    monotone = len(accs) > 2 and all(a > b for a, b in zip(accs, accs[1:]))
    add(
        "P18",
        "walk-forward degrades monotonically",
        monotone,
        f"fold accuracies={[round(a, 4) for a in accs]}",
    )

    metrics = (report.get("backtest", {}) or {}).get("metrics", {})
    sharpe, sortino = metrics.get("sharpe_ratio"), metrics.get("sortino_ratio")
    add(
        "P13",
        "Sortino below Sharpe on right-skewed returns",
        bool(sharpe and sortino and sortino < sharpe),
        f"sharpe={sharpe}, sortino={sortino}",
    )

    risk_pred = ((report.get("risk", {}) or {}).get("metrics", {}) or {}).get("prediction_stats", {})
    risk_target = ((report.get("risk", {}) or {}).get("metrics", {}) or {}).get("target_stats", {})
    add(
        "P8",
        "risk head cannot express a bad trade",
        bool(risk_pred) and risk_pred.get("min", 0) > risk_target.get("min", 0) * 3,
        f"prediction min={risk_pred.get('min')}, target min={risk_target.get('min')}",
    )
    add(
        "P28",
        "risk predictions exceed their documented domain",
        bool(risk_pred) and risk_pred.get("max", 0) > 1.0,
        f"prediction max={risk_pred.get('max')}",
    )

    exit_metrics = (report.get("exit", {}) or {}).get("metrics", {})
    tp_r2 = (exit_metrics.get("target_tp_pct", {}) or {}).get("r2")
    tr_r2 = (exit_metrics.get("target_trailing_pct", {}) or {}).get("r2")
    add(
        "P15",
        "trailing target is an affine image of the TP target",
        bool(tp_r2 is not None and tp_r2 == tr_r2),
        f"tp r2={tp_r2}, trailing r2={tr_r2}",
    )

    dupes = dataset.get("duplicate_feature_rows", 0)
    coverage = (dataset.get("split_coverage_pct", {}) or {}).get("validation", {})
    over = coverage.get("rows", 0) > coverage.get("capacity_rows", 1)
    add("P29", "duplicate rows present", bool(dupes) or over, f"duplicates={dupes}, rows>capacity={over}")

    summary = report.get("ai_summary", {})
    add(
        "P17",
        "summary blind to data problems",
        summary.get("biggest_data_problem") == "none measured" and bool(dead),
        f"status={summary.get('overall_status')}, data problem={summary.get('biggest_data_problem')!r}",
    )
    return out


FINDINGS: list[Finding] = [
    Finding("P1", "Direction metrics measure a discarded model", check_p1),
    Finding("P2", "Degenerate rows survive into training", check_p2),
    Finding("P3", "No nullity gate for dead features", check_p3),
    Finding("P4", "No span guard before isotonic calibration", check_p4),
    Finding("P5", "Entry bar never tested against barriers", check_p5),
    Finding("P6", "Backtest bypasses the Risk Guard", check_p6),
    Finding("P7", "Backtest selection differs from live", check_p7),
    Finding("P8", "Risk head trained on realised winners", check_p8),
    Finding("P9", "Recency half-life shrinks the window", check_p9),
    Finding("P10", "Validation block overloaded", check_p10),
    Finding("P11", "Threshold floor is only a fallback", check_p11),
    Finding("P12", "oos_fraction asserted, not measured", check_p12),
    Finding("P13", "Sortino denominator wrong", check_p13),
    Finding("P14", "opened_at is wall-clock in backtest", check_p14),
    Finding("P15", "Trailing target derived from TP target", check_p15),
    Finding("P16", "Calibration adopted on bare inequality", check_p16),
    Finding("P17", "Summary blind to data problems", check_p17),
    Finding("P18", "Walk-forward summary cannot report", check_p18),
    Finding("P19", "Relaxed settings discard configuration", check_p19),
    Finding("P20", "Only first failing rule recorded", check_p20),
    Finding("P21", "R4 logs a threshold it did not apply", check_p21),
    Finding("P22", "Trailing trigger hardcoded in _fill", check_p22),
    Finding("P23", "Train/serve population mismatch", check_p23),
    Finding("P24", "Sweep has no degeneracy flag", check_p24),
    Finding("P25", "DX saturates; NaN filled with zero", check_p25),
    Finding("P26", "Archive loader transport unusable", check_p26),
    Finding("P27", "class_distribution has no split awareness", check_p27),
    Finding("P28", "Risk metrics on unclipped predictions", check_p28),
    Finding("P29", "Duplicates counted, never dropped", check_p29),
    Finding("P30", "Dead code and unreachable logic", check_p30),
]


def run(diagnostic: Path | None) -> int:
    results: list[Result] = []
    for finding in FINDINGS:
        try:
            state, detail = finding.check()
        except Exception as error:  # noqa: BLE001 - an unevaluable check must be visible, not fatal
            state, detail = UNEVALUABLE, f"{type(error).__name__}: {error}"
        results.append(Result(finding.pid, finding.title, state, detail))

    width = max(len(r.title) for r in results)
    print("=" * (width + 34))
    print("SOURCE CHECKS (working tree)")
    print("=" * (width + 34))
    for r in results:
        print(f"{r.pid:<4} {r.state:<12} {r.title:<{width}}  {r.detail}")

    if diagnostic is not None:
        report = json.loads(diagnostic.read_text(encoding="utf-8"))
        json_results = json_checks(report)
        print()
        print("=" * (width + 34))
        print(f"DIAGNOSTIC CHECKS ({diagnostic.name})")
        print("=" * (width + 34))
        for r in json_results:
            print(f"{r.pid:<4} {r.state:<12} {r.title:<{width}}  {r.detail}")
        results += json_results

    reproduces = sum(1 for r in results if r.state == REPRODUCES)
    fixed = sum(1 for r in results if r.state == FIXED)
    unevaluable = sum(1 for r in results if r.state == UNEVALUABLE)
    print()
    print(f"summary: {reproduces} reproduce, {fixed} fixed, {unevaluable} unevaluable")
    return 1 if unevaluable else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, default=None, help="ML diagnostic JSON export to re-measure")
    args = parser.parse_args()
    return run(args.json)


if __name__ == "__main__":
    sys.exit(main())
