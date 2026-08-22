# Fix options — `claude/archive-features-and-data-recovery`

Companion to `docs/BRANCH_AUDIT_archive-features-and-data-recovery.md`. The audit establishes *what is
wrong and why*. This file gives, for each of the 30 findings, the **options that would actually solve
it**, what each option costs, what it breaks, which one to pick, and **how to prove the fix worked**.

**Every finding in the audit was re-verified against `git show c2968d8:<path>` and against the
diagnostic JSON before this file was written.** All 30 hold. One mechanism description was refined —
see [P25](#p25), where the ADX denominators turn out to be already guarded and the real cause is
one-sided directional movement. The fix that follows from that is different from the one the audit
proposed, so use this file's version.

## How to read the options

Each problem gets 2–4 options labelled **A / B / C**, then a **Pick**. The options are genuinely
different approaches, not a graded scale of effort — usually one is a local patch, one is structural,
and one is a detector that prevents recurrence without fixing the instance. Many problems want a
*combination* (fix + detector), and the Pick says so.

Three qualifiers appear throughout:

- **Cost** — rough implementation size. `1 line` / `small` (< 30 lines) / `medium` (one function or
  file) / `large` (cross-module, or requires a retrain to evaluate).
- **Breaks** — what stops working, or what number changes, when you apply it. Several of these fixes
  make the reported results *worse*. That is the intended direction; the current numbers are wrong in
  the optimistic direction.
- **Verify** — a concrete check. If you cannot run it, the fix is not done.

## Before you apply anything

Read [Ordering and dependencies](#ordering) first. Several fixes are meaningless or actively
misleading if applied out of order — in particular, **nothing about model quality can be judged until
[P1](#p1) lands and the diagnostic is re-run**, because until then every model metric in the report
describes an artifact that was thrown away.

---

## Contents

| # | Problem | Severity | Pick (short) |
|---|---|---|---|
| [P1](#p1) | Metrics measure a discarded model | Blocker | Score through `predict()` after calibration + identity guard |
| [P2](#p2) | 72% of training rows are degenerate | Blocker | Trace upstream, add zero-range QC, filter explicitly |
| [P3](#p3) | 7 of 57 features are dead | Blocker | Drop them now + add a nullity gate |
| [P4](#p4) | Stage 2 has no discrimination | Blocker | Re-measure on clean data, then span-guard the calibrator |
| [P5](#p5) | Entry bar never tested | Critical | Fill before resolve |
| [P6](#p6) | Risk Guard bypassed | Critical | Wire a real guard, report both runs |
| [P7](#p7) | Backtest selection ≠ live selection | Critical | Call `evaluate_many` |
| [P8](#p8) | Risk/Exit trained on winners only | Critical | Fix the target, drop the filter |
| [P9](#p9) | 45-day half-life on a 372-day window | Critical | Sweep it; default to 0 meanwhile |
| [P10](#p10) | Validation does four jobs | Critical | Split into fit/score halves |
| [P11](#p11) | Auto-tuned thresholds meaningless | High | Two-sided sweep + real floor + flat-sweep refusal |
| [P12](#p12) | Backtest window drifts | High | Pin the timeline; compute `oos_fraction` |
| [P13](#p13) | Sortino denominator wrong | High | One-line fix + trade-based metrics |
| [P14](#p14) | `opened_at` is wall-clock | High | Constructor parameter + invariant |
| [P15](#p15) | Exit SL head dead, trailing head trivial | High | Delete trailing, re-scale stop head, report override rate |
| [P16](#p16) | Calibration adopted on noise | High | Effect-size threshold + multi-fold agreement |
| [P17](#p17) | `_ai_summary` blind to data | High | Check registry + lift-over-chance scoring |
| [P18](#p18) | Walk-forward summary can't report | High | Trend test + surface latest fold |
| [P19](#p19) | Relaxed run resets all settings | High | `model_copy(update=...)` + decouple sizing base |
| [P20](#p20) | Rejection breakdown not comparable | Medium | Independent breakdown from the `checks` list |
| [P21](#p21) | Entry threshold override silent | Medium | Decide precedence, carry + log the applied value |
| [P22](#p22) | Trailing trigger hardcoded | Medium | Carry the pct on the signal + parity test |
| [P23](#p23) | Train/serve population mismatch | Medium | Training gates on `REQUIRED_FEATURE_COLUMNS` |
| [P24](#p24) | Confidence sweep rewards collapse | Medium | Degeneracy flag + generic detector |
| [P25](#p25) | ADX/DI saturate | Medium | Require both DIs positive; stop `fillna(0)` into the smoother |
| [P26](#p26) | Archive loader cannot work | Medium | Four independent small fixes |
| [P27](#p27) | Class distribution is whole-dataset | Medium | Per-split counts + shift metric |
| [P28](#p28) | Risk metrics unclipped | Low | Subsumed by P1's convention |
| [P29](#p29) | Duplicates counted, not removed | Low | DB unique index + drop + stop clamping coverage |
| [P30](#p30) | Dead code and unreachable logic | Low | Four independent small fixes |

---

<a name="p1"></a>
## P1 — Metrics measure a discarded model

**Verified:** `ml_models.py:881` computes `full_metrics` from the raw estimators; `:927` replaces
`self._model` via `_calibrate_cascade`. Report shows `production_calibration: {gate: isotonic,
direction: isotonic}`, `predicted_class_distribution.LONG_SUCCESS: 0`, and a backtest that is 79% LONG.

### Option A — Move the metric computation after calibration
Compute `probabilities` from `self._model["gate"]` / `self._model.get("direction")` *after*
`_calibrate_cascade` returns, so the metrics describe the artifact being saved.
- **Cost:** small (move ~8 lines, recompute on 1.4 M rows — seconds).
- **Breaks:** every Direction number in the report changes. Expect accuracy/balanced accuracy to move
  and `LONG_SUCCESS` to stop being structurally zero. Any baseline comparison against previous runs
  becomes invalid — that is correct, the old baseline was measuring the wrong thing.
- **Note:** the gate/direction *sweeps* and `feature_importance` must move too. Importance can only be
  extracted from the underlying booster, so reach through the calibrator wrapper
  (`calibrated.estimator` / `.calibrated_classifiers_[0].estimator`) rather than dropping it.

### Option B — Report both, labelled
Emit `metrics_raw` and `metrics_production` side by side, plus the delta.
- **Cost:** small.
- **Breaks:** nothing; doubles the Direction metrics block.
- **Why you might want it:** it makes [P16](#p16)'s adoption decision empirical instead of a log-loss
  coin flip — you would see directly that raw never goes long and calibrated goes long 79% of the time.

### Option C — Structural guard: refuse to save a mismatched artifact
Stamp the estimator identity (`id()` at fit time, or a hash of the model dict) into
`metadata["metrics_source"]`, and have `save()` raise when it differs from the model being written.
- **Cost:** small.
- **Breaks:** nothing at runtime; it turns this class of bug into a hard failure instead of a silent one.
- **Limitation:** does not fix the numbers on its own. Pair with A.

### Pick
**A + C**, and take **B** as well if you want [P16](#p16) settled in the same pass. C is the piece that
stops it recurring — this exact defect also exists in `RiskModel` ([P28](#p28)) and latently in
`ExitModel`, so the guard is worth more than the local fix.

### Verify
Reload the saved artifact, score a held-out slice through the public `predict()` path, and assert it
reproduces `metadata["metrics"]` within `1e-9`. Make that a test — nothing today compares the two paths.

---

<a name="p2"></a>
## P2 — 72% of the training set is degenerate candles

**Verified:** drift block gives `realized_vol_12_is_zero` train mean **0.7235** vs validation
**0.0241**; `wick_ratio` 2.349 σ, `whipsaw_rate` 1.834 σ, `adx` 1.341 σ. `processor.py:442` is
`dropna(subset=["label", "target_risk_score"])`. `features.py:434,438` are the `fillna(0.0)` calls that
turn flat bars into hard zeros.

### Option A — Filter degenerate rows explicitly, and add a QC check that catches them earlier
Two parts: a `high == low` ratio check in `qc_validator` at CRITICAL (**not** in the set `4384d08`
downgrades for exchange-confirmed bars — byte-identical re-service proves the feed is honest, not that
the bar is tradeable), and `usable = usable[usable["realized_vol_12_is_zero"] < 1.0]` in `_to_dataset`
with a `dropped_zero_volatility_rows` counter in `ProcessedDataset`.
- **Cost:** medium.
- **Breaks:** training set falls from 2.88 M to roughly 800 k rows. Split boundaries shift, so this run
  is not comparable to the current one. **Must be paired with [P9](#p9)** — at a 45-day half-life the
  post-filter effective sample would be very small.
- **Why filtering on the named cause beats the old behaviour:** the pre-`146b00c` code dropped these
  rows too, but as a side effect of "any NaN in any feature". That was unauditable, which is the
  legitimate half of `146b00c`'s complaint. A named filter with a counter is not the same thing.

### Option B — Keep the rows, weight them to zero
Set `sample_weight = 0` for rows where `realized_vol_12_is_zero == 1.0`, multiplied into the recency
weight. LightGBM and XGBoost both honour zero weights.
- **Cost:** small.
- **Breaks:** less — row counts, split boundaries and `capacity_pct` stay stable, so the run remains
  comparable to the Stage-1 baseline.
- **Trade-off:** the rows still flow through feature statistics, drift computation and the report, so
  the diagnostic keeps showing a contaminated distribution even though the model no longer learns from
  it. Confusing unless you also exclude them from the statistics.

### Option C — Fix it upstream (required regardless)
Determine whether the Aug 2024 – mid-2025 klines are what Binance actually serves. Two facts point at
synthesis rather than market reality: 26 of 27 symbols have *exactly* 212,352 rows over 738 days (a
perfect 5-minute grid with zero gaps, which real exchange history never has), and 72% of them are flat.
Check the heal/bootstrap path and the ingestion upsert for grid materialisation or padding.
- **Cost:** medium (investigation), unknown (repair).
- **Breaks:** nothing until you find something.
- **Why this is not optional:** A and B are mitigations against a symptom. If the pipeline is
  fabricating candles, every future backfill re-poisons the dataset and the filter silently discards
  three quarters of it forever.

### Pick
**C to establish the truth, A as the standing guard.** Use **B instead of A** only if you need this
run's split boundaries to stay identical to the Stage-1 baseline for a controlled comparison.

### Verify
After the fix: train-split `realized_vol_12_is_zero` mean < 0.05; the top entry in
`features.drift.most_drifted_features` below 1.0 train-σ; walk-forward fold accuracies no longer
monotone (see [P18](#p18)).

---

<a name="p3"></a>
## P3 — Seven of 57 features are dead

**Verified:** `ob_imbalance`, `ob_imbalance_delta`, `ob_spread_bps`, `ob_spread_rank` are null on all
5,722,703 rows. `open_interest_change`, `long_short_ratio`, `taker_buy_sell_ratio` have
`non_neutral_row_fraction == 0.0`. `liquidation_imbalance` 47.20% null; `funding_rate` 52.8% populated.

### Option A — Remove the dead columns from `FEATURE_COLUMNS` now
Drop the four `ob_*` columns and the three constant futures columns.
- **Cost:** small.
- **Breaks:** `FEATURE_COLUMNS` changes length, so existing artifacts are incompatible and a retrain is
  required — but a retrain is required anyway for [P1](#p1)/[P2](#p2).
- **Also remove** `microstructure_is_missing` if the four `ob_*` columns go, since it becomes constant.

### Option B — Keep the columns, fix the archive loader ([P26](#p26)) and re-backfill
- **Cost:** large (loader work plus a multi-day backfill).
- **Breaks:** nothing.
- **Reality check:** at 740 days × 27 symbols the current configuration will not complete. Even after
  the transport fixes, plan on a reduced day count for `bookTicker`.

### Option C — Add a nullity gate to the diagnostic (detector, not a fix)
Raise CRITICAL for any feature whose null rate exceeds ~0.98 or whose `unique_count` is 1.
- **Cost:** small.
- **Value:** this single check would have flipped this run's verdict from GOOD to CRITICAL. The four
  `ob_*` columns are currently covered by **no check at all** — `microstructure_coverage` inspects a
  hardcoded list of four *futures* features and never looks at the `ob_*` block.

### Pick
**A + C now, B when the loader is fixed.** Re-add features in the same commit that demonstrates their
coverage, never before.

### Verify
`dataset.null_counts_by_feature` has no entry at 100%; `feature_count` drops to 50 (or 49 without the
missingness flag); the diagnostic raises CRITICAL if you deliberately null a column.

---

<a name="p4"></a>
## P4 — Stage 2 has no discrimination; isotonic amplifies a 10-point band

**Verified:** `direction_threshold_sweep` gives `signals` = 762,529 (all rows) at t ∈ {0.30, 0.35, 0.40},
29,961 at 0.45, and **0 at 0.50**. Raw `p_long` is confined to `[0.40, 0.50)`. Precision at the loosest
threshold (0.4836) equals the LONG base rate exactly.

### Option A — Guard the calibrator on raw output span
Refuse isotonic when `np.ptp(raw_scores) < 0.20`; record `raw_probability_span` per stage every run.
- **Cost:** small.
- **Breaks:** with the raw estimator in place and `min_direction_given_trade_confidence = 0.60`, R1b
  rejects essentially everything — the system stops trading. **That is the honest outcome**, and it is
  strictly better than trading on isotonic-amplified noise while displaying "88% confidence".

### Option B — Drop the directional bet; trade the gate only
Size on `trade_probability` alone and either pick a side by a transparent rule or stop taking
directional positions.
- **Cost:** large (architectural).
- **Breaks:** the two-stage cascade's whole premise.
- **When it is right:** if stage 2 still cannot beat ~0.55 AUC after the data is clean, the label is not
  learnable at this horizon and this is the honest architecture.

### Option C — Re-label at a horizon where direction is learnable
`LONG_SUCCESS` vs `SHORT_SUCCESS` currently means "reached 2×ATR before 1×ATR within 48 bars". On
5-minute crypto that is close to a coin flip by construction, and the 0.4837 base rate says so. Try a
longer holding window, a larger TP multiple, or a trend-continuation label.
- **Cost:** large (relabel + retrain + re-evaluate).
- **Breaks:** every downstream target, since Entry/Exit/Risk all derive from the same simulation.

### Option D — Re-measure on clean data before deciding
The current measurement is confounded by [P2](#p2) (72% degenerate rows) and [P9](#p9) (effective
sample ≈ 2 months). Both push toward exactly this failure mode.
- **Cost:** none beyond Stage 2's work.

### Pick
**D first — the current number cannot support a decision.** Then **A** permanently, as the guard. Then
**B or C** depending on what D shows. Do not restructure the architecture on a measurement taken from
a poisoned training set.

### Verify
Report ROC-AUC for the gate and for stage 2 (neither is reported today — only Entry gets an AUC). If
stage-2 AUC on clean data is below ~0.55, Option B or C is the answer and no amount of calibration
will help.

---

<a name="p5"></a>
## P5 — The backtest never tests the entry bar's own high/low

**Verified:** `backtester.py:270` resolves barriers, `:284` fills signals. A position filled at bar *T*'s
open enters `positions` after step 1 has already run for bar *T*, so it is first tested on bar *T+1*.

### Option A — Reorder the loop: fill, then resolve
Move the fill block above the resolve block and clear `pending` before resolving.
- **Cost:** small.
- **Breaks:** win rate and profit factor both fall. With the stop floored at 1×ATR and 5-minute bars
  whose range is routinely comparable to ATR, a meaningful share of trades currently getting a free bar
  of recovery will book as immediate stops.

### Option B — Keep the order, resolve newly-filled positions in place
After filling, immediately call `_resolve_bar` on just the freshly-opened positions using the same bar.
- **Cost:** small.
- **Breaks:** same outcome as A, smaller diff to the existing loop structure.

### Both options need the same convention
Within one bar you cannot know whether the high or the low came first. `_resolve_bar` already handles
this pessimistically for existing positions (adverse level tested before target); apply the same rule to
entry-bar resolution, so a bar touching both stop and target on the entry bar books the **stop**. That
keeps the bias conservative rather than merely relocating it.

### Pick
**A.** It makes the invariant structural — "a position is always resolved against every bar it is open
for, including its first" — rather than a special case that a later refactor can drop.

### Verify
Test: construct a signal whose next bar gaps straight through the stop; assert the trade closes on that
bar, at the stop, with a negative MAE recorded.

---

<a name="p6"></a>
## P6 — The backtest bypasses the Risk Guard

**Verified:** `backtester.py:304` hardcodes `risk_guard_state="GREEN"` and `size_multiplier=1.0` for the
entire replay. `RiskGuard` (424 lines) is never instantiated. Reported `max_drawdown_pct` is 0.14217
against a configured `daily_drawdown_red_pct` of 0.05 with `require_manual_reset = True`.

### Option A — Instantiate a real `RiskGuard` in `_simulate`
Feed it every close and every equity mark; read `state` and `size_multiplier` into each bar's
`DecisionContext`. Honour `require_manual_reset` faithfully — once RED, stay RED for the rest of the
replay and report `halted_at`.
- **Cost:** medium.
- **Breaks:** the headline result, substantially. A run that halts on day 40 and sits out the remaining
  five months is a *result*, not a bug to work around.
- **Include `max_daily_trades`** in the same place. At 4.3 trades/day it would not bind here, but that
  is luck.

### Option B — Report constrained and unconstrained side by side
Keep an unconstrained "model-only" replay for isolating model quality from policy, and print it next to
the policy-constrained number.
- **Cost:** small on top of A.
- **Value:** genuinely useful — it separates "the model has no edge" from "the risk policy is too tight".
  It is only dishonest when it is reported *instead of* the constrained number, which is the current state.

### Option C — Detector only: report when thresholds would have tripped
Post-process the equity curve for daily-drawdown and consecutive-loss breaches without changing the
simulation.
- **Cost:** small.
- **Breaks:** nothing.
- **Limitation:** gives the reader the caveat without the corrected number. Acceptable as a stopgap; not
  a substitute for A.

### Pick
**A + B.** C only if you need the caveat in the report before A is ready.

### Verify
`BacktestReport` carries a `risk_guard_transitions` list; a deliberately loss-heavy synthetic run trips
RED and stops opening positions.

---

<a name="p7"></a>
## P7 — Backtest selection policy differs from live

**Verified:** `backtester.py:312` iterates `for symbol, rows in indexed.items()` and `:317` `break`s at
the concurrency cap. `DecisionEngine.evaluate_many` — which ranks by directional confidence and exists
precisely for this — is never called by the backtester. PIXEL takes 11.4% of strict-run trades and
25.8% of relaxed-run trades.

### Option A — Call `evaluate_many`
Build one `ModelInferenceResult` per eligible symbol per bar, hand the batch to `evaluate_many`, and
fill from its executable results.
- **Cost:** medium.
- **Breaks:** trade selection changes, so results change. `signals_generated` becomes meaningful (every
  symbol scored every bar) and the portfolio cap starts appearing as `Rule.PORTFOLIO_FULL` in the
  rejection breakdown instead of being an invisible `break`.
- **Watch the cost:** scoring all 27 symbols on every bar instead of stopping at the cap increases
  inference calls substantially. Acceptable, and it is what live does.

### Option B — Keep the loop, remove the `break`, sort before filling
Score every symbol, then sort executable signals by confidence before filling into free slots.
- **Cost:** small.
- **Breaks:** same as A.
- **Weakness:** duplicates `evaluate_many`'s logic in a second place, so the two can drift apart again.
  That divergence is the bug being fixed.

### Pick
**A.** The point is that backtest and live run *the same selection code*, not that they currently agree.

### Verify
Test: three symbols signal on one bar with one free slot; assert the slot goes to the highest-confidence
symbol, not the first in iteration order. Add a per-symbol concentration block to the report (trades and
P&L per symbol, share of net profit from the top symbol).

---

<a name="p8"></a>
## P8 — Risk and Exit are trained only on realised winners

**Verified:** `ml_models.py:1968` — `usable = dataset.direction_target != LabelClass.NO_TRADE_OR_FAIL.value`.
Risk and Exit both train on 1,536,896 rows vs Direction's 2,881,082. Risk predictions floor at **0.4217**
against a target floor of 0.0903, mean bias **+0.094**, R² 0.0789, and `R6_RISK_MODEL_ABORT` fires 281
times in 1,381,357 candidates (0.02%).

### Option A — Fix the target, drop the filter
`target_risk_score` is currently `0.0` by fiat on `NO_TRADE_OR_FAIL` rows, which is what made the
unfiltered population look unlearnable. Compute path heat (`1 − mae_ratio`) for *both* sides on every
bar and take the side the cascade would have chosen, so a losing setup gets a genuine low score rather
than a placeholder zero. Then train on all rows.
- **Cost:** medium (labeler change + retrain).
- **Breaks:** R² will likely *fall* below the current 0.0789 at first. That is not a regression — the
  current figure is measured against a population the head never sees live.
- **Why the docstring's defence fails:** it argues the filter matches inference because `predict()` is
  only asked about trades "Direction/Entry have already approved". But at inference those heads approve
  a *candidate* they predict will work; the filter selects on the *realised outcome*. The realised
  outcome is unknowable at decision time — that is the problem the system exists to solve.

### Option B — Keep a filter, but make it inference-observable
Filter on out-of-fold predicted gate probability ≥ threshold rather than on the label.
- **Cost:** large (needs OOF predictions across the training window).
- **Value:** legitimately matches the inference population.
- **Weakness:** expensive, and it inherits whatever bias the gate has.

### Option C — Train on all rows, down-weight `NO_TRADE` instead of excluding
Keep the current target but give `NO_TRADE` rows a weight of e.g. 0.3.
- **Cost:** 1 line.
- **Value:** a fast experiment that tells you how much of the R² gap is population and how much is the
  placeholder-zero target, before committing to A's labeler work.

### Pick
**A.** Run **C** first as a one-line probe — it costs a single training run and tells you whether A's
target rework is where the value is. Apply the same fix to `ExitModel`: TP/SL/trailing geometry learned
only from paths that reached TP is geometry for winners, and the head needs losers' paths to learn how
wide a stop must be to survive a trade that eventually fails.

### Verify
Add to every head's metrics block: training positive rate vs the positive rate of the population
`predict()` actually sees in the backtest. A gap from 100% to ~44% should be impossible to ship without
it appearing in the report. After the fix, Risk prediction `min` should approach the target `min`.

---

<a name="p9"></a>
## P9 — 45-day half-life on a 372-day window

**Verified:** `settings.py:408` — `recency_half_life_days: float = Field(default=45.0, ge=0.0)`. Weight
is `0.5 ** (age_days / 45)`. Integrated over 371.9 days that is **64.7 effective days — 17.4%**. A row
180 days old carries weight 0.0625; a row at the start of the window carries 0.0032.

### Option A — Raise the half-life to 90–180 days
- **Cost:** 1 line.
- **Breaks:** nothing structurally; changes what the model learns.
- **Rule of thumb:** a half-life below ~1/4 of the training span discards most of it. For a 12-month
  window, 90–180 days is defensible; 45 is not.

### Option B — Set it to 0 (uniform weighting)
- **Cost:** 1 line.
- **Rationale:** the chronological train/validation/test split already carries the recency argument —
  that is what a chronological split is *for*. Exponential weighting on top of it is a second, hidden
  recency policy.

### Option C — Sweep it
Run `{0, 45, 90, 180, ∞}` and report validation log loss for each.
- **Cost:** 5 training runs.
- **Value:** settles the question with evidence instead of intuition, and it is cheap.

### Pick
**C**, defaulting to **B** until the sweep is done. B is the neutral choice, and the current setting is
actively making `train_months: 12.0` and `rows: 2,881,082` fictions in the report.

This interacts with [P2](#p2) in both directions: the recovered rows `146b00c` added are mostly in the
older, degenerate part of the window where the weight is 0.003–0.06, so the two changes work against
each other. But once P2 removes those rows, the surviving window is *shorter*, which makes an
aggressive half-life more damaging, not less.

### Verify
Report `effective_train_rows` (`weights.sum()`) and `effective_train_days` alongside `rows`. Add a
config validator warning when `recency_half_life_days < train_months × 30 / 4`, matching the existing
`purge_bars < max_holding_bars` warning.

---

<a name="p10"></a>
## P10 — The validation block does four incompatible jobs

**Verified:** the same 1,418,472 rows drive early stopping (`_fit_cascade(..., early_stopping=True)`),
metric reporting, threshold auto-tuning, and — via `_fit_production_calibrator` — the shipped isotonic
calibrator, which is deliberately fit on *the entire* block.

### Option A — Split validation into fit and score halves
Use `_temporal_half_split` (already present and already used for the honest before/after measurement)
for the *production* calibrator too: fit calibration, thresholds and early stopping on the earlier half,
report metrics from the later half only.
- **Cost:** medium.
- **Breaks:** reported metrics change (they become genuinely held out); the calibrator sees half the
  rows.
- **On the docstring's objection** ("a shipped artifact should not throw away data the diagnostic report
  doesn't need"): that reasoning is backwards for isotonic regression, which is a free-form step
  function and will memorise whatever it is given. More rows make the overfitting worse, not better.

### Option B — Add a fourth split
Carve `train / fit / score / test` from the configured months.
- **Cost:** medium.
- **Breaks:** shortens the training window, which collides with [P2](#p2) and [P9](#p9) both shrinking
  the effective sample already.
- **When it is right:** if you have more history than the current 24 months.

### Option C — Provenance labelling only
Add `metrics_provenance` per head: `"held_out"` / `"used_for_early_stopping"` / `"used_for_calibration"`.
- **Cost:** small.
- **Breaks:** nothing.
- **Limitation:** documents the bias instead of removing it. Worth doing regardless — a clean-looking
  number that is not clean is the actual hazard.

### Pick
**A + C.** A reuses machinery that already exists; C is cheap and prevents the next reader from
over-trusting the numbers.

### Verify
The calibration block reports `calibration_rows` strictly less than `validation_rows`, and metrics are
computed on row indices disjoint from those used for calibration and early stopping.

---

<a name="p11"></a>
## P11 — Auto-tuned thresholds are arithmetically meaningless

**Verified:** `recommended_direction_threshold: 0.30` against a gate that computes `max(p, 1−p)` — which
is bounded below by 0.5, so 0.30 cannot reject anything. `recommended_gate_threshold: 0.35` against a
configured 0.55. `_select_recommended_threshold` uses `floor` only in the `if not candidates: return
floor` branch (verified at `ml_models.py` lines 1598 / 1609) and otherwise returns
`float(best["threshold"])` unbounded.

### Option A — Sweep the quantity the gate actually uses, and make `floor` a floor
Add `direction_confidence_sweep(is_correct_side, max(p, 1−p))` over `t ∈ [0.50, 0.95]` and select from
that; delete the sub-0.5 grid points from the direction sweep, since they describe a rule the engine
cannot express. Then `return max(float(best["threshold"]), floor)`, and rename the parameter
`configured_floor`.
- **Cost:** small.
- **Breaks:** recommended thresholds change and stop being no-ops.

### Option B — Refuse to recommend when the sweep is flat
If the best F-β is within a few percent of the loosest grid point's, return `NOT_AVAILABLE` with a reason.
- **Cost:** small.
- **Value:** "0.35" reads as a considered suggestion. "The sweep is flat — this model does not
  discriminate" is the same information, stated honestly. Look at the gate sweep: precision moves from
  0.576 to 0.842 only by discarding 99.8% of rows. There is no knee to find.

### Option C — Select on economics, not F-β
Every sweep row already has slots for `average_r`, `win_rate`, `profit_factor`, `expectancy`,
`net_pnl`, `max_drawdown` — and **every one reads `"NOT_AVAILABLE"` in this run**. Wire them up and
optimise expectancy.
- **Cost:** medium (needs the label simulation's realised outcomes joined to the sweep).
- **Value:** this is the real fix. F-β is indifferent to the 2.4:1 payoff structure that the strategy's
  entire edge depends on.

### Pick
**A + B now, C as the actual solution.** The slots exist; filling them is the difference between tuning
a classifier and tuning a strategy.

### Verify
No recommended threshold is ever below the gate's mathematical floor; a deliberately flat sweep returns
`NOT_AVAILABLE`.

---

<a name="p12"></a>
## P12 — Backtest window is not the test split, and the two runs differ

**Verified:** test split 2026-02-12T15:35 → 2026-08-14T06:35; strict backtest 2026-02-14T13:25 →
2026-08-14T11:10; relaxed 2026-02-14T16:40 → 2026-08-14T14:25. `main.py:472` sets `"oos_fraction": 1.0`
as a literal.

### Option A — Add `start_ms` / `end_ms` to `Backtester.run` and `_prepare`
Query the database for an explicit range instead of "the most recent N bars".
- **Cost:** medium (touches the DB loader signature).
- **Breaks:** nothing; makes the window deterministic.

### Option B — Pin the exact timeline from the split
Capture the test split's timestamps from `ProcessedDataset` and pass them to the backtester as the
timeline.
- **Cost:** medium.
- **Stronger than A** because it guarantees the replay covers exactly the rows the split defined, rather
  than a range that happens to overlap it. It also makes the strict and relaxed runs identical by
  construction, which is the entire point of the comparison.

### Option C — Compute `oos_fraction` instead of asserting it
Intersect the replay timeline with the train+validation index and report the real overlap; raise
CRITICAL below 1.0.
- **Cost:** small.
- **Required regardless.** As written the field can never detect the thing it exists to detect — if a
  future split change *did* introduce overlap, it would still read `1.0`.

### Also fix the double warm-up trim
`max_candles = test_bars + warmup_padding`, and then `_simulate` trims `warmup_bars` timestamps off the
front — but `_prepare` has *already* dropped un-warmed rows via `dropna(subset=REQUIRED_FEATURE_COLUMNS)`.
Drop one of the two. This is where the ~2-day shortfall at the front of the window comes from.

### Pick
**B + C**, plus the warm-up fix.

### Verify
Strict and relaxed runs report byte-identical `start`, `end` and `signals_generated`; `oos_fraction` is
computed and equals 1.0 for a correct split.

---

<a name="p13"></a>
## P13 — Sortino denominator; Sharpe on a degenerate curve

**Verified:** `backtester.py:686-697` — `downside = returns[returns < 0.0]` then
`np.sqrt(np.mean(np.square(downside)))`, dividing by `downside.size` rather than `returns.size`.
Reported Sharpe 4.2779, Sortino 2.0781 — Sortino below Sharpe on a right-skewed distribution
(avg win 3.876, avg loss −1.616) is the tell.

### Option A — Fix the denominator
```python
downside = np.minimum(returns, 0.0)                       # keep every period
deviation = float(np.sqrt(np.mean(np.square(downside))))  # divide by returns.size
```
- **Cost:** 2 lines.
- **Breaks:** Sortino rises to above Sharpe, which is the correct relationship here.

### Option B — Add trade-based risk-adjusted metrics
Per-trade Sharpe with an explicit trade count, plus a bootstrap confidence interval on expectancy.
- **Cost:** small.
- **Why it matters more than A:** the equity curve has 52,107 per-bar points but only **786 trades**.
  Annualising by `sqrt(365 × 24 × 12)` treats 105,120 near-degenerate, heavily autocorrelated
  observations per year as independent draws. Per-trade Sharpe is roughly 0.29; the standard error on
  the annualised figure is around 1.4. The honest reading is "positive, imprecise", not 4.28.

### Option C — Report the effective sample
`equity_curve_points`, `nonzero_return_bars`, `trades_per_year` next to the ratios.
- **Cost:** small.

### Pick
**All three.** A is two lines, B is the one that changes how the number gets read, C is the context that
stops it being quoted out of context.

### Verify
Unit test both functions on a synthetic curve with known Sharpe and Sortino. Both are pure functions and
neither is covered today.

---

<a name="p14"></a>
## P14 — `opened_at` is wall-clock time

**Verified:** `models.py:63` — `opened_at: datetime = field(default_factory=_utcnow)`. All 500 retained
trades "opened" within a 69-minute span (the backtest process's runtime). Mean holding period across
those trades is **−11,546 bars**; every single one is negative.

### Option A — Set it in `Backtester._fill`
`position.opened_at = ms_to_datetime(int(row["timestamp"]))` after construction.
- **Cost:** 1 line.
- **Weakness:** a future code path that constructs a `Position` without going through `_fill` reintroduces
  the bug silently.

### Option B — Make it a constructor parameter
Add `opened_at` to `Position.from_signal`, defaulting to `_utcnow()`, and pass the bar timestamp from the
backtester.
- **Cost:** small.
- **Stronger:** the simulated-vs-wall-clock choice becomes explicit at every construction site.

### Option C — Assert the invariant
`assert position.closed_at >= position.opened_at` in `_book_close`, plus a `holding_bars` field per trade
and a summary block in the report.
- **Cost:** small.
- **Value:** a backtest whose mean holding period is negative should not be able to produce a report.

### Pick
**B + C.**

### Verify
`holding_bars` is positive for every trade and its distribution respects `labels.max_holding_bars = 48`
— which is also the check that tells you whether executed trades honour the horizon the labels were
built on, the exact class of geometry mismatch `c2968d8` set out to eliminate.

---

<a name="p15"></a>
## P15 — Exit's stop head is dead code; its trailing head learns `y = 0.5x`

**Verified:** `labeler.py:522` — `simulation.optimal_trailing_pct[target] = optimal_tp * 0.5`. The two
R² values are identical to 17 significant figures (`0.20039042985291866`). The stop head has R² 0.0519
with prediction std 0.00193 against target std 0.00824, and `_assemble` floors the stop at
`atr_pct × sl_atr_multiple` (1.0), which exceeds the model's 0.00168 median prediction on essentially
every bar.

### Option A — Delete the trailing regressor
Compute `trailing_activation_pct = 0.5 × take_profit_pct` directly in `_assemble`.
- **Cost:** small.
- **Breaks:** nothing — the output is unchanged, because the target *is* that formula.
- **Saves:** a third of the Exit model's training cost and artifact size, plus a metric that cannot fail.

### Option B — Give the trailing head a real target
E.g. the excursion at which the realised path stopped making new favourable extremes.
- **Cost:** medium (labeler work).
- **When it is right:** if you want a learned trail. Note this interacts with [P22](#p22) — the
  backtester currently discards the model's trailing value entirely, so fixing the target without
  fixing P22 changes nothing observable.

### Option C — Re-scale the stop head to predict an ATR multiplier
Have it output a multiplier in `[1.0, 3.0]` instead of a raw percentage. The floor then becomes a lower
bound on a live parameter rather than a replacement for the model.
- **Cost:** medium (target rescale + retrain).
- **Breaks:** the stop head's metrics become comparable to something meaningful.

### Option D — Report the override rate (required regardless)
Fraction of validation rows where the ATR floor exceeded the model's prediction, and the same for the TP
clamp.
- **Cost:** small.
- **Value:** `floor_override_rate: 0.97` is the number that turns "this head has R² 0.05" into "this head
  is not in the loop". Also gate `beats_rule_based_baseline` on it — that badge currently describes an
  output nobody consumes.

### Pick
**A + C + D.** None of this is an argument against `c2968d8`'s stop-floor change, which fixed a real
defect correctly; it is about what the change left behind and did not report.

### Verify
Exit metrics carry `floor_override_rate` for both stop and TP; the trailing regressor is gone from the
artifact and the executed geometry is unchanged.

---

<a name="p16"></a>
## P16 — Calibration adopted on unguarded 0.001-nat differences

**Verified:** `ml_models.py:1268` and `metrics.py:400` — `improved: bool = calibrated_logloss <
raw_logloss`. Gate improvement is 0.6387282 → 0.6380239, i.e. **0.11%**. Joint is 0.14%. No effect-size
threshold, no significance test.

### Option A — Require a material effect size
```python
MIN_RELATIVE_IMPROVEMENT = 0.01          # 1%
improved = (raw_logloss - calibrated_logloss) / raw_logloss > MIN_RELATIVE_IMPROVEMENT
```
- **Cost:** 2 lines (in both places).
- **Breaks:** on this run all three heads would stay on the raw estimators — and the report would then
  honestly show a Direction model that never predicts LONG, which is the finding that matters.

### Option B — Require agreement across folds
Evaluate on 3–5 temporal folds within the validation block and require the improvement to hold in a
majority. `_temporal_half_split` already exists; it just needs applying more than once.
- **Cost:** small.
- **Value:** a 0.11% difference on a single draw carries no information about which estimator
  generalises. Multiple folds turn it into a signal or expose it as noise.

### Option C — Report the delta and gate promotion on it
`log_loss_improvement_pct` and `improvement_exceeds_threshold` in the calibration block; a human decides.
- **Cost:** small.
- **Limitation:** relies on someone reading it.

### Pick
**A + B**, with **C**'s reporting fields as well. This is the switch that decides whether the system
ships a model that never goes long or one that goes long 79% of the time ([P1](#p1), [P4](#p4)) — it
should not be a coin flip.

### Verify
Feed the calibration path a synthetic score with a known 0.1% improvement and assert it is rejected;
feed one with a 5% improvement and assert it is adopted.

---

<a name="p17"></a>
## P17 — `_ai_summary` is structurally blind to data problems

**Verified:** `biggest_data_problem` reads only `data_quality.symbol_exclusions_total`. `overall_status`
is GOOD unless a head is `NOT_TRAINED`, a symbol was excluded, the backtest has < 30 trades, or a
baseline regressed. Head comparison ranks balanced accuracy (0.3559), ROC-AUC (0.6021) and R² (0.0789)
against each other on raw magnitude. `exit` is collected into `heads` but never scored.

### Option A — A check registry
A list of named predicates over the report dict, each returning `(severity, message)`, driving both
`_ai_summary` and `_recommendations`.
- **Cost:** medium.
- **Value:** this is the structural fix. Everything needed for a correct verdict is *already computed and
  already in the JSON* — per-feature null counts, per-symbol-month null rates, drift in train-σ units,
  split coverage, per-fold walk-forward metrics, per-class confusion matrices. It simply is not read.
  With a registry, a check cannot exist in the data without appearing in the verdict.
- **Subsumes:** the detector halves of [P3](#p3), [P18](#p18), [P24](#p24), [P27](#p27), [P29](#p29).

### Option B — Point fixes to the existing functions
Add nullity and drift checks inline to `_ai_summary` / `_recommendations`.
- **Cost:** small.
- **Weakness:** the next check gets forgotten the same way these were.

### Option C — Normalise head scores to lift over chance
```python
scored = {
    "direction": (balanced_accuracy - 1/3) / (1 - 1/3),
    "entry":     (roc_auc - 0.5) / 0.5,
    "exit":      max(0.0, mean_r2_across_targets),
    "risk":      max(0.0, r2),
}
```
- **Cost:** small.
- **Breaks:** the strongest/weakest verdict flips. On lift over chance, direction (+0.023) is the weakest
  head by a factor of three, not risk (+0.079) — and direction is the head the entire edge depends on.
  The current "improve the risk model" recommendation is an artifact of R² living near zero while
  balanced accuracy lives near one third.

### Pick
**A + C.** Add a "not measured" section too: `regimes` reports
`"per-regime performance breakdown is not computed by this run"` and every economic column in every
threshold sweep reads `NOT_AVAILABLE`. A GOOD verdict resting partly on checks that did not run should
say which ones did not run.

### Verify
Re-run the diagnostic against the *current* (unfixed) JSON as a fixture and assert `overall_status`
comes out CRITICAL.

---

<a name="p18"></a>
## P18 — `_walk_forward_problem_summary` can only return "none measured"

**Verified:** `diagnostics.py:428-436` — once `status == "AVAILABLE"`, every return path begins with the
literal string `"none measured"`. Folds: accuracy 0.8034 → 0.6680 → 0.5322 → 0.4623, balanced 0.7208 →
0.6012 → 0.5491 → 0.3771, macro-F1 0.7249 → 0.6088 → 0.5268 → 0.3242 — strictly monotone in all three.

### Option A — Test the trend
```python
from scipy.stats import spearmanr
rho, p = spearmanr(range(1, len(folds) + 1), [f["accuracy"] for f in folds])
```
With ρ = −1.0 the summary should read: *"accuracy degrades monotonically across all 4 folds
(0.803 → 0.462, Spearman ρ = −1.0); the earliest fold is not representative of the period the model will
trade."*
- **Cost:** small (adds a `scipy` dependency if not already present — a rank correlation is ~10 lines to
  implement directly if you would rather not).
- **Why `accuracy_std` is the wrong statistic:** 0.131 is the standard deviation of a monotone sequence,
  and it takes the same value for that sequence shuffled into any order — including orders that would be
  benign noise. Reducing a trend to a spread destroys the only information it carries.

### Option B — Surface the latest fold
Report `latest_fold_accuracy` next to `accuracy_mean`. Fold 4 validates on the most recent, most
representative data; its 0.4623 is the closest thing in the report to an honest estimate, and it is
buried in an array while the 0.6165 mean is headlined.
- **Cost:** 2 lines.

### Option C — Escalate to `overall_status`
Force at least WARNING on a monotone collapse, CRITICAL when the final fold is within a few points of
the chance baseline — 0.4623 against a majority-class rate of ~0.46 means the last fold is, on accuracy
alone, indistinguishable from always predicting NO_TRADE.
- **Cost:** small (or free, under [P17](#p17)'s registry).

### Pick
**All three** — roughly 15 lines together, and they convert the single most important measurement in the
report from invisible to unmissable.

### Verify
Feed the current fold array as a fixture and assert the summary names the trend and the status is not GOOD.

---

<a name="p19"></a>
## P19 — Relaxed diagnostic backtest resets every other decision setting

**Verified:** `main.py:96-102` constructs `DecisionSettings(...)` with five fields, so every other field
reverts to its class default — `max_leverage`, `min/max_capital_allocation_pct`,
`max_concurrent_positions`, `max_positions_per_symbol`, `accepted_risk_tiers`, `blocked_hmm_regimes`,
`max_volatility_percentile`.

### Option A — Derive from the live settings
```python
relaxed_decision = self.settings.decision.model_copy(update=_RELAXED_OVERRIDES)
relaxed_settings = self.settings.model_copy(update={"decision": relaxed_decision})
```
- **Cost:** small.
- **Breaks:** nothing; the relaxed run starts respecting operator configuration.

### Option B — Decouple the sizing base from the gate threshold
`RiskModel.predict` uses `min_direction_given_trade_confidence` as the base of its sizing curve
(`confidence_span = 1 − threshold`), so strict (0.40) and relaxed (0.48) size every trade differently.
Give `RiskSettings` its own `sizing_confidence_base`.
- **Cost:** small.
- **Value:** this is a latent hazard in *live* tuning too, not just in the diagnostic — moving a decision
  threshold currently reprices every position as a side effect.

### Option C — Report the settings diff
Emit the field-by-field delta between strict and relaxed `DecisionSettings` into the relaxed block.
- **Cost:** small.
- **Value:** any unintended difference becomes visible in the report rather than in the source.

### Pick
**A + B + C.** Until they land, treat the report's LOW recommendation — that the low trade count "looks
like the cascade's intended selectivity" — as unsupported: the relaxed run differs from the strict one in
thresholds *and* sizing *and* reset settings *and* time window ([P12](#p12)), and none of it is separable.

### Verify
With `_RELAXED_OVERRIDES` applied, assert that every `DecisionSettings` field except the five overridden
ones equals the live value.

---

<a name="p20"></a>
## P20 — `rejection_breakdown` is first-match-wins and not comparable

**Verified:** `backtester.py:324` records only `decision.rule_triggered`. `R4_ENTRY_MODEL_SAYS_WAIT` is
568 in the strict run and 1,268,257 in the relaxed one — a factor of 2,233 — because R1a/R1b reject
1,338,511 of 1,381,357 candidates before R4 is ever evaluated.

### Option A — Count every failing check, not just the first
`DecisionEngine.evaluate` already builds a complete `checks` list via `_record` before returning. Keep
the early return for the *verdict*, but also accumulate every failed check into a second breakdown.
- **Cost:** small — the data is already there.
- **Value:** `rejection_breakdown` (which rule stopped it) plus `rejection_breakdown_independent` (how
  often each rule would object on its own). The second is the one that supports tuning.

### Option B — Per-rule reached/passed counts
`R4: reached 42,846, passed 42,278 (98.7%)` is unambiguous in a way that `R4: 568` is not.
- **Cost:** small.

### Pick
**A + B.** Also fix `signals_generated` via [P7](#p7) — the denominator is currently the number of
candidates scored *before the book filled*, which is why the two runs report 1,381,357 and 1,362,545 on
nominally the same window.

### Verify
Independent counts sum to more than the first-match counts, and `_recommendations`' "top rejection
reason" is computed from the independent breakdown.

---

<a name="p21"></a>
## P21 — Entry threshold override is silent and the audit log prints the wrong number

**Verified:** `ml_models.py:1621` falls back to `self._metadata["decision_threshold"]` (0.45) rather than
the configured `min_entry_probability` (0.55). `decision_engine.py:223` prints
`{self._config.min_entry_probability:.3f}` — 0.550 — in the R4 rejection message, while the gate that
actually fired was 0.45.

### Option A — Make the config authoritative, auto-tune advisory
Report `recommended_entry_threshold` but apply `settings.decision.min_entry_probability`.
- **Cost:** small.
- **Breaks:** the entry gate tightens from 0.45 to 0.55, so fewer trades.
- **Rationale:** an operator risk control should not be overridden by a value the model chose for itself
  on data it also early-stopped on ([P10](#p10)).

### Option B — Make auto-tune authoritative, but bound and announce it
`max(best, floor)` (see [P11](#p11)), rename the config field to `entry_probability_floor`, and log at
WARNING when the applied value differs from the configured one.
- **Cost:** small.
- **Rationale:** keeps the tuning benefit while making the semantics honest.

### Option C — Carry the applied threshold on the prediction (required either way)
Add `threshold` to `EntryPrediction` and use `entry.threshold` in both the `_record` detail and the
rejection message.
- **Cost:** small.
- **Value:** the audit log exists so decisions can be reconstructed. Right now every R4 line states a
  threshold the system did not use, and the structured `checks` payload does not carry the truth either.

### Pick
**Decide A vs B deliberately — that is the real question here — then apply C regardless.** The current
arrangement is the worst of the three: a config field that looks authoritative, an override that is not
announced, and a log line that prints the config value.

### Verify
A rejected entry's log line and its `checks` entry both quote the threshold that was actually compared
against.

---

<a name="p22"></a>
## P22 — Trailing trigger is hardcoded in `_fill`

**Verified:** `backtester.py:436` and `:442` set `trailing_trigger = fill_price * (1 ± signal.take_profit_pct * 0.5)`,
discarding `exit_params.trailing_activation_pct`. `TradeSignal` carries only the absolute
`trailing_trigger`, so `_fill` has nothing else to re-anchor from. 125 of 500 retained trades (25%) closed
at TRAILING_STOP.

### Option A — Carry the percentage on the signal
Add `trailing_activation_pct: float` to `TradeSignal` alongside the existing `take_profit_pct` /
`stop_loss_pct`, and use it in `_fill`.
- **Cost:** small (schema field + two call sites).
- **Breaks:** trailing behaviour in the backtest changes to match live, so results change.

### Option B — Re-anchor from the absolute trigger by ratio
`pct = abs(signal.trailing_trigger / signal.reference_price - 1.0)`, then apply to `fill_price`.
- **Cost:** 2 lines, no schema change.
- **Weakness:** derives a percentage back out of a price that was derived from a percentage. Works, but
  it is fragile to sign conventions and reads as a workaround.

### Option C — Parity test (required either way)
Given one `TradeSignal` and one fill price, assert `Backtester._fill` and `Position.from_signal` (the
live path) produce identical `take_profit`, `stop_loss` and `trailing_trigger`.
- **Cost:** small.
- **Value:** the backtest/live divergence is the actual defect; the test is what stops it recurring.

### Pick
**A + C.** Note the re-anchoring in `_fill` is itself correct and well-motivated — barriers computed off
the decision-bar close must be re-anchored to the actual fill. The bug is that the trailing line
re-anchors to a *different rule* than the one the model produced.

### Also worth deciding
`c2968d8` established that the executed stop must match the barrier the label priced. The same argument
applies to the trail: arming at half the TP converts labelled TP-winners into partial wins, and the
Direction model's probability says nothing about that outcome. Either model the trail in the labeler, or
report `trailing_exit_rate` as a known deviation and quantify its cost. 25% is large enough to matter.

### Verify
The parity test passes; `trailing_exit_rate` appears in the backtest metrics.

---

<a name="p23"></a>
## P23 — Train/serve population mismatch on `bb_position` and `volume_trend`

**Verified:** `processor.py:442` drops on labels only; `backtester.py:222-224` drops on
`REQUIRED_FEATURE_COLUMNS`. `bb_position` is 24.28% null, `volume_trend` 16.52%, and neither is in
`OPTIONAL_FEATURE_COLUMNS` (which holds only the five microstructure columns).

### Option A — Training gates on `REQUIRED_FEATURE_COLUMNS`
```python
usable = usable.dropna(subset=["label", "target_risk_score", *REQUIRED_FEATURE_COLUMNS])
```
- **Cost:** 1 line.
- **Breaks:** training set shrinks by ~24%. Combined with [P2](#p2)'s explicit filter, the degenerate
  rows then get dropped for a *stated* reason rather than as a side effect.
- **Restores the invariant** `146b00c`'s own commit message claims: *"REQUIRED_FEATURE_COLUMNS still gates
  warm-up, consistently across training, live inference and the backtester."*

### Option B — Inference stops gating; NaN everywhere
The more radical reading of `146b00c`'s thesis, and defensible — LightGBM does route NaN down a learned
default branch.
- **Cost:** medium.
- **Breaks:** the warm-up guarantee. A bar three candles into a symbol's history becomes tradeable.
- **Only viable with a separate explicit warm-up check** (minimum bars since the symbol's first candle).

### Option C — Parity test (required either way)
Assert that the rows surviving `_to_dataset` and those surviving `Backtester._prepare` agree on the same
input frame.
- **Cost:** small.
- **Value:** pins the invariant permanently, whichever direction you resolve it in.

### Pick
**A + C.** Keep `_align`'s NaN-passing fix from `146b00c` — that part is correct and is a genuine
train/serve correctness improvement. It just addresses a different half of the problem than this one.

### Verify
The parity test passes on a frame containing NaN in `bb_position`.

---

<a name="p24"></a>
## P24 — `confidence_threshold_analysis` rewards total class collapse

**Verified:** accuracy climbs 0.4559 → 0.9982 across the sweep while balanced accuracy sits at **exactly
0.33333** from threshold 0.55 upward — the value a 3-class classifier takes when it emits a single class
for every input.

### Option A — Flag degeneracy per sweep row
```python
row["distinct_predicted_classes"] = int(len(np.unique(predictions_above_threshold)))
row["is_degenerate"] = row["distinct_predicted_classes"] < 2
```
and omit degenerate rows from any headline or recommendation.
- **Cost:** small.

### Option B — Report per-class recall in the sweep
`LONG recall 0.0, SHORT recall 0.0` next to `accuracy 0.9835` makes the table self-explanatory.
- **Cost:** small.

### Option C — Generic detector
Raise a recommendation whenever accuracy rises while balanced accuracy falls across a sweep. That
divergence is the generic signature of threshold-induced class collapse, worth detecting once for every
head rather than per-table.
- **Cost:** small (free under [P17](#p17)'s registry).

### Pick
**A + C.** This is the most persuasive-looking table in the Direction section and it measures the model
collapsing into silence — acting on it by raising `min_gate_confidence` toward 0.7 would produce a system
that never trades, with a 98% "accuracy" to justify it.

### Longer term
Sweep the *production rule* (accuracy on rows passing R1a+R1b, restricted to the side actually chosen)
rather than `argmax`. That sweep would be directly actionable; this one is not. Same point as
[P1](#p1) and [P11](#p11).

### Verify
The degenerate rows are flagged and excluded from recommendations.

---

<a name="p25"></a>
## P25 — ADX and DI saturate — *corrected mechanism*

**Verified, with a correction to the audit.** The audit said the denominators were unguarded. They are
not. `indicators.py:180-207` reads:

```python
atr_values = atr(high, low, close, window)
safe_atr   = atr_values.replace(0.0, np.nan)                 # ← guarded
plus_di    = 100.0 * wilder_smooth(plus_dm, window) / safe_atr
minus_di   = 100.0 * wilder_smooth(minus_dm, window) / safe_atr
di_sum     = (plus_di + minus_di).replace(0.0, np.nan)        # ← guarded
directional_index = 100.0 * (plus_di - minus_di).abs() / di_sum
adx_values = wilder_smooth(directional_index.fillna(0.0), window)
return adx_values, plus_di.fillna(0.0), minus_di.fillna(0.0)
```

The real mechanism is different, and so is the fix:

1. **DX is scale-invariant.** `100 × |a − b| / (a + b)` equals exactly 100 whenever one DI is zero,
   regardless of how small the other is. On a run of flat bars punctuated by a single directional tick,
   `plus_dm` is one tick and `minus_dm` is exactly 0 — so DX = 100. No division by zero occurs; the
   guards never fire. This is why `p95 = p99 = 99.9999999999999` and the training median is ≈ 100.
2. **`fillna(0.0)` feeds a lie into the smoother.** Bars where DX is genuinely undefined get 0 injected
   into `wilder_smooth`, which then propagates that 0 through the whole smoothing window.

### Option A — Require both DIs meaningfully positive
```python
both_present = (plus_di > _MIN_DI) & (minus_di > _MIN_DI)
directional_index = (100.0 * (plus_di - minus_di).abs() / di_sum).where(both_present)
adx_values = wilder_smooth(directional_index, window)   # no fillna — let NaN propagate
```
- **Cost:** small.
- **Breaks:** ADX becomes NaN on degenerate stretches instead of 100. Since [P2](#p2) removes those rows
  anyway, the practical effect on a clean dataset is small — but it stops the feature lying.
- **Rationale:** this is `_align`'s own stated principle applied one layer earlier. Both 0 and 100 are
  assertions about a market that produced no information; NaN is the truth, and LightGBM splits on it
  natively.

### Option B — Just drop the `fillna(0.0)` calls
Let `wilder_smooth`'s `min_periods` handle NaN.
- **Cost:** 3 lines.
- **Weakness:** fixes the injected zeros but not the 100-ceiling, which is the larger distortion.

### Option C — Add `saturation_rate` to feature statistics (detector)
Fraction of rows within epsilon of the observed min or max; recommend above ~2%.
- **Cost:** small.
- **Value:** surfaces ADX, `di_spread` (saturated at ±1 for >5% of rows at each tail, and exceeding its
  documented domain at `±1.0000000000000002`) and any future equivalent automatically.

### Pick
**A + C.** Also clamp `di_spread` to `[-1, 1]` and check why it exceeds it — 2e-16 of float error is
harmless in itself but indicates the expression is not in its numerically stable form.

### Verify
`saturation_rate` for `adx` below 2% on clean data; `di_spread` within its documented domain.

---

<a name="p26"></a>
## P26 — The archive loader cannot work as configured

**Verified:** `archive_loader.py:176` — `aiohttp.ClientTimeout(total=archive_timeout_seconds)` with
`archive_timeout_seconds = 120.0`. `:329` — `raw: bytes = handle.read()`. `archive_backfill_days = 740`,
`archive_max_concurrent_downloads = 4`.

Four independent fixes; none blocks the others.

### A — Bound stalls, not total transfer size
```python
timeout = aiohttp.ClientTimeout(
    total=None, connect=30.0, sock_connect=30.0,
    sock_read=self._settings.data.archive_timeout_seconds,
)
```
`ClientTimeout(total=...)` bounds the entire request including the body. Generous for a kilobyte-scale
`liquidationSnapshot` day; a guaranteed abort for a `bookTicker` day of tens to hundreds of MB, with four
downloads sharing bandwidth. This is why liquidation reached 52.8% and bookTicker reached 0%.
- **Cost:** small.

### B — Stream from the ZIP member
`zipfile.ZipExtFile` is a file object and `pandas.read_csv` accepts it directly:
```python
with zipfile.ZipFile(io.BytesIO(payload)) as archive:
    name = next(n for n in archive.namelist() if n.lower().endswith(".csv"))
    with archive.open(name) as handle:
        yield from _read_csv_chunks_from_handle(handle, _HEADERS[dataset])
```
The docstring already promises chunked reading; today `chunksize` is applied to a `BytesIO` over a fully
materialised buffer, so peak memory is the whole uncompressed file (gigabytes for a bookTicker day).
Header detection needs a small rework — peek the first line, then `seek(0)`.
- **Cost:** medium.

### C — Per-dataset backfill depth
740 days × 27 symbols = **19,980 bookTicker symbol-days**, on the order of a terabyte of transfer. Default
`bookTicker` to 30–90 days and keep 740 only for the cheap `liquidationSnapshot`.
- **Cost:** small.
- **Note:** the disk cache stores *reduced* frames, which is a good design — once a day is processed it is
  cheap forever. The problem is only the first pass, which argues for an incremental, resumable backfill
  rather than all-or-nothing.

### D — Surface coverage and fail on it
`backfill_market_microstructure` already returns a `coverage_summary`. Put it in the diagnostic report per
dataset and raise CRITICAL when `days_failed / days_requested` exceeds ~0.2. Log the first few failures at
ERROR with the URL.
- **Cost:** small.
- **Value:** today a run where every bookTicker day failed is indistinguishable, in the report, from one
  where the feature was never requested.

### Pick
**All four**, in order D → A → C → B. D first because it tells you whether the others worked.

### Preserve through any rewrite
The parsing conventions here are the part that was done well: by-name column addressing rather than
positional, header auto-detection across the headerless/headered generations, microsecond/millisecond
normalisation in `_bucket_ms`, exact-bucket joins rather than as-of (so an uncovered bucket cannot inherit
a reading from hours earlier), and NaN preservation with an explicit missingness flag. Those are exactly
the details that would have silently corrupted every downstream imbalance. The design is sound; only the
transport is broken.

### Verify
`coverage_summary` appears in the report per dataset; a single-symbol single-day bookTicker fetch
completes and writes non-null `ob_imbalance` for that day's buckets.

---

<a name="p27"></a>
## P27 — Reported class distribution is the whole dataset

**Verified:** `processor.py:234-237` — `class_distribution()` runs `value_counts()` over the entire
`direction_target` with no notion of splits. The reported values sum to 5,722,703 (= `valid_samples`)
while the adjacent `rows` field is 2,881,082 (train only).

### Option A — Per-split counts
```python
def class_distribution(self, index: np.ndarray | None = None) -> dict[str, int]:
    target = self.direction_target if index is None else self.direction_target.iloc[index]
    return {str(k): int(v) for k, v in target.value_counts().items()}
```
then emit `{"train": ..., "validation": ..., "test": ..., "all": ...}`.
- **Cost:** small.
- **Value:** the current presentation actively conceals a substantial shift. Whole-dataset LONG:SHORT is
  1,916,915 : 1,151,345 = **1.66 : 1**; validation is 368,796 : 393,733 = **0.94 : 1**. Since validation is
  one of the three components, train must be *more* long-skewed than 1.66:1 — directly relevant to
  [P4](#p4)'s directional behaviour, and currently requiring the reader to do the subtraction by hand.

### Option B — Add a shift metric
Total-variation distance between train and validation class distributions, with a recommendation above
~0.05.
- **Cost:** small (free under [P17](#p17)'s registry).

### Pick
**A + B.** Note the same whole-dataset value is reproduced verbatim in the `Labels` section, so the report
states it twice and never states the per-split breakdown once.

### Verify
Per-split counts sum to their split's row count.

---

<a name="p28"></a>
## P28 — Risk metrics computed on unclipped predictions

**Verified:** `ml_models.py:2018` — `predictions = estimator.predict(validation_features)` (unclipped);
`:2081` — `score = clamp(float(self._model.predict(aligned)[0]), 0.0, 1.0)` in `predict()`. Reported
`prediction_stats.max` is 1.0108644678784353 for a quantity documented as `[0, 1]`. The same pattern
exists at `:1761` for `ExitModel`, whose `_assemble` also clamps at inference.

### Option A — Clip before measuring
```python
predictions = np.clip(estimator.predict(validation_features), 0.0, 1.0)
```
plus `"clipped_prediction_rate": float(np.mean((raw < 0.0) | (raw > 1.0)))`, which is diagnostically
useful on its own.
- **Cost:** 2 lines per head.

### Option B — Adopt the convention: every head measures through its own `predict()`
- **Cost:** medium.
- **Value:** this is [P1](#p1)'s Option A generalised, and it closes both findings — plus `ExitModel`,
  where `_assemble`'s clamps and the ATR floor mean the reported regression metrics describe a function
  the system never executes either ([P15](#p15)).

### Pick
**B**, with **A** as the immediate local patch if B is deferred.

### Verify
No reported prediction statistic falls outside its documented domain.

---

<a name="p29"></a>
## P29 — Duplicates counted, never removed

**Verified:** `processor.py:451` computes `duplicate_feature_rows` (1,141) and nothing acts on it.
`split_coverage_pct.validation` shows `rows: 1418472` against `capacity_rows: 1418445` — 27 rows more
than the split's own time span has slots, one per symbol — yet `coverage_pct` reports a clean `1.0`.

### Option A — Deduplicate on the key
```python
before = len(usable)
usable = usable.drop_duplicates(subset=["symbol", "timestamp"], keep="last").reset_index(drop=True)
dropped_duplicate_rows = before - len(usable)
```
Dedupe on `(symbol, timestamp)`, not on the feature vector — two different bars can legitimately share a
feature vector, but two rows cannot share a key.
- **Cost:** small.

### Option B — Stop clamping `coverage_pct`
A value above 1.0 is a real signal, and clamping it removes the only automatic detector of this problem.
- **Cost:** 1 line.

### Option C — `UNIQUE(symbol, timestamp)` index in the database
- **Cost:** small (migration).
- **Value:** the real fix. QC *does* check for duplicate open times (`_find_duplicates`, at CRITICAL) and
  duplicates reached the dataset anyway — so either QC runs on a different scope than the ingestion write,
  or the duplication is introduced after QC. A unique constraint makes it structurally impossible and
  answers the question.

### Pick
**C** as the fix, **A + B** as the guards. 1,141 rows in 5.7 M is 0.02% and negligible for training, but
duplicated feature vectors split across train and test are exact-match leakage by definition, and the
upsert-key problem they point at could be much larger under other conditions.

### Verify
`coverage_pct` never exceeds 1.0; a deliberate duplicate insert is rejected by the database.

---

<a name="p30"></a>
## P30 — Dead code and unreachable logic

Four independent items. Each is small; each is a maintenance hazard specifically because it reads as an
active safeguard.

### (a) The Risk head's direction veto is unreachable
`ml_models.py:2105` — `if direction_confidence < decision.min_direction_given_trade_confidence`. R1b gates
on the same quantity against the same threshold and returns first, so in the `evaluate` path this can never
fire. The 281 `R6_RISK_MODEL_ABORT` rejections all came from the volatility veto or the leverage floor.
- **Option A:** delete it.
- **Option B:** keep it as defence-in-depth for callers that bypass the engine — but say so in a comment,
  so the next reader does not spend time working out why it never appears in the logs.
- **Pick:** B. `infer_sync` is callable directly and the veto is cheap.

### (b) `max_positions_per_symbol` is dead config
`settings.py:464` declares it; a repo-wide grep finds no other reference. The only per-symbol check is
`_check_system_gates`' `inference.symbol in state.open_symbols`, which hardcodes a limit of 1.
- **Option A:** implement it (count positions per symbol rather than testing membership).
- **Option B:** delete the field.
- **Pick:** B unless someone wants the feature. A configurable that silently does nothing is worse than no
  configurable.

### (c) R5's `stop_vs_labelled_atr` telemetry is NaN in the backtest
`c2968d8` added it precisely so a future regression could not silently reintroduce the geometry mismatch.
But `backtester.py:387-394` builds the snapshot with `"atr"` and not `"atr_pct"`, so
`feature_snapshot.get("atr_pct", 0.0)` returns 0.0, `labelled_stop` is 0.0, and the ratio is `nan` on every
backtest evaluation. **The guard rail added by this branch's most recent commit does not work in the
environment where it would first be exercised.**
- **Option A:** add `"atr_pct": float(row.get("atr_pct", 0.0))` to the snapshot.
- **Option B:** replace the ad-hoc snapshot dict with a named constant listing the keys the decision engine
  reads, so a consumer added in one place cannot be forgotten in another.
- **Pick:** B, and assert in a test that `stop_vs_labelled_atr` is finite and ≥ 1.0 in a backtest — which
  is what the telemetry was for.

### (d) Liquidation fills are idealised
`backtester.py:555-556` exempts liquidation from slippage (`exit_price = raw_price`), which is backwards —
liquidation is the fill most likely to be worse than its trigger. Stops fill at exactly the stop level plus
a flat 5 bps regardless of bar range, and stop clustering in crypto perps produces materially worse fills on
precisely the moves that trigger stops.
- **Option A:** apply slippage on liquidation too.
- **Option B:** scale the penalty by the bar's range relative to ATR instead of a flat constant.
- **Option C:** report the sensitivity — rerun at 5, 15 and 30 bps and publish all three.
- **Pick:** A + C. The run reports `liquidations: 0.0` so A costs nothing here, but leverage reaches 8× in
  the trade book and it is a latent understatement of tail risk. C is what tells you how much of the 52%
  return survives a realistic fill assumption.

### Verify
(a) a comment or a deletion, either way no unexplained unreachable branch; (b) `grep -rn
max_positions_per_symbol` finds either a real consumer or nothing; (c) `stop_vs_labelled_atr` is finite and
≥ 1.0 in a backtest, asserted by a test; (d) the slippage sensitivity table (5 / 15 / 30 bps) appears in the
report.

---

<a name="crosscutting"></a>
## Fixes that subsume several findings

Three changes each close a cluster. If effort is limited, these are the highest-leverage places to spend it.

**1. "Every head measures through its own `predict()`, after every mutation, on the final model."**
Closes [P1](#p1), [P28](#p28), and the latent `ExitModel` case in [P15](#p15). Add the identity guard from
P1's Option C and the class cannot recur.

**2. A check registry driving `_ai_summary` and `_recommendations`.**
Closes [P17](#p17) and [P18](#p18) outright, and provides the detector halves of [P3](#p3), [P24](#p24),
[P27](#p27) and [P29](#p29) as registry entries rather than one-off edits. Everything these checks need is
already in the report JSON.

**3. Backtest/live parity tests.**
Closes [P22](#p22) and [P23](#p23), and would have caught [P7](#p7) and [P30](#p30)(c). Three assertions:
identical barrier geometry from one signal, identical surviving-row sets from one feature frame, identical
selection ordering from one batch of inferences.

---

<a name="ordering"></a>
## Ordering and dependencies

The order matters more than usual here, because one fix changes what every other measurement means.

### Stage 1 — Make the report honest. No model changes.
[P1](#p1) · [P28](#p28) · [P27](#p27) · [P17](#p17) · [P18](#p18) · [P24](#p24) · [P12](#p12)

Then **re-run the diagnostic with nothing else changed.** The resulting report will look far worse than the
current one. That report is the real baseline; every later comparison is against it, not against the run in
this audit.

### Stage 2 — Fix the data. Requires a retrain.
[P2](#p2) · [P3](#p3) · [P26](#p26) · [P23](#p23) · [P25](#p25) · [P29](#p29)

Compare against the Stage 1 baseline. This is where the walk-forward collapse should resolve, and where it
first becomes possible to say whether any head has an edge.

> **Do not deploy anything from this branch to live or paper trading before Stage 2 completes.** The current
> artifacts are fitted on a training set that is 72% degenerate bars, sized by a head that has only ever seen
> winners, and validated by a report that measures a different model.

### Stage 3 — Fix the backtest.
[P5](#p5) · [P6](#p6) · [P7](#p7) · [P14](#p14) · [P13](#p13) · [P22](#p22) · [P19](#p19) · [P20](#p20) · [P30](#p30)

Expect the reported return and Sharpe to fall substantially. That is the point.

### Stage 4 — Fix the models.
[P8](#p8) · [P9](#p9) · [P10](#p10) · [P16](#p16) · [P11](#p11) · [P21](#p21) · [P15](#p15) · [P4](#p4)

[P4](#p4) comes last deliberately: whether stage 2 has any directional edge cannot be judged until the data
is clean ([P2](#p2)), the effective sample is real ([P9](#p9)), and the metrics describe the shipped model
([P1](#p1)).

### Hard dependencies
- [P2](#p2) → [P9](#p9): filtering degenerate rows shortens the window, so the half-life must be revisited
  in the same pass.
- [P1](#p1) → everything in Stage 4: no model decision is supportable until the metrics describe the
  shipped artifact.
- [P16](#p16) → [P1](#p1): applying P16's effect-size threshold changes *which* model ships, so P1's fix
  must be in place first or the new metrics will again describe the wrong one.
- [P7](#p7) → [P20](#p20): `signals_generated` is the denominator for every rejection rate and is currently
  wrong.
- [P26](#p26) → [P3](#p3) Option B: features cannot be restored before the loader can fetch them.

---

## Environment note

`pandas`, `numpy`, `scikit-learn` and `lightgbm` are **not installed** in the audit environment, so the
repo's 160+ tests could not be run and no claim is made about whether they pass. Every finding above was
derived from reading source at `c2968d8` and from the diagnostic output. Before applying any fix, install
the requirements and establish a green baseline — several of these changes will legitimately break existing
tests (particularly around `146b00c`'s NaN-tolerance tests and `c2968d8`'s geometry tests), and you need to
be able to tell a legitimate break from a regression.
