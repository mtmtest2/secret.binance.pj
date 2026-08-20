# Branch audit — `claude/archive-features-and-data-recovery`

**Branch head:** `c2968d8bda9a5d0e554bb3ac0bbb67148d964b53`
**Evidence base:** `ml_diagnostic_d0eb385d-a0ec-413b-8414-c90b64fdea8e` (`.json` + `.md`), generated
2026-08-14T17:23:14Z from git `c2968d8bda9a` — i.e. the report was produced by *this exact commit*, so
every number quoted below is this branch's own output, not an inference about it.

**Branch-specific commits under audit**

| Commit | Title |
|---|---|
| `146b00c` | Recover the discarded training window and restore archive-backed microstructure |
| `c2968d8` | Make the executed stop match the barrier the Direction model was priced on |

The audit also covers defects those two commits inherited or exposed, because the diagnostic run
measures the whole stack as it stands at `c2968d8`.

---

## Verdict up front

The report's own AI summary says:

```
- Overall status: **GOOD**
- Biggest data problem: none measured
- Biggest validation problem: none measured
- Recommended next action: Improve the risk model
```

Every one of those four statements is wrong, and each is wrong for a reason that is reproducible from
the same JSON the summary was generated from. Concretely:

* **The report does not describe the model that was shipped.** Every Direction metric in it was
  computed against the *raw* cascade; the artifact saved to disk and replayed in the backtest is the
  *isotonic-calibrated* cascade. The two behave so differently that the raw model never emits a single
  LONG prediction across 1,418,472 validation rows, while the shipped model opens 79% of its backtest
  trades LONG.
* **72% of the training set is degenerate.** `realized_vol_12_is_zero` has a train mean of 0.7235 —
  2.08 M of 2.88 M training rows are bars where twelve consecutive closes were identical. Validation
  sits at 0.0241. `146b00c` did not "recover a discarded training window"; it re-admitted the rows the
  previous NaN filter had been (accidentally) keeping out.
* **The feature work this branch is named after produced nothing.** All four bookTicker-derived
  columns are **100.00% NaN across all 5,722,703 rows**. 605 lines of loader and 443 lines of tests
  yielded zero usable values.
* **The backtest's Sharpe of 4.28 is not a performance estimate.** It comes from a simulation that
  skips the entry bar's price range, never runs the Risk Guard, uses a different position-selection
  policy from the live path, and computes Sortino with the wrong denominator.

Nothing here is a style objection. Each item below states the defect, the evidence, the mechanism, the
consequence, and a concrete fix.

---

## Severity index

| # | Problem | Severity |
|---|---|---|
| [P1](#p1) | Reported Direction metrics measure a model that was discarded before saving | **Blocker** |
| [P2](#p2) | 72% of the training set is zero-volatility degenerate candles | **Blocker** |
| [P3](#p3) | 4 features are 100% NaN, 3 more are constant — 7 of 57 are dead | **Blocker** |
| [P4](#p4) | Stage-2 direction model has no discrimination; isotonic amplifies a 10-point noise band | **Blocker** |
| [P5](#p5) | Backtest never tests the entry bar's own high/low against the barriers | **Critical** |
| [P6](#p6) | Backtest bypasses the Risk Guard entirely | **Critical** |
| [P7](#p7) | Backtest uses a different position-selection policy from the live path | **Critical** |
| [P8](#p8) | Risk and Exit heads are trained only on winning trades | **Critical** |
| [P9](#p9) | Recency half-life of 45 days shrinks a 372-day window to ~17% effective sample | **Critical** |
| [P10](#p10) | The validation block does four incompatible jobs at once | **Critical** |
| [P11](#p11) | Auto-tuned thresholds are arithmetically meaningless | **High** |
| [P12](#p12) | Backtest window is not the test split, and the two backtests use different windows | **High** |
| [P13](#p13) | Sortino uses the wrong denominator; Sharpe annualises a degenerate curve | **High** |
| [P14](#p14) | `opened_at` is wall-clock time — every trade has a negative holding period | **High** |
| [P15](#p15) | Exit model's SL head is dead code; trailing head learns `y = 0.5x` | **High** |
| [P16](#p16) | Calibration is adopted on unguarded 0.001-nat differences | **High** |
| [P17](#p17) | `_ai_summary` is structurally incapable of reporting a data problem | **High** |
| [P18](#p18) | `_walk_forward_problem_summary` can only ever return "none measured" | **High** |
| [P19](#p19) | Relaxed diagnostic backtest silently resets every other decision setting | **High** |
| [P20](#p20) | `rejection_breakdown` is first-match-wins and not comparable across runs | **Medium** |
| [P21](#p21) | Entry threshold auto-tune undercuts the configured floor; audit log prints the wrong number | **Medium** |
| [P22](#p22) | Trailing trigger is hardcoded in `_fill`, ignoring the Exit model | **Medium** |
| [P23](#p23) | Train/serve population mismatch on `bb_position` and `volume_trend` | **Medium** |
| [P24](#p24) | `confidence_threshold_analysis` rewards total class collapse | **Medium** |
| [P25](#p25) | ADX and DI saturate; no guard against a degenerate true range | **Medium** |
| [P26](#p26) | Archive loader cannot work: 120 s total timeout, whole CSV read into RAM | **Medium** |
| [P27](#p27) | Reported class `distribution` is the whole dataset, printed next to train-only `rows` | **Medium** |
| [P28](#p28) | Risk metrics computed on unclipped predictions; `predict()` clips | **Low** |
| [P29](#p29) | `duplicate_feature_rows` is counted and never removed | **Low** |
| [P30](#p30) | Dead code: Risk head's direction veto, `max_positions_per_symbol`, R5 telemetry | **Low** |

---

<a name="p1"></a>
## P1 — The reported Direction metrics measure a model that was thrown away · **Blocker**

### Evidence

From the artifact metadata:

```json
"production_calibration": {"gate": "isotonic", "direction": "isotonic"}
```

From the reported validation metrics for the same run:

```json
"per_class": { "LONG_SUCCESS": {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 368796} },
"predicted_class_distribution": {"LONG_SUCCESS": 0, "SHORT_SUCCESS": 242150, "NO_TRADE_OR_FAIL": 1176322}
```

The model that produced those metrics **never predicts LONG — not once in 1,418,472 rows.** Yet the
backtest replay of the *saved* artifact, over the held-out test split, opens:

```
side: LONG 393, SHORT 107   (of the 500 trades retained in the report)
```

79% long. A model that cannot emit a LONG prediction cannot produce a 79%-long trade book. These are
two different models.

### Mechanism

`module_c_ml/ml_models.py::DirectionModel.train` does this, in this order:

```python
probabilities = self._combined_probabilities(gate_estimator, direction_estimator, validation_features)
full_metrics  = ml_metrics.direction_metrics(validation_target, probabilities, LABEL_ORDER)   # ← metrics
...
calibration, self._model, production_calibration = self._calibrate_cascade(...)               # ← model replaced
```

`_combined_probabilities` is called with the **raw** `gate_estimator` / `direction_estimator`.
`_calibrate_cascade` then returns a *new* `model` dict whose `"gate"` and `"direction"` entries are
`CalibratedClassifierCV` wrappers, and assigns it straight onto `self._model`. That wrapped model is
what `save()` writes and what `predict()` — and therefore the backtester and the live executor — uses
from then on.

So the following are all computed on an estimator that is discarded seconds later:

* `metrics.accuracy` (0.4559), `balanced_accuracy` (0.3559), `log_loss`, `macro_f1`
* the entire `confusion_matrix` and `predicted_class_distribution`
* `probability_stats` (which is why `LONG_SUCCESS.max` reads 0.4291 — the raw cascade's ceiling)
* `confidence_threshold_analysis`
* `per_symbol` direction accuracy for all 27 symbols
* `gate_threshold_sweep` and `direction_threshold_sweep`, and hence
  `recommended_gate_threshold` / `recommended_direction_threshold`
* `feature_importance` for both stages

The audit trail in the trades themselves shows how far apart the two models are. One backtest trade's
recorded payload:

```json
"direction_probabilities": {"LONG_SUCCESS": 0.5099, "SHORT_SUCCESS": 0.2354, "NO_TRADE_OR_FAIL": 0.2548}
```

`LONG_SUCCESS = 0.5099`. The raw cascade's reported maximum LONG probability over the whole validation
set is **0.4291**. The shipped model routinely produces values the reported model could never reach.

### Consequence

Every conclusion anyone could draw from the Direction section is unsound. "Balanced accuracy is 0.356,
barely above the 0.333 chance floor" may or may not be true of the shipped model — the run contains no
measurement of it. The report also contains no metric anywhere that evaluates the *production decision
rule* (R1a on `trade_probability`, then R1b on `max(p, 1-p)` of stage 2); the confusion matrix scores
`argmax` over the three-way joint distribution, which is not a rule the system ever executes.

Note the same defect, milder, in `EntryModel.train` (metrics computed at `cutoff` before
`self._model` may be replaced by a calibrated wrapper) and in `RiskModel.train` (see [P28](#p28)).

### Fix

1. **Compute metrics from the model you are about to save, not the one you just fitted.** Move the
   metrics block to *after* `_calibrate_cascade`, and drive it through the same code path `predict()`
   uses, so the numbers are the shipped behaviour by construction:

   ```python
   calibration, self._model, production_calibration = self._calibrate_cascade(...)
   probabilities = self._combined_probabilities(
       self._model["gate"], self._model.get("direction"), validation_features
   )
   full_metrics = ml_metrics.direction_metrics(validation_target, probabilities, LABEL_ORDER)
   ```

   If a raw-vs-calibrated comparison is wanted, report **both**, explicitly labelled — do not report
   one and ship the other.

2. **Add a production-rule metric.** The number that matters is not `argmax` accuracy; it is:
   *among rows that pass R1a and R1b, what fraction of the chosen sides were correct, and what was
   the realised expectancy?* Add a `production_rule_metrics` block computing precision/recall of the
   executed side on the subset `trade_probability >= min_gate_confidence AND
   max(p,1-p) >= min_direction_given_trade_confidence`. That is the only accuracy figure that maps
   onto P&L.

3. **Add a regression test** that asserts the metrics in `self._metadata` are reproducible by calling
   `predict()` on a held-out slice — a mismatch beyond floating-point tolerance should fail the build.
   This class of bug is invisible to every existing test because nothing compares the two paths.

---

<a name="p2"></a>
## P2 — 72% of the training set is degenerate, zero-volatility candles · **Blocker**

### Evidence

The drift block, comparing train against validation in train-standard-deviation units:

| feature | train mean | validation mean | shift (train σ) |
|---|---|---|---|
| `realized_vol_12_is_zero` | **0.7235** | 0.0241 | 1.564 |
| `wick_ratio` | 0.0596 | 0.4517 | 2.349 |
| `whipsaw_rate` | 0.1158 | 0.5218 | 1.834 |
| `adx` | **74.14** | 30.55 | 1.341 |
| `fdi_trending` | 0.2489 | 0.6761 | 0.988 |
| `hmm_regime_age` | 1.4116 | 0.0996 | 0.762 |

`realized_vol_12_is_zero` is 1.0 exactly when the standard deviation of the last 12 log returns is
zero — i.e. **twelve consecutive 5-minute closes at the identical price.** That is true of 72.35% of
training rows and 2.41% of validation rows.

The other four rows in that table are all downstream consequences of the same thing, which is what
makes the diagnosis certain rather than suggestive:

* `wick_ratio` is `1 - clip(body/range)`, rolled and `.fillna(0.0)`. On a flat bar `range == 0`, so
  `body/range` is `0/0 → NaN`, the rolling mean stays NaN, and `fillna(0.0)` writes **0**. Train mean
  0.0596 is what you get when three quarters of the rows are forced zeros.
* `whipsaw_rate` counts sign flips in the return series. A flat bar has `sign(0) == 0`, so no flips —
  again forced toward 0. Train 0.1158 vs validation 0.5218.
* `adx`: in `module_b_features/indicators.py`, the directional index is
  `|+DI − −DI| / (+DI + −DI) × 100`. On a run of flat bars punctuated by a single directional tick,
  one DI is 0 and the other is tiny, so `DX = |a − 0| / (a + 0) × 100 = 100`. The training window's
  median ADX is ≈ 100 (validation median is ~30, and the reported `median_shift` is **−71.6**).
  The overall percentiles confirm the pin: `p95 = p99 = 99.9999999999999`.
* `hmm_regime_age` collapses because a flat series gives the HMM nothing to switch on.

### Mechanism — how this branch let them in

Two changes, each defensible alone, removed both filters that had been excluding these bars:

1. **`4384d08`** (inherited into this branch) downgraded exchange-confirmed `EXCESSIVE_ZERO_VOLUME`
   and return-outlier verdicts from CRITICAL to WARNING. QC has a `max_zero_volume_ratio` check
   (default 0.25) but **no zero-*range* check at all** — nothing anywhere asserts
   `high > low` or `close != open` over a window. After the downgrade, a genuinely flat, zero-volume
   bar that Binance re-serves byte-identically is accepted as healthy data.

2. **`146b00c`** changed `DatasetProcessor._to_dataset` from `dropna(subset=feature_columns)` to
   `dropna(subset=["label", "target_risk_score"])`. Previously, a flat bar produced NaN in
   `bb_position`, `vol_of_vol`, `garch_vol_ratio` and others, and the row was dropped. That is exactly
   the "73% of the training window" the commit message describes recovering — measured as
   `train 26.9% populated` against `validation 96.9%`.

   **0.269 ≈ 1 − 0.7235.** The rows the old filter dropped and the rows flagged
   `realized_vol_12_is_zero` are, to within a percent, the same rows.

The commit message reads that 73% as "a systematic sample-selection bias" caused by "the sparsest
features clustering in low-liquidity symbols and quiet periods". The first half of that diagnosis is
right — the drop *was* non-uniform. The conclusion drawn from it is wrong: those rows were not
under-sampled good data, they were bars with no price action to learn from. The commit then went
further and hardened the pipeline against ever dropping them again — `_MIN_REALIZED_VOL` floors the
denominator so `vol_of_vol` and `garch_vol_ratio` stay finite, and `realized_vol_12_is_zero` carries
the state explicitly. The result is that 2.08 M dead bars now flow all the way into the boosters.

### Consequence — this is what the walk-forward is showing

```
fold 1: train 858,917    accuracy 0.8034   balanced 0.7208   log_loss 0.506
fold 2: train 1,719,488  accuracy 0.6680   balanced 0.6012   log_loss 0.754
fold 3: train 2,580,032  accuracy 0.5322   balanced 0.5491   log_loss 0.883
fold 4: train 3,440,603  accuracy 0.4623   balanced 0.3771   log_loss 1.043
```

Accuracy falls **monotonically** as the expanding window advances — 0.80 → 0.67 → 0.53 → 0.46 — and
log loss rises monotonically. More data makes the model strictly worse, in perfect order, across four
folds. That is not variance; the report's summary of it (`accuracy_std=0.131`) discards the only
feature of the sequence that matters.

The reason is that fold 1 validates on the earliest ~20% of the window, which is almost entirely dead
bars: with ATR ≈ 0 the labeler's barriers are degenerate and the outcome is trivially predictable.
Fold 4 validates on real market data and scores 0.462. **The 0.46 is the honest number; the 0.80 is
an artifact.** The `accuracy_mean` of 0.6165 that the report headlines is a blend of a real
measurement and a fabricated one.

The link into the model is direct: `adx` is the **3rd most-used feature in the gate** (6.44% of splits)
and the **2nd most-used in the Entry model** (6.00%). The gate learned its split thresholds against an
ADX distribution centred at 74 that does not exist in any period the system will ever trade.

### Fix

1. **Add a structural zero-range check to QC, at CRITICAL, not downgradeable.** A bar with
   `high == low` is not a market observation. Something like:

   ```python
   flat = (highs == lows)
   flat_ratio = float(flat.mean())
   if flat_ratio > self._qc.max_flat_candle_ratio:   # new setting, suggest 0.02
       issues.append(QCIssue(code="EXCESSIVE_FLAT_CANDLES", severity=CRITICAL,
                             timestamps=tuple(ts[i] for i in np.flatnonzero(flat)[:50]), ...))
   ```

   Crucially this must **not** be in the set that `4384d08` downgrades for exchange-confirmed bars.
   Byte-identical re-service proves the feed is not corrupt; it does not make a flat bar tradeable.

2. **Exclude degenerate rows from training explicitly, and say so in the report.** Reinstate a filter
   — but on the *cause*, not on incidental NaN:

   ```python
   usable = usable[usable["realized_vol_12_is_zero"] < 1.0]
   ```

   and record `dropped_zero_volatility_rows` in `ProcessedDataset` so it appears in `dataset` and can
   be trended run-over-run. Dropping on a named condition is auditable; dropping on "any NaN anywhere"
   was not, which is the legitimate half of `146b00c`'s complaint.

3. **Investigate the upstream source before retraining.** All 27 symbols report *exactly* 212,352 rows
   (POL: 201,551) — a perfectly complete 5-minute grid with zero gaps over 738 days, for 26 of 27
   symbols. Real exchange history has gaps (maintenance halts, listing dates, delisting pauses). A
   perfect grid plus 72% flat bars is the signature of a materialised grid backfilled with padding.
   Establish whether the Aug 2024 – mid-2025 klines are genuinely what Binance serves, or whether the
   bootstrap/heal path is synthesising them. Until that is answered, retraining will just re-poison
   the model with a different filter in front of it.

4. **Gate the split on a data-health assertion.** Refuse to train when any split's
   `realized_vol_12_is_zero` mean exceeds, say, 0.05, or when the train-vs-validation mean shift of
   any feature exceeds ~1.0 train σ. Four features cleared 1.0 σ in this run and the pipeline
   proceeded to a "GOOD" verdict.

---

<a name="p3"></a>
## P3 — Four features are 100% NaN and three more are constant · **Blocker**

### Evidence

`dataset.null_counts_by_feature`, against 5,722,703 valid samples:

| feature | nulls | rate |
|---|---:|---:|
| `ob_imbalance` | 5,722,703 | **100.00%** |
| `ob_imbalance_delta` | 5,722,703 | **100.00%** |
| `ob_spread_bps` | 5,722,703 | **100.00%** |
| `ob_spread_rank` | 5,722,703 | **100.00%** |
| `liquidation_imbalance` | 2,701,073 | 47.20% |
| `bb_position` | 1,389,457 | 24.28% |
| `volume_trend` | 945,145 | 16.52% |

And `features.microstructure_coverage`:

```json
"open_interest_change":  {"non_neutral_row_fraction": 0.0, "likely_populated": false},
"long_short_ratio":      {"non_neutral_row_fraction": 0.0, "likely_populated": false},
"taker_buy_sell_ratio":  {"non_neutral_row_fraction": 0.0, "likely_populated": false},
"funding_rate":          {"non_neutral_row_fraction": 0.528, "likely_populated": true}
```

And `null_rate_by_symbol_month` — every symbol, every month from `2024-08` through `2026-08`:
**1.0**. Not one bucket, for any of 27 symbols, over 25 months.

### The headline result for this branch

`146b00c` exists to restore five microstructure features from `data.binance.vision`. It added
`module_a_data/archive_loader.py` (605 lines), a `market_microstructure` table, a disk cache, negative
caching, chunked parsing, header auto-detection, and 443 lines of tests concentrated on "the
conventions a bug would hide in".

**Four of the five features came back completely empty.** The fifth (`liquidation_imbalance`, from
`liquidationSnapshot`) reached 52.8% coverage. Since both datasets travel the same download → unzip →
parse → aggregate → upsert → as-of-join path, and one of them works, the plumbing is sound and the
failure is specific to `bookTicker`.

Counting the three constant futures columns, the model is carrying **7 dead features out of 57 —
12.3% of the feature space** — plus `microstructure_is_missing`, which is constant `1.0` for the four
`ob_*` columns and therefore carries no information either.

### Why `bookTicker` fails (see also [P26](#p26))

`config/settings.py`:

```python
archive_timeout_seconds: float = Field(default=120.0, gt=0.0)
archive_backfill_days:   int   = Field(default=740, ge=1)
archive_max_concurrent_downloads: int = Field(default=4, ge=1, le=16)
```

`archive_loader.py` applies that as a **total** timeout on the whole request:

```python
timeout = aiohttp.ClientTimeout(total=self._settings.data.archive_timeout_seconds)
```

Binance's daily `liquidationSnapshot` ZIPs are kilobytes. Daily `bookTicker` ZIPs for a liquid USDT-M
perp are **tens to hundreds of megabytes**, uncompressing to multiple gigabytes of event-level rows.
Four such downloads in parallel, each with 120 seconds to complete in full, will time out essentially
always; `aiohttp` raises `ClientError`, the retry loop burns its 3 attempts, and `_download` returns
`None`. The day is booked as failed and the loop moves on — quietly, because the summary only reports
percentages.

The request volume alone rules the current configuration out: 740 days × 27 symbols = **19,980
symbol-days** of `bookTicker`, which at realistic file sizes is on the order of a terabyte of transfer
and tens of terabytes of parsing.

The parser compounds it. Despite the module docstring's claim of "streamed parsing … read in chunks":

```python
with archive.open(names[0]) as handle:
    raw: bytes = handle.read()          # entire uncompressed CSV into memory
frames = _read_csv_chunks(raw, _HEADERS[dataset])
```

`chunksize` is applied to an `io.BytesIO` over a fully-materialised buffer. Nothing is streamed; the
peak memory is the full uncompressed file. On a container with a normal memory budget this is an OOM,
not a slow path.

### Consequence

* The five features are declared `OPTIONAL_FEATURE_COLUMNS`, so their absence does not block a bar —
  good design, and it is why the run completed. But it also means the failure is silent.
* Four columns of pure NaN are handed to LightGBM on every one of 5.7 M rows. The booster will route
  them all down one default branch and split count will be ~0, so they mostly waste tree capacity
  rather than corrupting predictions — but they are also carried through `_align`, serialised into
  every artifact, and inflate `feature_count` to 57 when the real count is 50.
* The report's only warning about any of this is a single **HIGH** recommendation naming the three
  *futures* columns. The four 100%-NaN `ob_*` columns are **not covered by any check** —
  `microstructure_coverage` inspects a hardcoded list of four futures features and never looks at the
  `ob_*` block at all.

### Fix

1. **Add a nullity gate to the report, at CRITICAL.** Any feature with a null rate above ~0.98 or a
   `unique_count` of 1 is not a feature. In `_recommendations`:

   ```python
   dead = [name for name, count in report["dataset"]["null_counts_by_feature"].items()
           if count / report["dataset"]["valid_samples"] > 0.98]
   if dead:
       critical.append(f"{len(dead)} feature(s) are effectively empty and must be removed or "
                       f"backfilled before this artifact is used: {', '.join(sorted(dead))}")
   ```

   This one check would have turned this run's verdict from GOOD to CRITICAL.

2. **Make the archive backfill's coverage a first-class, failing signal.** `backfill_market_microstructure`
   already returns a `coverage_summary`. Log it at ERROR and surface it in the diagnostic report when
   any dataset's coverage is below a threshold, per dataset — right now a 0% `bookTicker` coverage and
   a 53% `liquidationSnapshot` coverage are indistinguishable from the report's point of view.

3. **Fix the loader's transport before re-enabling `bookTicker`** — see [P26](#p26) for the specifics
   (separate read timeout, streaming decompression, sensible default day count).

4. **In the meantime, drop the four dead columns from `FEATURE_COLUMNS`.** Carrying them costs
   artifact compatibility on every retrain and buys nothing. Re-add them when coverage is
   demonstrated, in the same commit that demonstrates it.

---

<a name="p4"></a>
## P4 — Stage 2 has no discrimination, and isotonic calibration amplifies its noise · **Blocker**

### Evidence

`direction_threshold_sweep`, run on the raw stage-2 estimator over 762,529 validation trade rows:

| threshold | signals | precision | recall |
|---:|---:|---:|---:|
| 0.30 | 762,529 | 0.4836 | 1.000 |
| 0.35 | 762,529 | 0.4836 | 1.000 |
| 0.40 | 762,529 | 0.4836 | 1.000 |
| 0.45 | 29,961 | 0.5985 | 0.0486 |
| 0.50 | **0** | 0.0 | 0.0 |
| ≥0.55 | 0 | 0.0 | 0.0 |

Read the `signals` column: **every one of the 762,529 rows has `p_long ≥ 0.40`, and not one has
`p_long ≥ 0.50`.** The raw long-vs-short model's entire output support is the interval **[0.40, 0.50)**
— a band 10 percentage points wide — and 96% of it sits in [0.40, 0.45).

Precision at the loosest threshold, 0.4836, is exactly the LONG base rate among trade rows
(368,796 / 762,529 = 0.4837). The model adds nothing at all until 0.45, where it buys 11 points of
precision on 3.9% of rows.

### Two consequences, and they point in opposite directions

**On the raw model, the system is short-only by construction.** `DecisionEngine.evaluate`:

```python
action = TradeAction.LONG if long_given_trade >= 0.5 else TradeAction.SHORT
```

With `p_long < 0.5` everywhere, `action` is always SHORT. And R1b gates on
`max(p, 1−p) ∈ (0.50, 0.60]` against a configured `min_direction_given_trade_confidence = 0.60` —
so under the raw model R1b would reject essentially 100% of candidates. The configured threshold sits
exactly at the model's output ceiling.

**On the shipped model, the band is stretched to fill [0,1].** `_fit_production_calibrator` wraps
stage 2 in `CalibratedClassifierCV(..., method="isotonic")`, fit on the whole validation block.
Isotonic regression is a free-form monotone step function: handed an input confined to a 10-point
band, it will map that band across whatever output range the labels support. The backtest confirms it
— 79% LONG, and `direction_probabilities.LONG_SUCCESS = 0.5099` on a trade whose raw joint LONG
probability could not have exceeded 0.4291.

So the shipped system's directional confidence is **a monotone rescaling of a signal with essentially
no discriminative power.** An "88% confident" signal in the audit log is not an 88% estimate of
anything; it is the isotonic image of a raw value somewhere around 0.44 ± 0.02, and the difference
between a 0.55-confidence signal and a 0.88-confidence one may be two or three thousandths of raw
probability. R1b's 0.60 threshold, applied to that, is not selecting anything meaningful — it is
partitioning noise.

This is the honest answer to the question `c2968d8` set out to answer ("a signal can clear R1B at 88%
and still close at its stop"). The commit's own answer — exit geometry mismatch plus conditional-vs-
joint confusion — is real and worth fixing, but it is second-order. The first-order reason an 88%
signal loses is that the 88% is manufactured by a calibrator from a model that cannot tell long from
short.

### Fix

1. **Stop shipping isotonic on stage 2 until the raw model discriminates.** Isotonic is appropriate
   for correcting a *miscalibrated but informative* score. Applied to a 10-point band, it converts
   invisible noise into displayed confidence. Guard it:

   ```python
   raw_span = float(np.ptp(long_given_trade_validation))
   if raw_span < 0.20:
       _LOGGER.error("stage-2 output spans only %.3f - refusing to calibrate a degenerate score", raw_span)
       # leave the raw estimator in place and mark the head as unusable
   ```

   Record `raw_probability_span` per stage in the calibration block so this is visible every run.

2. **Report stage-2 discrimination directly.** The report gives ROC-AUC for Entry but not for the
   gate or for stage 2. Add both. Stage 2's AUC, on this sweep, is barely above 0.5; that single
   number would have made the situation obvious without reading a sweep table.

3. **Fix the training population first ([P2](#p2), [P9](#p9)).** A model trained on 72% dead bars,
   with a 45-day half-life over a 372-day window, is being asked to learn long-vs-short from roughly
   two months of effective data. Retrain on clean data before concluding the architecture is at
   fault.

4. **Then reconsider the label.** `LONG_SUCCESS` vs `SHORT_SUCCESS` is defined by which side reached
   `2×ATR` before `1×ATR` within 48 bars. On 5-minute crypto that is close to a coin flip by
   construction, and the base rate (0.4837) says so. If stage 2 still cannot beat 0.55 AUC on clean
   data, the honest conclusion is that this label is not learnable at this horizon, and the system
   should size on the gate alone rather than pretending to a directional edge.

5. **Make the direction gate two-sided in the sweep.** `direction_threshold_sweep` sweeps `p ≥ t` for
   `t ∈ [0.30, 0.85]`, but R1b gates on `max(p, 1−p)`, which is bounded below by 0.5. Half the sweep
   grid describes a rule that cannot exist. Sweep `max(p,1−p) ≥ t` for `t ∈ [0.50, 0.95]` instead —
   see [P11](#p11).

---

<a name="p5"></a>
## P5 — The backtest never tests the entry bar's own high/low against the barriers · **Critical**

### Evidence

`module_e_execution/backtester.py::_simulate`, per timeline bar, in order:

```python
for timestamp in timeline:
    # --- 1. Resolve barriers on open positions with THIS bar ---
    for symbol in list(positions):
        ... self._resolve_bar(position, row) ...

    # --- 2. Fill signals raised on the PREVIOUS bar ---
    for signal in pending:
        opened = self._fill(signal, fill_row, equity)     # fills at THIS bar's open
        positions[signal.symbol] = opened
```

A position filled at bar *T*'s open enters `positions` **after** step 1 for bar *T* has already run.
It is first tested against barriers in step 1 of bar *T+1*, using bar *T+1*'s high and low.

**Bar *T*'s own high and low are never compared to the stop, the take-profit, or the liquidation
price.** The position gets one full 5-minute bar of free adverse movement.

### Consequence

This is a systematic, one-directional optimism, and it lands hardest on exactly the trades that decide
the result. The stop is floored at `1×ATR` (see `ExitModel._assemble` after `c2968d8`). A 5-minute bar
whose range is comparable to ATR is completely ordinary. So a meaningful share of trades that would
have been stopped on their entry bar are instead carried forward, given a chance to recover, and
counted as something other than an immediate loss.

The report's own exit mix shows how much rides on this: of 500 retained trades, 284 closed at
STOP_LOSS, 125 at TRAILING_STOP, 91 at TAKE_PROFIT. Win rate is 0.4364 and profit factor 1.857 — with
`average_win = 3.876` against `average_loss = −1.616`, the whole edge is a 2.4:1 payoff on a sub-50%
hit rate. Moving even a few percent of trades from "recovered to a trailing win" into "stopped on
entry bar" attacks both terms at once.

`update_excursions` is also never called for bar *T*, so every trade's recorded MAE and MFE understate
the true path by one bar — which in turn corrupts any downstream analysis of path heat, and would
corrupt the labeler's risk tiering if it were sourced the same way.

### Fix

Reorder the loop so a freshly-filled position is resolved against the remainder of its own bar.
Fill first, then resolve — and resolve newly-opened positions too:

```python
for timestamp in timeline:
    # --- 1. Fill signals raised on the PREVIOUS bar, at THIS bar's open ---
    for signal in pending:
        ...
        positions[signal.symbol] = opened
    pending = []

    # --- 2. Resolve barriers on ALL open positions, including ones just filled ---
    for symbol in list(positions):
        ...
```

One caveat this exposes: within a single bar the code cannot know whether the high or the low came
first. `_resolve_bar` already handles that pessimistically for existing positions (the adverse level is
tested before the target), and the same convention should apply to entry-bar resolution — a bar that
touches both the stop and the target on the entry bar must book the stop. That keeps the bias
conservative rather than merely moving it.

Add a test that constructs a signal whose next bar gaps straight through the stop and asserts the
trade closes on that bar, at the stop, with a negative MAE recorded.

---

<a name="p6"></a>
## P6 — The backtest bypasses the Risk Guard entirely · **Critical**

### Evidence

`backtester.py::_simulate` builds one context and reuses it for every bar of the entire replay:

```python
context = DecisionContext(
    risk_guard_state="GREEN",
    trading_enabled=True,
    trading_mode="backtest",
    ...
    size_multiplier=1.0,
)
```

`risk_guard_state` is hardcoded `"GREEN"` and `size_multiplier` hardcoded `1.0`. The Risk Guard
(`module_e_execution/risk_guard.py`, 424 lines) is never instantiated, never consulted, and never
updated with realised P&L.

Meanwhile `config/settings.py::RiskSettings` configures, by default:

```python
daily_drawdown_red_pct    = 0.05     # halt on a 5% daily drawdown
daily_drawdown_yellow_pct = 0.03     # throttle at 3%
total_drawdown_red_pct    = 0.20
consecutive_losses_red    = 5        # halt after 5 consecutive losses
consecutive_losses_yellow = 3
yellow_size_multiplier    = 0.4      # size at 40% while YELLOW
max_daily_trades          = 40
require_manual_reset      = True
```

`DecisionEngine._check_system_gates` implements the RED block correctly:

```python
halted = state.risk_guard_state.upper() == "RED"
if halted:
    return self._reject(..., "Blocked: Risk Guard is RED - all new risk is forbidden until manual reset", ...)
```

It just never sees anything other than `"GREEN"` in a backtest.

### Consequence

The backtest reports `max_drawdown_pct = 0.14217`. The configured daily RED trip is 5% and the total
RED trip is 20%, with `require_manual_reset = True`. A 14.2% peak-to-trough excursion over 181 days
will cross the 5% *daily* threshold — and once RED trips, the live system stops opening positions and
stays stopped until a human resets it.

So the equity curve that produced `total_return_pct = 0.522`, `sharpe_ratio = 4.278` and
`calmar_ratio = 9.384` is **not reachable under the configured risk settings.** The live system would
have halted somewhere inside it and sat out the remainder. The report presents these numbers under an
`oos_disclosure` asserting the replay is a faithful out-of-sample measurement; it is faithful about the
*data* and silent about the *policy*.

The YELLOW path matters just as much and is subtler: at 3% daily drawdown or 3 consecutive losses,
live sizing drops to 40% via `size_multiplier`, which flows into
`DecisionEngine._build_signal`'s `allocation` computation. The backtest sizes every trade at full
conviction, forever. Since drawdowns cluster, the backtest is systematically largest exactly where
live would be smallest — which inflates recovery and deflates realised drawdown simultaneously.

### Fix

1. **Instantiate a real `RiskGuard` inside `_simulate` and feed it every close.** After
   `balance += self._book_close(...)`, push the realised P&L and the current equity into the guard,
   then read `guard.state` and `guard.size_multiplier` when building each bar's `DecisionContext`:

   ```python
   guard = RiskGuard(self._settings)
   ...
   for timestamp in timeline:
       ...
       state, multiplier = guard.evaluate(equity=equity, realised=closed_this_bar, timestamp=timestamp)
       context = DecisionContext(
           risk_guard_state=state,
           trading_enabled=True,
           trading_mode="backtest",
           ...,
           size_multiplier=multiplier,
       )
   ```

   Implement `require_manual_reset` faithfully — once RED, stay RED for the rest of the replay, and
   report `halted_at` in the `BacktestReport`. A backtest that ends in a permanent halt is a *result*,
   not a failure to be worked around.

2. **Enforce `max_daily_trades` in the same place.** At 786 trades over 181 days (4.3/day) the cap
   would not bind in this particular run, but that is luck, not design.

3. **Add `risk_guard_state` transitions to the report.** A block listing every GREEN→YELLOW→RED
   transition with timestamp and trigger would make the difference between "the strategy made 52%" and
   "the strategy made 52% and would have been halted on day 40" impossible to miss.

4. **Run both, and report both.** Keeping an unconstrained "model-only" replay is genuinely useful for
   isolating model quality from policy. It should be labelled as such and printed *next to* the
   policy-constrained number, never instead of it.

---

<a name="p7"></a>
## P7 — The backtest uses a different position-selection policy from the live path · **Critical**

### Evidence

`DecisionEngine.evaluate_many` exists precisely to allocate scarce position slots well, and documents
why:

```python
"""Evaluate a batch, respecting the portfolio cap across the batch.

Symbols are ranked by directional confidence so that when the portfolio
can only absorb two more positions, they go to the two strongest signals
rather than to whichever symbol happened to be first alphabetically.
"""
ranked = sorted(inferences, key=_direction_given_trade_confidence, reverse=True)
```

The backtester does not call it. `_simulate` does this instead:

```python
for symbol, rows in indexed.items():
    row = rows.get(timestamp)
    if row is None or symbol in positions:
        continue
    if len(positions) + len(pending) >= self._settings.decision.max_concurrent_positions:
        break
    decision = self._decide(symbol, row, context)
    generated += 1
    ...
```

`indexed` is built from `featured`, which `_prepare` builds by iterating `symbols` in universe order.
So the backtest allocates its 5 concurrent slots (`max_concurrent_positions = 5`) **in dictionary
insertion order, first-come-first-served**, and `break`s out of the symbol loop the moment the book is
full — never scoring the remaining symbols at all.

### Consequence

**The backtest measures a strategy nobody will run.** Live, the five slots go to the five
highest-confidence signals across the universe. In the backtest they go to whichever symbols appear
earliest in the iteration order and happen to fire first.

The trade book shows the resulting concentration. Of 500 retained trades:

```
strict  : PIXEL 57 (11.4%), ZEC 47, ALGO 43, MANA 32, ATOM 32, 1INCH 30, BCH 29, ZIL 29
relaxed : PIXEL 129 (25.8%), MANA 35, ATOM 30, GRT 29, ZIL 27, 1INCH 23, ALGO 23, STX 22
```

In the relaxed run, **one symbol accounts for a quarter of all trades** — a low-cap, thin-book perp,
on a 27-symbol universe. That is not a strategy characteristic; it is an artifact of iteration order
interacting with the `break`.

The `break` also corrupts the counters. `generated` is only incremented for symbols reached *before*
the break, so `signals_generated` is not the number of opportunities evaluated — it is the number
evaluated before the book filled, which depends on how fast the book fills, which depends on the
thresholds. That is why the strict and relaxed runs report different `signals_generated` (1,381,357 vs
1,362,545) over what the report claims is the same window, and it makes every rejection-rate percentage
in the report a function of an implementation detail. (See also [P20](#p20) and [P12](#p12).)

### Fix

1. **Have the backtester call `evaluate_many`, exactly as live does.** Collect one `ModelInferenceResult`
   per eligible symbol for the bar, then hand the whole batch over:

   ```python
   inferences = [self._infer(symbol, rows[timestamp])
                 for symbol, rows in indexed.items()
                 if timestamp in rows and symbol not in positions]
   generated += len(inferences)
   for decision in self._decisions.evaluate_many(inferences, context):
       if decision.is_executable and decision.signal is not None:
           pending.append(decision.signal)
       else:
           rejected += 1
           rejection_breakdown[decision.rule_triggered] = rejection_breakdown.get(decision.rule_triggered, 0) + 1
   ```

   This scores every symbol every bar (so `generated` becomes meaningful), applies the portfolio cap
   through `_check_system_gates` as `Rule.PORTFOLIO_FULL` rather than a silent `break`, and allocates
   slots by confidence.

2. **Add `Rule.PORTFOLIO_FULL` to the rejection breakdown legitimately.** Right now "we were full" is
   invisible; after the change it becomes a counted, reportable reason, which is diagnostically much
   more useful than its current absence.

3. **Add a per-symbol concentration block to the report.** Trades per symbol, P&L per symbol, and the
   share of net profit from the top symbol. A strategy taking 26% of its trades in one thin alt is a
   finding regardless of its cause.

4. **Add a test** asserting that when three symbols signal on the same bar and only one slot is free,
   the slot goes to the highest-confidence symbol — not the first in iteration order.

---

<a name="p8"></a>
## P8 — The Risk and Exit heads are trained only on winning trades · **Critical**

### Evidence

`RiskModel.train`:

```python
usable = dataset.direction_target != LabelClass.NO_TRADE_OR_FAIL.value
features = dataset.features[usable]
target   = dataset.risk_target[usable]
```

`direction_target != NO_TRADE_OR_FAIL` means `LONG_SUCCESS` or `SHORT_SUCCESS` — and per the labeler,
those are precisely the bars where a trade **reached take-profit before its stop, within the holding
window**. Not "bars where a trade was plausible": bars where a trade *won*.

The report records the filter and the resulting row count:

```json
"training_row_filter": "direction_target != NO_TRADE_OR_FAIL - see RiskModel.train docstring",
"rows": 1536896      // vs Direction's 2881082
```

`ExitModel` applies the same restriction implicitly — `_attach_model_targets` leaves
`target_tp_pct` / `target_sl_pct` / `target_trailing_pct` as `NaN` on every non-selected row, so the
regressors only ever see winners' geometry. Same row count: 1,536,896.

### The measurable damage

`risk` metrics on validation:

```json
"r2": 0.0789,
"target_stats":     {"mean": 0.5572, "median": 0.5525, "min": 0.0903, "max": 0.9983},
"prediction_stats": {"mean": 0.6507, "median": 0.6519, "min": 0.4217, "max": 1.0109}
```

Three things, all consequences of the same bias:

* **Prediction floor 0.4217 against a target floor of 0.0903.** The head's output covers only the top
  half of the target's range. *It has no way to say "this is a bad trade."* It has never seen one.
* **Mean bias of +0.094** (0.4 target-σ), systematically upward. Since `composite` — which drives both
  leverage and capital allocation — is linear in `score`, this is a systematic over-sizing of every
  position the system takes.
* **R² of 0.0789.** Effectively no explanatory power on the validation population.

The decision cascade confirms the head is inert: `R6_RISK_MODEL_ABORT` fired **281 times out of
1,381,357** candidate signals — 0.02%. A veto that fires on one signal in five thousand is not a veto.

### Why the docstring's justification does not hold

The docstring argues the filter is right because `predict()` is "only ever asked … how clean is the
path of a trade Direction/Entry have *already* approved."

That conflates two different things. At inference, Direction and Entry approve a **candidate** — a bar
they *predict* will work. The training filter selects on the **realised outcome** — bars that *did*
work. The realised outcome is not knowable at decision time; that is the entire problem the system
exists to solve. Direction's own balanced accuracy is near chance, so of the candidates it approves,
well under half are winners. The head is trained on a 100%-winner population and deployed on a
population that is at best ~44% winners. That is textbook survivorship bias, and the prediction floor
of 0.4217 is exactly its fingerprint.

The docstring's *stated* motivation is also revealing: the filter was introduced to raise R² ("the
single largest driver of this head's R² sitting far below Exit's"). Restricting a regressor's
training population to the easy half of the distribution will reliably raise in-sample R² while
destroying the head's actual job.

### Fix

1. **Train Risk on every bar a trade could be opened on, with the outcome-dependent target defined for
   all of them.** `target_risk_score` is currently `0.0` by fiat on `NO_TRADE_OR_FAIL` rows, which is
   what made the unfiltered population look unlearnable. Fix the target instead of the population:
   compute path heat (`1 − mae_ratio`) for *both* sides on every bar and take the side the cascade
   would have chosen, so a losing setup gets a genuine low score rather than a placeholder zero.

2. **If a filter is kept, it must be an inference-time-observable one.** Filtering on
   "Direction's predicted probability exceeded the gate" is legitimate — that is knowable live.
   Filtering on the realised label is not.

3. **Same fix for `ExitModel`.** TP/SL/trailing geometry learned only from paths that reached TP is
   geometry for winners. The head needs losers' paths to learn how wide a stop has to be to survive a
   trade that eventually fails.

4. **Report the population overlap explicitly.** Add to each head's metrics block: training positive
   rate vs the positive rate of the population `predict()` actually sees in the backtest. A gap from
   100% to 44% should be impossible to ship without noticing.

5. **Interim mitigation:** subtract the measured mean bias (+0.094) before sizing, and clamp
   `composite` so the sizing curve cannot express more confidence than the head has demonstrated.
   This is a band-aid, not a fix — but the head is currently over-sizing every position and that is
   live-capital risk.

---

<a name="p9"></a>
## P9 — A 45-day half-life shrinks the 372-day training window to ~17% effective sample · **Critical**

### Evidence

```python
recency_half_life_days: float = Field(default=45.0, ge=0.0)
```

```python
age_days = (timestamps.max() - timestamps) / 86_400_000.0
return np.power(0.5, age_days / half_life_days)
```

Against a training span of 371.9 days:

| row age | sample weight |
|---:|---:|
| 30 d | 0.630 |
| 60 d | 0.397 |
| 90 d | 0.250 |
| 180 d | 0.0625 |
| 372 d | **0.0032** |

Integrating the weight over the window: **64.7 effective days out of 371.9 — 17.4%.**

### Consequence

The report presents the training configuration as:

```json
"train_months": 12.0,
"rows": 2881082
```

Both numbers are nominal fictions. In weighted terms the model is fitted on roughly **two months** of
data. The oldest 180 days — half the window — carry, in aggregate, less weight than the newest 20 days.

This collides head-on with `146b00c`. That commit's entire purpose was to grow the training set 3.7×
by recovering rows the NaN filter had removed. Almost all of those recovered rows are in the older,
degenerate part of the window ([P2](#p2)), where the recency weight is between 0.003 and 0.06. The
commit paid a large correctness cost (poisoning the feature distributions the model splits on) to add
rows that the weighting then discards. The two design decisions actively work against each other, and
nothing in the report shows it.

It also plausibly explains the stage-2 short bias described in [P4](#p4). The effective training
period is the final ~2 months of the window (roughly June–August 2025). Whatever the directional
character of that stretch, the model has learned it and little else, and it is being asked to
generalise across the six-month validation block and a further six-month test block.

### Fix

1. **Report the effective sample size, not just the nominal one.** Two lines in the metadata:

   ```python
   weights = self._recency_weights(train_timestamps)
   "effective_train_rows": float(weights.sum()) if weights is not None else float(len(train_index)),
   "effective_train_days": float(weights.sum() / rows_per_day),
   ```

   `effective_train_rows ≈ 500,000` sitting next to `rows: 2,881,082` makes the trade-off visible.

2. **Set the half-life against the window, not in the abstract.** A half-life below ~1/4 of the
   training span discards most of it. For a 12-month window, 90–180 days is a defensible range;
   45 days is not. Alternatively set `recency_half_life_days = 0` (uniform) and let the
   train/validation/test split carry the recency argument — that is what a chronological split is
   *for*.

3. **Make it an experiment, not a constant.** Sweep the half-life over `{0, 45, 90, 180, ∞}` and
   report validation log loss for each. That is a 5-run experiment that settles the question with
   evidence rather than intuition.

4. **Add a config validator** warning when `recency_half_life_days < train_months × 30 / 4`, in the
   same style as the existing `purge_bars < max_holding_bars` warning in `config/settings.py`.

---

<a name="p10"></a>
## P10 — The validation block does four incompatible jobs at once · **Critical**

### Evidence

The same 1,418,472-row validation block is used, in `DirectionModel.train`, for all of:

1. **Early stopping** — `_fit_cascade(..., early_stopping=True)` passes it to `_fit_estimator` as the
   eval set, with `early_stopping_rounds = 50`. The number of trees is selected on this data.
2. **Metric reporting** — `direction_metrics(validation_target, probabilities, ...)`, the headline
   accuracy/log-loss/confusion matrix.
3. **Threshold auto-tuning** — `gate_threshold_sweep` and `direction_threshold_sweep` are run on it,
   and `_select_recommended_threshold` picks from those sweeps.
4. **Calibration fitting** — `_fit_production_calibrator(gate_estimator, validation_features, ...)`
   fits the shipped isotonic calibrator on **the entire** validation block, by design:

   ```python
   """This is the separate, production-facing step: fit the calibrator that will
   actually be used for inference, on the *entire* validation block (every held-out
   row available, not half of it) …"""
   ```

The same pattern appears in `EntryModel.train` and `RiskModel.train`.

### Consequence

The reported validation metrics are not held-out. Tree count was chosen to minimise loss *on this
data*; the metrics then report loss *on this data*. The optimism is small for a single early-stopping
decision on 1.4 M rows, but it is not zero and it is not disclosed.

The more serious issue is the calibrator. `calibrate_classifier` honestly measures whether isotonic
helps using `_temporal_half_split` (fit on the earlier half, score on the later half) — that part is
well done. But then the *shipped* calibrator is refit on all 1,418,472 rows, including the half used
to make the adoption decision. And the isotonic step function it learns is now fitted to the base
rates of a specific six-month period (Aug 2025 – Feb 2026) with no held-out data left to check whether
that mapping transfers. Given that the calibrator is doing the heavy lifting in stage 2
([P4](#p4)) — stretching a 10-point band across the whole probability range — this is the single most
overfit-prone component in the system and it has no validation at all.

The `oos_disclosure` block on the backtest is careful and correct about the *test* split:

```json
"note": "…never used for training, early stopping, calibration, threshold selection or model
         selection for any of the four heads…"
```

That is true, and it is the reason the backtest window is worth anything. But it does not rescue the
validation metrics, which the report presents with equal confidence and which carry all four
contaminations.

### Fix

1. **Split validation in two, permanently.** Give each head a `fit` sub-block (early stopping,
   calibration, threshold selection) and a `score` sub-block (metrics only, touched by nothing else).
   The existing `_temporal_half_split` already does the temporal division correctly; the change is to
   use it for the *production* calibrator too, not just the measurement:

   ```python
   calib_x, calib_y, score_x, score_y = _temporal_half_split(validation_features, is_trade_validation)
   calibrated_gate = self._fit_production_calibrator(gate_estimator, calib_x, calib_y)   # earlier half only
   full_metrics    = ml_metrics.direction_metrics(score_y, probs_on(score_x), LABEL_ORDER)  # later half only
   ```

   The docstring's argument for using every row ("a shipped artifact should not throw away data the
   diagnostic report doesn't need") is exactly backwards for isotonic regression, which will happily
   memorise whatever it is given.

2. **Label every metric with its contamination status.** A `metrics_provenance` field per head —
   `"held_out"` / `"used_for_early_stopping"` / `"used_for_calibration"` — costs nothing and prevents
   a clean-looking number from being read as clean.

3. **Move threshold selection to the fit half too**, and report the selected threshold's performance
   on the score half. A threshold picked and measured on the same rows is not a measurement.

---

<a name="p11"></a>
## P11 — The auto-tuned thresholds are arithmetically meaningless · **High**

### Evidence

```json
"recommended_gate_threshold": 0.35,       // configured min_gate_confidence:                     0.55
"recommended_direction_threshold": 0.30   // configured min_direction_given_trade_confidence:    0.60
```

### Two separate defects

**(a) The direction recommendation is not a valid threshold.** R1b gates on
`max(p, 1−p)`, which is bounded below by 0.5 by definition. A threshold of **0.30 cannot reject
anything** — it is a no-op that would disable R1b entirely. The recommendation is produced by
`EntryModel._select_recommended_threshold`, a one-sided F-β maximiser, applied to a sweep that also
sweeps one-sided (`p ≥ t`). Half the sweep grid (`t < 0.5`) describes a rule the decision engine
cannot express. The tuner is optimising over a space that does not correspond to the gate it feeds.

**(b) Both recommendations sit at or near the loosest grid point**, which is the signature of a model
with no usable discrimination rather than a genuine recommendation. Look at the gate sweep:

| threshold | signals | precision | recall | F0.5 |
|---:|---:|---:|---:|---:|
| 0.30 | 1,321,128 | 0.5760 | 0.9979 | ~0.63 |
| 0.35 | 1,318,893 | 0.5762 | 0.9967 | ~0.63 |
| 0.55 | 1,094,272 | 0.5875 | 0.8432 | — |
| 0.70 | 55,451 | 0.7253 | 0.0527 | — |
| 0.85 | 2,131 | 0.8423 | 0.0024 | — |

Precision moves from 0.576 (the trade base rate is 0.5375, so ~0.04 of lift) to 0.842 only by
discarding 99.8% of rows. F-β=0.5 is maximised by the loosest threshold in the grid because there is
no knee to find. The tuner returns 0.35 not because 0.35 is good but because nothing is.

**(c) `floor` is not a floor.** `_select_recommended_threshold(sweep, floor)` documents itself as
"Falls back to `floor` (the configured default) when nothing in the sweep qualifies" — and that is
exactly what it does, a *fallback*, never a *bound*:

```python
candidates = [row for row in sweep if row.get("meets_min_sample_size")]
if not candidates:
    return floor
...
return float(best["threshold"])          # may be far below `floor`
```

The parameter's name promises a constraint the code does not implement. For the Entry model this has
a direct live effect — see [P21](#p21).

### Fix

1. **Sweep the two-sided quantity for the two-sided gate.** Add a dedicated
   `direction_confidence_sweep(is_correct_side, max(p, 1−p))` over `t ∈ [0.50, 0.95]`, and select
   from that. Delete the one-sided grid points below 0.5 from the direction sweep entirely — they
   describe a rule that cannot be configured.

2. **Make `floor` actually floor.** One line:

   ```python
   return max(float(best["threshold"]), floor)
   ```

   and rename the parameter to `configured_floor` so the contract is unambiguous. If the intent was a
   default rather than a floor, rename it to `default` and stop passing
   `DecisionSettings.min_entry_probability` into it.

3. **Refuse to recommend when the sweep has no knee.** If the best F-β is within a few percent of the
   loosest grid point's, return `NOT_AVAILABLE` with a reason rather than a number:

   ```python
   if f_beta(best) - f_beta(candidates[0]) < 0.02:
       return None   # report "no informative threshold; sweep is flat"
   ```

   A recommendation of 0.35 reads as a considered suggestion. "The sweep is flat — this model does
   not discriminate" is the same information, honestly stated.

4. **Select thresholds on economics, not on F-β.** Note that every sweep row already has slots for
   `average_r`, `win_rate`, `profit_factor`, `expectancy`, `net_pnl`, `max_drawdown` — and every one
   of them reads `"NOT_AVAILABLE"` in this run. Wiring those up would let the tuner optimise the thing
   that matters instead of a classification proxy that is indifferent to the 2.4:1 payoff structure.

---

<a name="p12"></a>
## P12 — The backtest window is not the test split, and the two backtests differ · **High**

### Evidence

| | start | end |
|---|---|---|
| test split (`oos_disclosure`) | 2026-02-12T15:35 | 2026-08-14T06:35 |
| `backtest` | 2026-02-14T13:25 | 2026-08-14T11:10 |
| `backtest_diagnostic_relaxed` | 2026-02-14T16:40 | 2026-08-14T14:25 |

The relaxed run's disclaimer states: *"Same out-of-sample window as `backtest`."* The two windows
differ by 3h15m at the start and 3h15m at the end. Neither matches the test split. Both **end after
the test split ends** — by 4.6 and 7.8 hours respectively. Both **start ~2 days after** it begins,
discarding roughly 576 bars per symbol from the front of the held-out window.

`signals_generated` differs correspondingly: 1,381,357 vs 1,362,545.

### Mechanism

`main.py::_final_backtest_window` computes a **bar count**, and `Backtester.run` turns it into "the
most recent N bars":

```python
""" ``Backtester.run`` loads the *most recent* ``max_candles`` bars per symbol from the database;
    since this replay always runs immediately after training on the same dataset, "most recent N
    candles" and "the test split's own date range" are the same window …"""
```

That equivalence is false for two reasons, both operational:

* **The database keeps growing.** The live ingestion pipeline writes a new candle every 5 minutes
  while training and backtesting run. "The most recent N bars" is a moving target. The strict run and
  the relaxed run execute at different wall-clock moments, so they load different windows — the
  3h15m offset between them is exactly the time the strict run took.
* **Warm-up is subtracted twice.** `max_candles = test_bars + warmup_padding`, and then `_simulate`
  additionally trims `warmup_bars` timestamps off the front of the timeline — but `_prepare` has
  *already* dropped un-warmed rows via `dropna(subset=REQUIRED_FEATURE_COLUMNS)`. The front of the
  test window is cut twice, which is where the ~2-day shortfall comes from.

Meanwhile `oos_disclosure` is a **hardcoded literal**, not a measurement:

```python
oos_disclosure = {
    "note": "… Always 100% out-of-sample by construction.",
    "test_start": ms_to_datetime(split.test_start_ms).isoformat(),
    "test_end":   ms_to_datetime(split.test_end_ms).isoformat(),
    "test_rows_in_training_dataset": int(len(split.test_index)),
    "oos_fraction": 1.0,                       # ← asserted, never computed
}
```

It is attached verbatim to both reports, describing a window neither of them ran.

### Consequence

The out-of-sample claim happens to survive — the drift is *forward* in time, so the extra bars are
newer than any training data and the missing bars are simply unused. So this is not a leakage bug.
But:

* The strict and relaxed runs are **not comparable**, which is the entire point of running the relaxed
  one. Some of the difference in their results is window, not thresholds (see also [P19](#p19)).
* The report contains a false statement of fact ("Same out-of-sample window") and an asserted
  `oos_fraction` that nothing verifies. If a future change to the split logic *did* introduce overlap,
  this field would still read `1.0`.
* Roughly 1.1% of the held-out window is silently discarded every run, and which 1.1% depends on when
  the run started.

### Fix

1. **Select the backtest window by timestamp range, not by bar count.** Add a
   `start_ms` / `end_ms` parameter to `Backtester.run` and `_prepare`, and query the database for that
   explicit range. "Most recent N" is only correct on a static database and this database is not
   static.

2. **Compute `oos_fraction`, do not assert it.** After the replay, intersect the actual timeline with
   the training dataset's train+validation index and report the real overlap:

   ```python
   overlap = np.intersect1d(replay_timestamps, train_validation_timestamps)
   oos_fraction = 1.0 - len(overlap) / len(replay_timestamps)
   ```

   Then raise a CRITICAL recommendation when it is below 1.0. As written, the field can never detect
   the thing it exists to detect.

3. **Run the relaxed replay against a pinned window.** Capture the strict run's exact timeline and
   pass it to the relaxed run, so the only difference between them is the decision settings — which
   is what the comparison claims to isolate.

4. **Stop double-trimming warm-up.** `_prepare` already guarantees warm rows; drop the extra
   `warmup_bars` trim in `_simulate`, or stop adding `warmup_padding` in `_final_backtest_window`.
   Doing both is a bug.

---

<a name="p13"></a>
## P13 — Sortino uses the wrong denominator; Sharpe annualises a degenerate curve · **High**

### Evidence

```json
"sharpe_ratio":  4.2779,
"sortino_ratio": 2.0781
```

Sortino is *below* Sharpe on a distribution with `average_win = 3.876` and
`average_loss = −1.616` — strongly right-skewed. For a right-skewed return series, downside deviation
should be *smaller* than total standard deviation, so Sortino should exceed Sharpe. Getting the
opposite is a signal that the two are not computed on comparable denominators.

```python
@classmethod
def _sortino(cls, equity):
    returns  = cls._period_returns(equity)
    downside = returns[returns < 0.0]
    if downside.size == 0:
        return 0.0
    deviation = float(np.sqrt(np.mean(np.square(downside))))       # ← divides by downside.size
    return float(returns.mean() / deviation * math.sqrt(_BARS_PER_YEAR))
```

The standard downside deviation divides the sum of squared negative returns by the **total** number of
periods, not by the number of negative ones. Dividing by `downside.size` inflates the deviation by
`sqrt(N_total / N_downside)` and deflates Sortino by the same factor. The observed
`Sharpe / Sortino = 2.06` is consistent with roughly a quarter of bars carrying a negative return —
which, on an equity curve with 786 trades over ~52,000 bars, is about right.

### A second, larger problem with both

The equity curve has one point per timeline bar: `52,107` points over the replay
(`years = periods / _BARS_PER_YEAR` back-solves to 0.4957, and `0.4957 × 105,120 = 52,107`, matching
181 days of 5-minute bars). With 786 trades, the *overwhelming* majority of those per-bar returns are
either exactly zero or tiny mark-to-market moves on at most five open positions.

Annualising that by `sqrt(365 × 24 × 12)` treats 105,120 near-degenerate, heavily autocorrelated
observations per year as if they were independent draws. The effective sample is the **786 trades**,
not 52,107 bars. Per-trade, the numbers are far more modest: with expectancy 0.780 and a trade-P&L
standard deviation around 2.7, per-trade Sharpe is roughly 0.29.

A Sharpe of 4.28 also has to be read against [P5](#p5) (entry bar never tested), [P6](#p6) (no Risk
Guard), and [P7](#p7) (dict-order symbol selection). It is not a performance estimate.

### Fix

1. **Fix the Sortino denominator:**

   ```python
   downside = np.minimum(returns, 0.0)                       # keep every period
   deviation = float(np.sqrt(np.mean(np.square(downside))))  # divide by returns.size
   ```

2. **Report trade-based risk-adjusted metrics alongside the bar-based ones.** Per-trade Sharpe with an
   explicit trade count, and a bootstrap confidence interval on expectancy, are far more honest for
   786 observations than an annualised bar Sharpe.

3. **Report the effective sample.** Add `equity_curve_points`, `nonzero_return_bars`, and
   `trades_per_year` next to the ratios so a reader can see that the 4.28 rests on 786 events.

4. **Add a deflated Sharpe or a simple significance note.** With 786 trades and per-trade Sharpe ≈ 0.29,
   the standard error on annualised Sharpe is roughly `sqrt(trades_per_year / trades) ≈ 1.4` — i.e.
   the honest reading is "positive, imprecise", not 4.28.

5. **Add a unit test** on a synthetic curve with known Sharpe and Sortino. Both functions are pure and
   trivially testable, and neither is covered today.

---

<a name="p14"></a>
## P14 — `opened_at` is wall-clock time; every trade has a negative holding period · **High**

### Evidence

From the report's own trade records:

```
opened_at range across all 500 trades: 2026-08-14 12:59:44.444845+00:00 → 2026-08-14 14:08:30.280886+00:00
```

All 500 trades "opened" inside a 69-minute span — which is the wall-clock duration of the backtest
process, not the simulated window. And an individual trade:

```json
"opened_at": "2026-08-14 12:59:44.444845+00:00",
"closed_at": "2026-05-30 12:35:00+00:00"
```

**The close precedes the open by 76 days.** Computed across the 500 retained trades, the mean holding
period is **−11,546 bars**; the maximum is **−310.7 bars**. Not one trade has a positive duration.

### Mechanism

`module_e_execution/models.py`:

```python
opened_at: datetime = field(default_factory=_utcnow)
```

`Position.from_signal` never overrides it, and `Backtester._fill` never sets it — so it takes the
process's current time at construction. `_book_close`, by contrast, sets the close correctly from the
simulated bar:

```python
position.closed_at = datetime.fromtimestamp(timestamp / 1_000.0, tz=timezone.utc)
```

One end of the interval is simulated, the other is wall-clock.

### Consequence

* Any holding-period analysis is garbage. The labeler's `max_holding_bars = 48` cannot be validated
  against the backtest, so nobody can check whether executed trades respect the horizon the labels
  were built on — which is the *exact* class of geometry mismatch `c2968d8` set out to eliminate.
* The downloadable trades CSV added in `4844365` inherits this. Anyone opening it in a spreadsheet
  gets 786 rows with negative durations and identical open timestamps.
* Time-of-day, day-of-week, and regime attribution are all impossible, which matters given that
  `hour_cos` is the Entry model's 8th most-used feature.
* Paper and live trading are unaffected (there, wall-clock ≈ bar time), so this reproduces only in
  backtest — which is precisely where it does the most analytical damage.

### Fix

Set the open time from the simulated bar. In `Backtester._fill`, after constructing the position:

```python
position.opened_at = ms_to_datetime(int(row["timestamp"]))
```

Better, make it explicit in the constructor so it cannot be forgotten — add an `opened_at` parameter
to `Position.from_signal` defaulting to `_utcnow()`, and pass the bar timestamp from the backtester.

Then add an invariant that catches the whole class:

```python
assert position.closed_at >= position.opened_at, "closed before opened"
```

and a report field `holding_bars` per trade, with a summary block. A backtest whose mean holding
period is negative should not be able to produce a report.

---

<a name="p15"></a>
## P15 — Exit's stop head is dead code; its trailing head learns `y = 0.5x` · **High**

### Evidence — the trailing head

```json
"target_tp_pct":       {"r2": 0.20039042985291866,
                        "target_stats": {"mean": 0.020640876719458544, "median": 0.013167520117044636,
                                         "std": 0.022701554726192653}},
"target_trailing_pct": {"r2": 0.20039042985291866,
                        "target_stats": {"mean": 0.010320438359729272, "median": 0.006583760058522318,
                                         "std": 0.011350777363096327}}
```

The R² values are identical to **17 significant figures**. The mean, median and standard deviation of
the trailing target are each exactly half the TP target's. The labeler confirms it directly:

```python
simulation.optimal_tp_pct[target]       = optimal_tp
simulation.optimal_sl_pct[target]       = optimal_sl
simulation.optimal_trailing_pct[target] = optimal_tp * 0.5      # module_b_features/labeler.py:522
```

`target_trailing_pct` is a deterministic constant multiple of `target_tp_pct`. A separate LightGBM
regressor — 400 trees over 57 features on 1.5 M rows — is trained to learn `y = 0.5x`, and the report
presents its R² as an independent measurement of model quality, alongside a
`beats_rule_based_baseline: true` verdict.

### Evidence — the stop head

```json
"target_sl_pct": {"r2": 0.0519,
                  "target_stats":     {"mean": 0.003894, "median": 0.001790, "std": 0.008242},
                  "prediction_stats": {"mean": 0.002219, "median": 0.001677, "std": 0.001934}}
```

R² of 0.052 — essentially nothing. The prediction standard deviation (0.00193) is under a quarter of
the target's (0.00824): the regressor is emitting a near-constant.

Then `c2968d8` added this to `ExitModel._assemble`:

```python
if atr_pct > 0.0:
    stop_loss = max(stop_loss, atr_pct * self._settings.labels.sl_atr_multiple)   # sl_atr_multiple = 1.0
```

The model's median stop prediction is 0.00168 (0.168%). For any symbol whose ATR exceeds 0.168% of
price — which on 5-minute crypto perps is the normal case, not the exception — the floor wins and the
prediction is discarded. **The stop is, in practice, always exactly 1 × ATR.**

### Consequence

This is not an argument against `c2968d8`'s change, which fixed a real and well-diagnosed problem: a
stop floored at `0.5 × ATR` while the label assumed `1.0 × ATR` meant every MEDIUM- and HIGH-tier
labelled winner was stopped out by construction. The reasoning in that commit message is correct.

The problem is what it left behind and did not report. After the change:

* The SL regressor's output is discarded on essentially every bar. One of the Exit model's three heads
  is dead weight — still trained, still serialised, still measured, still reported with a
  `beats_rule_based_baseline: true` badge that describes a number nobody consumes.
* The TP head is partially overridden too: `take_profit = clamp(take_profit, stop_loss * 1.1,
  stop_loss * _MAX_REWARD_RISK)`, where `stop_loss` is now the ATR floor. TP is therefore pinned to a
  band defined by ATR rather than by the model.
* Together with the trailing head being a constant multiple, **the Exit model contributes almost
  nothing to the executed geometry**, which is now approximately "stop at 1 ATR, target in
  [1.1, 6] × stop". That may well be the right geometry — but it is a rule, and it should be stated as
  one rather than dressed as three learned regressions.

### Fix

1. **Delete the trailing regressor.** Compute `trailing_activation_pct = 0.5 × take_profit_pct`
   directly in `_assemble`. This removes a third of the Exit model's training cost and artifact size,
   and removes a metric that cannot fail. If a genuinely independent trailing target is wanted, define
   one — e.g. the excursion at which the *realised* path stopped making new highs — and label it
   properly.

2. **Decide the stop head's fate explicitly.** Either:
   * remove it, and state in `_assemble` that the stop is `labels.sl_atr_multiple × ATR` by design; or
   * keep it, but let it predict a *multiplier on ATR* in `[1.0, 3.0]` rather than a raw percentage.
     Then it has room to express something the floor does not already say, and the floor becomes a
     lower bound on a live parameter rather than a replacement for it.

3. **Report the override rate.** Add to the Exit metrics block: the fraction of validation rows where
   the ATR floor exceeded the model's prediction, and the same for the TP clamp. `floor_override_rate:
   0.97` is the number that turns "this head has R² 0.05" into "this head is not in the loop".

4. **Do not report `beats_rule_based_baseline` for an output that is discarded.** At minimum, gate the
   flag on the override rate.

---

<a name="p16"></a>
## P16 — Calibration is adopted on unguarded 0.001-nat differences · **High**

### Evidence

```json
"gate":      {"log_loss_raw": 0.6387282315683127, "log_loss_calibrated": 0.6380239320070472, "improved": true},
"direction": {"log_loss_raw": 0.6942705654225932, "log_loss_calibrated": 0.6877537584588903, "improved": true},
"joint":     {"log_loss_raw": 1.0165591544817398, "log_loss_calibrated": 1.0151163137984403, "improved": true}
```

The gate's improvement is **0.00070 nats — 0.11%**. The joint's is 0.00144 nats — 0.14%. The decision
rule is a bare strict inequality:

```python
improved = calibrated_logloss < raw_logloss
```

No effect-size threshold, no significance test, no minimum. Any improvement in the 16th decimal place
flips the shipped artifact onto a different estimator.

And the report's LOW recommendations then present these as findings:

```
- "direction gate (trade vs no-trade) isotonic calibration measurably improves log loss and is wired into inference"
- "direction long/short isotonic calibration measurably improves log loss and is wired into inference"
- "entry isotonic calibration measurably improves log loss and is wired into inference"
```

"Measurably" is doing a lot of work for a 0.11% delta on a single temporal half-split.

### Consequence

This is the switch that decides whether the system ships the raw cascade or the isotonic one — and
those two models behave completely differently ([P1](#p1), [P4](#p4): one never goes long, the other
goes long 79% of the time). That decision is currently made by a coin flip dressed as a measurement.
On a rerun with a different random seed or a slightly different validation boundary, the sign could
easily flip, and the shipped system would change character entirely with nothing in the report to
indicate why.

The evaluation split is also a single temporal half — one draw, no repetition, no error bar. A 0.11%
difference on one draw carries no information about which estimator generalises better.

### Fix

1. **Require a material effect size:**

   ```python
   MIN_RELATIVE_IMPROVEMENT = 0.01          # 1%
   improved = (raw_logloss - calibrated_logloss) / raw_logloss > MIN_RELATIVE_IMPROVEMENT
   ```

   On this run, that alone would have kept all three heads on the raw estimators — and the report
   would then have honestly shown a model that never predicts LONG, which is the finding that
   matters.

2. **Report the delta, not just the boolean.** Add `log_loss_improvement_pct` and
   `improvement_exceeds_threshold` to the calibration block, so a reader sees 0.11% rather than
   `improved: true`.

3. **Evaluate on more than one split.** Use 3–5 temporal folds within the validation block and require
   the improvement to hold in a majority of them. The machinery (`_temporal_half_split`) is already
   there; it just needs to be applied more than once.

4. **Guard against degenerate inputs** — see [P4](#p4): refuse to calibrate a score whose raw span is
   under ~0.2, regardless of what the log loss says. Isotonic on a 10-point band is not calibration,
   it is amplification.

---

<a name="p17"></a>
## P17 — `_ai_summary` is structurally incapable of reporting a data problem · **High**

### Evidence

The whole of the data-problem logic in `module_f_panel/diagnostics.py::_ai_summary`:

```python
dq = report.get("data_quality", {})
if isinstance(dq.get("symbol_exclusions_total"), int) and dq["symbol_exclusions_total"] > 0:
    warnings.append(f"{dq['symbol_exclusions_total']} symbol-cycle exclusion(s) recorded")
...
"biggest_data_problem": (
    f"{dq['symbol_exclusions_total']} symbol exclusion(s) this run"
    if isinstance(dq.get("symbol_exclusions_total"), int) and dq["symbol_exclusions_total"] > 0
    else "none measured"
),
```

`biggest_data_problem` reads exactly one integer: the QC symbol-exclusion count. It cannot see
`dataset.null_counts_by_feature`, `dataset.null_rate_by_symbol_month`, `features.drift`,
`features.statistics`, or `features.microstructure_coverage`. All of those are in the same report
object, three lines away.

So with four features at 100% NaN, three constant, 72% of training rows degenerate, and four features
drifting more than one train-σ between train and validation, `biggest_data_problem` reads
**"none measured"** — accurately describing what the function measures, which is nothing.

And `overall_status`:

```python
if critical:                       status = "CRITICAL"     # only when a head is literally NOT_TRAINED
elif warnings or regressions:      status = "WARNING"      # only from symbol exclusions / <30 trades / baseline regression
else:                              status = "GOOD"
```

Four heads trained, zero symbol exclusions, 786 ≥ 30 trades, no previous baseline to regress against →
**GOOD**.

### The strongest/weakest comparison is not a comparison

```python
scored["direction"] = direction_metrics["balanced_accuracy"]     # 0.3559
scored["entry"]     = entry_metrics["roc_auc"]                   # 0.6021
scored["risk"]      = risk_metrics["r2"]                         # 0.0789
strongest = max(scored, key=scored.get)
weakest   = min(scored, key=scored.get)
```

Three metrics on three incommensurable scales with three different null baselines, ranked against each
other on raw magnitude:

| head | metric | value | chance baseline | lift over chance |
|---|---|---:|---:|---:|
| direction | balanced accuracy | 0.3559 | 0.3333 | **+0.023** |
| entry | ROC-AUC | 0.6021 | 0.5000 | **+0.102** |
| risk | R² | 0.0789 | 0.0000 | **+0.079** |

The report concludes *"Strongest component: entry, Weakest component: risk"*. On lift over chance, the
weakest head is **direction** by a factor of three — and direction is the head the entire system's
edge depends on. The verdict is an artifact of R² living near zero while balanced accuracy lives near
one third.

The downstream recommendation inherits the error:

```python
elif weakest != NOT_AVAILABLE and scored.get(weakest, 1.0) < 0.55:
    next_action = f"Improve the {weakest} model - it is the weakest measured component"
```

→ *"Recommended next action: Improve the risk model."* Given [P1](#p1)–[P4](#p4), that is not the next
action.

Note also that **`exit` is collected into `heads` but never scored** — no key is extracted for it — so
the Exit model can never be named strongest or weakest regardless of its metrics. [P15](#p15) is
invisible to the summary by construction.

### Fix

1. **Score heads on lift over their own chance baseline, not on raw metric values:**

   ```python
   scored = {
       "direction": (balanced_accuracy - 1/3) / (1 - 1/3),
       "entry":     (roc_auc - 0.5) / 0.5,
       "exit":      max(0.0, mean_r2_across_targets),
       "risk":      max(0.0, r2),
   }
   ```

   Every entry is then "fraction of the available headroom captured", on one comparable [0,1] scale,
   and `exit` is included.

2. **Feed the data blocks into `biggest_data_problem`.** Rank the candidates that already exist in the
   report and take the worst:

   ```python
   candidates = []
   dead = [n for n, c in null_counts.items() if c / valid_samples > 0.98]
   if dead:
       candidates.append((3.0, f"{len(dead)} feature(s) are ~100% null: {', '.join(sorted(dead))}"))
   worst_drift = report["features"]["drift"]["most_drifted_features"][0]
   if worst_drift["mean_shift_in_train_std"] > 1.0:
       candidates.append((worst_drift["mean_shift_in_train_std"],
                          f"{worst_drift['feature']} shifts {worst_drift['mean_shift_in_train_std']:.2f} "
                          f"train-σ between train and validation"))
   if dq.get("symbol_exclusions_total", 0) > 0:
       candidates.append((0.5, f"{dq['symbol_exclusions_total']} symbol exclusion(s)"))
   biggest = max(candidates)[1] if candidates else "none measured"
   ```

3. **Let data problems drive `overall_status`.** Any ~100%-null feature, or any drift above ~1.5
   train-σ, should force CRITICAL. Every check listed above already has its inputs sitting in the
   report; only the wiring is missing.

4. **Add a "not measured" section.** `regimes` reports
   `"per-regime performance breakdown is not computed by this run"` and every economic column in every
   threshold sweep reads `NOT_AVAILABLE`. A GOOD verdict that rests partly on checks that did not run
   should say which ones did not run.

---

<a name="p18"></a>
## P18 — `_walk_forward_problem_summary` can only ever return "none measured" · **High**

### Evidence

The function, in full:

```python
def _walk_forward_problem_summary(walk_forward: dict[str, Any]) -> str:
    if not isinstance(walk_forward, dict) or walk_forward.get("status") != "AVAILABLE":
        return "walk-forward evaluation not available (single train/validation split only)"
    std   = walk_forward.get("accuracy_std")
    folds = walk_forward.get("n_folds", "?")
    if isinstance(std, (int, float)):
        return f"none measured (walk-forward across {folds} folds, accuracy std={std:.3f})"
    return f"none measured (walk-forward across {folds} folds)"
```

Once `status == "AVAILABLE"`, **every return path begins with the literal string "none measured"**.
There is no threshold on `accuracy_std`, no trend test, no comparison of the last fold against the
mean. The function is named as a problem detector and is incapable of reporting a problem.

### What it failed to report

```
fold 1: accuracy 0.8034   balanced 0.7208   log_loss 0.506   macro_f1 0.725
fold 2: accuracy 0.6680   balanced 0.6012   log_loss 0.754   macro_f1 0.609
fold 3: accuracy 0.5322   balanced 0.5491   log_loss 0.883   macro_f1 0.527
fold 4: accuracy 0.4623   balanced 0.3771   log_loss 1.043   macro_f1 0.324
```

Four folds, four metrics, **strictly monotone degradation in every one of them.** Accuracy falls 34
points; macro-F1 falls 40 points; log loss doubles. The probability of four independent metrics all
ordering monotonically by chance is negligible — this is a structural trend, and it is the single most
important measurement in the entire report ([P2](#p2) explains why).

The report renders it as: *"Biggest validation problem: none measured (walk-forward across 4 folds,
accuracy std=0.131)"*.

`accuracy_std = 0.131` is the wrong summary statistic. It is the standard deviation of a monotone
sequence, and it would take the same value for the sequence shuffled into any order — including an
order that would be genuinely benign noise. Reducing a trend to a spread destroys the only information
it carries.

### Fix

1. **Test the trend and report it.** Rank correlation between fold index and accuracy is three lines:

   ```python
   from scipy.stats import spearmanr
   rho, p = spearmanr(range(1, len(folds) + 1), [f["accuracy"] for f in folds])
   ```

   With rho = −1.0, the summary should read something like: *"accuracy degrades monotonically across
   all 4 folds (0.803 → 0.462, Spearman ρ = −1.0); the earliest fold is not representative of the
   period the model will trade."*

2. **Threshold on the spread as well.** `accuracy_std > 0.05` across folds is worth reporting on its
   own; 0.131 is very large.

3. **Report the last fold separately, and prefer it.** Fold 4 validates on the most recent, most
   representative data. Its accuracy of 0.4623 is the closest thing in the report to an honest
   estimate — and it is buried in an array while `accuracy_mean = 0.6165` is headlined. Surface
   `latest_fold_accuracy` next to the mean.

4. **Escalate to `overall_status`.** A monotone four-fold collapse should force at least WARNING, and
   arguably CRITICAL when the final fold is within a few points of the chance baseline (0.4623 vs a
   majority-class rate of ~0.46 — the last fold is, on accuracy alone, indistinguishable from always
   predicting NO_TRADE).

---

<a name="p19"></a>
## P19 — The relaxed diagnostic backtest silently resets every other decision setting · **High**

### Evidence

```python
_RELAXED_DECISION_SETTINGS: Final[DecisionSettings] = DecisionSettings(
    min_gate_confidence=0.50,
    min_direction_given_trade_confidence=0.52,
    max_no_trade_probability=0.50,
    min_entry_probability=0.50,
    min_reward_risk_ratio=1.0,
)
...
relaxed_settings = self.settings.model_copy(update={"decision": _RELAXED_DECISION_SETTINGS})
```

`DecisionSettings(...)` is constructed with five fields. Every other field on that model — and there
are many — **reverts to its class default**, discarding whatever the operator configured:

* `max_leverage`, `min_leverage`
* `min_capital_allocation_pct`, `max_capital_allocation_pct`
* `max_concurrent_positions`, `max_positions_per_symbol`
* `accepted_risk_tiers`
* `blocked_hmm_regimes`
* `max_volatility_percentile`

If an operator has tuned any of these in `.env`, the relaxed run silently ignores them.

### The second, subtler problem

`min_direction_given_trade_confidence` is not only a decision threshold. `RiskModel.predict` reads it
as the base of its sizing curve:

```python
confidence_span   = max(1e-6, 1.0 - decision.min_direction_given_trade_confidence)
confidence_factor = clamp((direction_confidence - decision.min_direction_given_trade_confidence)
                          / confidence_span, 0.0, 1.0)
composite = score * (0.35 + 0.65 * confidence_factor) * volatility_factor * tier_factor
```

Strict: `confidence_span = 1 − 0.60 = 0.40`. Relaxed: `1 − 0.52 = 0.48`. **Changing the threshold
changes the sizing of every trade**, because `confidence_factor` is rescaled against it. So the
relaxed run is not "the same trades with a looser filter" — it is a different filter *and* a different
position-sizing curve.

### Consequence

The comparison the relaxed run exists to enable is invalid, and the report draws a conclusion from it
anyway:

```
LOW: "The same window with loosened decision thresholds … produced 1342 trade(s) vs 786 under the
      live thresholds - the low live trade count looks like the cascade's intended selectivity
      rather than a data/signal-generation bug"
```

The relaxed run returns +117.5% against the strict run's +52.2%. Some of that gap is thresholds, some
is sizing, some is reset settings, and some is a different time window ([P12](#p12)). None of it is
separable as the code stands. The conclusion "the cascade's selectivity is intended" is not supported
by the experiment that produced it — and the opposite reading is at least as available: the configured
thresholds are strictly worse than looser ones on the held-out window, which points at [P11](#p11).

### Fix

1. **Derive the relaxed settings from the live ones instead of constructing a fresh object:**

   ```python
   _RELAXED_OVERRIDES = {
       "min_gate_confidence": 0.50,
       "min_direction_given_trade_confidence": 0.52,
       "max_no_trade_probability": 0.50,
       "min_entry_probability": 0.50,
       "min_reward_risk_ratio": 1.0,
   }
   relaxed_decision = self.settings.decision.model_copy(update=_RELAXED_OVERRIDES)
   relaxed_settings = self.settings.model_copy(update={"decision": relaxed_decision})
   ```

   Now only the five intended fields differ.

2. **Decouple the sizing base from the gate threshold.** Give `RiskSettings` its own
   `sizing_confidence_base` (defaulting to the current 0.60) so that moving a decision threshold does
   not silently reprice every position. This is a latent hazard in live tuning too, not just in the
   diagnostic.

3. **Report the effective settings diff.** Emit the actual field-by-field delta between the strict and
   relaxed `DecisionSettings` into the relaxed block, so any unintended difference is visible in the
   report rather than in the source.

4. **Soften the LOW recommendation's conclusion** until the comparison is clean, and state explicitly
   what the relaxed run controls for and what it does not.

---

<a name="p20"></a>
## P20 — `rejection_breakdown` is first-match-wins and not comparable across runs · **Medium**

### Evidence

| rule | strict | relaxed |
|---|---:|---:|
| `R1B_DIRECTION_CONFIDENCE_TOO_LOW` | 1,033,527 | 0 |
| `R1A_GATE_CONFIDENCE_TOO_LOW` | 304,984 | 85,780 |
| `R3_NO_TRADE_MASS_TOO_HIGH` | 41,070 | 0 |
| `R4_ENTRY_MODEL_SAYS_WAIT` | **568** | **1,268,257** |
| `R6_RISK_MODEL_ABORT` | 281 | 7,166 |
| `R5_REWARD_RISK_BELOW_FLOOR` | 141 | 0 |

R4 goes from 568 to 1,268,257 — a factor of **2,233** — when thresholds are *loosened*.

### Mechanism

`DecisionEngine.evaluate` returns on the first failing rule, and `_simulate` records only that one:

```python
rejection_breakdown[decision.rule_triggered] = rejection_breakdown.get(decision.rule_triggered, 0) + 1
```

In the strict run, R1a and R1b together reject 1,338,511 of 1,381,357 candidates before R4 is ever
evaluated. R4's count of 568 is not "the entry model rarely objects" — it is "the entry model was
almost never asked". Loosen R1a/R1b and R4's true objection rate becomes visible: it wants to reject
roughly 93% of everything it sees.

### Consequence

The breakdown reads as an attribution of *why* signals are rejected, and it is not one. Anyone tuning
this cascade on the strict numbers would conclude the Entry model is nearly inert and focus on
Direction; the relaxed numbers say the Entry model is by far the most restrictive gate in the system.
Both conclusions come from the same run.

The `_recommendations` code takes `max(breakdown, key=breakdown.get)` as "top rejection reason", which
inherits the bias directly.

Compounding this, `signals_generated` is itself unreliable ([P7](#p7)) because the `break` on
`max_concurrent_positions` stops the symbol loop early, so the denominator differs between runs
(1,381,357 vs 1,362,545) even on nominally the same window.

### Fix

1. **Evaluate every rule and record every failure.** `DecisionEngine.evaluate` already builds a
   complete `checks` list via `_record` before returning; it just returns early. Keep the early return
   for the *verdict*, but count all failures:

   ```python
   rejection_breakdown_all[rule] += 1 for every check in decision.checks where passed is False
   ```

   Report both: `rejection_breakdown` (first-match, which rule stopped it) and
   `rejection_breakdown_independent` (how often each rule would object on its own). The second is the
   one that supports tuning.

2. **Add a per-rule pass rate conditional on reaching that rule** — `reached` and `passed` counts per
   rule. `R4: reached 42,846, passed 42,278 (98.7%)` is unambiguous in a way that `R4: 568` is not.

3. **Fix `signals_generated`** via [P7](#p7) so the denominator is the number of candidates actually
   scored.

---

<a name="p21"></a>
## P21 — Entry threshold auto-tune undercuts the configured floor, and the audit log prints the wrong number · **Medium**

### Evidence

```json
"decision_threshold": 0.45,
"configured_floor_threshold": 0.55
```

`EntryModel.predict` uses the auto-tuned value, not the configured one:

```python
cutoff = (threshold if threshold is not None
          else self._metadata.get("decision_threshold", self._settings.decision.min_entry_probability))
...
should_enter = probability >= cutoff
```

So the live gate is 0.45. The operator configured 0.55. And `_select_recommended_threshold` was passed
0.55 as `floor` but only uses it as a fallback ([P11](#p11)), so nothing prevents the tuner from
selecting below it.

Then `DecisionEngine.evaluate`'s R4 rejection message reports a threshold that was never applied:

```python
return self._reject(
    inference, Rule.ENTRY_REJECTED,
    f"Rejected: entry model says wait for the next 5m candle "
    f"(p={entry.probability:.3f} < {self._config.min_entry_probability:.3f})",   # ← prints 0.550
    checks,
)
```

A bar rejected at `p = 0.42` logs *"0.420 < 0.550"*, implying a 0.55 gate. The actual gate was 0.45.
Every R4 line in the audit log — 568 in the strict run, 1,268,257 in the relaxed one — states a
threshold the system did not use.

### Consequence

* An operator's configured risk control is silently overridden by a value the model chose for itself
  on validation data it also early-stopped on ([P10](#p10)).
* The audit log, which exists so decisions can be reconstructed after the fact, misstates the decision
  rule. Anyone reconstructing a rejected trade from the log will compute the wrong counterfactual.
* The `_record` call just above the rejection has the same problem — it stringifies
  `entry.probability` and `entry.reason` but not the actual cutoff, so the structured `checks` payload
  does not carry the truth either.

### Fix

1. **Make the floor a floor** (see [P11](#p11)): `return max(float(best["threshold"]), floor)`.

2. **Log the threshold that was applied.** Surface it on `EntryPrediction`:

   ```python
   return EntryPrediction(probability=..., should_enter=probability >= cutoff,
                          threshold=cutoff, source=..., reason=...)
   ```

   and use `entry.threshold` in both the `_record` detail and the rejection message.

3. **Warn loudly when the two diverge.** At load time, if
   `metadata["decision_threshold"] != settings.decision.min_entry_probability`, log at WARNING and put
   it in the report's HIGH recommendations. An auto-tuned override of an operator setting should never
   be silent.

4. **Decide the precedence deliberately.** Either the config is authoritative and the auto-tuned value
   is advisory (reported, not applied), or the auto-tuned value is authoritative and the config field
   is renamed to `entry_probability_floor`. The current arrangement — a config field that looks
   authoritative, an override that is not announced, and a log line that prints the config value — is
   the worst of the three.

---

<a name="p22"></a>
## P22 — The trailing trigger is hardcoded in `_fill`, ignoring the Exit model · **Medium**

### Evidence

`ExitModel._assemble` computes a trailing activation level and `DecisionEngine._build_signal` converts
it to a price:

```python
trailing_activation = clamp(trailing, take_profit * 0.25, take_profit * 0.95)
...
trailing_trigger = price * (1.0 + trailing_pct)          # _build_signal
```

`Backtester._fill` then throws it away and substitutes a constant:

```python
position.take_profit     = fill_price * (1.0 + signal.take_profit_pct)
position.stop_loss       = fill_price * (1.0 - signal.stop_loss_pct)
position.trailing_trigger = fill_price * (1.0 + signal.take_profit_pct * 0.5)   # ← hardcoded 0.5
```

The re-anchoring itself is correct and well-motivated (the comment explains it: barriers were computed
off the decision-bar close, and must be re-anchored to the actual fill). But `take_profit_pct * 0.5`
is not the re-anchored form of `signal.trailing_trigger` — it is a different rule. The re-anchored
form would be `fill_price * (1.0 + signal.trailing_activation_pct)`, and `trailing_activation_pct` is
not even carried on `TradeSignal` (only the absolute `trailing_trigger` is).

### Consequence

* **The Exit model's trailing head has no effect in the backtest.** Combined with [P15](#p15) (the
  target is `0.5 × TP` by construction, so the head learns a constant multiple), the entire trailing
  path is a hardcoded rule wearing a model's clothes. `_assemble`'s clamp to
  `[TP × 0.25, TP × 0.95]` never binds because the value is discarded downstream.
* **Backtest and live diverge.** `Executor._maybe_advance_trailing` and `PaperTrader` both call
  `position.advance_trailing(price)`, which arms against `position.trailing_trigger` — set from
  `signal.trailing_trigger` in `Position.from_signal`, i.e. the model's value. Only the backtest
  overrides it. So the backtest and the live system use different trailing activation levels.
* This is not marginal: **125 of 500 retained trades (25%) closed at TRAILING_STOP.** A quarter of all
  exits are governed by a rule that differs between backtest and production, and that the labeler never
  modelled — the labels ask about a fixed `2×ATR` / `1×ATR` barrier pair, and a trailing stop is a
  third barrier that changes the payoff distribution the Direction model's probability describes.

### Fix

1. **Carry the percentage on the signal and re-anchor it properly.** Add
   `trailing_activation_pct: float` to `TradeSignal` alongside the existing `take_profit_pct` /
   `stop_loss_pct`, and in `_fill`:

   ```python
   position.trailing_trigger = fill_price * (1.0 + signal.trailing_activation_pct)   # long
   position.trailing_trigger = fill_price * (1.0 - signal.trailing_activation_pct)   # short
   ```

2. **Add a backtest-vs-live geometry test.** Given one `TradeSignal` and one fill price, assert that
   `Backtester._fill` and `Position.from_signal` (the live path) produce identical `take_profit`,
   `stop_loss` and `trailing_trigger`. That invariant should be impossible to break silently.

3. **Extend `c2968d8`'s reasoning to the trailing stop.** That commit correctly established that the
   executed stop must match the barrier the label priced. The same argument applies to the trailing
   stop: a trail that arms at half the TP converts labelled TP-winners into partial wins, and the
   model's probability says nothing about that outcome. Either model the trail in the labeler, or
   report `trailing_exit_rate` as a known deviation and quantify its cost — the current 25% is large
   enough to matter.

---

<a name="p23"></a>
## P23 — Train/serve population mismatch on `bb_position` and `volume_trend` · **Medium**

### Evidence

```
bb_position:  1,389,457 nulls (24.28%)
volume_trend:   945,145 nulls (16.52%)
```

Neither is in `OPTIONAL_FEATURE_COLUMNS` (which contains only the five microstructure columns), so
both are in `REQUIRED_FEATURE_COLUMNS`.

Training keeps them, because `146b00c` changed the dataset filter to labels-only:

```python
usable = usable.dropna(subset=["label", "target_risk_score"])       # _to_dataset
```

Inference and backtest drop them:

```python
usable = frame.replace([np.inf, -np.inf], np.nan).dropna(subset=list(REQUIRED_FEATURE_COLUMNS))   # Backtester._prepare
```

The same gate applies on the live path via `DatasetProcessor.build_inference_payload`.

### Consequence

The commit message for `146b00c` states the intent clearly:

> `REQUIRED_FEATURE_COLUMNS` still gates warm-up, consistently across training, live inference and
> the backtester.

That consistency does not hold. Training now keeps rows that inference and the backtester reject. The
model is fitted on a population ~24% larger than, and systematically different from, the population it
is ever scored on — specifically, it learns from bars where Bollinger position is undefined
(`bb_width == 0`, i.e. flat candles again — see [P2](#p2)) and is then never asked about such a bar in
production.

The direction of the error is benign for *this* run (the nulls concentrate in the degenerate training
window, and the test window is nearly clean, so the backtest drops few rows). But the invariant is
broken, and a future period with genuine `bb_position` sparsity would produce a model trained on rows
it can never score.

The same commit fixed the *other* half of this mismatch correctly — `BaseModelHead._align` now passes
NaN through instead of filling 0.0, with a good rationale. That fix is right and should stay. It just
does not address the population question.

### Fix

Pick one contract and enforce it in one place:

* **Option A (recommended)** — training gates on `REQUIRED_FEATURE_COLUMNS` exactly as inference does:

  ```python
  usable = usable.dropna(subset=["label", "target_risk_score", *REQUIRED_FEATURE_COLUMNS])
  ```

  This restores the invariant the commit message claims, and — combined with [P2](#p2)'s explicit
  zero-volatility filter — drops the degenerate rows for a stated reason rather than as a side effect.

* **Option B** — inference stops gating on required columns and lets NaN through to the booster
  everywhere. This is the more radical reading of `146b00c`'s thesis and is defensible (LightGBM does
  route NaN), but it removes the warm-up guarantee, so a bar 3 candles into a symbol's history becomes
  tradeable. Not recommended without a separate warm-up check.

Either way, **add a test** asserting that the set of rows surviving `_to_dataset` and the set surviving
`Backtester._prepare` agree on the same input frame. That single test pins the invariant permanently.

---

<a name="p24"></a>
## P24 — `confidence_threshold_analysis` rewards total class collapse · **Medium**

### Evidence

| threshold | n_predictions | % of samples | accuracy | balanced accuracy |
|---:|---:|---:|---:|---:|
| 0.30 | 1,418,472 | 100.0% | 0.4559 | 0.3559 |
| 0.45 | 329,388 | 23.2% | 0.6305 | 0.3477 |
| 0.50 | 188,780 | 13.3% | 0.7460 | 0.3383 |
| 0.55 | 122,822 | 8.7% | 0.8810 | **0.33333** |
| 0.60 | 103,586 | 7.3% | 0.9556 | **0.33333** |
| 0.70 | 97,344 | 6.9% | **0.9835** | **0.33333** |

Accuracy climbs to 98.35%. Balanced accuracy sits at **exactly 1/3** — the value it takes when a
3-class classifier emits a single class for every input.

At thresholds of 0.55 and above, the only predictions surviving are `NO_TRADE_OR_FAIL` (the only class
whose probability reaches high values, since `NO_TRADE = 1 − trade_probability`). The 98% accuracy is
the base rate of the surviving subset, not skill.

### Consequence

This table reads as "raise the confidence threshold and accuracy improves dramatically". It is the most
persuasive-looking table in the Direction section and it is measuring the model's collapse into
silence. Acting on it — raising `min_gate_confidence` toward 0.7 — would produce a system that never
trades, with a 98% "accuracy" to justify it.

Balanced accuracy pinned at exactly 0.33333 is the tell, and it is present in the data, three columns
away. Nothing in the report flags it.

### Fix

1. **Add a degeneracy flag to each sweep row:**

   ```python
   row["distinct_predicted_classes"] = int(len(np.unique(predictions_above_threshold)))
   row["is_degenerate"] = row["distinct_predicted_classes"] < 2
   ```

   and omit degenerate rows from any headline or recommendation.

2. **Report per-class recall in the sweep**, not just aggregate accuracy. `LONG recall 0.0, SHORT
   recall 0.0` next to `accuracy 0.9835` makes the table self-explanatory.

3. **Sweep the production rule, not `argmax`.** As with [P1](#p1) and [P11](#p11), the quantity that
   matters is accuracy on rows passing R1a+R1b, restricted to the side actually chosen. That sweep
   would be directly actionable; this one is not.

4. **Raise a recommendation** whenever accuracy rises while balanced accuracy falls across a sweep —
   that divergence is the generic signature of threshold-induced class collapse and is worth detecting
   once, generically, for every head.

---

<a name="p25"></a>
## P25 — ADX and DI saturate; no guard against a degenerate true range · **Medium**

### Evidence

From `features.statistics`:

```json
"adx":       {"mean": 51.88, "median": 37.82, "p95": 99.9999999999999, "p99": 99.9999999999999, "max": 99.9999999999999}
"di_spread": {"mean": 0.0116, "p1": -1.0, "p5": -0.9999999999998078, "min": -1.0000000000000002, "max": 1.0000000000000002}
```

* ADX is pinned at its 100 ceiling for **more than 5% of all rows**, and its training-window median is
  ≈ 100 ([P2](#p2)).
* `di_spread` is saturated at ±1 for **more than 5% of rows at each tail** — the p5 value is
  `−0.9999999999998`, i.e. numerically at the boundary.
* `di_spread` also exceeds its mathematical range: `min = −1.0000000000000002`,
  `max = 1.0000000000000002`.

For context, a median ADX of 37.8 on 5-minute crypto is already implausibly high — typical values sit
in the 15–30 band. A median near 100 in the training window is not a market condition.

### Mechanism

`module_b_features/indicators.py::adx`:

```python
adx_values = wilder_smooth(directional_index.fillna(0.0), window)
return adx_values, plus_di.fillna(0.0), minus_di.fillna(0.0)
```

The directional index is `|+DI − −DI| / (+DI + −DI) × 100`. When true range collapses toward zero
(flat bars), both DIs collapse toward zero, but rarely to *exactly* the same value — one tick of
movement in one direction gives `|a − 0| / (a + 0) × 100 = 100`. The `fillna(0.0)` handles the exact
`0/0` case but not the far more common near-degenerate case, which is what produces the saturation.

`kama_distance` shows a related asymmetry (`min −5.00`, `max 0.50`) that is consistent with division by
a near-zero KAMA value.

### Consequence

`adx` is the gate's **3rd most-used feature** (6.44% of splits) and the Entry model's **2nd**
(6.00%). A feature that saturates at its ceiling on the majority of training rows and on 5% of all
rows is contributing a boolean, not a magnitude — and the splits the boosters learned against a
train-median of 100 do not transfer to a validation median of 30.

`di_spread` at ±1 for >10% of rows has the same character, and its out-of-range values will fail any
downstream assertion that assumes the documented `[-1, 1]` domain.

### Fix

1. **Guard the denominator explicitly.** Rather than `fillna(0.0)` after the fact, refuse to compute
   the index when the true range is not meaningfully positive:

   ```python
   di_sum = plus_di + minus_di
   directional_index = np.where(di_sum > _MIN_DI_SUM,
                                (plus_di - minus_di).abs() / di_sum * 100.0,
                                np.nan)          # NaN, not 0 and not 100
   ```

   NaN is correct here for exactly the reason `_align`'s docstring gives: "undefined" must stay
   distinguishable from a real value. Both 0 and 100 are lies about a flat market.

2. **Clamp `di_spread` to its documented domain** and investigate why it exceeds it — floating-point
   error of 2e-16 is harmless, but it indicates the expression is not computed in the numerically
   stable form.

3. **Add saturation to the feature-statistics report.** A `saturation_rate` column (fraction of rows
   within epsilon of the observed min or max) would surface ADX, `di_spread`, and any future
   equivalent automatically. Raise a recommendation above ~2%.

4. **Re-examine the ADX window** once the flat-candle contamination is removed ([P2](#p2)). A median
   of 37.8 on clean 5-minute data would still be worth a look.

---

<a name="p26"></a>
## P26 — The archive loader cannot work as configured · **Medium**

This is the operational root cause behind [P3](#p3). Four separate issues compound.

### 1. A total timeout of 120 s on multi-hundred-megabyte files

```python
archive_timeout_seconds: float = Field(default=120.0, gt=0.0)
...
timeout = aiohttp.ClientTimeout(total=self._settings.data.archive_timeout_seconds)
```

`ClientTimeout(total=...)` bounds the **entire** request-response cycle including the body transfer.
For a `liquidationSnapshot` day (kilobytes) that is generous; for a `bookTicker` day (tens to hundreds
of MB) it is a guaranteed abort, especially with `archive_max_concurrent_downloads = 4` sharing
bandwidth. The observed coverage — 52.8% liquidation, 0% bookTicker — is exactly what this predicts.

**Fix:** use `sock_read` rather than `total`, so the timeout bounds *stalls* rather than total size:

```python
timeout = aiohttp.ClientTimeout(
    total=None,
    connect=30.0,
    sock_connect=30.0,
    sock_read=self._settings.data.archive_timeout_seconds,
)
```

### 2. The whole CSV is materialised in memory

```python
with archive.open(names[0]) as handle:
    raw: bytes = handle.read()
frames = _read_csv_chunks(raw, _HEADERS[dataset])
```

The docstring claims "bookTicker files are large. They are read in chunks" — but `chunksize` is applied
to a `BytesIO` over an already fully-decompressed buffer. Peak memory is the full uncompressed file,
which for a bookTicker day is measured in gigabytes.

**Fix:** stream from the ZIP member directly. `zipfile.ZipExtFile` is a file object and
`pandas.read_csv` accepts it:

```python
with zipfile.ZipFile(io.BytesIO(payload)) as archive:
    name = next(n for n in archive.namelist() if n.lower().endswith(".csv"))
    with archive.open(name) as handle:
        yield from _read_csv_chunks_from_handle(handle, _HEADERS[dataset])
```

The header-detection logic needs a small rework (peek the first line, then `seek(0)`) but the chunked
reader is otherwise unchanged. This makes memory O(chunk), which is what the docstring already
promises.

### 3. The default backfill depth is off by an order of magnitude

```python
archive_backfill_days: int = Field(default=740, ge=1)
```

740 days × 27 symbols = **19,980 bookTicker symbol-days**. At realistic file sizes that is on the order
of a terabyte of transfer per full backfill.

**Fix:** default `bookTicker` depth to something achievable (30–90 days) and keep 740 only for the
cheap `liquidationSnapshot` dataset. Make the depth per-dataset rather than global. Note that the
disk cache stores *reduced* frames (a good design), so once a day is processed it is cheap forever —
the problem is only the first pass, which argues for an incremental, resumable backfill rather than an
all-or-nothing one.

### 4. Failures are invisible

`_download` returns `None` on both "Binance has no such file" (404, logged at DEBUG) and "we gave up
after 3 attempts" (logged at ERROR, then swallowed). `ArchiveCoverage` distinguishes `days_absent`
from `days_failed` — good — but the resulting summary never reaches the diagnostic report, so a run
where every single bookTicker day failed is indistinguishable, in the report, from one where the
feature was never requested.

**Fix:** surface `coverage_summary` per dataset in the report, and raise a CRITICAL recommendation when
any dataset's `days_failed / days_requested` exceeds ~0.2. Log the first few failures at ERROR with the
URL, so the cause is diagnosable without a rerun.

### 5. Positive note

The parsing conventions this branch got right are worth preserving through any rewrite: header
auto-detection with by-name column addressing (rather than positional), the microsecond/millisecond
timestamp normalisation in `_bucket_ms`, exact-bucket joining rather than as-of (so an uncovered
bucket cannot inherit a reading from hours earlier), and NaN-preservation with an explicit
`microstructure_is_missing` flag. Those are the details that would have silently corrupted every
downstream imbalance, and the 443 lines of tests are concentrated on exactly the right conventions.
The design is sound; only the transport layer is broken.

---

<a name="p27"></a>
## P27 — Reported class `distribution` is the whole dataset, printed next to train-only `rows` · **Medium**

### Evidence

The Direction metadata block:

```json
"rows": 2881082,                 // train only:  len(train_index)
"validation_rows": 1418472,
"distribution": {"NO_TRADE_OR_FAIL": 2654443, "LONG_SUCCESS": 1916915, "SHORT_SUCCESS": 1151345}
```

Those three counts sum to **5,722,703** — which is `dataset.valid_samples`, i.e. train + validation +
test. The source confirms it:

```python
"rows": int(len(train_index)),
"distribution": dataset.class_distribution(),      # value_counts() over the ENTIRE dataset
```

`ProcessedDataset.class_distribution()` has no notion of splits.

### Consequence

The two fields sit adjacent and invite the reading "here is the training set and its class balance".
The distribution shown is not the training set's, and the difference is material:

| | LONG : SHORT |
|---|---:|
| whole dataset (as reported) | 1,916,915 : 1,151,345 = **1.66 : 1** |
| validation (from `metrics.class_distribution`) | 368,796 : 393,733 = **0.94 : 1** |

The overall figure is long-skewed; validation is balanced-to-short. Since validation is one of the
three components of the overall figure, the training split must be *more* long-skewed than 1.66:1. That
is a substantial label-distribution shift between train and validation — directly relevant to
[P4](#p4)'s directional bias — and the report as written actively conceals it.

The same `distribution` value is reproduced verbatim in the `Labels` section, so the report states it
twice and never states the per-split breakdown once.

### Fix

Report per-split counts:

```python
def class_distribution(self, index: np.ndarray | None = None) -> dict[str, int]:
    target = self.direction_target if index is None else self.direction_target.iloc[index]
    return {str(k): int(v) for k, v in target.value_counts().items()}
...
"distribution": {
    "train":      dataset.class_distribution(train_index),
    "validation": dataset.class_distribution(validation_index),
    "test":       dataset.class_distribution(test_index),
    "all":        dataset.class_distribution(),
},
```

And add a `label_distribution_shift` field — e.g. the total-variation distance between the train and
validation class distributions — with a recommendation above ~0.05. A shift this size is a
first-order explanation for a directional model's behaviour and should never require the reader to do
the subtraction by hand.

---

<a name="p28"></a>
## P28 — Risk metrics are computed on unclipped predictions; `predict()` clips · **Low**

### Evidence

```json
"prediction_stats": {"mean": 0.6507, "median": 0.6519, "std": 0.1243, "min": 0.4217, "max": 1.0108644678784353}
```

The docstring says the head "estimates an opportunity score in `[0, 1]`", and `predict()` enforces it:

```python
score = clamp(float(self._model.predict(aligned)[0]), 0.0, 1.0)
```

But `train()` measures the raw estimator:

```python
predictions = estimator.predict(validation_features)          # unclipped
full_metrics = ml_metrics.regression_metrics(target.iloc[validation_index].to_numpy(), predictions)
```

Hence a reported maximum of 1.0109 for a quantity that can never exceed 1.0 in production.

Commit `895ef21` on this branch's ancestry is titled *"Fix Risk model target leakage; confirm output
clipping already in place"*. The clipping is indeed in place — in `predict()`. The confirmation did not
extend to the metrics path.

### Consequence

Small in magnitude but the same class of error as [P1](#p1): the reported metrics describe a function
the system does not execute. MAE, RMSE and R² are all computed against out-of-range predictions, so
each is slightly pessimistic relative to production — and, more importantly, the visible `max > 1.0`
is the only clue anyone has that the two paths differ. It should not require noticing a 17th decimal
place to establish which function was measured.

### Fix

Clip before measuring, using the same expression `predict()` uses:

```python
predictions = np.clip(estimator.predict(validation_features), 0.0, 1.0)
full_metrics = ml_metrics.regression_metrics(target.iloc[validation_index].to_numpy(), predictions)
```

and record the clip rate, which is diagnostically useful in its own right:

```python
"clipped_prediction_rate": float(np.mean((raw < 0.0) | (raw > 1.0))),
```

More generally: every head should measure through its own `predict()` path rather than through the
bare estimator. That single convention closes [P1](#p1) and [P28](#p28) together.

---

<a name="p29"></a>
## P29 — `duplicate_feature_rows` is counted and never removed · **Low**

### Evidence

```json
"duplicate_feature_rows": 1141,
"dropped_missing_or_inf_rows": 0
```

```python
usable = usable.reset_index(drop=True)
duplicate_feature_rows = int(usable.duplicated(subset=feature_columns).sum())
```

The count is computed and stored. Nothing acts on it.

There is a second, subtler indication of the same thing in `split_coverage_pct`:

```json
"validation": {"rows": 1418472, "span_days": 182.4, "capacity_rows": 1418445, "coverage_pct": 1.0}
```

`rows` (1,418,472) **exceeds** `capacity_rows` (1,418,445) by 27 — one extra row per symbol. A split
cannot contain more rows than its own time span has slots unless some timestamps appear twice.
`coverage_pct` is presumably clamped, so it reports a clean 1.0 and hides the overflow.

### Consequence

1,141 rows out of 5.7 M is 0.02% — negligible for training. But duplicated 57-feature vectors split
across train and test are, by definition, exact-match leakage; and duplicate `(symbol, timestamp)`
pairs mean at least one candle is stored twice, which points at an upsert-key problem in the ingestion
path that could be larger under other conditions.

QC does check for duplicate open times (`_find_duplicates`, at CRITICAL), yet duplicates reached the
dataset anyway. Either the QC check runs on a different scope than the ingestion write, or the
duplication is introduced after QC.

### Fix

1. **Drop them, and report what was dropped:**

   ```python
   before = len(usable)
   usable = usable.drop_duplicates(subset=["symbol", "timestamp"], keep="last").reset_index(drop=True)
   dropped_duplicate_rows = before - len(usable)
   ```

   Dedupe on `(symbol, timestamp)` rather than on the feature vector — two different bars can
   legitimately share a feature vector, but two rows cannot share a key.

2. **Do not clamp `coverage_pct`.** A value above 1.0 is a real signal; clamping it removes the only
   automatic detector of this problem. Report the raw ratio and flag anything above 1.0.

3. **Trace the source.** Find whether the duplicate `(symbol, timestamp)` pairs originate in the
   candle table, the futures-metrics table, or the join in `FeatureEngineer.build`, and fix the
   uniqueness constraint there. A `UNIQUE(symbol, timestamp)` index on the candle table would make
   this structurally impossible.

---

<a name="p30"></a>
## P30 — Dead code and unreachable logic · **Low**

Four items, each small, each a maintenance hazard because they read as active safeguards.

### (a) The Risk head's direction veto is unreachable

```python
# RiskModel.predict
if direction_confidence < decision.min_direction_given_trade_confidence:
    return RiskAllocation(leverage=0, ..., abort_reason=f"direction confidence {…} < {…}")
```

`DecisionEngine.evaluate` runs R1b — gating on the *same quantity* against the *same threshold* — and
returns before R6 is ever reached:

```python
if directional_confidence < self._config.min_direction_given_trade_confidence:
    return self._reject(inference, Rule.DIRECTION_CONFIDENCE, …)
```

So in the `evaluate` path this branch can never fire. The 281 `R6_RISK_MODEL_ABORT` rejections all
come from the volatility veto or the leverage floor.

**Fix:** either delete it, or keep it explicitly as a defence-in-depth check for callers that bypass
the engine — and say so in a comment, so the next reader does not spend time working out why it never
appears in the logs.

### (b) `max_positions_per_symbol` is dead config

```python
max_positions_per_symbol: int = Field(default=1, ge=1, le=5)
```

The only per-symbol check anywhere is `_check_system_gates`'s `inference.symbol in state.open_symbols`,
which hard-codes a limit of 1. Setting `max_positions_per_symbol = 3` changes nothing.

**Fix:** implement it (count positions per symbol rather than testing membership) or delete the field.
A configurable that silently does nothing is worse than no configurable.

### (c) R5's `stop_vs_labelled_atr` telemetry is NaN in the backtest

`c2968d8` added this precisely so a future regression could not silently reintroduce the geometry
mismatch:

```python
atr_pct = float(inference.feature_snapshot.get("atr_pct", 0.0) or 0.0)
labelled_stop = atr_pct * self._settings.labels.sl_atr_multiple
stop_ratio = exit_params.stop_loss_pct / labelled_stop if labelled_stop > 0.0 else float("nan")
```

But `Backtester._decide` builds the snapshot with a fixed key set that does **not** include `atr_pct`:

```python
snapshot={
    "hmm_regime": ..., "garch_volatility": ..., "garch_vol_rank": ...,
    "kama_slope": ..., "fdi": ..., "atr": ...,          # "atr", not "atr_pct"
}
```

So `atr_pct` defaults to 0.0, `labelled_stop` is 0.0, and `stop_ratio` is `nan` on every backtest
evaluation. The guard rail added by this branch's most recent commit does not work in the environment
where it would first be exercised.

**Fix:** add `"atr_pct": float(row.get("atr_pct", 0.0))` to the backtester's snapshot, and — better —
replace the ad-hoc snapshot dict with a named constant listing the keys the decision engine reads, so
a consumer added in one place cannot be forgotten in another. Then assert in a test that
`stop_vs_labelled_atr` is finite and ≥ 1.0 in a backtest, which is what the telemetry was for.

### (d) Exit slippage is applied but liquidation and stop fills are idealised

`_book_close` applies `slippage_bps` (default 5.0) symmetrically on exit, which is right in principle.
But:

* Liquidation is explicitly exempted (`if reason is CloseReason.LIQUIDATION: exit_price = raw_price`),
  which is backwards — liquidation is the fill most likely to be worse than its trigger price.
* Stops fill at exactly the stop level plus a flat 5 bps, regardless of bar range or volatility. Stop
  clustering in crypto perps produces materially worse fills on the moves that trigger stops in the
  first place.
* The run reports `liquidations: 0.0`, so the first point costs nothing in *this* run — but it is a
  latent understatement of tail risk, and leverage reaches 8× in the trade book.

**Fix:** apply slippage on liquidation too (it is the worst case, not the best), and consider scaling
the slippage penalty by the bar's range relative to ATR rather than using a flat constant. At minimum,
report the sensitivity: rerun the backtest at 5, 15 and 30 bps and publish all three, so the reader can
see how much of the 52% return survives a realistic fill assumption.

---

## Cross-cutting themes

Three patterns account for most of the individual findings, and fixing them at the pattern level would
prevent recurrence better than fixing each instance.

### 1. Metrics are computed on a different object than the one that ships

[P1](#p1) (Direction: raw vs calibrated), [P28](#p28) (Risk: unclipped vs clipped), and to a lesser
degree Entry's threshold/calibration ordering all have the same shape: `train()` measures the bare
estimator, then mutates `self._model` into something else before saving.

**Systemic fix:** make every head measure through its own public `predict()` path, on the final model,
after every mutation. Add a shared base-class helper — `self._score(validation_features,
validation_target)` — called as the *last* step of every `train()`, and forbid metric computation
anywhere else. Add a test per head asserting metric reproducibility from a reloaded artifact.

### 2. Absence is encoded as a neutral value rather than as absence

This branch fixed several instances of this well — `_align` now passes NaN instead of 0.0, the
microstructure columns stay NaN with an explicit `microstructure_is_missing` flag, `_MIN_REALIZED_VOL`
floors a denominator rather than producing NaN. The reasoning in `146b00c` on this point is correct
and worth keeping.

But the same pattern survives elsewhere and causes [P2](#p2) and [P25](#p25):
`wick_ratio ... .fillna(0.0)`, `whipsaw_rate ... .fillna(0.0)`, `directional_index.fillna(0.0)`,
`plus_di.fillna(0.0)`. In each case a flat bar produces "undefined", and the code writes a value that
is indistinguishable from a real observation — and, worse, a value at one end of the feature's range,
so it does not merely add noise but shifts the distribution.

**Systemic fix:** apply `_align`'s own stated principle to the feature layer. Audit every `.fillna()`
in `features.py` and `indicators.py`; each should either be removed (let NaN through — the boosters
handle it) or justified in a comment explaining why the fill value is a genuine observation rather
than a placeholder.

### 3. The report's health checks are decoupled from the report's data

[P17](#p17), [P18](#p18), [P3](#p3) and [P24](#p24) all reduce to the same gap. `diagnostics.py`
assembles a rich, well-structured report — per-feature null counts, per-symbol-month null rates, drift
in train-σ units, split coverage, per-fold walk-forward metrics, per-class confusion matrices — and
then evaluates the run's health against roughly three scalars: `symbol_exclusions_total`,
`status == "NOT_TRAINED"`, and `total_trades >= 30`.

Everything needed to produce a correct verdict is already computed and already in the JSON. It simply
is not read.

**Systemic fix:** introduce an explicit check registry — a list of named predicates over the report
dict, each returning `(severity, message)` — and drive both `_ai_summary` and `_recommendations` from
it. Adding a check then becomes one function, and a check can never exist in the data without
appearing in the verdict. The checks proposed throughout this document
([P3](#p3), [P17](#p17), [P18](#p18), [P24](#p24), [P27](#p27)) would all be entries in that registry.

---

## Suggested order of work

The dependencies matter here — several fixes are pointless until earlier ones land, and one of them
changes what every other measurement means.

**Stage 1 — make the report honest (no model changes).** Nothing else can be evaluated until the
report describes reality. [P1](#p1) (measure the shipped model), [P28](#p28) (clip before measuring),
[P27](#p27) (per-split distributions), [P17](#p17)/[P18](#p18)/[P24](#p24) (wire the health checks to
the data), [P12](#p12) (compute `oos_fraction`, pin the window). Then **rerun the diagnostic with no
other change.** The resulting report will look far worse than this one, and that report is the real
baseline.

**Stage 2 — fix the data.** [P2](#p2) (zero-range QC check, explicit degenerate-row filter, and
resolve whether the Aug 2024–2025 klines are genuine), [P3](#p3)/[P26](#p26) (archive transport, or
drop the dead columns), [P23](#p23) (one filtering contract), [P25](#p25) (indicator degeneracy),
[P29](#p29) (deduplicate). Retrain. Compare against the Stage 1 baseline — this is where the walk-
forward collapse should resolve, and where it becomes possible to say whether the heads have any edge.

**Stage 3 — fix the backtest.** [P5](#p5) (entry-bar resolution), [P6](#p6) (Risk Guard),
[P7](#p7) (`evaluate_many`), [P14](#p14) (`opened_at`), [P13](#p13) (Sortino), [P22](#p22) (trailing
geometry), [P19](#p19) (relaxed settings), [P20](#p20) (rejection accounting), [P30](#p30)(c)/(d)
(telemetry and fill realism). Expect the reported return and Sharpe to fall substantially. That is the
point.

**Stage 4 — fix the models.** [P8](#p8) (train Risk and Exit on a realistic population),
[P9](#p9) (half-life vs window), [P10](#p10) (split validation into fit and score),
[P16](#p16) (calibration effect-size guard), [P11](#p11)/[P21](#p21) (threshold selection),
[P15](#p15) (retire the trailing regressor, decide the stop head's role), [P4](#p4) (re-evaluate
whether stage 2 has an edge at all on clean data).

**Do not deploy anything from this branch to live or paper trading before Stage 2 completes.** The
current artifacts are fitted on a training set that is 72% degenerate bars, sized by a head that has
only ever seen winners, and validated by a report that measures a different model.

---

## What this branch got right

The audit above is unsparing, and it would be misleading to leave the impression that the work is
without merit. Several things here are genuinely well done and should survive any rework:

* **`_align` passing NaN instead of 0.0.** The reasoning — that `0.0` is an ordinary value for
  `log_return_*`, `di_spread` and `funding_rate`, so filling with it makes the model read "unknown" as
  "flat" and act on it confidently — is exactly right, and it is a real train/serve correctness fix.
* **`c2968d8`'s stop-floor diagnosis.** The argument that a `0.5 × ATR` stop against a `1.0 × ATR`
  label makes every MEDIUM- and HIGH-tier labelled winner stop out *by construction* is correct,
  well-evidenced from the labeler's own tiering, and identifies a genuine structural defect that no
  accuracy metric could have surfaced.
* **The conditional-vs-joint confidence distinction.** Surfacing both `directional_confidence` and
  `joint_success_probability` — and stating plainly that 88% conditional on a 55% gate is a 48% trade —
  is a real improvement in the audit trail.
* **The archive loader's parsing conventions.** By-name column addressing, header auto-detection,
  microsecond/millisecond normalisation, exact-bucket joins rather than as-of, NaN preservation with an
  explicit missingness flag, and 443 lines of tests aimed at the conventions a bug would hide in. The
  design is sound; only the transport is broken.
* **`split_coverage_pct` and the null diagnostics** added by `c2968d8` and `146b00c`. These are what
  made [P2](#p2) and [P3](#p3) diagnosable at all. The instrumentation is good — it is the health
  checks reading it that are missing.
* **The `oos_disclosure` discipline** around the test split. Reserving a six-month block untouched by
  training, early stopping, calibration, threshold selection and model selection is the right
  architecture, and it is why the backtest window is worth arguing about at all.

The pattern across the branch is consistent: the diagnosis of each problem it set out to solve is
careful and usually correct, the instrumentation added alongside is good, and the fix goes one step
further than the evidence supports — recovering rows that turn out to be degenerate, restoring
features that turn out to be empty, widening a stop in a way that retires a trained head. The
instrumentation this branch added is what makes all of that visible. It just is not wired into
anything that would stop a bad run from being labelled GOOD.

---

*Audit performed against `c2968d8bda9a` with the diagnostic run `d0eb385d-a0ec-413b-8414-c90b64fdea8e`
as the primary evidence base. Every quoted figure is from that run's JSON export or from source at that
commit. The test suite could not be executed in this environment — `pandas`, `numpy`, `scikit-learn`
and `lightgbm` are not installed here — so no claim is made about whether the branch's 160+ tests pass;
all findings above are derived from source reading and from the diagnostic output itself.*
