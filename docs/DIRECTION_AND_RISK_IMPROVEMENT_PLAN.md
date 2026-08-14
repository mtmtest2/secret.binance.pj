# Directional Accuracy & Risk Model — Findings and Implementation Plan

**Analysed run:** `480ccd49-19dd-4e52-b327-3de22fd7ad6d`, generated 2026-08-11T23:15:30Z
**Analysed code:** git `4384d08d` (tip of `origin/lastchance`) — the commit that produced the report.
**Also reviewed:** `origin/claude/ml-orderflow-tp-ladder-9k4m2p` (`a4e045a`), which is 3 commits ahead of the
report state and already contains an aggTrades order-flow block and a three-stage TP ladder. Where that
branch has already solved something, this document says so and builds on it rather than repeating it.

> **How to read this document.** Sections 1–4 are diagnosis: what the numbers actually say, including
> several places where the report's headline metrics are measuring something other than what they appear
> to measure. Sections 5–8 are the proposed work, ordered by expected value per unit of effort.
> Section 9 is the measurement harness that has to exist before any of this can be judged.
> Every proposal is written to be actionable as an implementation prompt.

---

## 0. Executive summary

| # | Finding | Severity | Section |
|---|---|---|---|
| 1 | 73% of the intended 12-month training window is silently discarded by a `dropna` over all 50 features. Train is 26.9% populated; validation 96.9%; test 99.8%. | **Critical** | §1 |
| 2 | The direction stage (long-vs-short) carries ~1.0% of the available information. The gate stage carries ~7.4%. The system knows *whether* to trade roughly 7× better than it knows *which way*. | **Critical** | §2 |
| 3 | The headline `accuracy 0.477` / `balanced_accuracy 0.355` / `LONG recall 4.8%` are structural artifacts of taking `argmax` over a product-decomposed distribution. They do not measure directional skill and must not be optimised. | **High** | §2.3 |
| 4 | Decision rule **R3** is documented as a consistency check but is arithmetically a *second, stricter gate threshold*. It silently requires `p_trade ≥ 0.65` (~2.7% of bars) on top of R1A's `≥ 0.55` (~52% of bars). | **High** | §2.5 |
| 5 | The "30-day retention" premise is true only of the **REST API**. Binance's own bulk archive (`data.binance.vision`) carries open interest, both long/short ratios and the taker buy/sell ratio at full contract history. The deleted features do not need replacing — they need a different loader. | **High** | §3 |
| 6 | The feature matrix is almost devoid of independent *signed* information. Of the direction model's top 10 features, 6 are unsigned volatility/structure measures. There is no cross-sectional, no BTC-relative, and no timeframe longer than the label horizon itself. | **Critical** | §4 |
| 7 | The risk model's target mixes a quantity that is *unknown at t* (path heat) with one that is *already a model input* (`garch_vol_rank`). Most of its R²=0.089 is the model recovering a deterministic factor it was handed. Genuine predictive content on path risk is near zero. | **Critical** | §5 |
| 8 | The risk model is trained only on bars whose forward path succeeded, then applied at inference to bars that include failures. Result: +0.076 systematic optimism and an output range of [0.30, 1.03] against a true range of [0.09, 1.00] — it cannot represent a high-risk trade at all. | **Critical** | §5 |
| 9 | The exit model's TP and trailing heads are **23.5% worse than the rule-based ATR baseline** and are in production. Trailing is `0.5 × TP` by construction — a second regressor is being trained on a rescaled copy of the first. | **Medium** | §6 |
| 10 | Thresholds are selected by F1 on the LONG class. The SHORT side is never measured. No trading economics (`average_r`, `win_rate`, `profit_factor`, `expectancy`) are computed anywhere. | **High** | §9 |

**The single sentence version:** the direction stage is being asked to solve a signed problem using an
unsigned feature set, on roughly 5,000 effectively-independent samples, and is then graded by a metric
that cannot see whether it succeeded. Fixing the data loss (§1), restoring the derivatives block from the
archive (§3), and adding signed/cross-sectional features (§4) attack the cause. Everything else is
downstream of those three.

---

## 1. The dataset is not what the config asked for

### 1.1 The arithmetic

`history_bootstrap_candles = 212_400` (≈737 days of 5m bars) and the OHLCV backfill **worked**:

```
total_candidate_rows      5,723,214  ÷ 27 symbols = 211,971 bars/symbol   ✓ matches config
rejected_invalid_label_rows   1,296  = 27 × 48                            ✓ exactly the un-simulatable tail
dropped_missing_or_inf_rows 2,153,211 = 37.6% of all candidate rows       ✗ this is the problem
valid_samples             3,568,707
```

Now compare each split against its own theoretical capacity (`days × 288 bars × 27 symbols`):

| Split | Window | Capacity | Actual | **Populated** |
|---|---|---|---|---|
| Train | 2024-08-05 → 2025-08-11 (370.2 d) | 2,878,524 | 773,873 | **26.9%** |
| Validation | 2025-08-11 → 2026-02-09 (182.4 d) | 1,418,418 | 1,374,788 | **96.9%** |
| Test | 2026-02-09 → 2026-08-11 (182.6 d) | 1,420,119 | 1,417,887 | **99.8%** |

The row loss is **almost entirely inside the training window**, and it increases monotonically as you go
back in time. This is not rolling-window warm-up: warm-up would remove an identical ~1,000 rows per
symbol, not 100,000.

### 1.2 It is also a *population* shift, not just a volume shift

Per-symbol validation counts are in the report; per-symbol totals are too. Subtracting gives approximate
per-symbol training counts:

| Symbol | Total valid | Validation | ~Train rows | ~% of train capacity |
|---|---|---|---|---|
| SOL | 200,073 | 52,484 | ~95,000 | ~89% |
| XRP | 195,622 | 52,226 | ~90,800 | ~85% |
| BCH | 170,338 | 52,287 | ~65,500 | ~61% |
| LTC | 167,886 | 52,447 | ~62,800 | ~59% |
| … | | | | |
| PIXEL | 111,796 | 49,729 | ~12,000 | ~11% |
| ZIL | 112,296 | 49,299 | ~12,700 | ~12% |
| GRT | 112,464 | 49,441 | ~12,500 | ~12% |

*(estimates — the report gives per-symbol validation counts but not per-symbol train counts)*

Eight symbols (SOL, XRP, BCH, LTC, DOGE, LINK, TRX, XLM) supply roughly **65% of the training set**
while being **30% of the panel**. In validation and test they are ~29%. The model is trained
predominantly on large-cap behaviour and evaluated on a uniform 27-symbol panel — a textbook covariate
shift, entirely self-inflicted.

### 1.3 Root cause

`DatasetProcessor._to_dataset` does:

```python
usable = usable.replace([np.inf, -np.inf], np.nan)
usable = usable.dropna(subset=feature_columns + ["label", "target_risk_score"])
```

**A single NaN in any one of 50 columns destroys the entire row.** The leading suspects, in order:

1. **`vol_of_vol` and `garch_vol_ratio` both divide by `realized_vol_12` with `.replace(0.0, np.nan)`.**
   `realized_vol_12` is exactly `0.0` whenever the last 12 log-returns are all zero — i.e. whenever a
   low-tick-size alt printed 12 consecutive flat 5m candles. That is common for PIXEL/ZIL/GRT/1INCH in
   quiet 2024–2025 conditions and rare for SOL/XRP in 2026. **The drop is therefore correlated with
   exactly the two axes we observe: symbol liquidity and market quietness.**
2. **`garch_volatility` holes.** `_rolling_garch_forecast` leaves `NaN` and `continue`s while
   `fitted is False`. `_fit_garch` returns `None` whenever `alpha + beta >= 1.0` — which is the *normal*
   outcome for GARCH(1,1) on high-frequency crypto returns (near-IGARCH behaviour). Until one fit
   happens to land inside the stationarity constraint, every row is NaN. The EWMA fallback only fires
   if the **entire** series is NaN, so partial failure produces a long leading hole rather than a
   fallback.
3. `slope()` divides by `series.abs().replace(0.0, np.nan)` → `kama_slope` NaN if KAMA is ever 0.

### 1.4 Fix

**F1.1 — Instrument first (30 min, do this before anything else).**
In `_to_dataset`, before the `dropna`, record per-column null counts and per-column null counts bucketed
by symbol and by month. Emit into `ProcessedDataset` and into the diagnostics report as
`null_counts_by_feature` and `null_rate_by_symbol_month`. This turns the hypothesis above into a fact in
one training run and tells you which of the three causes dominates.

**F1.2 — Stop dropping rows for NaN at all.**
LightGBM handles missing values natively (`use_missing=True`, default). The `dropna` should be reduced to
the label columns only:

```python
usable = usable.dropna(subset=["label", "target_risk_score"])
```

Add an explicit boolean `<feature>_is_missing` companion column for any feature whose missingness is
*informative* (the derivatives block, §3). For features whose missingness is pure numerical accident
(`vol_of_vol`, `garch_vol_ratio`), fix the arithmetic instead (F1.3).

**F1.3 — Make the ratio features total.**
```python
# was: .div(frame["realized_vol_12"].replace(0.0, np.nan))
denominator = frame["realized_vol_12"].clip(lower=_FLOOR_VOL)   # e.g. 1e-8
```
Or better, define them on a floored denominator and add a `realized_vol_12_is_zero` indicator so the
tree can still learn "this is a dead market" — which is genuine information, currently thrown away.

**F1.4 — Make the GARCH fallback per-row, not per-series.**
When `fitted is False` at row `i`, emit the EWMA value for row `i` rather than `NaN`. Relax the
stationarity rejection to `alpha + beta >= 1.0 + tol` or clamp to `0.999` instead of rejecting, and log
the rejection rate.

**F1.5 — Fix the train/serve inconsistency this creates.**
`BaseModelHead._align` currently does `.fillna(0.0)` at inference. So a row that would have been
*deleted* during training is *scored with a fabricated zero* in production. Zero is a meaningful value
for `log_return_*`, `di_spread`, `funding_rate`, `bb_position` — the model reads it as "flat", not as
"unknown". Once F1.2 lands, `_align` must pass `NaN` through unchanged so training and inference see the
same encoding of missingness.

**Expected effect:** training rows go from 773,873 to roughly 2.8M, the train/validation population shift
largely disappears, and every downstream head gets ~3.5× more data. This is the highest
value-per-hour change in this entire document and it is a prerequisite for judging any of the others.

---

## 2. What the direction numbers actually say

### 2.1 The gate is fine. The direction stage is nearly empty.

Both stages are binary, so both can be scored against their own base rate.

**Stage 1 (gate — trade vs no-trade).** Base rate = 733,186 / 1,374,788 = 53.33%.

| Metric | Model | Base rate | Skill |
|---|---|---|---|
| Log loss | 0.64009 | 0.69089 | **pseudo-R² = 7.35%** |
| Brier (per-class) | 0.22918 | 0.24889 | **BSS = 7.92%** |

**Stage 2 (direction — long vs short, given trade).** Base rate = 355,788 / 733,186 = 48.53%.

| Metric | Model | Base rate | Skill |
|---|---|---|---|
| Log loss | 0.68621 | 0.69315 (`ln 2`) | **pseudo-R² = 1.00%** |
| Brier (per-class) | 0.24658 | 0.24978 | **BSS = 1.28%** |

> **The gate carries about 7× more information than the direction stage.** This is the finding that
> should drive the roadmap. It is also why the stage-2 isotonic calibration reported
> `improved: false` — there is almost nothing there to calibrate.

The threshold sweep is consistent with ~1% pseudo-R²: at `p_long|trade ≥ 0.50` the model calls LONG on
276,820 of 733,186 trade rows (37.8%) with 56.9% precision against a 48.5% base — a real but thin +8.4
point lift, concentrated in a minority of bars.

### 2.2 Why it is thin — the effective sample size

Three multiplicative reductions apply to the nominal 773,873 training rows:

1. **Coverage loss (§1):** the window is 26.9% populated.
2. **Recency weighting:** `recency_half_life_days = 45.0` against a 370-day window. The oldest training
   row carries weight `0.5^(370/45) = 0.0033`. Kish effective sample size of the weighting alone is
   roughly 35% of nominal.
3. **Label overlap:** `max_holding_bars = 48`, so consecutive bars share up to 47/48 of their forward
   label horizon. No uniqueness weighting or event-based sampling is applied anywhere.

Order-of-magnitude: `773,873 × 0.35 / 48 ≈ 5,600` effectively independent observations, for a 50-feature
gradient-boosted model with 700 trees. **That is the quantitative reason stage 2 has ~1% skill**, and it
is why the fix is data and features, not hyperparameters and not thresholds.

Note that (1) and (2) *compound in the same direction*: coverage is worst in the oldest part of the
window, which is also where recency weight is lowest. The model is effectively trained on roughly the
last two months before 2025-08-11, then validated on the following six months and tested on the six
after that.

### 2.3 The headline accuracy metric is measuring an artifact

`predicted_class_distribution` shows NO_TRADE chosen on 1,299,860 of 1,374,788 rows (**94.5%**), giving
LONG recall 4.8% and balanced accuracy 0.355. This looks catastrophic. It is arithmetic.

The cascade produces `p_long = p_trade × c` and `p_short = p_trade × (1 − c)` where `c` is the
conditional confidence. `argmax` can pick a direction only when:

```
p_trade × max(c, 1−c) > 1 − p_trade      ⟺      p_trade > 1 / (1 + c)
```

With `c ∈ [0.5, 1.0]`, that requires `p_trade` between **0.50 and 0.667**. From the gate sweep, only
101,707 rows (7.4%) have `p_trade ≥ 0.60` and 36,544 (2.7%) have `p_trade ≥ 0.65`. The model predicted a
direction on 74,928 rows (5.45%) — exactly what the geometry predicts.

**Consequences:**
- `accuracy`, `balanced_accuracy`, `macro_f1`, `per_class.recall` and the confusion matrix in the
  direction block describe a decision rule the production system **never uses** (production uses
  independent thresholds on the two stages, per R1A/R1B).
- The AI diagnostic's summary verdict ("Overall status: GOOD, weakest component: risk") is comparing
  incommensurable scales — a 3-class argmax accuracy against a regression R². It should not be trusted
  to rank components.
- **The `confidence_threshold_analysis` block is actively misleading.** It shows accuracy climbing to
  93.7% at threshold 0.60 and 99.8% at 0.85. Those are high-confidence *NO_TRADE* predictions. It is
  reporting that the model is extremely good at recognising bars it will not trade. It says nothing
  about directional skill and must not be read as an argument for raising thresholds.

**Fix:** report a **two-sided direction metric** (§9) and demote the argmax block to a footnote, clearly
labelled as not corresponding to the production decision rule.

### 2.4 The direction threshold sweep only measures LONG

`direction_threshold_sweep` delegates to `entry_threshold_sweep`, which computes
`precision_score(y_is_long, p_long >= t)`. For a symmetric long/short problem this measures **one side
only**. SHORT precision is never reported anywhere in the diagnostic, even though the confusion matrix
hints SHORT is the better side (recall 5.28% vs 4.76%, precision 0.498 vs 0.485).

`recommended_direction_threshold: 0.45` is therefore selected by maximising **F1 on the LONG class** — a
quantity with no economic meaning and no symmetry. See §9 for the replacement.

### 2.5 R3 is a hidden second gate — and it is stricter than R1A

`DecisionEngine` rule R3 rejects when `direction.no_trade_probability > max_no_trade_probability`
(default **0.35**). Its docstring describes it as a consistency check against the two stages
disagreeing.

But in `DirectionModel.predict`:

```python
long_probability  = trade_probability * long_given_trade
short_probability = trade_probability * (1.0 - long_given_trade)
no_trade_probability = max(0.0, 1.0 - long_probability - short_probability)   # == 1 - trade_probability
```

`no_trade_probability` is **identically `1 − trade_probability`**. There is no independent information in
it and no disagreement it could detect. R3 is arithmetically:

```
1 - p_trade <= 0.35     ⟺     p_trade >= 0.65
```

Meanwhile R1A requires only `p_trade >= min_gate_confidence = 0.55`.

| Rule | Effective requirement | Bars passing (validation) |
|---|---|---|
| R1A | `p_trade ≥ 0.55` | 720,693 (52.4%) |
| **R3** | **`p_trade ≥ 0.65`** | **36,544 (2.66%)** |

R3 discards **95% of the signals R1A admits**, for reasons the code does not intend and the operator
cannot see from the rule name. (The joint isotonic calibrators are applied to the reported distribution
before R3 reads it, which perturbs the exact cut point but not the structure — it remains a monotone
re-gate on `p_trade`.)

**This is directly relevant to your constraint.** You asked not to buy accuracy by raising thresholds and
producing more no-trades. R3 is an *unintentional* threshold raise that is already in place, costing 95%
of trade candidates. Removing or repairing it **increases** trade count.

**Fix — pick one:**
- **(a) Delete R3.** It is redundant with R1A by construction. This is the honest fix.
- **(b) Re-purpose it** into the check its docstring describes: compare the gate's own
  `trade_probability` against the *joint-calibrated* NO_TRADE mass and reject only when the two differ by
  more than a tolerance (e.g. `abs((1 - p_trade) - calibrated_no_trade) > 0.15`). That is an actual
  consistency check.
- In either case, if a `p_trade ≥ 0.65` requirement is genuinely wanted, it should be stated as
  `min_gate_confidence = 0.65` where the operator can see it — not hidden inside a rule named
  "NO_TRADE_MASS".

---

## 3. The 30-day retention problem is a REST-API problem, not a data problem

### 3.1 The premise is only half true

Your report and the code comments state that `open_interest_change`, `long_short_ratio` and
`taker_buy_sell_ratio` are limited to ~30 days and therefore unusable. That is correct **for the REST
endpoints**:

- [`/futures/data/openInterestHist`](https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Open-Interest-Statistics) — "Only the data of the latest 1 month is available."
- [`/futures/data/topLongShortAccountRatio`](https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Top-Long-Short-Account-Ratio) — "Only the data of the latest 30 days is available."

But Binance publishes the **same series** as daily bulk archives with full contract history at
[data.binance.vision](https://data.binance.vision/?prefix=data%2Ffutures%2Fum%2Fdaily%2Fmetrics%2F):

```
https://data.binance.vision/data/futures/um/daily/metrics/{SYMBOL}/{SYMBOL}-metrics-{YYYY-MM-DD}.zip
```

with columns:

```
create_time, symbol,
sum_open_interest, sum_open_interest_value,
count_toptrader_long_short_ratio, sum_toptrader_long_short_ratio,
count_long_short_ratio, sum_taker_long_short_vol_ratio
```

**These are exactly the features that were deleted.** They do not need replacing. They need a different
loader.

> **Verify before building:** confirm row granularity on a single downloaded file (expected: 288 rows/day
> = 5-minute buckets, matching the REST `period=5m`). One `curl` + `unzip` + `head` settles it. Everything
> in §3.3 assumes 5m; if the archive turns out to be coarser for some symbols, `merge_asof(backward)`
> onto the 5m grid with an explicit staleness cap still works.

### 3.2 The other five deleted features are also available

Commit `2b56c8a` removed `ob_imbalance`, `ob_imbalance_delta`, `ob_spread_bps`, `ob_spread_rank` and
`liquidation_imbalance` with the comment *"Binance exposes no historical endpoint for either, ever."*
That is true of the REST API and false of the archive. `data/futures/um/daily/` also carries:

| Archive | Restores | Notes |
|---|---|---|
| `bookTicker` | `ob_spread_bps`, `ob_spread_rank`, a best-bid/ask size imbalance | Event-level best bid/ask + sizes. Large; aggregate to 5m before storing. |
| `bookDepth` | `ob_imbalance`, `ob_imbalance_delta` | Periodic depth snapshots at fixed percentage levels. |
| `liquidationSnapshot` | `liquidation_imbalance` | Forced-order stream. **Verify availability per symbol** — coverage is less uniform than metrics. |
| `markPriceKlines` + `indexPriceKlines` | **`basis`** (new — see §4) | Currently unused. Perp basis is a direct signed positioning measure. |
| `premiumIndexKlines` | **`predicted_funding`** (new) | Forward-looking funding pressure, not the stale realised rate. |
| `fundingRate` | already used | Full history. |
| `aggTrades` | already used on the orderflow branch | Full history. |

### 3.3 Implementation — archive ingestion

**F3.1 — New module `module_a_data/archive_loader.py`.**

```
class BinanceArchiveLoader:
    async def download_daily(symbol, dataset, date) -> pd.DataFrame | None
    async def backfill_range(symbol, dataset, start_ms, end_ms) -> int
```

Requirements:
- URL pattern `https://data.binance.vision/data/futures/um/daily/{dataset}/{SYMBOL}/{SYMBOL}-{dataset}-{YYYY-MM-DD}.zip`.
- A 404 is normal (symbol not yet listed, or dataset absent for that day). Treat as "no data", record it,
  do not retry, do not fail the run.
- Verify the accompanying `.CHECKSUM` file.
- Cache extracted CSVs on disk keyed by `(symbol, dataset, date)` so re-runs are free. 27 symbols × 737
  days × 1 dataset ≈ 20k files; budget for it and make the cache directory configurable.
- Bounded concurrency (the existing `_symbol_semaphore` pattern), with a global rate limiter.
- Monthly archives exist for some datasets and are far fewer files — prefer `monthly/` for dates older
  than ~2 months and `daily/` for the recent tail.

**F3.2 — Rewrite `PipelineOrchestrator.backfill_futures_metrics`** to source from the archive first and
fall back to the REST endpoints only for the trailing ~30 days that the archive has not yet published
(the archive lags real time by roughly a day). This gives seamless full-history coverage with no gap at
the join.

**F3.3 — Widen `FuturesMetrics`** (`module_a_data/models.py`, `db_models.py`) to carry all six metric
columns rather than the current subset — in particular keep `count_toptrader_long_short_ratio` and
`sum_toptrader_long_short_ratio` **separately** (see §4.3, they mean different things).

**F3.4 — Restore the deleted feature columns** in `FEATURE_COLUMNS` once the loader is proven, with
`*_is_missing` indicators (§1) so early-listing gaps are represented honestly rather than as neutral
constants.

**F3.5 — Reconcile with the orderflow branch.** `d781381` deleted these three features on the explicit
premise that they could not be backfilled. That premise is wrong. Keep that branch's aggTrades block
(it is genuinely valuable and additive) and *restore* the derivatives block alongside it rather than
treating them as alternatives.

---

## 4. Why direction has no signal: the feature matrix is unsigned

### 4.1 The diagnosis

The direction model's top 10 features by gain:

| Rank | Feature | Signed? |
|---|---|---|
| 1 | `adx` | ✗ unsigned (trend *strength*) |
| 2 | `di_spread` | ✓ signed |
| 3 | `atr_pct` | ✗ unsigned |
| 4 | `close_ema_slow_ratio` | ✓ signed |
| 5 | `rsi` | ✓ signed |
| 6 | `fdi` | ✗ unsigned |
| 7 | `wick_ratio` | ✗ **unsigned — and it need not be** |
| 8 | `realized_vol_48` | ✗ unsigned |
| 9 | `garch_volatility` | ✗ unsigned |
| 10 | `log_return_48` | ✓ signed |

**Six of the top ten cannot express a direction at all.** The model is spending most of its capacity on
volatility and structure — which is the *gate's* question, not the direction stage's. And the four signed
features that remain (`di_spread`, `close_ema_slow_ratio`, `rsi`, `log_return_48`, plus `kama_distance`
and `ema_fast_slow_spread` further down) are all near-collinear readings of the same slow trend.

Three structural gaps compound it:

1. **No cross-sectional information.** `FeatureEngineer.build` is strictly per-symbol. The model cannot
   see that all 27 alts are down together (market beta) versus this one alt being down alone
   (idiosyncratic). These are completely different trades and the model cannot distinguish them.
2. **No timeframe longer than the label horizon.** The longest lookback is `log_return_48` = 4 hours,
   which is *exactly* `max_holding_bars`. The model is asked to predict 4 hours ahead with no context
   beyond 4 hours back.
3. **No symbol identity.** Per-symbol accuracy ranges 0.438 (ZEC) to 0.530 (TRX) — a 9-point spread the
   model has no way to condition on.

### 4.2 Block A — Cross-sectional / market-relative *(highest value; needs no new data source)*

This requires an architectural change: a **second, panel-wide feature pass** keyed by timestamp, after
per-symbol features are built. Add `FeatureEngineer.build_cross_sectional(pooled: pd.DataFrame)` called
from `DatasetProcessor` after the per-symbol concat, and mirror it on the live path (which already builds
payloads for all symbols each cycle, so the data is there).

Add BTCUSDT and ETHUSDT to the **fetch** universe as context symbols even if they are never traded.

| Feature | Definition | Why |
|---|---|---|
| `btc_return_12`, `btc_return_48` | BTC log return over same windows | The market factor. Nothing in the current set knows it exists. |
| `eth_return_12` | ETH log return | Second factor; ETH/BTC rotation is a real regime. |
| `beta_btc_288` | Trailing 24h OLS beta of symbol returns on BTC returns | How much of this symbol's move is market. |
| **`residual_return_12`** | `log_return_12 − beta_btc_288 × btc_return_12` | **Idiosyncratic momentum. Probably the single most valuable feature in this document.** Separates "everything is up" from "this is up". |
| `residual_return_48` | same at 48 bars | |
| `market_breadth_12` | Fraction of the universe with `log_return_12 > 0` at time `t` | A regime read no single-symbol feature can produce. |
| `cs_return_rank_12` | Rank of `log_return_12` within the universe at `t`, in [0,1] | Cross-sectional momentum / reversal — a well-established crypto factor. |
| `cs_return_rank_48` | same at 48 bars | |
| `cs_vol_rank` | Rank of `realized_vol_12` within the universe at `t` | Relative, not absolute, risk. |
| `btc_corr_288` | Trailing 24h correlation with BTC | Regime of coupling: decoupled alts trade differently. |
| `btc_dominance_delta` | `btc_return_12 − median(universe log_return_12)` | Rotation proxy. |

**Look-ahead warning:** every cross-sectional statistic must be computed from bars at time `t` only,
across symbols. Any use of `t+1` from another symbol is leakage. Unit-test this explicitly — shuffle the
future and assert the features are unchanged.

### 4.3 Block B — Derivatives, restored (§3) and used *directionally*

The previous encoding wasted these features by using levels and ranks. The information is in the
**interaction with price**.

| Feature | Definition | Why |
|---|---|---|
| `oi_change_12`, `oi_change_48` | `pct_change` of `sum_open_interest` | Raw flow of positioning. |
| **`oi_price_quadrant`** | Categorical: `sign(return_12) × sign(oi_change_12)` → 4 states | **The classic directional read.** price↑ OI↑ = new longs (continuation); price↑ OI↓ = short covering (exhaustion); price↓ OI↑ = new shorts; price↓ OI↓ = long liquidation. Pass as a LightGBM categorical. |
| `oi_price_divergence` | `oi_change_12 × sign(log_return_12)`, continuous | Same signal, continuous form. |
| `taker_ls_ratio` | `log(sum_taker_long_short_vol_ratio)` | Full-history aggressive flow, from the archive. |
| `taker_ls_ratio_ewma_48` | EWMA of the above | A single 5m bucket is noise; persistence is signal. |
| `top_trader_ls_account` | `log(count_toptrader_long_short_ratio)` | *How many* top accounts are long. |
| `top_trader_ls_position` | `log(sum_toptrader_long_short_ratio)` | *How much* top position is long. |
| **`top_trader_skew`** | `top_trader_ls_position − top_trader_ls_account` | **Divergence between the two = concentration.** Few accounts holding a large long is a different state from many accounts holding small longs. Neither series alone shows it. |
| `ls_ratio_z_288` | Trailing 24h z-score of `count_long_short_ratio` | Positioning extremes mean-revert. |
| `basis` | `(mark_close − index_close) / index_close` from `markPriceKlines` / `indexPriceKlines` | Direct signed perp pressure. Currently unused data. |
| `basis_z_288` | Trailing z-score of basis | |
| `funding_z_288` | Trailing z-score of `funding_rate` (replaces/augments `funding_rate_rank`) | |
| `funding_x_oi` | `funding_z_288 × oi_change_12` | Crowded longs *being added to* vs *being unwound* — opposite trades. |

### 4.4 Block C — Order flow, extending the orderflow branch

`a4e045a` already provides `order_flow_imbalance_5m`, `volume_delta_5m`, `relative_volume_5m` from
aggTrades. Extend, do not replace:

| Feature | Definition | Why |
|---|---|---|
| `ofi_ewma_12`, `ofi_ewma_48` | EWMA of `order_flow_imbalance_5m` | A single bucket is dominated by noise. |
| `cvd_slope_48` | Slope of cumulative volume delta over 48 bars, normalised by dollar volume | Sustained aggression. |
| **`cvd_price_divergence`** | `sign(cvd_slope_48) ≠ sign(price_slope_48)`, magnitude-weighted | **Classic reversal setup**: price making highs on falling CVD. |
| `large_trade_imbalance` | Order-flow imbalance restricted to prints above the trailing 90th percentile of trade size | Separates whale flow from retail flow. aggTrades gives per-print quantity, so this is free. |
| `trade_count_imbalance` | Imbalance computed on *counts* rather than volume | `count` vs `volume` imbalance diverging = few large orders vs many small. |
| `vpin_50` | Volume-bucketed order-flow toxicity (Easley/López de Prado) over 50 buckets | [Validated on BTC perps](https://www.sciencedirect.com/science/article/pii/S0275531925004192); predicts jumps. Note published evidence of [alpha decay](https://medium.com/coinmonks/i-used-a-2012-market-microstructure-paper-to-find-alpha-in-btc-it-worked-but-its-dying-500f9bc0fc94) — treat as regime, not as a standalone signal. |
| `kyle_lambda_288` | Regression coefficient of `|return|` on signed volume over 24h | Price impact per unit flow — conditions *how much* a given imbalance should move price. |

### 4.5 Block D — Multi-timeframe *(klines only, no new data)*

Resample the existing 5m frame; do not fetch anything.

`ema_slope_1h`, `rsi_1h`, `adx_1h`, `di_spread_1h`, `close_vs_vwap_1h`, `close_vs_vwap_4h`,
`dist_to_prior_day_high`, `dist_to_prior_day_low`, `dist_to_prior_day_close`, `dist_to_week_open`.

Rationale: a 4-hour-horizon model with no context beyond 4 hours is structurally blind. Prior-day
high/low/close pivots are among the most reliable signed levels in intraday crypto.

### 4.6 Block E — Signed shape *(cheapest changes in this document)*

| Change | Detail |
|---|---|
| **Split `wick_ratio`** | Currently `1 − body/range`, symmetric — it is the **7th most important direction feature and it discards its own sign**. Replace with `upper_wick_ratio = (high − max(open,close))/range` and `lower_wick_ratio = (min(open,close) − low)/range`. Keep the combined version for the gate. **This is a ~10-line change with genuine expected value.** |
| `close_location_value` | `(close − low)/(high − low)` and its 12-bar mean. Signed cousin of `wick_ratio`. |
| `realized_skew_48` | Third moment of 48-bar log returns. Realized skewness is one of the more robust short-horizon signed predictors in crypto. |
| `realized_kurt_48` | Fourth moment — tail risk, feeds the risk head too. |
| `semivar_ratio_48` | `upside_vol / downside_vol` over 48 bars. Signed volatility asymmetry. |
| `signed_volume_trend` | Existing `volume_trend × sign(log_return_12)`. |

### 4.7 Block F — Symbol identity

Add `symbol` as a **native LightGBM categorical feature**, plus continuous descriptors that generalise to
unseen symbols: `listing_age_days`, `avg_dollar_volume_rank_30d`, `tick_size_relative_to_price`.

The last one matters: it is likely the *mechanical* cause of the §1 NaN pattern (flat candles on
coarse-tick alts), so it encodes a real behavioural difference the model should know about.

### 4.8 Feature hygiene

- **Install `shap`.** Every importance block in the report says `"shap is not an installed project
  dependency"`. Native gain importance is biased toward high-cardinality continuous features and gives
  **no sign** — which for a directional problem is exactly the information you need. Add `shap` to
  `requirements.txt` and populate the existing `shap` slots. Cheap, and it will immediately show whether
  the new signed features are being used directionally.
- Add a **feature-count guard**: this plan roughly doubles the matrix. With ~5,600 effectively
  independent samples (§2.2), adding 40 features without first fixing §1 will overfit. **Ship §1 before
  §4.**
- Prune as you add: run permutation importance on the walk-forward folds and drop features that do not
  survive.

---

## 5. The risk model

### 5.1 The target is partly a feature

```python
heat_component       = 1.0 - clip(mae_ratio, 0, 1)          # unknown at t — this is the real question
volatility_component = 1.0 - clip(volatility_percentile, 0, 1)
target_risk_score    = heat_component * (0.5 + 0.5 * volatility_component)
```

`volatility_percentile` is `garch_vol_rank` — **already column 25 of the feature matrix**. So the target
is `f(unknown) × g(known input)`. A gradient-boosted tree will trivially learn `g` and report the result
as R².

The feature importances confirm it exactly: `garch_vol_rank` is **#1 at 8.96%** and `atr_rank` **#2 at
5.96%** — the model's top two splits are recovering the deterministic multiplier it was handed. The
genuine predictive content on path heat is **materially below the reported R² = 0.089**, and quite
possibly near zero.

**F5.1 — Split the target.**
Predict `mae_ratio` (or better `mae_pct`, in price terms, which is what sizing actually needs) directly.
Apply the volatility adjustment *deterministically after* prediction, at the sizing layer where it
belongs. Then R² becomes an honest measurement of the only thing the model is for.

### 5.2 It is trained on survivors and scored on everyone

`RiskModel.train` filters to `direction_target != NO_TRADE_OR_FAIL` — i.e. only bars whose forward path
actually reached take-profit. At inference it is asked about every bar Direction and Entry approved.
Direction's precision at its recommended threshold is ~52–57%, so **roughly half the bars it scores live
come from the population it was never trained on.**

The evidence is unambiguous in the metrics:

| | Target | Prediction | |
|---|---|---|---|
| Mean | 0.5501 | 0.6263 | **+0.076 systematic optimism (+13.8%)** |
| Median | 0.5469 | 0.6130 | +0.066 (so it is not a mean-vs-median artifact of the L1 objective) |
| Std | 0.2308 | 0.1223 | **reproduces 53% of the true dispersion** |
| Min | 0.0903 | 0.3014 | **the model can never output a genuinely dangerous score** |
| Max | 0.9983 | 1.0264 | out of the target's definitional range |

The `min` row is the important one. The true risk score reaches 0.09; the model's floor is 0.30 —
roughly the bottom **~14%** of the true risk distribution is unreachable. **It is not a risk model; it is
a slightly noisy constant of ~0.63.** That is precisely the "very weak" behaviour you observed.

The commit that introduced this filter (`895ef21` / the `RiskModel.train` docstring) argues it is correct
because "Direction/Entry have already approved" the row. That reasoning holds only if Direction is
accurate. At 57% precision it is not.

**F5.2 — Train on the population it will actually score.**
Fit on all bars that pass a *realistic simulated* Direction+Entry filter (using out-of-fold predictions,
not the labels), taking the path of the side the model **would have taken** — including the bars where
that side lost. The label `mae_ratio` is well-defined for every bar and both sides; the current code
already computes `long_mae_ratio` and `short_mae_ratio` for all rows and then throws most of them away.

### 5.3 A point estimate is the wrong output

Sizing needs a *tail*, not a mean. Predicting "expected heat is 0.55" is nearly useless; predicting "there
is a 5% chance heat exceeds 0.92" is directly actionable.

**F5.3 — Replace the single L1 regressor with a quantile ensemble.**

```python
RISK_QUANTILES = (0.50, 0.80, 0.95)
# LGBMRegressor(objective="quantile", alpha=tau) per tau, on target = mae_pct
```

Then:
- **Stop distance** = `q₀.₉₀(mae_pct) × buffer` — an empirically grounded stop instead of a fixed ATR
  multiple. This also gives the Exit head a far better SL target than the current
  `heat × 1.15` formula (which, per the labeler's own comments, collapses onto the `_MIN_SL_PCT` floor
  and destroys the variance the regressor needs — see §6).
- **Position size** ∝ `1 / q₀.₉₅(mae_pct)` — risk parity on *predicted* heat rather than on trailing
  volatility.
- **Veto** when `q₀.₅₀(mae_pct)` alone already exceeds the affordable stop.

Enforce monotonicity across quantiles (sort the three outputs) to avoid crossing.

**F5.4 — Conformalise.**
Wrap the quantile heads in [conformalized quantile regression](https://arxiv.org/pdf/2602.01912) with a
rolling calibration window, so "95th percentile MAE" is an honest 95%-coverage bound on live data rather
than an in-sample artifact. Report realised coverage in the diagnostic — if the 95% band covers 78% of
live trades, that is a headline number the operator needs.

### 5.4 Sizing is four hand-tuned fudge factors

```python
composite = score * (0.35 + 0.65 * confidence_factor) * volatility_factor * tier_factor
leverage  = floor(composite * leverage_cap)
```

Four multiplicative constants (`0.35`, `0.65`, the `**2` in `volatility_factor`, and the
`{LOW:1.0, MEDIUM:0.75, HIGH:0.5}` map), none calibrated against realised PnL. And `tier_factor` is
derived from `risk_tier_from_score(score)`, which inverts the training-time score formula — so `score`
enters the product **twice**, once directly and once through the tier, giving it an unintended
super-linear weight.

**F5.5 — Replace with expected value / fractional Kelly.**

```
p  = calibrated P(direction correct)        # from the direction head
b  = predicted reward / risk                # from the exit head (TP distance / SL distance)
f* = (p * (b + 1) - 1) / b                  # Kelly fraction
size = clamp(kelly_fraction * f*, 0, max_allocation)     # kelly_fraction ∈ [0.25, 0.5]
```

Then scale down by the predicted MAE quantile (§5.3) and by portfolio heat (§5.5).

**Hard prerequisite:** Kelly requires a *calibrated* `p`. The direction stage's isotonic calibration
reported `improved: false` and production runs on the **raw** estimator — so `p` is not calibrated today.
[Machine learning only helps Kelly if it improves the calibrated posterior edge](https://coriva.eu.org/en/kelly-criterion-position-sizing/),
not if it merely ranks well. Sequence: fix §1 and §4 → re-measure calibration → then adopt Kelly. Until
calibration passes, keep the heuristic but *reduce* it to two terms (score and confidence) and delete the
double-counted tier factor.

Use half-Kelly or quarter-Kelly. Full Kelly on a model with ~1% pseudo-R² would be reckless — [half-Kelly
gives ~75% of the growth rate at half the volatility](https://coriva.eu.org/en/kelly-criterion-position-sizing/).

### 5.5 There is no portfolio-level risk layer at all

Every position is sized independently. But 27 USDT-M altcoin perps are one factor bet: `beta_btc_288`
(§4.2) will show most of them at 0.8–1.2 to BTC. Six concurrent 2% positions in the same direction is
not 12% diversified risk — it is a single ~12% BTC-beta position.

**F5.6 — Add `module_e_execution/portfolio_risk.py`:**
- Rolling correlation matrix across open positions (reuse the cross-sectional pass from §4.2).
- **Correlation-adjusted gross exposure cap**: `effective_exposure = sqrt(wᵀ Σ w)`, capped.
- **Aggregate beta cap**: `Σ wᵢ × betaᵢ ≤ max_portfolio_beta`.
- **Same-direction concurrency cap**: at most N simultaneous longs (or shorts) across the universe.
- **Drawdown-state throttle**: feed current strategy drawdown in as a sizing multiplier so the system
  de-risks after losses rather than sizing identically into a losing streak.
- **Regime multiplier** from the existing `hmm_prob_high_vol` — already computed, currently unused by the
  risk head.

### 5.6 Also worth fixing

- **Bounded output.** The target is in [0,1] but the regressor is unconstrained (predicted max 1.026).
  Either fit on `logit(target)` or use `objective="cross_entropy"`, which LightGBM supports for targets
  in [0,1]. Clamping at predict time hides the miscalibration rather than fixing it.
- **`risk_hyperparameters`** are the most complex of any head (`num_leaves: 95`, `max_depth: 7`,
  `min_child_samples: 25`) on the *smallest* dataset (457,731 rows). That is backwards. Once the target is
  honest (F5.1), re-tune — expect to need *less* capacity, not more.

---

## 6. The exit model (in scope because it feeds risk and R5)

`beats_rule_based_baseline` is **false** for two of three targets:

| Target | Model MAE | Rule-based MAE | Verdict |
|---|---|---|---|
| `target_tp_pct` | 0.014958 | 0.012111 | **23.5% worse** |
| `target_trailing_pct` | 0.007479 | 0.006055 | **23.5% worse** |
| `target_sl_pct` | 0.001979 | 0.003588 | 44.8% better ✓ |

**F6.1 — Ship the baseline where it wins.** Until the model beats it, use the rule-based ATR geometry for
TP and trailing, and the model only for SL. Make this a config switch driven by the measured comparison,
not a manual decision — the diagnostic already computes `beats_rule_based_baseline`; wire it to actually
select the production path, the same way `calibration.improved` already selects the calibrated estimator.

**F6.2 — Delete the trailing regressor.** `target_trailing_pct = optimal_tp × 0.5` in the labeler, so it
is a rescaled copy of the TP target. The identical R² (`0.2026280610667266` for both, bit-for-bit) and
identical feature importances confirm it. Training a second booster on `0.5 × y` costs a third of exit
training time and buys nothing — and after independent clamping in `_assemble` the two can even become
geometrically inconsistent. Compute trailing from the TP prediction.

**F6.3 — The TP head is biased upward** (prediction mean 0.0278 vs target mean 0.0206, +35%). Combined
with R5's `min_reward_risk_ratio = 1.3` floor, this systematically inflates targets. Fit on
`log(tp_pct)` or use a quantile objective; check bias explicitly in the diagnostic.

**F6.4 — Fix the SL target floor.** The labeler's own comments (lines ~79–100) document that
`_MIN_SL_PCT = 0.0015` binds for a large share of rows and collapses their true optimal SL into one
constant, destroying the variance the regressor needs (`r² ≈ 0.062`). The right fix is not to lower the
floor (the comment correctly notes a ~0.2% round-trip cost makes tighter stops economically unsound) —
it is to **not train on a clamped target at all**. Train on the raw `heat_to_peak / entry` and apply the
economic floor at inference, where it belongs. Better still, replace this head entirely with the MAE
quantile model of §5.3, which answers the same question properly.

---

## 7. Labelling and training changes for directional accuracy

None of these raise a threshold; all of them increase or improve the *information* reaching stage 2.

**F7.1 — Stop folding high-volatility winners into NO_TRADE.**
`_classify` relabels a correct directional call as `NO_TRADE_OR_FAIL` whenever
`discard_very_high_risk = True` and volatility percentile ≥ 0.95 (or MAE ratio > 0.85). Those rows then
vanish from stage 2's training population entirely. This teaches the *direction* stage about *risk* —
which is the risk head's job, and the risk head has its own target for it. Set
`discard_very_high_risk = False`, keep the tier as a separate column, and let the risk head handle it.
This is a pure gain in stage-2 training data at no cost to safety, since sizing still sees the tier.

**F7.2 — Give stage 2 a continuous, symmetric target on *all* bars.**
Stage 2 currently trains only on the ~53% of bars where one side reached TP. Bars that drifted +1.8 ATR
without touching 2.0 are discarded, even though their direction was unambiguous. Replace the binary
`is_long` with a continuous signed target defined on every bar:

```
directional_strength = (long_mfe_ratio - short_mfe_ratio)      # both already computed in _SideSimulation
```

and train either a regressor on it, or a classifier on soft labels
`p_long = sigmoid(k × directional_strength)`. This roughly **doubles stage 2's training population** and
removes the survivorship structure in its labels. Expected to be the single biggest modelling gain after
the data fixes.

**F7.3 — Weight by label uniqueness.**
With `max_holding_bars = 48`, consecutive labels overlap up to 47/48. Implement López de Prado's average
uniqueness: for each row, `uniqueness = 1 / (mean number of concurrent labels over its horizon)`, and
pass it as `sample_weight` (multiplied with the existing recency weight). Alternatively — or
additionally — use **CUSUM event sampling** to select bars rather than training on all of them.
This is the standard correction for exactly the overfitting mechanism in §2.2.

**F7.4 — Weight by outcome magnitude.** A trade that reached TP in 3 bars with 0.1 MAE and one that
scraped in at bar 47 with 0.9 MAE are currently the same label. Weight by
`mfe_ratio / (1 + mae_ratio)` or by `1/bars_to_exit`.

**F7.5 — Lengthen the recency half-life.** 45 days against a 370-day window is throwing away most of the
data (§2.2). Sweep `recency_half_life_days ∈ {45, 90, 180, ∞}` on the walk-forward folds and pick
empirically. Once §1 lands and there is 3.5× more training data, a longer half-life is likely to win.

**F7.6 — Sweep the barrier geometry.** `tp=2.0 ATR / sl=1.0 ATR / 48 bars` is one arbitrary point in a
3-D space, never tested against alternatives. Fit at `(1.5, 0.75, 24)`, `(2.0, 1.0, 48)`,
`(3.0, 1.5, 96)` and select on out-of-sample **economics** (§9), not accuracy. Directional predictability
in crypto is strongly horizon-dependent, and a 4-hour horizon at 2:1 may simply not be where the edge is.

**F7.7 — Feed the gate's output into stage 2.** Add `p_trade` (out-of-fold) as a feature to stage 2 so it
can condition on regime. Cheap stacking; commonly worth 1–3% log loss.

**F7.8 — Seed-average stage 2.** Fit 5 seeds and average probabilities. On targets this noisy this
reliably buys 1–3% log loss for a 5× training cost on the smallest head.

---

## 8. Suggested sequencing

| Phase | Work | Effort | Expected impact |
|---|---|---|---|
| **0** | §9 measurement harness — two-sided direction metric, economics in the sweeps, per-column null counts, `shap` installed | 1–2 days | **None directly — but nothing below is judgeable without it.** Do this first. |
| **1** | §1 F1.1–F1.5 (stop dropping 73% of training data) + §2.5 (repair R3) | 1–2 days | **Largest single gain.** ~3.5× training data, removes the population shift, and R3's repair increases trade count without touching quality. |
| **2** | §4.6 signed shape features + §4.7 symbol identity + §7.1 (stop folding winners into NO_TRADE) | 2–3 days | Cheap, high ratio. Splitting `wick_ratio` alone is ~10 lines against the 7th most important feature. |
| **3** | §4.2 cross-sectional block (BTC-relative, residual momentum, breadth) | 4–6 days | The largest *feature* gain. Requires the panel-wide pass and live-path mirroring. |
| **4** | §3 archive loader + §4.3 derivatives used directionally | 4–6 days | Restores the deleted block at full history and uses it properly (`oi_price_quadrant`, `top_trader_skew`, `basis`). |
| **5** | §5 risk redesign — split the target, fix the training population, quantile heads, conformal calibration | 4–6 days | Turns a near-constant into an actual risk model. |
| **6** | §7.2 continuous stage-2 target, §7.3 uniqueness weighting, §7.6 barrier sweep | 3–5 days | Modelling gains that only become measurable after phases 1–4. |
| **7** | §4.4 order-flow extensions, §4.5 multi-timeframe, §5.5 portfolio risk, §5.4 Kelly sizing | ongoing | Kelly is gated on calibration passing — re-check after phase 4. |

**Re-baseline after every phase.** With ~5,600 effectively independent samples today, several of these
changes are individually within noise; only the walk-forward economics will tell you which stuck.

---

## 9. Measurement harness (build this first)

The current diagnostic cannot detect success at the thing you are trying to improve.

**F9.1 — Two-sided direction metric.** Replace the LONG-only `direction_threshold_sweep` with a
symmetric margin sweep:

```python
for margin in (0.00, 0.02, 0.05, 0.10, 0.15, 0.20):
    take = abs(p_long_given_trade - 0.5) >= margin
    side = where(p_long_given_trade > 0.5, LONG, SHORT)
    report: n_signals, long_precision, short_precision, overall_accuracy,
            long_share, and the economics below
```

**Report `overall_accuracy` on taken signals as the project's headline "directional accuracy".** That is
the number your goal is actually about, and it does not currently exist anywhere in the report.

**F9.2 — Wire the backtester into the sweeps.** Every sweep row currently reports
`average_r / win_rate / profit_factor / expectancy / net_pnl / max_drawdown` as `NOT_AVAILABLE`, so all
thresholds are being chosen by classification F1. The backtester exists (`module_e_execution/backtester.py`).
Run it per sweep row with realistic fees and slippage and populate those fields. **Then select thresholds
by expectancy, not F1.** This is also the only way to verify that a change improved *trading* rather than
*classification*.

**F9.3 — Data health.** Per-feature null counts; null rate by symbol × month; per-split row counts
against theoretical capacity with an explicit `coverage_pct` that would have made §1 obvious at a glance.

**F9.4 — Calibration curves,** not just Brier/log-loss scalars: 10-bin reliability diagrams for the gate,
the direction stage, and the entry head, plus realised coverage for the risk quantiles (§5.3).

**F9.5 — Regime-conditional breakdowns.** Direction accuracy by HMM regime, by volatility quartile, by
symbol liquidity tier, by hour-of-day. A model with 1% average skill may have 6% skill in one regime and
negative skill elsewhere — which would change the strategy entirely and is currently invisible.

**F9.6 — Fix the AI summary's component ranking.** "Weakest component: risk" is derived by comparing a
regression R² against a 3-class argmax accuracy. Either normalise every head to a skill score against its
own baseline (as done in §2.1), or drop the ranking.

---

## 10. Direct answers to the three questions

**"How do I increase directional accuracy without raising the threshold?"**
Three things, in order. (1) Stop discarding 73% of the training window — §1 — this is not a modelling
problem, it is a `dropna` over 50 columns. (2) Give the direction stage *signed* information: split
`wick_ratio`, add BTC-relative residual momentum and cross-sectional rank, add the OI/price quadrant.
Today six of its top ten features cannot express a direction at all — §4. (3) Give stage 2 more and
better-targeted rows: stop folding high-volatility winners into NO_TRADE, and train it on a continuous
signed target across all bars rather than a binary one across the surviving 53% — §7.1, §7.2.
Separately: **repair R3** — you are already running a hidden `p_trade ≥ 0.65` threshold that discards 95%
of the signals R1A admits (§2.5). Removing it moves in exactly the direction you asked for.

**"Replace the 30-day features with retrievable ones."**
Mostly you don't have to. The 30-day limit is a REST-API limit; `data.binance.vision` publishes open
interest, both long/short ratios and the taker buy/sell ratio at full contract history in daily archives
(§3.1), along with `bookTicker`, `bookDepth`, `markPriceKlines` and `indexPriceKlines`, which also restore
the five order-book/liquidation features deleted earlier and add `basis` for free. Build the archive
loader, then use those series *directionally* rather than as levels and ranks (§4.3). Where you do want
genuinely new features, the highest-value additions need no new data source at all: BTC-relative residual
momentum and cross-sectional rank from klines you already have (§4.2), and multi-timeframe context
(§4.5).

**"The risk model is very weak."**
It is weaker than R²=0.089 suggests, and for structural reasons rather than tuning ones. Its target
multiplies an unknown (path heat) by a factor that is already one of its own inputs (`garch_vol_rank`),
so most of the measured R² is the model recovering something it was handed — §5.1. It is trained only on
bars whose forward path succeeded but scored live on bars that include failures, producing +13.8%
systematic optimism and an output floor of 0.30 against a true floor of 0.09 — §5.2. In practice it emits
a near-constant ~0.63 and cannot flag a dangerous trade. Fix: predict raw `mae_pct` directly, train on
the population it will actually score, and replace the point estimate with conformalised quantile heads
so sizing can use a tail rather than a mean — §5.3, §5.4. Then replace the four hand-tuned sizing
multipliers with fractional Kelly on a calibrated probability (§5.5, gated on calibration actually
passing), and add the portfolio-level correlation and beta caps that do not exist today (§5.6).

---

## Sources

- [Binance — Open Interest Statistics (REST, 1-month retention)](https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Open-Interest-Statistics)
- [Binance — Top Trader Long/Short Account Ratio (REST, 30-day retention)](https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Top-Long-Short-Account-Ratio)
- [data.binance.vision — futures/um/daily/metrics](https://data.binance.vision/?prefix=data%2Ffutures%2Fum%2Fdaily%2Fmetrics%2F)
- [binance/binance-public-data — bulk archive documentation](https://github.com/binance/binance-public-data)
- [binance-public-data issue #211 — metrics dataset columns](https://github.com/binance/binance-public-data/issues/211)
- [López de Prado — Advances in Financial Machine Learning (triple barrier, meta-labelling, uniqueness)](https://toc.library.ethz.ch/objects/pdf03/e01_978-1-119-48208-6_01.pdf)
- [Hudson & Thames — Does meta-labelling add to signal efficacy?](https://hudsonthames.org/does-meta-labeling-add-to-signal-efficacy-triple-barrier-method/)
- [Algorithmic crypto trading using information-driven bars, triple-barrier labelling and deep learning](https://link.springer.com/article/10.1186/s40854-025-00866-w)
- [Bitcoin wild moves: evidence from order-flow toxicity and price jumps](https://www.sciencedirect.com/science/article/pii/S0275531925004192)
- [Easley et al. — Microstructure and market dynamics in crypto markets](https://stoye.economics.cornell.edu/docs/Easley_ssrn-4814346.pdf)
- [Empirical note on VPIN alpha decay in BTC](https://medium.com/coinmonks/i-used-a-2012-market-microstructure-paper-to-find-alpha-in-btc-it-worked-but-its-dying-500f9bc0fc94)
- [A trend factor for the cross-section of cryptocurrency returns](https://www.cambridge.org/core/services/aop-cambridge-core/content/view/4C1509ACBA33D5DCAF0AC24379148178/S0022109024000747a.pdf/trend_factor_for_the_cross_section_of_cryptocurrency_returns.pdf)
- [Machine learning and the cross-section of cryptocurrency returns](https://affi2023.eventsadmin.com/Papers/ViewContribution?cid=8390&h=A0CBBA4C296557EB2205D3ECCD9DBD7F)
- [Reliable real-time VaR estimation via quantile regression forest with conformal calibration](https://arxiv.org/pdf/2602.01912)
- [Temporal Conformal Prediction — adaptive risk forecasting](https://arxiv.org/pdf/2507.05470)
- [Conformal Kelly — conformal prediction intervals as the scale in fractional Kelly sizing](https://arxiv.org/html/2608.01494)
- [Kelly criterion and position sizing: from formula to quant practice](https://coriva.eu.org/en/kelly-criterion-position-sizing/)
