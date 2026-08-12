# New Idea — feature overhaul, Direction rework, three-stage TP/SL

This document describes what changed on the `new-idea` branch, why, and how to
verify it. It is written to be read alongside the code, not instead of it.

---

## 1. Three unreliable features removed

`long_short_ratio`, `open_interest_change` and `taker_buy_sell_ratio` are gone —
**deleted, not zero-filled**.

Binance publishes the underlying series (`futures/data/globalLongShortAccountRatio`,
`futures/data/takerlongshortRatio`, and open-interest history) for a rolling
window of roughly 30 days. The training period spans a year. There is no honest
way to reconstruct those columns across it, and back-filling them with `0`, `NaN`
or a neutral constant would have taught the models that "we have no data" is a
market state — a feature that correlates with *when* a row was collected rather
than with anything about the market.

Removed from:

| Layer | File | What went |
| --- | --- | --- |
| Feature contract | `module_b_features/features.py` | the three columns, from `FEATURE_COLUMNS` and every per-head contract |
| Feature computation | `module_b_features/features.py` | `_add_microstructure_features` no longer computes them |
| Raw schema | `module_a_data/models.py` | `FuturesMetrics.long_short_ratio`, `.top_trader_long_short_ratio`, `.taker_buy_sell_ratio` |
| Storage | `module_a_data/db_models.py` | the matching `futures_metrics` columns |
| Persistence & reads | `module_a_data/db_handler.py` | upsert payload and the loaded frame |
| Ingestion | `module_a_data/fetcher.py` | the `futures/data` ratio calls and their `_safe_ratio` helper |

`open_interest_rank` survives: it is a rolling percentile of open interest,
which *is* available for the full period, and only the 12-bar *change* feature
was affected by the coverage gap.

Because the training data no longer carries those columns, an old artifact and a
new dataset are no longer compatible — retrain (`python main.py train`).

---

## 2. Three new 5-minute order-flow features

The 5-minute architecture is unchanged. No 15-minute or 1-hour dataset was
introduced; all three features are 5-minute quantities aligned one-to-one with
the existing 5-minute candles.

### Data path

Binance Futures **aggTrades** are folded into 5-minute buckets that share the
candles' own grid:

```
fetcher.fetch_agg_trade_flow()  ->  AggTradeFlow  ->  agg_trade_flow table
                                                          |
                                    exact-key join on bucket open time
                                                          v
                                              FeatureEngineer._add_order_flow_features
```

Aggressor side comes from `isBuyerMaker`: `false` means the buyer lifted the
offer (aggressive **buy**), `true` means the seller hit the bid (aggressive
**sell**). Only fully closed buckets are ever written, so a partially observed
interval cannot reach the feature stack.

The join is on the exact bucket key, never `merge_asof`. An as-of join would
silently attach the *previous* bar's flow to a candle whose own bucket is
missing; an exact join makes a missing bucket read as what it is — a bar with no
recorded aggressive flow.

Backfill runs as part of `bootstrap` and of the panel's setup stage; the live
cycle refreshes the last `DATA__AGG_TRADE_LIVE_BUCKETS` closed buckets.

### `order_flow_imbalance_5m`

```
(buy_volume - sell_volume) / (buy_volume + sell_volume)
```

Range `[-1, +1]`. Positive means aggressive buying dominated the bar. An empty
bucket yields `0.0` — the neutral value. The feature never emits `Inf` or `NaN`
(asserted in `tests/test_order_flow_features.py`).

### `volume_delta_5m`

```
buy_volume - sell_volume
```

Raw aggTrades quantities, exactly as specified — not derived from OHLCV, which
cannot distinguish an aggressive buyer from a passive one.

> **Flagged for the reader:** this column is in *base units*, so its scale
> differs by orders of magnitude across the universe (1000SHIB vs BCH). In a
> pooled cross-sectional model that makes it a weaker feature than a normalised
> equivalent would be. It is implemented exactly as specified, and the
> normalised information is already carried by `order_flow_imbalance_5m`, which
> is the same signal made scale-free. If you later want the delta to pull its
> weight cross-sectionally, rank-normalising it per symbol is the change to
> make — it would not add a feature, only rescale one.

### `relative_volume_5m`

```
volume[t] / mean(volume[t-20 : t-1])
```

The current candle is excluded from its own baseline by a `shift(1)` before the
rolling mean. Without it the feature would be partly a function of the very
quantity it is meant to contextualise.

The project had no pre-existing baseline that excluded the current bar (the
96-bar mean behind `volume_trend` includes it), so the specified default of 20
candles is used, configurable via `FEATURES__RELATIVE_VOLUME_LOOKBACK`. A
zero/near-zero baseline degrades to the neutral `1.0`; warm-up rows stay `NaN`
and are dropped like every other rolling feature.

### Routing to the heads

Features are routed to the head whose question they answer, and nowhere else
(`HEAD_FEATURE_COLUMNS` in `module_b_features/features.py`):

| Feature | Direction | Entry | Risk | Exit |
| --- | :---: | :---: | :---: | :---: |
| `order_flow_imbalance_5m` | ✅ | ✅ | — | — |
| `volume_delta_5m` | ✅ | ✅ | — | — |
| `relative_volume_5m` | ✅ | ✅ | ✅ | — |

Net feature count: **−3 removed, +3 added**. Direction and Entry see 53 columns,
Risk 51, Exit 50.

---

## 3. Direction model — better at the same threshold

**The decision threshold did not change.** `DECISION__MIN_DIRECTION_CONFIDENCE`,
`DECISION__MIN_DIRECTION_MARGIN` and `DECISION__MAX_NO_TRADE_PROBABILITY` are
untouched, and the evaluation harness reads them from live configuration rather
than taking its own — `tests/test_direction_evaluation.py` pins the evaluator's
accept/reject decision to the Decision Engine's own cascade, row for row.

### Diagnosis

The old head was one flat 5-class booster. Roughly 45 % of rows are
`NO_TRADE_OR_FAIL` and the two directional outcomes are near-symmetric, so
nearly all of the achievable log-loss reduction sits in the *trade vs no-trade*
question. That is where the capacity went: in the supplied diagnostic, LONG
recall was 4.8 % and SHORT recall 5.3 % while `NO_TRADE` recall was 96.5 %. The
model was not bad at direction so much as it was barely attempting it.

Mass was also fragmented: the aggregation in `DirectionPrediction` sums
`LONG_SUCCESS_LOW_RISK + LONG_SUCCESS_HIGH_RISK`, but the booster's objective
treated them as unrelated classes, so it spent splits separating two tiers whose
distinction the decision layer immediately discards.

### The change: a two-stage cascade

| Stage | Trained on | Learns |
| --- | --- | --- |
| `gate` | every row | `P(tradeable \| x)` |
| `direction` | **tradeable rows only** | `P(LONG \| x, tradeable)` |
| `tier` | tradeable rows only | `P(low risk \| x, tradeable)` |

The restriction on stage 2 is the whole point: it never sees a `NO_TRADE` row,
so none of its capacity is spent re-deriving the gate, and every split it makes
is spent on long-vs-short.

The joint distribution is reassembled into the *identical* five-class contract:

```
p(NO_TRADE)  = 1 - p_gate
p(LONG_*)    = p_gate *      p_dir  * {p_tier, 1 - p_tier}
p(SHORT_*)   = p_gate * (1 - p_dir) * {p_tier, 1 - p_tier}
```

so `DirectionPrediction`, the R1–R3 cascade, the implied risk tier, the audit
log and every threshold see exactly what they saw before. Nothing downstream
knows the architecture changed.

Also addressed:

- **Class weighting** — `class_weight="balanced"` on both binary stages, so a
  mild long/short imbalance cannot become a systematic directional bias.
- **Stage-2 hyper-parameters** — the long/short stage sees a fraction of the
  rows the gate does and overfits readily, so it is regularised harder
  (`ML__DIRECTION_STAGE2_*`).
- **Probability calibration** — isotonic calibrators for the gate and direction
  stages, fitted on a purged slice carved out of the **training** block, and
  kept only when they improve the Brier score on that slice. Validation is
  already spent on early stopping; the test block is untouchable.
- **Feature redundancy** — no features added beyond the three specified, and
  three removed.
- **Feature scaling** — not applicable to gradient-boosted trees; scaling was
  checked and deliberately not added.

### The baseline is frozen

`DirectionBaselineModel` is pinned to `single_stage` **and** to the pre-change
feature block (no order-flow columns). It writes its own artifact
(`direction_model_baseline.joblib`), is excluded from `MLSubsystem.heads` so
routine training can never overwrite it, and can never serve a live prediction.
Flipping `ML__DIRECTION_ARCHITECTURE` does not move it — asserted in the tests.

---

## 4. Three-stage take profit + dynamic stop loss

One state machine — `module_e_execution/tp_ladder.py` — drives backtest, paper
and live. They differ only in what they feed it: `on_bar(high, low)` for OHLC
replay, `on_tick(price)` for live observation. A ladder re-implemented per engine
is a ladder that behaves differently in the backtest than in production.

| Stage | Trigger | Closes | Stop moves to |
| --- | --- | --- | --- |
| TP1 | ⅓ of the target distance | 30 % | entry (breakeven) |
| TP2 | ⅔ of the target distance | 30 % | the TP1 price |
| TP3 | the model's full target | remaining 40 % | — position closed |

Levels are fractions of the take-profit distance the Exit model produced, so the
ladder inherits the model's volatility-scaled geometry instead of bolting a
fixed one on top, and TP3 *is* the model's target — the reward/risk the Decision
Engine checked in R5 is the reward/risk the position carries. The spec's worked
example falls out exactly: entry 100, TP 3 %, SL 2 % → TP1 101, TP2 102, TP3 103,
SL 98.

Shorts are the exact inverse, through the same code path.

Everything is configurable, nothing is hard-coded: `TAKE_PROFIT__LEVEL_FRACTIONS`,
`TAKE_PROFIT__CLOSE_FRACTIONS`, `TAKE_PROFIT__BREAKEVEN_OFFSET_PCT`,
`TAKE_PROFIT__MIN_LEG_FRACTION`, and `TAKE_PROFIT__ENABLED=false` restores the
original single-target behaviour end to end.

### Integration, not duplication

The ladder plugs into the existing `Position` object that the backtester, paper
trader and live executor already share. `effective_stop()` now composes the
signal stop, the trailing stop and the ladder stop by taking the **most
protective** of them, so the pre-existing trailing mechanism and the ladder
cannot loosen each other. Partial closes go through one accounting method
(`Position.book_partial_close`), so a leg costs the same in all three engines.

In live trading the take-profit legs are placed as three resting reduce-only
`TAKE_PROFIT_MARKET` orders sized to their allocations — the exchange executes
the scale-out even if the process dies — while the single stop order is
cancelled and re-placed as stages advance. Stop breaches are left to the resting
stop order; racing the exchange with a second closing order is how positions get
double-closed.

### The intrabar ordering rule

5-minute OHLCV records four numbers and **cannot** say whether the high came
before the low. Rather than assume, the rule below is applied identically in the
labeler, the backtester and the reporting, and is documented in the module:

1. **The stop is tested first, against the bar's adverse extreme.** If the bar
   could have stopped us out, it did — even when that same bar also reached a
   take-profit level. A bar touching both TP1 and the initial stop is booked as
   a stop, never as "TP1 first, then a protected exit".
2. **Only then may the ladder advance**, strictly in order TP1 → TP2 → TP3.
   Those levels are monotone in the favourable direction, so filling them in
   sequence assumes nothing about intrabar order.
3. **The tightened stop is re-tested against the same bar.** A bar that ran to
   TP2 and collapsed gives back the TP1-locked stop *inside that bar*, not on
   the next one.

Rule 1 is the pessimistic assumption; rule 3 closes the loophole rule 2 would
otherwise open. Liquidation is checked before all of it. Existing spread, fee,
slippage, funding and candle-sequencing assumptions are untouched — signals
still fill at the next bar's open, never at the close that produced them.

If intrabar data (the aggTrades already being ingested) is later fed to the
backtester, the sequence could be *proven* rather than assumed. Until then this
is a documented conservative rule, not a silent guess.

---

## 5. Data leakage and validation

- **Point-in-time features.** `tests/test_leakage_and_splits.py` rebuilds the
  feature matrix over a prefix of history and asserts every overlapping row is
  bit-identical, and separately asserts that rewriting *future* flow buckets
  leaves earlier features untouched.
- **`relative_volume_5m` excludes the current candle** — asserted both against
  the explicit formula and behaviourally.
- **A real test block now exists.** The repository previously had only
  train/validation. `ML__TEST_FRACTION` reserves a chronological tail that
  nothing reads except final evaluation: every head fits through
  `BaseModelHead._split()`, which removes it, so "the test set is never trained
  on" is structural rather than a convention.
- **Purging is measured in time, not rows.** This is a leakage fix, not a
  refactor: the dataset is pooled across ~30 symbols, so one bar of history is
  ~30 rows, and the old row-based purge of 60 removed *two bars* where the
  48-bar label horizon needed far more. The gap is now
  `(purge_bars + embargo_bars) × timeframe_ms` and is applied on both sides of
  every boundary.
- **Calibration comes out of train**, never validation or test.
- **Walk-forward folds are carved out of train+validation only**, so a
  walk-forward number can never be a disguised test-set number.

---

## 6. Running the before/after comparison

```bash
python main.py compare                 # direction + walk-forward + trading
python main.py compare --skip-backtest # direction metrics only (much faster)
```

Writes `artifacts/reports/baseline_vs_new_idea.{json,md}` and prints the
Markdown.

What it does, in order, all on one dataset:

1. Trains the frozen single-stage baseline and the new cascade **on identical
   rows**, scores both with the **same evaluator at the same threshold**, and
   reports accuracy, balanced accuracy, macro F1, per-action
   precision/recall/F1, both confusion matrices, a threshold-free LONG-vs-SHORT
   AUC, and — deliberately alongside every quality metric — the **number of
   directional signals**, so a model that "improves" by trading less cannot hide
   it.
2. Refits both per walk-forward fold and reports per-fold scores plus their
   dispersion.
3. Backtests the same history once per direction head, swapping only that head
   in and out of the live `MLSubsystem` so entry, exit, risk, the decision
   cascade, the ladder, fees, slippage and funding are byte-for-byte identical.
   Reports win rate, profit factor, expectancy, net PnL, max drawdown, trade
   count, the LONG/SHORT split, and the ladder statistics (% reaching TP1/TP2/
   TP3, % stopped at breakeven, % stopped at TP1-protected profit, average
   realised R, average trade return).

The test block is read exactly once, at the end of step 1, and nothing is tuned
on what it says.

### Status of the numbers

**No real market data was available in the environment where this branch was
written** — the repository ships no database and the environment has no exchange
access, so the tables above cannot be filled in with real Binance results here.
Running `python main.py compare` against a populated database produces them.

The harness itself is verified end to end: it was run against a synthetic market
whose order flow genuinely leads price, and produced a complete report with
every required section. Those numbers describe generated data and are **not** a
claim about live performance; they are evidence that the measurement apparatus
works and that the ladder and cascade behave as designed.

---

## 7. What was deliberately not touched

Per the instruction to keep the Direction work isolated: the Entry, Exit and
Risk heads were changed **only** where the shared plumbing required it — their
feature contracts (routing, as specified) and the shared purged/embargoed split
(so they too stop training on the test block). Their objectives,
hyper-parameters, targets and heuristics are unchanged.

---

## 8. Test coverage

```bash
python -m pytest tests/ -q     # 59 tests
```

| File | Covers |
| --- | --- |
| `test_order_flow_features.py` | feature definitions, bounds, zero-volume safety, exact-key join, current-candle exclusion, removal of the three metrics, per-head routing |
| `test_tp_ladder.py` | the worked example, all stage transitions, short inversion, configurable allocations, dust merging, and each intrabar ordering rule |
| `test_leakage_and_splits.py` | point-in-time correctness, split disjointness/ordering, the time-based purge, and that no head or fold touches the test block |
| `test_direction_evaluation.py` | evaluator/Decision-Engine threshold parity, the five-class output contract, baseline pinning, long/short discrimination, full comparison run |
| `test_backtest_ladder.py` | partial fills, full ladder runs, stop precedence, liquidation precedence, and the reported TP/SL and LONG/SHORT statistics |
