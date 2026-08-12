# Order-flow features and the three-stage take-profit ladder

Branch: `claude/ml-orderflow-tp-ladder-9k4m2p`, cut from
`claude/ml-position-detection-quality-7f23rb` (`4384d08`).

---

## Scope

Five things were requested. Two of them this codebase had already solved before
this branch, and saying so is more useful than re-litigating them:

| Task | Status |
| --- | --- |
| 1. Remove 3 unreliable features | **Done here** |
| 2. Add 3 aggTrades order-flow features | **Done here** |
| 3. Improve Direction at the same threshold | **Mostly pre-existing** — see below |
| 4. Three-stage TP + dynamic SL | **Done here** |
| 5. Leakage / validation | **Pre-existing**, extended to the new code |

On task 3, this branch already had the two-stage cascade (`gate` +
`direction`), isotonic calibration wired into production inference, a
calendar-based train/validation/test split with a purge+embargo gap,
walk-forward validation, threshold sweeps and the ML diagnostic report. What
this branch adds to Direction is the order-flow block feeding the cascade —
routed to it deliberately (below) — and nothing else. **No threshold was
changed**: `min_gate_confidence` (0.55) and
`min_direction_given_trade_confidence` (0.60) are untouched.

---

## 1. Three features removed

`open_interest_change`, `long_short_ratio` and `taker_buy_sell_ratio` are
deleted — from the feature contract, the `FuturesMetrics` schema, the
`futures_metrics` table, the live fetch, the historical backfill and the
diagnostics coverage map. Not zero-filled, not NaN-filled.

This is the same argument that already removed five order-book/liquidation
columns on this branch, one step milder. Those endpoints do not exist at all;
these exist but Binance retains only ~30 days of them against a multi-month
training window. Across most of that window they could not be reconstructed
either, and a placeholder teaches the models that "no data" is a market state —
a feature correlated with *when a row was collected* rather than with the
market.

What survives: `funding_rate` (full history since contract inception) and
`open_interest_rank` (a rolling percentile that degrades gracefully where open
interest is sparse). Only the 12-bar *change* was fatally affected.

Retraining is required — an old artifact and a new dataset no longer share a
feature contract.

## 2. Three order-flow features added

`aggTrades` is the one aggressive-flow source Binance serves from **full
contract history**. That is the entire reason this block can back a feature set
where `taker_buy_sell_ratio` could not.

```
fetch_agg_trade_flow()  ->  AggTradeFlow  ->  agg_trade_flow table
                                                    |
                              exact-key join on bucket open time
                                                    v
                                  FeatureEngineer._add_order_flow_features
```

Aggressor side comes from `isBuyerMaker`: `false` = the buyer lifted the offer
(aggressive **buy**), `true` = the seller hit the bid (aggressive **sell**).
Only fully closed buckets are ever written, so a partially observed interval
cannot reach the feature stack.

- **`order_flow_imbalance_5m`** — `(buy - sell) / (buy + sell)`, range
  `[-1, +1]`. An empty bucket is `0.0`, never `Inf` or `NaN`.
- **`volume_delta_5m`** — `buy - sell` in raw aggTrades quantities. Not derived
  from OHLCV, which cannot tell an aggressive buyer from a passive one.
- **`relative_volume_5m`** — `volume[t] / mean(volume[t-20 : t-1])`. The
  `shift(1)` is what excludes the current candle from its own baseline.

No existing lookback was reusable for the last one: `volume_trend`'s 12/96 means
and the HMM's 96-bar relative volume all *include* the current bar, which is
precisely what this feature must not do. Hence the specified 20-candle default
(`FEATURES__RELATIVE_VOLUME_LOOKBACK`).

### Why an exact-key join, not `merge_asof`

The funding/OI snapshots are joined backward-as-of, and correctly so. These
buckets are different: they sit on the candles' *own* 5-minute grid. An as-of
match would silently attach the previous bar's aggression to a candle whose own
bucket is missing. An exact-key join makes a missing bucket read as what it is —
a bar with no recorded aggressive flow. Pinned by
`test_flow_is_joined_on_the_exact_bucket`.

### Routing

Features go to the head whose question they answer, and nowhere else
(`HEAD_FEATURE_COLUMNS`):

| Feature | Direction | Entry | Risk | Exit |
| --- | :---: | :---: | :---: | :---: |
| `order_flow_imbalance_5m` | yes | yes | — | — |
| `volume_delta_5m` | yes | yes | — | — |
| `relative_volume_5m` | yes | yes | yes | — |

Imbalance and delta answer a directional/timing question. Relative volume is
also a genuine sizing input — a signal fired on a quarter of the usual volume
deserves less capital. Exit keeps the base block; its targets are excursion
geometry, driven by volatility rather than by who crossed the spread.

Net feature count: **−3 / +3**.

The Entry heuristic's order-flow term also moves from `taker_buy_sell_ratio` to
`order_flow_imbalance_5m` — a closer substitute for the `ob_imbalance` it stands
in for, being real order flow and available for the whole history.

> **Flagged:** `volume_delta_5m` is in base units, so its scale varies by orders
> of magnitude across the universe (1000SHIB vs BCH). In a pooled
> cross-sectional model that makes it weaker than a normalised equivalent. It is
> implemented exactly as specified, and the scale-free version of the same
> signal is already present as `order_flow_imbalance_5m`. If you later want the
> delta to pull its weight cross-sectionally, rank-normalising it per symbol is
> the change — it would rescale a feature, not add one.

## 4. Three-stage take profit + dynamic stop

One state machine — `module_e_execution/tp_ladder.py` — drives backtest, paper
and live. They differ only in what they feed it: `on_bar(high, low)` for OHLC
replay, `on_tick(price)` for live.

| Stage | Trigger | Closes | Stop moves to |
| --- | --- | --- | --- |
| TP1 | ⅓ of the target distance | 30 % | entry (breakeven) |
| TP2 | ⅔ of the target distance | 30 % | the TP1 price |
| TP3 | the model's full target | remaining 40 % | — closed |

Levels are fractions of the Exit model's own take-profit distance, so the ladder
inherits its volatility scaling, and TP3 *is* that target — the reward/risk the
Decision Engine checked in R5 is the one the position carries. Shorts are the
exact inverse through the same code path.

Everything is configurable: `TAKE_PROFIT__LEVEL_FRACTIONS`,
`TAKE_PROFIT__CLOSE_FRACTIONS`, `TAKE_PROFIT__BREAKEVEN_OFFSET_PCT`,
`TAKE_PROFIT__MIN_LEG_FRACTION`, and `TAKE_PROFIT__ENABLED=false` restores the
original single-target behaviour end to end.

### The intrabar ordering rule

5-minute OHLCV records four numbers and cannot say whether the high came before
the low. Rather than guess, the rule below is applied identically in the
backtester and the reporting, and is documented in the module:

1. **The stop is tested first, against the bar's adverse extreme.** If the bar
   could have stopped us out, it did — even when that same bar also reached a
   take-profit. A bar touching both TP1 and the initial stop is booked as a
   stop, never as "TP1 first, then a protected exit".
2. **Only then may the ladder advance**, strictly in order TP1 → TP2 → TP3.
   Those levels are monotone in the favourable direction, so filling them in
   sequence assumes nothing about intrabar order.
3. **The tightened stop is re-tested against the same bar.** A bar that ran to
   TP2 and collapsed gives back the TP1-locked stop *inside that bar*.

Rule 1 is the pessimistic assumption; rule 3 closes the loophole rule 2 would
otherwise open. Liquidation precedes all of it.

The aggTrades this branch now ingests could in principle *prove* the intrabar
sequence rather than assume it. They are not wired into the backtester for that
purpose, so this remains a documented conservative rule.

### Integration, not duplication

The ladder plugs into the existing shared `Position`. `effective_stop()` now
composes the signal stop, the trailing stop and the ladder stop by taking the
**most protective** of them, so the pre-existing trailing mechanism and the
ladder cannot loosen each other. Partial closes go through one accounting
method, so a leg costs the same in all three engines.

Live trading places the legs as three resting reduce-only
`TAKE_PROFIT_MARKET` orders sized to their allocations — the exchange completes
the scale-out even if the process dies — while the single stop order is
cancelled and re-placed as stages advance. Stop breaches are left to the resting
stop order; racing the exchange with a second closing order is how positions get
double-closed.

### Reporting

`/ml-report` gains a **Take-Profit Ladder** section: the configuration
alongside the measured outcome — % reaching TP1/TP2/TP3, % stopped at
breakeven, % stopped at TP1-protected profit, average realised R, average trade
return, and the LONG/SHORT split. A configured-but-never-triggered ladder is
therefore distinguishable from a disabled one.

## 5. Leakage and validation

The purged, embargoed, calendar-based split and the walk-forward validator were
already here and are unchanged. What this branch adds is coverage of the new
code by the same standard:

- `test_order_flow_features_are_point_in_time` rebuilds the feature matrix over
  a prefix of history and requires every overlapping row to match exactly.
- `test_future_flow_cannot_leak_backwards` rewrites future buckets and requires
  earlier features to be untouched.
- `relative_volume_5m` excluding the current candle is asserted against the
  explicit formula *and* behaviourally (a volume spike must not move any earlier
  bar's ratio).
- Only closed buckets are written, so the live path cannot see data the
  backtest could not.

---

## Running it

```bash
python main.py bootstrap   # candles, futures metrics, then aggTrade order flow
python main.py train       # backfills both, then fits the four heads
python -m pytest tests/ -q # 217 tests
```

The order-flow backfill is paced: `aggTrades` returns individual prints rather
than an aggregate, so it walks a bounded number of pages per run
(`DATA__AGG_TRADE_MAX_PAGES`) and resumes from the oldest stored bucket. Deep
history therefore fills in across several runs rather than in one pass.

## Test coverage

| File | Covers |
| --- | --- |
| `test_agg_trade_flow_ingestion.py` | aggressor-side convention, the one-hour request cap, quiet-hour advancement, closed-bucket-only emission, bucket folding |
| `test_order_flow_features.py` | definitions, bounds, zero-volume safety, exact-key joining, current-candle exclusion, per-head routing, point-in-time correctness |
| `test_tp_ladder.py` | the worked example, every stage transition, short inversion, configurable allocations, dust merging, each intrabar ordering rule, the report section |
| `test_backtest_ladder.py` | partial fills, full ladder runs, stop and liquidation precedence, the size-reporting regression |
| `test_feature_removal_and_entry_heuristic.py` | the three removals, updated for this round |

## Status of the numbers

No before/after trading comparison is included, because no market data was
available in the environment this was written in — the repository ships no
database and there is no exchange access. `python main.py train` produces the
full diagnostic report, including the new ladder section, against a populated
database.
