# AI Quant Trading System — Binance USDT-M Perpetual Futures (5m)

A modular, fully asynchronous algorithmic trading system for Binance USDT-M
perpetual futures on the **5-minute timeframe**. It covers the whole path from
raw market data to an audited execution decision: a quality-controlled data
pipeline, causal feature engineering (KAMA, FDI, GARCH, HMM), forward-looking
risk-tiered labelling, four independent ML heads, a rule-based Decision Engine,
paper/live execution with hard kill-switches, and a web panel.

> **Status.** The code is complete and runnable end to end. The shipped model
> artifacts are *not* — you train them on your own data. No claim is made that
> this system is profitable; see [Honest expectations](#honest-expectations).

---

## Architecture

```
                 ┌──────────────────────────────────────────────┐
                 │  main.py — APScheduler (5m cron) + uvicorn   │
                 └───────────────────────┬──────────────────────┘
                                         │ one asyncio event loop
   ┌────────────┐   ┌────────────┐   ┌───▼────────┐   ┌──────────┐   ┌─────────┐
   │  Module A  │──▶│  Module B  │──▶│  Module C  │──▶│ Module D │──▶│Module E │
   │ Ingest+QC  │   │ Features + │   │  4 ML      │   │ Decision │   │Execution│
   │            │   │  Labels    │   │  heads     │   │ Engine   │   │+ Risk   │
   └─────┬──────┘   └────────────┘   └────────────┘   └────┬─────┘   └────┬────┘
         │                                                 │              │
         └───────────────────► SQLite ◄────────────────────┴──────────────┘
                                  ▲
                    ┌─────────────┴──────────────┐
                    │ Module F: Audit + Web panel │
                    └────────────────────────────┘
```

| Module | Directory | Responsibility |
|---|---|---|
| **A** | `module_a_data/` | Async ccxt ingestion, QC gatekeeper with auto-healing, async SQLite persistence |
| **B** | `module_b_features/` | Causal features (KAMA, FDI, GARCH, HMM, micro-structure) + risk-tiered labels |
| **C** | `module_c_ml/ml_models.py` | Direction, Entry, Exit and Risk models — stateless inference |
| **D** | `module_c_ml/decision_engine.py` | Deterministic 10-rule cascade; the only producer of `TradeSignal` |
| **E** | `module_e_execution/` | Risk Guard, live executor, paper trader, event-driven backtester |
| **F** | `module_f_panel/` | Audit Engine + FastAPI dashboard on `IP:8000` |

Modules communicate only through validated Pydantic schemas
(`ModelInferenceResult` → `TradeSignal`). No module mutates another's state.

---

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env          # edit thresholds and (for live) API keys

python main.py bootstrap      # backfill ~6000 5m candles per symbol
python main.py train          # build the dataset and fit all four heads
python main.py backtest       # replay history through the full pipeline
python main.py run            # scheduler + web panel on 0.0.0.0:8000
```

Then open `http://<vps-ip>:8000/`.

| Command | What it does |
|---|---|
| `run` | Scheduler + web panel (default) |
| `bootstrap` | Backfill historical candles for the universe |
| `train` | Build the training dataset and fit the four heads |
| `backtest [--candles N] [--equity X]` | Event-driven replay with full metrics |
| `cycle` | Run exactly one trading cycle, then exit |
| `--mode paper\|live` | Override the execution mode |

---

## How it works

### Module A — ingestion & quality control

Fetches 5m OHLCV, L2 order-book snapshots, funding rate, open interest,
long/short ratios and (where exposed) liquidation flow for ~30 liquid perps,
concurrently and under a bounded semaphore.

The **QC gatekeeper** (`qc_validator.py`) runs four families of checks before
anything reaches the ML stack:

1. **Timestamp integrity** — every candle must open on an exact multiple of
   300 000 ms; gaps, duplicates and future-dated bars are detected and the
   precise list of missing grid timestamps is returned.
2. **Price logic** — `high ≥ max(open, close)`, `low ≤ min(open, close)`,
   all prices strictly positive.
3. **Volume logic** — non-negative; glitch spikes screened against the rolling
   median; a dead feed detected via the zero-volume ratio.
4. **Statistical anomalies** — log returns screened with a **MAD** z-score
   (median absolute deviation × 1.4826), not a standard deviation: a single 40 %
   glitch print inflates σ enough to hide itself, whereas the MAD has a 50 %
   breakdown point.

On a CRITICAL failure the pipeline **auto-heals**: it re-fetches only the damaged
timestamp window with exponential backoff, merges the fresh rows over the corrupt
ones and re-validates, up to `qc.max_heal_attempts`. A symbol that cannot be
healed is **skipped for the cycle** — never interpolated. Trading on invented
candles is worse than not trading.

### Module B — features & labels

**Every feature at bar `t` uses only data available at the close of `t`.** This
is verified, not assumed: truncating the input series leaves all earlier feature
values bit-identical.

- **KAMA** — Kaufman's adaptive MA; the efficiency ratio makes it track price in
  a clean trend and flatten in chop.
- **FDI** — Fractal Dimension Index over a rolling window. `≈1.0` straight
  trend, `≈1.5` random walk, `≈2.0` violent chop. Verified: a straight line
  yields 1.25, an alternating series 1.99.
- **GARCH(1,1)** — refit on a trailing window every `garch_refit_every` bars;
  between refits the conditional-variance recursion rolls forward on realised
  returns. The output at `t` is the **one-step-ahead** forecast, which is by
  definition known at `t`. Degrades to EWMA volatility if a fit fails.
- **HMM** — Gaussian HMM refit on a trailing window, with regimes produced by a
  **forward (α) filter**, never by Viterbi or forward–backward smoothing —
  smoothing at `t` would use data from `t+1…T`. Raw states are remapped onto a
  stable taxonomy (bull / bear / high-volatility / sideways) from the fitted
  emission means, so the feature keeps its meaning across refits.
- **Micro-structure** — order-book imbalance, spread, funding deltas, OI change,
  joined with `merge_asof(direction="backward")`.

The **labeler** simulates a long *and* a short at every bar close with
ATR-scaled triple barriers, and is deliberately pessimistic: when one candle
touches both barriers the **stop is assumed to have been hit first**. Path heat
(MAE as a fraction of the stop distance) plus the entry volatility percentile
produce the risk tier, yielding the five classes
`LONG_SUCCESS_{LOW,HIGH}_RISK`, `SHORT_SUCCESS_{LOW,HIGH}_RISK`,
`NO_TRADE_OR_FAIL`.

### Modules C & D — models and the arbiter

Four heads, each answering exactly one question and knowing nothing about the
others: **direction**, **entry timing**, **exit geometry**, **sizing**.
Inference is stateless and runs off the event loop via `asyncio.to_thread`.

When an artifact is missing a head falls back to a documented heuristic and
stamps `source=HEURISTIC` — and rule **R9** refuses to let a heuristic place a
live order.

The **Decision Engine** contains no ML. It is a fixed 10-rule cascade
(system gates → direction confidence → margin → NO_TRADE mass → entry timing →
reward/risk → sizing → risk tier → regime → provenance) where the first failing
rule short-circuits, so the audit log always names one precise cause.

### Module E — execution & risk

The **Risk Guard** is a `GREEN → YELLOW → RED` state machine. RED cancels every
order, flattens every position, disables execution and **latches** — it persists
to SQLite so a crash-loop cannot silently re-enable trading, and requires a
manual reset. Triggers: daily drawdown, peak-to-trough drawdown, consecutive
losses, or a burst of exchange errors inside a sliding window.

The **live executor** forces ISOLATED margin and the signal's leverage before
every order, places reduce-only TP/SL immediately after the fill, and **flattens
the position if the protective leg cannot be placed** — a naked leveraged
position is the worst state to be in.

The **paper trader** mirrors that interface exactly, so the trading loop contains
no `if paper:` branches, and charges real fees, slippage, funding and
liquidation.

### Module F — audit & panel

Every decision cycle is logged, **including every `NO_TRADE`**, with the feature
snapshot, all four model outputs and the exact rule that fired. Writes are
batched through an `asyncio.Queue`, so the trading loop never waits on SQLite.

The panel serves `/` (dashboard), `/audit`, `/trades` plus a JSON API and the
control endpoints `POST /api/toggle_trading`, `POST /api/kill_switch`,
`POST /api/reset_risk_guard`.

---

## Honest expectations

**The backtester is built to disappoint you, on purpose.** Signals fill on the
*next* bar's open (never the close that produced them), a bar that could resolve
several barriers always books the adverse one, and fees, slippage, funding and
liquidation are all charged.

That honesty is measurable. Training the models on one synthetic series and
replaying them on an independent one gives:

| | Trades | Win rate | Profit factor | Return |
|---|---|---|---|---|
| In-sample (trained on this data) | 27 | 92.6 % | 31.6 | +0.57 % |
| **Out-of-sample (unseen data)** | **5** | **0.0 %** | **0.0** | **−0.06 %** |

That gap is the point. In-sample numbers from `python main.py backtest` right
after `python main.py train` are **meaningless** — the models saw that data.
Hold out a period the models never touched before you believe anything.

---

## Operational notes

- **Security.** The panel is not hardened for the public internet. Set
  `WEB__API_TOKEN` to protect the control endpoints, and put the port behind a
  firewall allow-list or an SSH tunnel.
- **Going live.** Requires `EXCHANGE__TESTNET=false`, real API keys with futures
  permission, and trained artifacts for all four heads. Start on the futures
  testnet, then paper, then live with the smallest size that clears the
  exchange minimums.
- **Leverage** is capped at 10x by schema validation, not just by configuration.
- **Time** is UTC end to end; the scheduler fires a few seconds past each 5m
  boundary (`DATA__CYCLE_SECOND_OFFSET`) because firing at exactly `:00` races
  the exchange's own candle close.

## Requirements

Python 3.11+. Dependencies in `requirements.txt`. `ta-lib` is deliberately *not*
used: every indicator is implemented from first principles in
`module_b_features/indicators.py`, so there is no compiled C extension to build
on a small VPS.

## Disclaimer

Trading leveraged perpetual futures carries a substantial risk of total loss.
This software is provided for research and educational purposes, with no
warranty and no representation that it is profitable. You are responsible for
any capital you put at risk with it.
