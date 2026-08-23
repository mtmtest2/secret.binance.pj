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

## Quick start — one command

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # optional: thresholds, and API keys for live

python main.py                # that's it
```

Open `http://<vps-ip>:8000/` and follow the panel:

1. **Pick pairs.** The system parks in `AWAITING_UNIVERSE` and the *Pairs* page
   lists every USDT-M perpetual **fetched live from the Binance API**, annotated
   with 24 h volume, spread, exchange minimum, lot-step cost, listing age and a
   screening verdict. Tick what you want (or press *Suggest top 30*) and save.
2. **Watch setup.** Saving immediately starts data collection for exactly those
   pairs, then trains the four models on them. The dashboard shows a live
   progress bar through `COLLECTING_DATA → TRAINING → READY`.
3. **Arm paper trading.** Press *START PAPER*. Nothing trades before you do.
4. **Go live when ready.** Press *GO LIVE* — open paper positions are flattened
   first, and live is refused unless all four models are genuinely trained.
   *STOP TRADING* disarms at any time.

Everything is re-runnable: restarting only backfills the missing tail and skips
training when the models already match the saved universe.

### Lifecycle

```
STARTING ──▶ AWAITING_UNIVERSE ──(you tick pairs)──▶ COLLECTING_DATA ──▶ TRAINING
                                                                            │
                     PAPER_TRADING ⇄ LIVE_TRADING ◀──(you arm it)──── READY ◀┘
```

### Other commands

| Command | What it does |
|---|---|
| *(none)* / `run` | Panel + automatic setup + 5m scheduler |
| `universe` | Print the screened pair list in the terminal |
| `bootstrap` | Backfill historical candles only |
| `train` | Build the training dataset and fit the four heads |
| `backtest [--candles N] [--equity X]` | Event-driven replay with full metrics |
| `cycle` | Run exactly one trading cycle, then exit |
| `--mode paper\|live` | Override the execution mode |

The one-shot commands use the universe saved from the panel; if none exists they
auto-select the top screened pairs so the CLI is usable standalone.

---

## How it works

### Universe selection

The tradeable universe is discovered from Binance at runtime, not hard-coded.
Four independent screens run over every active USDT-M perpetual, and the panel
shows exactly which one a pair failed:

| Screen | Default | Why |
|---|---|---|
| 24 h quote volume | ≥ 50 M USDT | Thin books turn a 10x position into its own adverse price move |
| Bid/ask spread | ≤ 6 bps | Paid on every 5m round trip |
| Listing age | ≥ 90 days | Below this there is not enough 5m history to train on |
| **Small-account fit** | calibrated to `reference_equity` | See below |

The last one is the screen most universes get wrong, and it is the reason this
system does **not** simply pick the biggest coins. A pair is unusable on a small
account when either the exchange's minimum notional exceeds the smallest
position the risk model can open, or one lot-size step costs so much that sizing
becomes hopelessly coarse. On a 1 000 USDT account whose smallest position is
10 USDT, a 0.001 BTC step is ~95 USDT of notional — BTC simply cannot be sized
correctly no matter how liquid it is, so it is flagged ineligible with that exact
reason. You can still tick it deliberately; the system honours the choice and
logs a warning.

Changing the universe invalidates the models (they were fitted on a different
cross-section), so saving a different selection forces a retrain.

### Module A — ingestion & quality control

Fetches 5m OHLCV, L2 order-book snapshots, funding rate, open interest,
long/short ratios and (where exposed) liquidation flow for the selected pairs,
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
produce a risk tier, but the tier is kept out of the Direction model's own
target: fusing "which way" with "how clean was the path" into one label
diluted the discrete outcome the Direction model can actually learn and
swamped the majority NO_TRADE class in extra classes worth of noise. Direction
predicts three classes — `LONG_SUCCESS`, `SHORT_SUCCESS`, `NO_TRADE_OR_FAIL` —
and the risk tier instead trains the Risk model's continuous opportunity
score, which is turned back into a discrete tier (for the R7 gate and
leverage sizing) by inverting that same score formula at inference time.

### Modules C & D — models and the arbiter

Four heads, each answering exactly one question and knowing nothing about the
others: **direction**, **entry timing**, **exit geometry**, **sizing**.
Inference is stateless and runs off the event loop via `asyncio.to_thread`.

The **Direction** head is itself a two-stage cascade rather than one 3-way
softmax: a binary gate decides trade-vs-NO_TRADE, and — only on rows the gate
calls a trade — a second binary model decides long-vs-short. The two
questions lean on different signal (whether-to-trade skews toward
volatility/regime features, long-vs-short toward directional/momentum ones),
so splitting them sharpens each decision boundary instead of forcing one
model to serve both. The **Entry** head auto-selects its decision threshold
from its own validation sweep (F-beta=0.5, precision-weighted — a
false-positive entry costs real capital, a missed true positive only costs a
smaller position count) rather than trusting one fixed config constant. The
**Risk** head fits the same L1 (robust) objective as Exit, since
`target_risk_score` is right-skewed the same way exit-geometry percentages
are, and is fed two additional causal features (`wick_ratio`,
`whipsaw_rate`) aimed specifically at path-heat/whipsaw rather than raw
move magnitude.

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

The panel serves `/` (dashboard), `/universe` (pair picker), `/audit`,
`/trades` and `/ml-report` (ML diagnostic report), plus a JSON API:

| Endpoint | Purpose |
|---|---|
| `GET /api/universe/available` | Screened Binance perpetuals |
| `GET /api/universe/suggest` | Top-scoring eligible pairs |
| `POST /api/universe/select` | Save the selection and start setup |
| `GET /api/setup/status` | Live collection/training progress |
| `POST /api/setup/start` | Re-run collection, `{"force_retrain": true}` to refit |
| `POST /api/trading/start` | Arm `{"mode": "paper"｜"live"}` |
| `POST /api/trading/stop` | Disarm (`{"flatten": true}` to close positions too) |
| `POST /api/kill_switch` | Trip RED, flatten everything |
| `POST /api/reset_risk_guard` | Clear a RED latch |
| `GET /api/ml/diagnostics` | Full ML diagnostic report for the last training run (JSON) |
| `GET /api/ml/diagnostics/export.json` | Same report, as a download |
| `GET /api/ml/diagnostics/export.md` | Human-readable Markdown rendering, as a download |

### ML diagnostic report

Every completed training run (`python main.py train`, or the panel's
COLLECTING_DATA → TRAINING step) builds one structured report covering
dataset health, QC/healing telemetry, per-head metrics (Direction, Entry,
Exit, Risk) with confusion matrices, threshold sweeps, calibration and
feature importance, label distribution, per-symbol breakdown, expanding-window
walk-forward evaluation of the Direction model across multiple rolling folds,
microstructure/derivatives data-coverage (is the order-book/funding/OI feed
actually populated, or silently defaulting to neutral values), pipeline
timing, and a before/after comparison against the previous run. Training also
automatically replays the out-of-sample validation window through the full
decision pipeline (`Backtester`), so win rate, profit factor, expectancy, max
drawdown and Sharpe are real measured numbers rather than placeholders — see
`TradingSystem._run_validation_backtest` in `main.py` for the exact window it
covers and the one documented caveat (production calibrators are fit on that
same validation block, so it is the best available out-of-sample
approximation rather than a third, fully untouched split). An AI-ready
summary sits at the top so the report can be handed to another AI to
diagnose what changed. Every field with no real source data is the literal
string `NOT_AVAILABLE` rather than a guess (e.g. walk-forward on a dataset
too small to carve out honest folds). Reports are written to
`<model_dir>/../reports/ml_diagnostic_<run_id>.{json,md}` and viewable at
`/ml-report`. See `AUDIT_REPORT.md` for the full pipeline audit this was
built from.

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
- **Mainnet only.** All market data - training, backtest and paper trading
  alike - comes from mainnet, and the exchange client never enters sandbox
  mode. Reading it needs no API keys. `EXCHANGE__TESTNET` is deprecated and
  ignored; setting it is reported as an error at startup.
- **Going live.** Requires real API keys with futures permission and trained
  artifacts for all four heads (heuristic fallbacks are refused). Go paper
  first, on the same mainnet data the live system reads, then live with the
  smallest size that clears the exchange minimums.
- **Nothing trades on its own.** Trading is armed only from the panel (or
  `POST /api/trading/start`). Set `AUTOSTART_PAPER_TRADING=true` if you
  genuinely want paper trading to begin the moment setup finishes.
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
