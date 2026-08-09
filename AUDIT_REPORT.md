# ML Pipeline Audit & Diagnostic Reporting — Final Report

This report documents a full audit of the existing ML trading pipeline and the
changes made as a result. It follows the structure requested for this task:
what was audited, what was found, what was changed and why, before/after
comparison, and what remains open.

**Scope note, stated up front and honestly:** this audit and implementation
were done in a sandboxed environment with no access to live Binance market
data and no multi-hour training budget. That means Parts of the original
request that require running real training/backtests against real historical
data — a baseline report from real candles, hyperparameter search under
temporal validation, walk-forward evaluation, a real backtest comparison —
could not be executed here and are marked `NOT_AVAILABLE` throughout, exactly
as the project's own reporting rules require ("do not invent values"). What
*was* done: a full read-through audit of every pipeline stage, concrete fixes
for the two real bugs found, and a genuine, wired-in diagnostic reporting
system that will produce real numbers the next time someone runs
`python main.py train` against real data.

---

## A. What was audited

Every stage of the pipeline was read in full:

- `module_a_data/qc_validator.py`, `pipeline.py`, `fetcher.py`, `db_handler.py`, `models.py`
- `module_b_features/features.py`, `indicators.py`, `labeler.py`, `processor.py`
- `module_c_ml/ml_models.py`, `decision_engine.py`, `schemas.py`
- `module_e_execution/backtester.py`, `paper_trader.py`, `risk_guard.py`
- `module_f_panel/web_app.py`, `templates.py`, `audit_engine.py`
- `main.py` (scheduler, cycle orchestration, CLI commands)

## B. What was found — and what was already solid

The audit's most important finding is that **most of the system is already
built carefully and does not need "fixing"**:

- **No feature leakage found.** All 53 features in `FEATURE_COLUMNS` were
  individually traced. Every rolling window is causal (`min_periods` set,
  never `center=True`), GARCH and HMM refit only on trailing windows and use
  a forward-only filter (never smoothing), and all rank/percentile transforms
  are per-row trailing computations — never fit on the full dataset or across
  symbols. No scaler or encoder exists anywhere in the repo that could leak
  train/validation boundary information (the models are tree-based and need
  none).
- **No label leakage found.** The triple-barrier labeler
  (`module_b_features/labeler.py`) is documented and implemented consistently:
  when both TP and SL could resolve in the same bar, the stop is assumed to
  hit first (the conservative choice); the horizon and ATR-multiple logic
  match the docstrings exactly.
- **Train/validation split is genuinely temporal**, with a purge gap
  (`purge_bars`) sized to cover the label horizon, applied identically by all
  four heads (`ProcessedDataset.train_validation_split`).
- **Scheduler overlap is already handled correctly.** Both APScheduler
  (`max_instances=1`) and an application-level `asyncio.Lock` prevent
  concurrent cycles; an overrunning cycle is *skipped*, never queued or run
  concurrently. This directly answers Part 6 of the original request: no fix
  was needed here, only the instrumentation to prove it (see below).

Concrete, real gaps found and fixed:

1. **Unbounded heal fallback window** (`module_a_data/qc_validator.py`,
   `_heal_windows`). When QC damage was scattered into more windows than
   `max_heal_window_groups`, the healer collapsed everything into **one**
   re-fetch window spanning the entire damaged range — for a long,
   discontinuous corruption this is exactly the "tens of thousands of
   candles in one request" failure mode named in the task description.
2. **No persisted QC/healing telemetry.** `validate_and_heal` returned only a
   pass/fail verdict; per-attempt detail (windows tried, bars
   requested/received, duration, result) existed only as a single log line
   and was never retained anywhere queryable.
3. **No per-symbol exclusion tracking.** A symbol dropped from a cycle by QC
   was only visible as an `ERROR`-level log line — no structured record of
   *which* symbol, *why*, or *when*.
4. **No per-stage cycle timing.** Only one coarse overall cycle duration was
   measured; there was no way to see which stage (ingestion, features,
   prediction, decision, execution, database) was actually slow.
5. **ML metrics were narrow and never persisted outside the joblib blob.**
   Direction had only accuracy/balanced-accuracy/log-loss (no confusion
   matrix, no per-class P/R/F1, no confidence analysis); Entry had no F1, no
   confusion matrix, no PR-AUC, no threshold sweep; Exit had only MAE (no
   RMSE/R², no baseline comparison); nothing was written anywhere a report
   generator could read without unpickling an estimator.
6. **No calibration, no SHAP/feature importance, no diagnostic report, no
   `/ml-report` dashboard page** — all genuinely absent, confirmed by
   grepping the whole repo, not assumed.

## C. What was changed, and why

### Data layer (`module_a_data/`)

- **`qc_validator.py`** — `_heal_windows` now batches the fallback case into
  multiple windows capped at a new `max_heal_window_bars` setting (default
  2,000 bars ≈ 7 days) instead of one unbounded window. `validate_and_heal`
  now returns a third value, `heal_attempts: list[HealAttempt]` — real,
  measured per-round telemetry (symbol, reason, window span, bars
  requested/received/written, remaining damage, duration, result) — attached
  to the raised `DataIntegrityError.context` even on failure, so nothing is
  lost when a heal ultimately fails.
- **`models.py`** — new `HealAttempt` Pydantic model backing the above.
- **`pipeline.py`** — `DataPipeline` now accumulates
  `last_cycle_heal_attempts` and `last_cycle_exclusions` (symbol + reason)
  per cycle and persists both as a bounded rolling history (500 entries) to
  durable state (`qc_heal_telemetry`, `qc_symbol_exclusions` keys), so the
  diagnostic report can show real healing/exclusion history across many
  cycles, not just the last one.
- **`config/settings.py`** — added `QCSettings.max_heal_window_bars`.

### Orchestration (`main.py`)

- `_run_cycle` now records per-stage timings (`ingestion_and_qc`,
  `risk_guard`, `feature_generation`, `prediction`, `decision`,
  `execution_and_audit`, `database`, `total`) and persists a bounded rolling
  history (`cycle_timings_history` state key).
- A `cycles_skipped_overlap` counter now directly answers "is the 5-minute
  cycle silently skipping slots because the previous cycle overran" —
  incremented every time the overlap lock rejects a new cycle.
- Both training entry points (`command_train` CLI and `_setup_train` panel
  flow) now call `module_f_panel.diagnostics.build_report(...)` after
  `MLSubsystem.train_all` and store the run ID. Report generation failures
  are caught and logged — they must never abort a successful training run.
- `training_summary` pushed through `/api/setup/status` was changed from the
  full (now much larger) per-head metrics dict to a small fixed-size
  headline summary (`_headline_metrics`), so the periodic polling endpoint
  doesn't balloon; the full report is always available via
  `/api/ml/diagnostics`.

### ML layer (`module_c_ml/`)

- **New `metrics.py`** — pure, independently-tested functions:
  `direction_metrics` (accuracy family, per-class P/R/F1/support, confusion
  matrix raw+normalized, class/predicted-class distribution, probability
  stats, confidence-threshold analysis at 8 thresholds), `entry_metrics` +
  `entry_threshold_sweep` (9 thresholds; trading-level fields — average R,
  win rate, profit factor, expectancy, net PnL, max drawdown — are reported
  as `NOT_AVAILABLE` rather than approximated, since they require a real
  backtest simulation this sweep does not run), `regression_metrics`
  (MAE/RMSE/R²/median-AE + target/prediction distributions, shared by Exit
  and Risk), `calibrate_classifier` (isotonic calibration fit on a
  temporally-earlier half of the validation slice, scored on the later half
  — Brier score and log loss before/after), `feature_importance` (native
  gain/split importance; SHAP explicitly reported `NOT_AVAILABLE` since
  `shap` is not a project dependency — not silently skipped), and
  `per_symbol_direction_accuracy`.
- **`ml_models.py`** — all four heads' `train()` methods now call into
  `metrics.py` for the full metric set (previously: hand-rolled 2–3-metric
  dicts). `BaseModelHead.save()` now also writes a `{name}.metrics.json`
  sidecar next to the joblib artifact containing the full metadata:
  `trained_at`, `git_commit`, hyperparameter snapshot, feature list, dataset
  size, class distribution, and every computed metric — the joblib blob
  alone was previously the only record of a training run and is opaque
  without unpickling an estimator (satisfies Part 25, model artifact
  versioning). `ExitModel` now also compares every regressor against a
  vectorised twin of its own rule-based ATR heuristic
  (`_baseline_predictions`) and reports `beats_rule_based_baseline` per
  target (Part 15: prove the ML model adds value over the baseline it would
  replace).
- Calibration is **measured and reported, not wired into live inference** —
  swapping the decision engine onto calibrated probabilities is a
  live-trading behaviour change that deserves its own explicit review, not a
  side effect of an audit. This is stated in the report's `calibration`
  section (`note` field) and listed as a follow-up below.

### Dataset health (`module_b_features/processor.py`)

- `ProcessedDataset` gained four real, measured counters —
  `total_candidate_rows`, `rejected_invalid_label_rows`,
  `dropped_missing_or_inf_rows`, `duplicate_feature_rows` — set in
  `_to_dataset` from the actual cleaning step, instead of only being logged
  at `INFO` level and discarded.

### Reporting (`module_f_panel/diagnostics.py`, new)

- `build_report(...)` assembles one JSON-serialisable dict per training run
  from the sources above: run metadata (run ID, git commit, model versions,
  training/validation period), dataset health, QC/healing telemetry, feature
  statistics/correlation/drift (train-vs-validation, in train-std units),
  labels (configuration + class distribution), the four heads' full metrics,
  calibration, walk-forward (`NOT_AVAILABLE`, honestly, with the reason —
  only a single split is currently performed), backtest (`NOT_AVAILABLE`
  unless a `BacktestReport` is passed in), per-symbol breakdown, pipeline
  timing, an AI-ready summary, a before/after comparison against the
  previous stored report, and rule-based recommendations by priority
  (CRITICAL/HIGH/MEDIUM/LOW) — every recommendation is derived from a
  measured field in the same report, never invented.
- Every float is sanitised before being persisted or served: `NaN`/`Infinity`
  (which can legitimately occur, e.g. log loss on a degenerate single-class
  validation slice) are converted to JSON `null` rather than left as the
  non-standard `NaN`/`Infinity` tokens `json.dumps` would otherwise emit —
  which a browser's native `JSON.parse` cannot read. This was caught and
  fixed during testing, not assumed away.
- Reports are written to `reports/ml_diagnostic_<run_id>.{json,md}` next to
  the model directory, with a durable-state pointer to the latest one for
  before/after comparison and for the panel to read back after a restart.

### Dashboard (`module_f_panel/web_app.py`, `templates.py`)

- New page `GET /ml-report` and API routes `GET /api/ml/diagnostics`,
  `GET /api/ml/diagnostics/export.json` (download), `GET
  /api/ml/diagnostics/export.md` (download) — added to the `SystemController`
  Protocol and implemented on `TradingSystem`.
- The page shows the AI summary, dataset health, data quality/healing,
  per-head metrics (with confusion matrices, threshold sweeps, calibration,
  top features rendered as tables), feature drift, labels, backtest, per-symbol
  table, pipeline timing, the before/after comparison table, prioritised
  recommendations, and a raw-JSON viewer at the bottom — so nothing important
  exists only inside a chart, per the task's explicit requirement.

## D. Data quality — before / after

| | Before | After |
|---|---|---|
| Unbounded heal fallback | Possible (single window spanning full damaged range) | Bounded batches, capped at `max_heal_window_bars` (2,000 bars default) |
| Heal attempt detail | One log line per round | Structured `HealAttempt` record: symbol, reason, window span, bars requested/received/written, remaining damage, duration, result |
| Symbol exclusion visibility | `ERROR` log line only | Structured record with reason, persisted with bounded rolling history |
| Per-stage cycle timing | Overall duration only | Per-stage breakdown (ingestion, features, prediction, decision, execution, database), persisted history |
| Scheduler overlap evidence | Implicit (a log line per skip) | `cycles_skipped_overlap` counter, directly answers "is the interval being overrun" |

## E–L. Direction / Entry / Exit / Risk / Features / Labels / Walk-forward / Backtest

**NOT AVAILABLE for this run.** No real training data was fetched from
Binance in this sandboxed environment (no network access to the exchange),
so no real baseline metrics, real before/after comparison, real feature
importance/SHAP ranking, real walk-forward folds, or real backtest exist to
report here. Reporting fabricated numbers for these sections would violate
the task's own core rule ("do not invent values", "do not fabricate
metrics"), so they are left honestly blank.

What *is* verified: the full pipeline that produces these sections was
exercised end-to-end against synthetic data (72 automated tests, see below,
including two full training + report-generation runs), confirming the
mechanism is correct and will produce real numbers the next time
`python main.py train` runs against real historical data. The very first
real run after this change becomes the new baseline; every run after that
gets an automatic before/after comparison against it.

## M. Symbol-level results

Not available for the reason above. The per-symbol breakdown mechanism
(`per_symbol_direction_accuracy`) is implemented and tested
(`tests/test_ml_metrics.py::test_per_symbol_direction_accuracy_groups_correctly`)
and will populate on the next real training run.

## N. Regime-level results

**NOT AVAILABLE — not implemented.** The `hmm_regime` feature exists and is
audited as leakage-safe, but a regime-conditioned performance breakdown
(Part 22) was not built in this pass. Listed as a follow-up below.

## O. Warnings / errors encountered

- `sklearn.calibration.CalibratedClassifierCV(cv="prefit")` was removed in
  scikit-learn ≥ 1.6 in favour of `sklearn.frozen.FrozenEstimator`. Fixed
  with a version-tolerant code path (`metrics.calibrate_classifier`) that
  tries `FrozenEstimator` first and falls back to `cv="prefit"` for the
  project's pinned floor (`scikit-learn>=1.4.0`).
- `json.dumps`/`JSONResponse` emit non-standard `NaN`/`Infinity` tokens for
  degenerate metrics (e.g. R² on a near-constant target), which a browser's
  `JSON.parse` rejects. Fixed with `_sanitize_non_finite` before any report
  is persisted or served.
- No HMM/GARCH convergence warnings were observed or investigated in this
  pass, since no real training run against real market data was performed.
  This audit item (Part 26) is open.

## P. Remaining problems — stated honestly

- **No walk-forward evaluation.** Only a single temporal train/validation
  split is performed. The report says so explicitly
  (`walk_forward.status == "NOT_AVAILABLE"`) rather than hiding it.
- **No real baseline/after metrics exist yet** — see D–L above. This is the
  single biggest gap: everything needed to generate one is now in place, but
  generating it requires a real training run this sandboxed session could
  not perform.
- **Calibration is measured but not wired into live inference.** A
  deliberate scope boundary, not an oversight — see section C.
- **No regime-conditioned performance breakdown** (Part 22).
- **No hyperparameter search** (Part 11) — was in scope but requires many
  real training runs against real data to do honestly; not attempted rather
  than faked.
- **No SHAP values** — `shap` is not an installed dependency; native
  gain-based feature importance is available instead, reported as such.
- **Backtest is not automatically bundled into the training report.**
  `python main.py backtest` produces a `BacktestReport` independently;
  wiring it into the same diagnostic report as `train` is a small, safe
  follow-up (the `build_report(backtest=...)` parameter already exists for
  it).

## Q. Recommended next actions

**CRITICAL**
- None. No head is untrained, no data-integrity safeguard was weakened.

**HIGH**
- Run `python main.py bootstrap && python main.py train` against real market
  data to produce the first real ML diagnostic report — the actual baseline
  everything else in this task was meant to be measured against.
- Add walk-forward evaluation (multiple rolling train→validate folds) before
  trusting any single split's metrics.
- Run `python main.py backtest` and pass its `BacktestReport` into
  `diagnostics.build_report(backtest=...)` so trading-level metrics
  (win rate, profit factor, Sharpe, drawdown) appear in the same report as
  the predictive metrics.

**MEDIUM**
- Add regime-conditioned performance breakdown (bull/bear/sideways/high-vol)
  using the existing, leakage-safe `hmm_regime` feature.
- Run the label-configuration experiment matrix (Part 10 — TP/SL/horizon
  variants) using the offline framework this session did not have the data
  to exercise.
- If calibration continues to show a measured log-loss improvement on real
  data, decide deliberately (with a human review, not automatically) whether
  to wire it into the decision engine.

**LOW**
- Consider adding `shap` as an optional dependency for per-prediction
  explanation, now that the report has a defined slot for it.

---

## Modified / added files

**Data layer:** `config/settings.py`, `module_a_data/models.py`,
`module_a_data/qc_validator.py`, `module_a_data/pipeline.py`

**Orchestration:** `main.py`, `core/utils.py`

**ML layer:** `module_c_ml/metrics.py` (new), `module_c_ml/ml_models.py`

**Dataset:** `module_b_features/processor.py`

**Reporting/dashboard:** `module_f_panel/diagnostics.py` (new),
`module_f_panel/web_app.py`, `module_f_panel/templates.py`

**Tests (new):** `tests/test_heal_window_batching.py`,
`tests/test_pipeline_qc_telemetry.py`, `tests/test_ml_metrics.py`,
`tests/test_ml_diagnostics_report.py`, `tests/test_ml_models_metrics_sidecar.py`

**Docs:** `README.md`, `AUDIT_REPORT.md` (this file)

## Tests executed

`.venv/bin/python -m pytest tests/ -q` — **72 passed** (45 pre-existing + 27
new), including two full train→report-generation runs against synthetic data,
JSON/Markdown export validity checks, NaN/Infinity sanitisation, heal-window
batching bounds, and QC/heal telemetry persistence. `python -m compileall`
over every module and `python main.py --help` were also verified.

## Where to find the report in the dashboard

`/ml-report` (linked from the panel's top nav as "ML Report"), backed by
`GET /api/ml/diagnostics` (JSON), `GET /api/ml/diagnostics/export.json`
(download) and `GET /api/ml/diagnostics/export.md` (download). Report
artifacts are written to `<model_dir>/../reports/ml_diagnostic_<run_id>.json`
and `.md` on every completed training run.
