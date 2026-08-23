# Fixes from the first full live run

The audit (`BRANCH_AUDIT_archive-features-and-data-recovery.md`) was written
against the code. This document is written against a *run*: the log of the first
complete bootstrap on real infrastructure, before training. Four problems showed
up there that no amount of reading the source would have found, because each one
is a case of the code working exactly as written and the written behaviour being
wrong.

They share a shape worth naming, because it is the thing to look for next time:

> **A transient condition was recorded as a permanent fact, and nothing
> downstream could tell the difference afterwards.**

A timeout became "this file does not exist". A budget expiring became "this
symbol's data is corrupt". A long backfill became "candles are being missed". In
each case the *recording* is what did the damage - the original condition was
recoverable, and the record of it was not.

---

## L1 - The system was reading the wrong market

**Symptom.** 162 requests rejected over the run, all of them `fapiData`:
`openInterestHist`, `globalLongShortAccountRatio`, `takerlongshortRatio`.

**Mechanism.** `EXCHANGE__TESTNET=true` (the shipped default in `.env.example`)
made `BinanceDataFetcher._build_client` call `exchange.set_sandbox_mode(True)`.
Those three endpoints do not exist on Binance's futures testnet at all.

**Why it mattered more than 162 errors.** This system has *one* exchange client.
It serves the candles training learns from, the funding and open-interest
history behind the derivatives features, and the prices the backtest and the
paper trader replay. Pointing it at testnet does not produce a test run - it
produces a model fitted on a different market, whose order book is largely
synthetic. And the failure was silent where it counted: `open_interest_change`,
`long_short_ratio` and `taker_buy_sell_ratio` were simply constant for the
entire history, and nothing in the pipeline treats a constant feature as a
failure. The model would have trained on three dead columns and reported
perfect health.

**Fix.** `set_sandbox_mode` is never called, by any module - asserted against the
source tree in `tests/test_mainnet_only_and_recovery.py`, because the regression
to guard against is a *new* call site somewhere else. `exchange.testnet` remains
as a deprecated field defaulting to `False`, solely so a stale flag in an
existing `.env` is reported as an error at startup instead of being dropped in
silence. `.env.example` and the README no longer suggest testnet at all.

**What replaces testnet.** Paper trading, on the same mainnet data the live
system reads. Reading mainnet market data requires no API keys; whether real
orders are sent is decided by arming trading from the panel, which is where that
decision belongs.

---

## L2 - The archive cache poisoned itself, permanently

**Symptom.** `bookTicker` coverage: **0.0%** on every symbol, for every day,
across 25 months. Zero failures reported. An earlier run had reached 53%.

**Mechanism.** `_download` returned `None` for a 404 *and* for a timeout.
`_one_day` wrote the same permanent `.absent` marker for either. So:

1. one run behind a slow link times out on a large `bookTicker` day;
2. that day is marked as never-published;
3. every later run reads the marker and skips the day without requesting it;
4. coverage is 0% forever, and `days_failed` is 0, so nothing looks wrong.

The zero *failure* rate is what made this invisible for an entire release. The
existing escalation checked failures, and there were none - because the days
were never requested again.

**Fix, in three parts.**

* `_download` returns a `DownloadOutcome` (`OK` / `ABSENT` / `FAILED`).
  Only `ABSENT` - an actual 404 - justifies a marker. `FAILED` counts toward
  `days_failed` and is retried on the next run.
* Markers written before the distinction existed cannot be told apart from
  poisoned ones, so they are purged at the start of each backfill
  (`purge_legacy_absent_markers`) and retried once, which rewrites them
  correctly. `_one_day` checks for them too, so a loader used outside the
  pipeline recovers as well.
* Zero coverage with zero failures now escalates on its own. Binance publishes
  these files for the life of each contract, so an entire universe reporting
  nothing over years is an unreachable CDN or a stale cache - not an empty
  archive. The error carries the `curl` command to confirm which.

**Recovering an already-poisoned cache.** Nothing to do: the next backfill
purges the old markers itself. (`rm -rf` on the archive cache directory also
works, and is what an operator would have had to know to do before.)

---

## L3 - Three symbols discarded over one bad bar each

**Symptom.** `ATOM`, `PIXEL` and `ALGO` failed bootstrap with
`heal loop exceeded its wall-clock budget` after 130-170s, each with
`bars_invalid_after_heal=1`, having already re-written roughly 120,000 bars.

**Mechanism.** Two independent errors compounding.

*The budget was sized for the wrong job.* `max_heal_duration_seconds=90` exists
to stop one stuck symbol from starving the shared request-rate budget in a
5-minute live cycle, where healing touches a handful of bars. Bootstrap healing
spans two years and legitimately takes minutes. The same constant governed both.

*The outcome was all-or-nothing.* When the budget expired with anything still
flagged, the whole symbol was discarded. That is right for a feed serving
corrupt data and wrong for a residue of one bar - which is what a two-year
backfill routinely leaves behind (an exchange-side halt, a single stale print).

**Fix.** Backfills pass `max_bootstrap_heal_duration_seconds` (900s), leaving the
tight per-cycle ceiling to do its actual job. And when the budget does expire,
the validator asks whether what remains is damage before discarding anything: a
residue within **both** a relative tolerance (0.05% of the block) and an absolute
cap (50 bars) is accepted. Both bounds are needed - the ratio alone would let a
long history absorb hundreds of bad bars, and the cap alone would not scale.

**Accepting is not forgetting.** The surviving `CRITICAL` issues are re-emitted
as `WARNING`s carrying the same code, message and timestamps, so those bars stay
named in the QC report and travel downstream with their verdict intact. They
simply no longer condemn the block. The boosters route them down their
missing-value branch either way.

> Implementation note: the first version of this fix wrote
> `report.model_copy(update={"passed": True})`, which does nothing - `passed` is
> a computed property derived from the issue severities, so the update is
> silently dropped and the report still read as failed. The fix had to be
> expressed where the verdict actually lives. A test now asserts
> `accepted_report.passed`.

---

## L4 - 16 warnings for something entirely expected

**Symptom.** Sixteen `Previous cycle is still running` warnings, plus
APScheduler's own context-free `max_instances` warnings, during a bootstrap
where single cycles ran 2536s and 2465s against a 5-minute schedule.

**Mechanism.** `max_instances=1` is correct - two ingestion cycles at once would
race each other's writes, so an overrunning cycle skips the next firing rather
than stacking. The skip was the right behaviour being reported as a problem.

**Why it is worth fixing.** The identical warning is what *real* data loss looks
like. Sixteen false ones during setup teach an operator to scroll past the one
that means live candles are being missed.

**Fix.** Both paths - the in-cycle lock, and APScheduler's `EVENT_JOB_MISSED` /
`EVENT_JOB_MAX_INSTANCES` listener, which previously logged with no context at
all - now name the phase and pick their level from it: `INFO` while a backfill,
heal or training run legitimately owns the cycle, `WARNING` once the system is
past setup and a skip means data loss. A test asserts the two paths agree.

---

## Observed but deliberately not "fixed"

**Long-biased label distribution.** The barrier labeller produced materially
more long outcomes than short over the bootstrap window. This is a property of
the window - a broadly rising market - not a defect in the labeller, and
rebalancing it would mean training on a market that did not happen. It is
handled where it should be: the split-aware class distribution
(`class_distribution_by_split`) and the label-shift check added in the audit
work now report it per split, so a train/test distribution *shift* is caught
even though the overall skew is left alone. Watch the diagnostic's label-shift
line rather than the headline balance.

---

## Verification

| | |
|---|---|
| Test suite | 290 passed |
| Audit re-verification | `scripts/reverify_audit_findings.py`: 0 reproduce, 30 fixed, 0 unevaluable |
| New tests | `tests/test_mainnet_only_and_recovery.py`, `tests/test_scheduler_skip_reporting.py`, residual-acceptance cases in `tests/test_qc_healing.py` |

**Not verified here, and it matters.** This environment has no database, no
candles, no trained models and no route to Binance, so none of the four fixes
has been exercised against the real thing. The next run is the real test, and
these are the lines to check:

1. `Archive dataset bookTicker returned data for ...%` - should be well above
   0%, and no `near-zero coverage` error;
2. no `fapiData` rejections at all, and `open_interest_change`,
   `long_short_ratio`, `taker_buy_sell_ratio` non-constant in the diagnostic;
3. `ATOM`, `PIXEL`, `ALGO` present in the universe, possibly with an
   `accepting N bar(s) still flagged` warning rather than an error;
4. cycle-skip lines at `INFO` during bootstrap, and none at all afterwards.
