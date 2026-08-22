---
name: quant-audit-fix
description: >-
  Diagnose and fix the AI quant trading system in this repo (Binance USDT-M perpetual futures, 5m
  timeframe) using the committed audit and fix-options documents in docs/. Use this whenever the user
  asks about problems, bugs, failures, weaknesses or fixes in this trading system; whenever they mention
  the ML diagnostic report, an ml_diagnostic_*.json / .md file, a branch audit, or a finding by its P-number
  (P1..P30); whenever they ask why the models underperform, why a backtest result looks wrong or too good,
  why the direction/entry/exit/risk heads behave oddly, why features are empty or drifting, or whether a
  run is safe to deploy; and whenever they ask to retrain, re-run the diagnostic, or act on the
  recommendations. Also use it when they hand over a NEW diagnostic run for this system and want it
  analysed. Do not wait for the words "audit" or "P-number" — any request to investigate or repair this
  trading system belongs here.
---

# Quant audit & fix

This repo is a four-head ML trading system (Direction / Entry / Exit / Risk) over Binance USDT-M
perpetual futures on a 5-minute timeframe, with its own diagnostic report generator. A full audit has
already been done. Your job is to use it rather than redo it — and, critically, to **re-verify before
you act**, because the audit is evidence-derived and the evidence changes every time someone re-runs
the pipeline.

## The two documents

Both are committed in `docs/`:

| File | What it holds |
|---|---|
| `BRANCH_AUDIT_archive-features-and-data-recovery.md` | 30 findings (P1–P30). Each has the evidence, the mechanism in code, the consequence, and a proposed fix. Read a finding's section before touching the code it describes. |
| `BRANCH_FIXES_archive-features-and-data-recovery.md` | For each finding: 2–4 genuinely different **options** with cost / what-it-breaks / verification, a **Pick**, plus a cross-cutting section and a staged ordering with hard dependencies. |

They were written against commit `c2968d8` on branch `claude/archive-features-and-data-recovery`, using
diagnostic run `d0eb385d-a0ec-413b-8414-c90b64fdea8e`. Both facts matter — see below.

The fixes file's tables are the fastest way in: its **Contents** table maps every P-number to a one-line
Pick, and its **Ordering and dependencies** section gives the four stages and the hard dependencies
between them. Start there, then read the specific finding in both files before writing code.

## Orient before you do anything else

Three questions decide everything that follows. Answer them from the environment, not from memory:

1. **Which commit is the user actually working on?** The audit describes `c2968d8`. If HEAD is
   elsewhere, some findings may already be fixed and some line numbers will have moved. Check
   `git log --oneline -5` and `git branch --show-current`.
2. **Is there a newer diagnostic run?** Look for `ml_diagnostic_*.json` in the working tree, in
   `/root/.claude/uploads/`, or wherever the user points. The audit's numbers come from one specific
   run. A newer run supersedes them.
3. **What does the user want — analysis, or repair?** "Why is X broken" wants the finding explained
   with its evidence. "Fix X" wants an option chosen and code written, which pulls in the whole
   ordering discipline below.

## Re-verify before you fix

This is the part that separates useful work from confidently wrong work. The audit's claims are
statements about a specific commit and a specific diagnostic run. Both can be stale.

For a **code** claim, confirm against the tree you are actually editing:

```bash
git show <commit>:module_c_ml/ml_models.py | grep -n "full_metrics = ml_metrics.direction_metrics"
```

or, when a whole-tree view is easier, extract once and grep freely:

```bash
mkdir -p /tmp/verify && git archive <commit> | tar -x -C /tmp/verify
```

For a **number** claim, read it out of the diagnostic JSON rather than trusting the prose. The JSON is
large (~2 MB), so query it with a small Python snippet instead of loading it into context:

```bash
python3 -c "
import json; d=json.load(open('<path>/ml_diagnostic_<run>.json'))
print(d['dataset']['null_counts_by_feature']['ob_imbalance'], d['dataset']['valid_samples'])
"
```

The report's top-level keys are: `run`, `dataset`, `data_quality`, `features`, `labels`, `direction`,
`entry`, `exit`, `risk`, `walk_forward`, `backtest`, `backtest_reliability`,
`backtest_diagnostic_relaxed`, `regimes`, `timings`, `warnings`, `errors`, `calibration`, `symbols`,
`comparison_to_previous_baseline`, `ai_summary`, `recommendations`.

If a finding no longer reproduces, say so plainly and move on — do not apply a fix for a problem that
is gone. If a finding reproduces with *different numbers*, use the new numbers; the mechanism usually
survives even when the magnitude changes.

## Why the ordering is not optional

The fixes file defines four stages. The reason they cannot be reshuffled is specific, not procedural:

- **Stage 1 makes the report honest.** Until P1 lands, every model metric in the report describes an
  artifact that was discarded before saving — the measured Direction model never predicts LONG, while
  the shipped one goes long 79% of the time. Any model decision taken before that is taken on a
  measurement of the wrong object. After Stage 1, **re-run the diagnostic with nothing else changed**.
  That run is the real baseline; every later comparison is against it.
- **Stage 2 fixes the data**, and requires a retrain to evaluate.
- **Stage 3 fixes the backtest.** Expect return and Sharpe to fall. That is the intended direction.
- **Stage 4 fixes the models**, and P4 (does the direction model have any edge?) comes last on purpose
  — that question is unanswerable while the training set is 72% degenerate bars.

Several fixes make the reported results **worse**. When that happens, report it as success. The current
numbers are wrong in the optimistic direction, and a fix that lowers them is doing its job.

Carry this warning into any deployment conversation: **nothing from this branch should reach live or
paper trading before Stage 2 completes.**

## Choosing between options

Each finding offers real alternatives, and the Pick is a recommendation, not an instruction. Override
it when the user's constraints differ — but say what you are trading away.

The recurring shapes:

- **Local patch vs structural fix.** The patch fixes the instance; the structural change stops the
  class. P1's identity guard, P17's check registry and the parity tests in P22/P23 are the structural
  ones, and each closes several findings at once. If effort is limited, spend it there — the fixes
  file's *"Fixes that subsume several findings"* section lists all three.
- **Fix vs detector.** Many findings pair a repair with a report check that would have caught it. The
  detector is often the cheaper half and the more durable one. Prefer shipping both.
- **Mitigation vs root cause.** P2 is the clearest case: filtering degenerate rows is a guard, but if
  the ingestion path is fabricating candles, every future backfill re-poisons the dataset. Chase the
  root cause even when the mitigation is tempting.

When a Pick depends on a policy decision that is genuinely the user's — P21's config-authoritative vs
auto-tune-authoritative is the clearest — put the choice to them with the consequence of each, rather
than picking silently.

## Verification discipline

Every finding in the fixes file ends with a **Verify** step. Treat it as part of the fix: a change
without its check is not done, because most of these defects were invisible precisely because nothing
checked for them.

Where a check can be a test, write the test. The audit repeatedly found that a whole class of bug
survived because no test compared two paths that were supposed to agree — training metrics vs the
saved artifact, backtest geometry vs live geometry, training row survival vs inference row survival.
Those comparisons are cheap to assert and they pin the invariant permanently.

## Environment

The dependencies (`pandas`, `numpy`, `scikit-learn`, `lightgbm`) were **not installed** when the audit
was written, so the repo's 160+ tests were never run and the audit makes no claim about them. Before
applying fixes:

```bash
pip install -r requirements.txt -r requirements-dev.txt
python -m pytest tests/ -q
```

Establish a green baseline first. Several of these changes will legitimately break existing tests —
particularly `tests/test_nan_tolerant_dataset.py` and `tests/test_feature_removal_and_entry_heuristic.py`
(which encode `146b00c`'s NaN-tolerance behaviour) and `tests/test_direction_geometry_consistency.py`
(which encodes `c2968d8`'s stop geometry). You need to be able to tell a legitimate break from a
regression, and you can only do that against a known-green starting point.

Git work goes on the branch the user names; do not push to another branch without asking.

## When the user brings a new diagnostic run

This is the case the documents are most useful for and least specific about. Work it like this:

1. Load the new run's `ai_summary`, `recommendations`, `warnings` and `errors` first — they are small
   and they tell you what the generator thinks. Then distrust the verdict: `_ai_summary` reads only
   three scalars, so a `GOOD` status means almost nothing until P17 is fixed.
2. Walk the findings that have machine-checkable evidence and re-test each against the new JSON:
   feature null rates (P3), drift magnitudes and `realized_vol_12_is_zero` (P2), the direction sweep's
   `signals` column (P4), walk-forward fold monotonicity (P18), predicted class distribution (P1),
   Sharpe-vs-Sortino ordering (P13), risk prediction floor and bias (P8), the identical R² pair (P15),
   backtest window vs test split (P12), trade `opened_at` range (P14).
3. Report which findings resolved, which persist, and which are new. A new run is the only way to know
   whether Stage 1 or Stage 2 actually worked.

## The 30 findings at a glance

Blockers first. Anchors are `#p1`…`#p30` in both documents.

| # | Problem |
|---|---|
| P1 | Reported Direction metrics measure a model discarded before saving |
| P2 | 72% of the training set is zero-volatility degenerate candles |
| P3 | 4 features 100% NaN, 3 constant — 7 of 57 dead |
| P4 | Stage-2 direction model has no discrimination; isotonic amplifies a 10-point band |
| P5 | Backtest never tests the entry bar's own high/low against the barriers |
| P6 | Backtest bypasses the Risk Guard entirely |
| P7 | Backtest selection policy differs from live (`evaluate_many` unused) |
| P8 | Risk and Exit heads trained only on realised winners |
| P9 | 45-day recency half-life shrinks a 372-day window to ~17% effective sample |
| P10 | Validation block does four incompatible jobs at once |
| P11 | Auto-tuned thresholds are arithmetically meaningless |
| P12 | Backtest window is not the test split; the two runs differ |
| P13 | Sortino denominator wrong; Sharpe annualises a degenerate curve |
| P14 | `opened_at` is wall-clock — every trade has a negative holding period |
| P15 | Exit SL head is dead code; trailing head learns `y = 0.5x` |
| P16 | Calibration adopted on unguarded 0.001-nat differences |
| P17 | `_ai_summary` structurally incapable of reporting a data problem |
| P18 | `_walk_forward_problem_summary` can only ever return "none measured" |
| P19 | Relaxed diagnostic backtest resets every other decision setting |
| P20 | `rejection_breakdown` is first-match-wins, not comparable across runs |
| P21 | Entry threshold auto-tune undercuts the config; audit log prints the wrong number |
| P22 | Trailing trigger hardcoded in `_fill`, ignoring the Exit model |
| P23 | Train/serve population mismatch on `bb_position` and `volume_trend` |
| P24 | `confidence_threshold_analysis` rewards total class collapse |
| P25 | ADX/DI saturate — mechanism corrected in the fixes file, use that version |
| P26 | Archive loader cannot work: total timeout, whole CSV into RAM, 740-day default |
| P27 | Class `distribution` is whole-dataset, printed next to train-only `rows` |
| P28 | Risk metrics computed on unclipped predictions |
| P29 | `duplicate_feature_rows` counted, never removed |
| P30 | Dead code: Risk veto, `max_positions_per_symbol`, R5 telemetry, liquidation slippage |

## What the branch got right

Worth knowing so you do not "fix" it: `_align` passing NaN rather than 0.0 is a genuine train/serve
correctness improvement and should stay. `c2968d8`'s stop-floor diagnosis is correct. The archive
loader's parsing conventions (by-name columns, header auto-detection, exact-bucket joins, NaN
preservation) are the part that was done well — only its transport is broken. And the instrumentation
these commits added (`split_coverage_pct`, per-feature null counts, symbol×month null rates, the drift
block) is what made the audit possible at all; the failure is that nothing reads it.

The pattern across the branch is consistent: careful diagnosis, good instrumentation, and a fix that
goes one step further than the evidence supports. Keep that in mind when proposing your own.
