# Diagnostic runs from the audit-fix branch

Produced by `scripts/run_synthetic_diagnostic.py` on synthetic data with the
audit's own pathologies injected (30% flat candles, two blanked feature
columns). The model-quality numbers here mean nothing - the data is generated.
What they demonstrate is that the report's detectors, invariants and fields
behave correctly on data whose defects are known by construction.

| file | when |
|---|---|
| `01_baseline.json` / `.md` | before any fix |
| `03_final.json` / `.md` | after P1-P30 |

Headline change on identical inputs:

| field | before | after |
|---|---|---|
| `ai_summary.overall_status` | WARNING | CRITICAL |
| `biggest_data_problem` | "none measured" | names all 5 empty feature columns |
| `recommendations.CRITICAL` | 0 | 1 |
| `dataset.valid_samples` | 35,808 | 24,944 (degenerate rows removed) |
| `duplicate_feature_rows` | 5,410 | 0 |
| worst feature drift | 1.46 train-sigma | 0.45 train-sigma |
| `direction.metrics_source` | absent | present (identity guard) |
| risk prediction floor | 0.419 | 0.201 |
| risk training rows | 4,041 (winners only) | 12,428 (all bars) |
| exit regressors trained | 3 | 2 (trailing derived) |

The run also exercised two guards on real degeneracy: the calibrator refused a
stage-2 score spanning 0.029, and the build-time nullity check named the empty
columns before any model was trained.

A real Binance-backed diagnostic has **not** been run - see the branch summary
for why and for what that leaves unverified.
