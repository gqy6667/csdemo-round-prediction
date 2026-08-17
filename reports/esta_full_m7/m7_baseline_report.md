# M7 Baseline Comparison Report

All models use the same fixed 70/20/10 split, encoded feature columns,
test rows, classification threshold (0.5), and probability metrics.

| Model | Accuracy | AUC | Log Loss | Brier | ECE10 |
|---|---:|---:|---:|---:|---:|
| constant_train_prior | 0.524449 | 0.500000 | 0.692640 | 0.249745 | 0.018501 |
| logistic_regression | 0.658437 | 0.727229 | 0.592538 | 0.205508 | 0.008624 |
| xgboost_tuned | 0.647411 | 0.727122 | 0.591733 | 0.205294 | 0.023198 |

## Acceptance

XGBoost minus logistic-regression test AUC: -0.000107.
Target (at least 0.01): **NOT MET**.

A NOT MET result is retained as an honest experimental result, as required
by the M7 specification; it is not a pipeline failure.
