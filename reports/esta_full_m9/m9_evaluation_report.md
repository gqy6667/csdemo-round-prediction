# M9 Unified Evaluation Report

The fixed M8 XGBoost model was evaluated once on the fixed test split.
Confidence intervals use series-level bootstrap resampling.

| Metric | Point | 95% CI | Minimum | Stage target | Result |
|---|---:|---:|---:|---:|---|
| auc | 0.727122 | [0.713125, 0.740907] | >= 0.700 | >= 0.730 | minimum passed |
| log_loss | 0.591733 | [0.580192, 0.603874] | <= 0.610 | <= 0.580 | minimum passed |
| accuracy | 0.647411 | [0.632426, 0.662448] | >= 0.640 | >= 0.660 | minimum passed |
| brier_score | 0.205294 | [0.200853, 0.209890] | <= 0.210 | <= 0.195 | minimum passed |

Test rounds: 4,172; series: 79.
Bootstrap repetitions: 2,000; seed: 42.

Minimum thresholds passed: 4/4; stage targets passed: 0/4.
No parameter or decision threshold was selected using these test results.
