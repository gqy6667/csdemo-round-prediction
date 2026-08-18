# M11 Robustness and Error Analysis Report

LAN-online absolute AUC gap: 0.009003 (signed 95% CI [-0.018130, 0.036925]).
Maps with at least 300 rounds: 7; minimum AUC: 0.695993.
Lowest large-map AUC CI lower bound: 0.658606.
High-confidence wrong rounds available: 90; reviewed: 30.

## Acceptance

LAN-online AUC gap <= 0.04: True.
Large-map minimum AUC >= 0.67: True.
Large-map minimum AUC >= 0.69: True.
Every large-map AUC CI lower bound >= 0.67: False.
At least 30 high-confidence errors reviewed: True.
All group tables include rounds, series counts, and series-level 95% intervals.
First-kill data is used only as a post-hoc diagnostic outcome.

## External Benchmark Differences

The closest published task also extracts equipment at `RoundFreezetimeEnd`. Its DNN
reports accuracy 0.679220 and log loss 0.567860. The current XGBoost is 3.18
percentage points lower in accuracy and 0.023873 higher in log loss. The datasets and
split protocols differ, so this is a reference gap rather than a controlled algorithm
comparison.

Mid-round snapshot studies report random-forest accuracy of 0.8841 and 0.88. The
current accuracy is numerically 23.67 and 23.26 percentage points lower, respectively,
but those snapshots include post-combat state and are not directly comparable. See
`external_benchmark_comparison.md` for sources and the row-level calculations.
