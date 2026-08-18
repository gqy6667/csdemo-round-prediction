# CSDemo Round Winner Prediction

This project builds three round-win prediction tasks from ESTA/AWPY demo data:

1. purchase-complete, pre-combat win probability
2. post-first-kill win probability
3. real-time win probability

The first milestone focuses on the first two tasks with XGBoost, using a 70/20/10 train/validation/test split. LightGBM can be added after the baseline is stable.

## Milestones

The first prediction point is the frame nearest `freezeTimeEndTick`: players have
finished buying and combat has not started yet.

Chinese project documentation:

- `docs/project_paths.md`
- `docs/metrics_guide.md`
- `docs/pre_round_xgb_module_spec.md`
- `docs/m6_feature_dictionary.md`
- `reports/esta_full_m6/m6_feature_report.md`
- `reports/m5_split_leakage_audit.md`
- `docs/m7_baseline_spec.md`
- `reports/esta_full_m7/m7_baseline_report.md`
- `reports/esta_full_m8_tuned/m8_controlled_tuning_report.md`
- `docs/m9_evaluation_spec.md`
- `reports/esta_full_m9/m9_evaluation_report.md`
- `docs/m10_calibration_spec.md`
- `reports/esta_full_m10/m10_calibration_report.md`
- `docs/m11_robustness_spec.md`
- `docs/m12_explanation_spec.md`
- `docs/m13_prediction_interface_spec.md`
- `docs/m14_final_acceptance_spec.md`
- `docs/external_benchmark_policy.md`
- `reports/esta_full_m11/m11_robustness_report.md`
- `reports/esta_full_m11/external_benchmark_comparison.md`
- `reports/esta_full_m12/m12_explanation_report.md`
- `reports/esta_full_m12/external_benchmark_comparison.md`
- `reports/esta_full_m13/m13_interface_report.md`
- `reports/esta_full_m13/external_benchmark_comparison.md`
- `reports/esta_full_m14/m14_final_acceptance_report.md`
- `reports/esta_full_m14/external_benchmark_comparison.md`

Current purchase-complete XGBoost test metrics after controlled tuning:

- AUC: `0.7271`
- Log loss: `0.5917`
- Accuracy: `0.6474`

M7 baseline comparison on the identical test rows:

- Constant train-prior AUC: `0.5000`
- Logistic-regression AUC: `0.7272`
- Tuned XGBoost AUC: `0.7271`
- XGBoost minus logistic-regression AUC: `-0.0001` (the `+0.01` target was not met)

M9 series-level bootstrap evaluation of tuned XGBoost:

- AUC: `0.7271` (95% CI `[0.7131, 0.7409]`)
- Log loss: `0.5917` (95% CI `[0.5802, 0.6039]`)
- Accuracy: `0.6474` (95% CI `[0.6324, 0.6624]`)
- Brier score: `0.2053` (95% CI `[0.2009, 0.2099]`)

M10 grouped validation selected `uncalibrated` (identity): raw XGBoost test ECE10
is `0.0232`, and both sigmoid and isotonic calibration worsened Log Loss and Brier.

M11 found a LAN-online AUC gap of `0.0090` (95% CI `[-0.0181, 0.0369]`).
All seven maps with at least 300 test rounds pass the `0.69` point-estimate target,
while some map confidence intervals still cross below `0.67`. Thirty high-confidence
errors were reviewed and categorized.

Closest published freeze-time benchmark differences:

- DNN accuracy `0.6792`; current XGBoost `0.6474`: `-3.18` percentage points.
- DNN log loss `0.5679`; current XGBoost `0.5917`: `+0.0239` (higher is worse).
- Different datasets and split protocols make these reference gaps, not a controlled
  DNN-versus-XGBoost comparison.

Mid-round snapshot studies report approximately `0.88` accuracy, but use alive-player,
health, time and bomb-state information observed after combat starts. Their numerical
gaps are reported separately and are not interpreted as model superiority.

M12 explains the unchanged model with deployment-tree gain, 20-repeat test AUC
permutation importance, and native XGBoost TreeSHAP. Equipment-value difference is
the most stable feature (gain/permutation/SHAP ranks `2/1/1`). All 43 encoded inputs
pass the pre-round leakage audit, and TreeSHAP reconstructs deployed probabilities with
maximum absolute error `0.0000002512`.

The full ESTA dataset and historical models are not committed. The small tuned M8
pre-round model is included for reproducibility. See `data/README.md`.

### Stage 0: inventory

- Confirm local ESTA LAN/online dataset paths.
- Confirm the Anaconda environment name and installed packages.
- Inspect the raw file format: parsed JSON, CSV, parquet, or original demos.

### Stage 1: static round samples

- Build one sample per round at freeze-time/buy-end.
- Predict final round winner.
- Use simple, robust features: map, side, team economy, equipment value, weapon counts, armor/helmet/defuse kit, health, alive players, score, round number.
- Train an XGBoost baseline.

### Stage 2: first-kill samples

- Build one sample per round at the first kill timestamp.
- Add first-kill features: killer side/team, victim side/team, weapon, headshot, trade window, alive-player differential after the kill.
- Train an XGBoost baseline with the same split.

### Stage 3: real-time samples

- Generate multiple snapshots per round at fixed intervals or after key events.
- Add time-left, bomb state, positional, and event-history features.
- Start with XGBoost/LightGBM tabular snapshots before considering sequence models.

## Recommended Structure

```text
data/
  raw/                 # original ESTA/AWPY files, not committed
  interim/             # normalized round/event tables
  processed/           # model-ready samples
models/
notebooks/
reports/
src/csdemo/
```

## First Run

After setting the dataset path, run:

```powershell
python -m src.csdemo.make_dataset --input <DATASET_PATH> --output data/processed
python -m src.csdemo.train_xgb --task pre_round --data data/processed/pre_round.csv
python -m src.csdemo.train_xgb --task first_kill --data data/processed/first_kill.csv
```

Run the reproducible M7 baseline comparison:

```powershell
C:\Users\admin\11\envs\game\python.exe -m src.csdemo.m7_baselines --data data\processed\esta_full\pre_round.parquet --model-dir models\esta_full_m7 --report-dir reports\esta_full_m7
```

Run the M9 fixed-test evaluation:

```powershell
C:\Users\admin\11\envs\game\python.exe -m src.csdemo.m9_evaluation --data data\processed\esta_full\pre_round.parquet --model models\esta_full_m8_tuned\pre_round_xgb.joblib --report-dir reports\esta_full_m9 --bootstrap-samples 2000 --seed 42
```

Run M10 validation-only calibration selection:

```powershell
C:\Users\admin\11\envs\game\python.exe -m src.csdemo.m10_calibration --data data\processed\esta_full\pre_round.parquet --base-model models\esta_full_m8_tuned\pre_round_xgb.joblib --model-dir models\esta_full_m10 --report-dir reports\esta_full_m10 --folds 5
```

Run M11 grouped robustness and error analysis:

```powershell
C:\Users\admin\11\envs\game\python.exe -m src.csdemo.m11_robustness --predictions reports\esta_full_m9\test_predictions.csv --data data\processed\esta_full\pre_round.parquet --kills data\interim\esta_full\kills.parquet --report-dir reports\esta_full_m11 --bootstrap-samples 2000 --seed 42 --review-cases 30
```

Generate the external benchmark difference table for M11 or a later stage:

```powershell
C:\Users\admin\11\envs\game\python.exe -m src.csdemo.benchmark_comparison --metrics reports\esta_full_m9\m9_summary.json --benchmarks benchmarks\external_round_model_metrics.csv --report-dir reports\esta_full_m11 --stage-label M11
```

Run M12 model explanation and regenerate the required external comparison:

```powershell
C:\Users\admin\11\envs\game\python.exe -m src.csdemo.m12_explanation --data data\processed\esta_full\pre_round.parquet --model models\esta_full_m8_tuned\pre_round_xgb.joblib --report-dir reports\esta_full_m12 --permutation-repeats 20 --seed 42 --case-features 10 --shap-plot-rows 1500
```

Predict one purchase-end, pre-combat snapshot with the M13 interface:

```powershell
C:\Users\admin\11\envs\game\python.exe -m src.csdemo.predict_pre_round --input examples\pre_round_snapshot.json --model models\esta_full_m8_tuned\pre_round_xgb.joblib --calibrator models\esta_full_m10\pre_round_calibrator.joblib
```

Re-run all M13 interface checks and external benchmark comparisons:

```powershell
C:\Users\admin\11\envs\game\python.exe -m src.csdemo.m13_interface --model models\esta_full_m8_tuned\pre_round_xgb.joblib --calibrator models\esta_full_m10\pre_round_calibrator.joblib --json-example examples\pre_round_snapshot.json --csv-example examples\pre_round_snapshot.csv --metrics reports\esta_full_m9\m9_summary.json --benchmarks benchmarks\external_round_model_metrics.csv --report-dir reports\esta_full_m13
```

Run the M14 final acceptance against existing artifacts:

```powershell
.\scripts\run_pre_round_pipeline.ps1
```

Rebuild everything from the local ESTA dataset before M14 acceptance:

```powershell
.\scripts\run_pre_round_pipeline.ps1 -FullRebuild
```

M14 passes all 15 blocking checks and all 70 automated tests. The four core test
metrics pass the agreed minimum gates, but none reaches the higher stage target.
The project is ready to start a fresh post-first-kill XGBoost pipeline using the
repaired round identity and the persisted 782-series split manifest.
