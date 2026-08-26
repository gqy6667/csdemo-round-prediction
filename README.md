# CSDemo Round Winner Prediction

This project builds three round-win prediction tasks from ESTA/AWPY demo data:

1. purchase-complete, pre-combat win probability
2. post-first-kill win probability
3. real-time win probability

The first milestone focuses on the first two tasks with XGBoost, using a 70/20/10
train/validation/test split. The pre-round LightGBM comparison and final acceptance
are complete through M27. The post-first-kill LightGBM controlled baseline and
validation-only tuning and frozen-model evaluation are complete through M30 under
the frozen M21 data, feature, split, and evaluation contract.

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
- `docs/m15_first_kill_data_spec.md`
- `docs/m16_first_kill_baseline_spec.md`
- `docs/m17_first_kill_tuning_spec.md`
- `docs/m18_first_kill_evaluation_spec.md`
- `docs/m19_first_kill_explanation_spec.md`
- `docs/m20_first_kill_prediction_interface_spec.md`
- `docs/m21_first_kill_final_acceptance_spec.md`
- `docs/m22_pre_round_lightgbm_baseline_spec.md`
- `docs/m23_pre_round_lightgbm_tuning_spec.md`
- `docs/m24_pre_round_lightgbm_evaluation_spec.md`
- `docs/m25_pre_round_lightgbm_explanation_spec.md`
- `docs/m27_pre_round_lightgbm_final_acceptance_spec.md`
- `docs/m28_post_first_kill_lightgbm_controlled_baseline_spec.md`
- `docs/m29_post_first_kill_lightgbm_tuning_spec.md`
- `docs/m30_post_first_kill_lightgbm_evaluation_spec.md`
- `docs/teacher_review_reports_spec.md`
- `docs/external_benchmark_policy.md`
- `reports/esta_full_m11/m11_robustness_report.md`
- `reports/esta_full_m11/external_benchmark_comparison.md`
- `reports/esta_full_m12/m12_explanation_report.md`
- `reports/esta_full_m12/external_benchmark_comparison.md`
- `reports/esta_full_m13/m13_interface_report.md`
- `reports/esta_full_m13/external_benchmark_comparison.md`
- `reports/esta_full_m14/m14_final_acceptance_report.md`
- `reports/esta_full_m14/external_benchmark_comparison.md`
- `reports/esta_full_m15/m15_first_kill_data_report.md`
- `reports/esta_full_m15/external_benchmark_comparison.md`
- `reports/esta_full_m16/m16_first_kill_baseline_report.md`
- `reports/esta_full_m16/external_benchmark_comparison.md`
- `reports/esta_full_m17/m17_first_kill_tuning_report.md`
- `reports/esta_full_m17/external_benchmark_comparison.md`
- `reports/esta_full_m18/m18_first_kill_evaluation_report.md`
- `reports/esta_full_m18/external_benchmark_comparison.md`
- `reports/esta_full_m19/m19_first_kill_explanation_report.md`
- `reports/esta_full_m19/external_benchmark_comparison.md`
- `reports/esta_full_m20/m20_first_kill_interface_report.md`
- `reports/esta_full_m20/external_benchmark_comparison.md`
- `reports/esta_full_m21/m21_first_kill_final_acceptance_report.md`
- `reports/esta_full_m21/external_benchmark_comparison.md`
- `reports/esta_full_m23/m23_pre_round_lightgbm_tuning_report.md`
- `reports/esta_full_m24/m24_pre_round_lightgbm_evaluation_report.md`
- `reports/esta_full_m25/m25_pre_round_lightgbm_explanation_report.md`
- `reports/esta_full_m27/m27_pre_round_lightgbm_final_acceptance_report.md`
- `reports/esta_full_m28/m28_post_first_kill_lightgbm_controlled_baseline_report.md`
- `reports/esta_full_m29/m29_post_first_kill_lightgbm_tuning_report.md`
- `reports/esta_full_m30/m30_post_first_kill_lightgbm_evaluation_report.md`
- `reports/teacher_review/01_pre_round_xgboost_report.md`
- `reports/teacher_review/02_pre_round_lightgbm_report.md`
- `reports/teacher_review/03_post_first_kill_xgboost_report.md`
- `reports/m6_to_m24_progress_report.md`

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

- Build one sample at the earliest valid enemy-kill tick within each repaired
  `series_id + game_id + round_id` key.
- Add first-kill side, victim side, weapon, headshot, and the 5v4/4v5 alive state.
- Treat ESTA `seconds` as a candidate feature, never as the event-ordering key.
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
M15 can now rebuild and audit the repaired post-first-kill dataset:

```powershell
.\scripts\run_first_kill_data_stage.ps1
```

M15 passes all 12 blocking checks and all 80 automated tests. It keeps 41,027
post-first-kill samples, explicitly excludes 47 rounds with no valid enemy kill,
and reuses the persisted 782-series split manifest. Replacing seconds ordering
with tick ordering changes the selected event in 14,357 old rows (34.99%). M15
does not train a model.

Run the M16 fixed-split first-kill baseline comparison:

```powershell
.\scripts\run_first_kill_baselines.ps1
```

M16 passes all 8 blocking checks and all 90 automated tests. On the untouched
4,170-row test split, untuned XGBoost reaches Accuracy `0.7453`, AUC `0.8089`,
Log Loss `0.5248`, and Brier `0.1763`. Logistic regression reaches AUC `0.8091`,
so the tree model is not yet clearly better. Against an identical-row pre-round
XGBoost control, the first-kill feature profile adds `0.0880` validation AUC and
`0.0860` test AUC.

Run M17 validation-only controlled XGBoost tuning:

```powershell
.\scripts\run_first_kill_tuning.ps1
```

M17 evaluates 39 frozen candidates across eight sequential phases and passes all
12 blocking checks plus all 100 automated tests. The selected model uses a 1,500-tree
cap, 50-round early stopping, depth 2, and subsample 0.9; seed 42 stops at 409 trees.
Test AUC improves from `0.8089` to `0.8098`, Log Loss from `0.5248` to `0.5231`, and
Brier from `0.1763` to `0.1757`. Accuracy falls by `0.12` percentage points and ECE10
worsens by `0.0045`, so those tradeoffs remain explicit.

Run M18 fixed-model evaluation and validation-only calibration diagnosis:

```powershell
.\scripts\run_first_kill_evaluation.ps1
```

M18 replays the frozen M17 probabilities to `1.11e-16` maximum absolute error and
makes zero XGBoost fit calls. It passes all 13 blocking checks and all 108 automated
tests. Test AUC is `0.8098` with series-level 95% CI `[0.7977, 0.8221]`; Log Loss is
`0.5231` with CI `[0.5097, 0.5361]`. The LAN-online AUC difference is `-0.0103`
with CI `[-0.0346, 0.0148]`, and all maps with at least 300 test rounds exceed the
`0.740` minimum AUC. Grouped validation selects the uncalibrated identity method.

Run M19 first-kill model explanation and feature-leakage audit:

```powershell
.\scripts\run_first_kill_explanation.ps1
```

M19 explains the unchanged M17/M18 model with deployment-tree Gain, 20-repeat
encoded and raw-feature grouped permutation importance, and native TreeSHAP. All
82 encoded inputs map to 40 allowed raw features with zero leakage failures, and
TreeSHAP reconstructs the fixed probabilities within `4.03e-7`. All ten formal
performance and robustness targets pass. The strongest raw feature is
`first_kill_advantage_ct`, followed by `eq_value_diff_ct`.

Predict one post-first-kill snapshot with the M20 interface:

```powershell
C:\Users\admin\11\envs\game\python.exe -m src.csdemo.predict_first_kill --input examples\first_kill_snapshot.json --model models\esta_full_m17\first_kill_xgboost_tuned.joblib --calibrator models\esta_full_m18\first_kill_calibrator.joblib
```

Run the complete M20 interface acceptance:

```powershell
.\scripts\run_first_kill_interface.ps1
```

M20 passes all 10 blocking checks and all 131 automated tests. It validates 27
purchase fields plus four first-kill fields, derives nine CT-minus-T differences,
and aligns one row to the frozen 40 raw/82 encoded feature contract. JSON and CSV
produce identical probabilities; ten invalid examples are rejected. The stage makes
zero XGBoost fit calls and leaves all M18/M19 metrics and target margins unchanged.

Run the M21 final acceptance and reproducibility entrypoint:

```powershell
.\scripts\run_first_kill_pipeline.ps1
```

M21 passes all 17 blocking checks and all 145 automated tests. It inventories all
1,558 ESTA files, verifies the grouped 69.44/20.40/10.16 percent split has zero
cross-split series/game/round overlap, and replays all 4,170 test probabilities to
`1.11e-16` maximum absolute error. All five metrics reproduce exactly, all ten
formal targets retain zero remaining gap, and no XGBoost fit call occurs. The
post-first-kill XGBoost track is complete. M22 then starts an independent LightGBM
controlled comparison on the identical data, split, features, and metrics.

Use `-RebuildFirstKill` to rerun M15-M20 from accepted M14 artifacts, or
`-FullRebuild` to rebuild the whole local pipeline from the raw ESTA files.

Run the M22 pre-round LightGBM controlled baseline:

```powershell
.\scripts\run_pre_round_lightgbm_baseline.ps1
```

M22 keeps the accepted 41,074 pre-round rows, grouped 70/20/10 split, 36 raw/43
encoded features, and five metrics fixed. All 13 blocking checks and all 155 tests
pass. The untuned LightGBM baseline reaches Accuracy 0.650767, AUC 0.727846, Log
Loss 0.591437, Brier 0.205201, and ECE10 0.018875. It is slightly better than the
frozen XGBoost on all five point metrics, but the margins are small. The next stage
is M23 validation-only controlled LightGBM tuning; test metrics must not be used to
select candidates.

Run the M23 validation-only LightGBM tuning stage:

```powershell
.\scripts\run_pre_round_lightgbm_tuning.ps1
```

M23 evaluates 36 candidates across nine one-parameter phases and five fixed seeds.
No candidate improves validation Log Loss by the predeclared 0.0001 minimum, so the
accepted model deliberately retains the M22 parameters and test metrics. All 14
blocking checks and 165 tests pass. See `reports/lightgbm_xgboost_external_metrics.md`
for the consolidated LightGBM, final XGBoost, logistic-regression, and public-model
comparison. M24 will evaluate the frozen model with grouped confidence intervals,
robustness slices, and calibration diagnostics without further tuning.

Run the M24 frozen LightGBM evaluation stage:

```powershell
.\scripts\run_pre_round_lightgbm_evaluation.ps1
```

M24 replays the frozen M23 model without calling LightGBM `fit`. All 16 blocking
checks and 176 tests pass. Test AUC is 0.727846 with series-level 95% CI
[0.714169, 0.741427]; Log Loss is 0.591437 with CI [0.580635, 0.602921]. LightGBM
has better point estimates than XGBoost on all five metrics, but every paired
bootstrap advantage interval includes zero, so the current evidence does not show
a statistically stable win. LAN-online AUC difference is 0.008711 with CI
[-0.017073, 0.034198], and validation OOF correctly selects no calibration.

Run the M25 frozen LightGBM explanation stage:

```powershell
.\scripts\run_pre_round_lightgbm_explanation.ps1
```

M25 replays all 4,172 M24 probabilities without calling `fit`, keeps the model hash
unchanged, and completes 20-repeat encoded, raw-feature, and five-group permutation
importance plus native TreeSHAP. All 43 encoded columns map to the accepted 36 raw
features with zero leakage failures; SHAP reconstructs probability to `7.77e-16`.
`eq_value_diff_ct` ranks first by Gain, grouped permutation, and SHAP. LightGBM and
M12 XGBoost share 8/10 Gain Top-10 features and 9/10 for permutation, SHAP, and mean
rank. All 14 blockers and 192 tests pass.

Run the M26 frozen LightGBM JSON/CSV prediction interface acceptance:

```powershell
.\scripts\run_pre_round_lightgbm_interface.ps1
```

For one JSON prediction, run:

```powershell
C:\Users\admin\11\envs\game\python.exe -m src.csdemo.predict_pre_round_lightgbm `
  --input examples\pre_round_snapshot.json `
  --model models\esta_full_m23\pre_round_lightgbm_tuned.joblib `
  --calibrator models\esta_full_m24\pre_round_lightgbm_calibrator.joblib
```

M26 validates 27 input fields, derives 9 CT-minus-T fields, and aligns the result to
the frozen 36 raw and 43 encoded features. JSON and CSV predictions match exactly;
all 10 invalid examples are rejected. The formal run passes all 15 blockers and 201
tests without a LightGBM `fit` call or artifact hash change.

Run the M27 final acceptance and one-command reproduction gate:

```powershell
.\scripts\run_pre_round_lightgbm_pipeline.ps1
```

M27 replays all 4,172 test probabilities with a maximum error of `1.11e-16`, keeps
all five metrics unchanged, preserves the data/model/calibrator hashes, and records
the honest paired conclusion that zero of five metrics show statistically stable
LightGBM superiority. All 19 blockers, 211 tests, source compilation, and three
reproduction modes pass. The pre-round LightGBM line is closed; teacher-review
reports are prepared before the post-first-kill LightGBM controlled baseline.

Run the M28 post-first-kill LightGBM controlled baseline on Windows:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_post_first_kill_lightgbm_baseline.ps1
```

M28 keeps the 41,027 M21 rows, 547/156/79 series split, and exact 40 raw/82
encoded feature contract, changing only the model algorithm. The fixed LightGBM
baseline stops at 160 trees and reaches Accuracy 0.746043, AUC 0.809070, Log Loss
0.523799, Brier 0.175894, and ECE10 0.013622. Against the frozen M21 XGBoost,
Accuracy and ECE10 point estimates improve while AUC, Log Loss, and Brier worsen
slightly; all five paired series-bootstrap intervals include zero, so neither model
has a statistically stable advantage. All 16 blockers, 225 tests, source compilation,
and 27 artifact hash checks pass. M29 is validation-only controlled tuning; no fourth
teacher report is created until the later evaluation and final acceptance stages pass.

Run the M29 post-first-kill LightGBM validation-only tuning stage:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_post_first_kill_lightgbm_tuning.ps1
```

M29 evaluates 36 frozen-order candidates and five stability seeds without exposing
test metrics to selection. Only `max_depth=-1` to `max_depth=3` clears the 0.0001
validation Log Loss improvement rule; validation Log Loss improves by 0.000738 and
the final model stops at 211 trees. Frozen test Accuracy/AUC/Log Loss/Brier/ECE10 are
0.742926/0.808255/0.524063/0.176003/0.014191. These point estimates are slightly
worse than M28 on all five metrics and worse than M21 XGBoost on four of five, so M29
does not claim a model win. All 16 blockers, 236 tests, source compilation, and 28
artifact hashes pass. M30 will determine uncertainty with series-level paired analysis.

Run the M30 frozen post-first-kill LightGBM evaluation:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_post_first_kill_lightgbm_evaluation.ps1
```

M30 replays all 4,170 M29 test probabilities with maximum absolute error
`1.11e-16` and zero LightGBM fit calls. Series-level 95% intervals are
AUC `[0.796194, 0.820185]` and Log Loss `[0.510784, 0.537071]`. Against the
same-row M21 XGBoost, all five paired performance-advantage intervals include zero;
there is no statistically stable winner. All eight robustness families pass, the
LAN-online AUC gap is -0.012561 with CI `[-0.036757, 0.011680]`, and validation OOF
selects the uncalibrated identity method. All 18 blockers, 246 tests, source
compilation, and 40 artifact hashes pass. M31 is frozen-model explanation.
