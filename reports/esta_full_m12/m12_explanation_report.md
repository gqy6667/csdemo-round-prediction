# M12 Model Explanation Report

This stage explains the fixed M8 XGBoost on the unchanged M9 test split.
No model, feature, threshold, or test probability was changed.

## Explanation Integrity

The saved booster contains 313 trees, while deployed
predictions use the first 213 trees selected by early stopping.
Gain and native TreeSHAP were both limited to those deployment trees.
TreeSHAP reconstructed all probabilities with maximum absolute error
0.0000002512.

## Global Importance

| Feature | Gain rank | Normalized gain | Permutation rank | Test AUC decrease | SHAP rank | Mean abs SHAP |
|---|---:|---:|---:|---:|---:|---:|
| eq_value_diff_ct | 2 | 0.163236 | 1 | 0.076410 | 1 | 0.425467 |
| ct_eq_value | 7 | 0.037703 | 2 | 0.011400 | 2 | 0.146753 |
| helmet_diff_ct | 3 | 0.159426 | 4 | 0.002921 | 4 | 0.069526 |
| t_eq_value | 11 | 0.021568 | 3 | 0.004556 | 3 | 0.095839 |
| grenade_diff_ct | 1 | 0.265843 | 12 | 0.000392 | 5 | 0.046907 |
| ct_armor | 8 | 0.031025 | 8 | 0.000924 | 7 | 0.036169 |
| ct_m4a1_s | 12 | 0.008605 | 6 | 0.001906 | 6 | 0.036575 |
| t_rifles | 5 | 0.061180 | 10 | 0.000700 | 10 | 0.024746 |
| armor_diff_ct | 4 | 0.063761 | 11 | 0.000567 | 13 | 0.017964 |
| score_diff_ct | 14 | 0.005382 | 5 | 0.002032 | 9 | 0.032513 |
| t_armor | 9 | 0.028577 | 9 | 0.000835 | 12 | 0.021523 |
| map_name_de_inferno | 13 | 0.005617 | 7 | 0.001484 | 11 | 0.023140 |
| rifle_diff_ct | 10 | 0.026314 | 13 | 0.000342 | 8 | 0.033920 |
| t_helmets | 6 | 0.043048 | 18 | 0.000045 | 19 | 0.002383 |
| map_name_de_ancient | 24 | 0.003441 | 14 | 0.000196 | 18 | 0.002667 |

`eq_value_diff_ct` is the most stable signal: gain rank 2, permutation rank 1,
and SHAP rank 1. `grenade_diff_ct` is gain rank 1 but permutation rank 12,
showing why gain alone is not sufficient. Rank Spearman correlations are
gain-permutation 0.644,
gain-SHAP 0.873, and
permutation-SHAP 0.714.

Gain measures average loss reduction at tree splits. Permutation reports test AUC
loss after shuffling one encoded column. Mean absolute TreeSHAP reports average
contribution magnitude in log-odds. Correlated raw and difference features can
share importance, and none of these measures establishes causality.

## Leakage Audit

All 43 encoded model columns passed the pre-round schema audit.
The top 20 TreeSHAP features contain no ID, first-kill, damage, bomb-state, or
round-result fields.

| SHAP rank | Feature | Source | Result |
|---|---|---|---|
| 1 | eq_value_diff_ct | eq_value_diff_ct | pass |
| 2 | ct_eq_value | ct_eq_value | pass |
| 3 | t_eq_value | t_eq_value | pass |
| 4 | helmet_diff_ct | helmet_diff_ct | pass |
| 5 | grenade_diff_ct | grenade_diff_ct | pass |
| 6 | ct_m4a1_s | ct_m4a1_s | pass |
| 7 | ct_armor | ct_armor | pass |
| 8 | rifle_diff_ct | rifle_diff_ct | pass |
| 9 | score_diff_ct | score_diff_ct | pass |
| 10 | t_rifles | t_rifles | pass |
| 11 | map_name_de_inferno | map_name | pass |
| 12 | t_armor | t_armor | pass |
| 13 | armor_diff_ct | armor_diff_ct | pass |
| 14 | map_name_de_overpass | map_name | pass |
| 15 | map_name_de_dust2 | map_name | pass |
| 16 | map_name_de_nuke | map_name | pass |
| 17 | ct_score | ct_score | pass |
| 18 | map_name_de_ancient | map_name | pass |
| 19 | t_helmets | t_helmets | pass |
| 20 | t_grenades | t_grenades | pass |

## Round Cases

| Case | Round | Map | Actual | Predicted CT probability | Correct |
|---|---|---|---|---:|---|
| ct_high_probability | online:946b0351-728d-41c6-9964-9b20f21df71d_5 | de_ancient | CT | 0.982434 | True |
| t_high_probability | lan:fe5f9da6-2d03-4b91-9786-425ce96f2631_3 | de_inferno | T | 0.034359 | True |
| high_confidence_error | online:1f4b1506-6f45-43ee-8b4c-82961eb4bd5b_2 | de_vertigo | T | 0.977234 | False |

### ct_high_probability

| Rank | Feature | Value | SHAP log-odds | Direction |
|---:|---|---:|---:|---|
| 1 | t_eq_value | 1000 | 1.601606 | toward_ct |
| 2 | eq_value_diff_ct | 25050 | 1.023211 | toward_ct |
| 3 | helmet_diff_ct | 5 | 0.290184 | toward_ct |
| 4 | t_armor | 0 | 0.262389 | toward_ct |
| 5 | rifle_diff_ct | 4 | 0.149214 | toward_ct |

### t_high_probability

| Rank | Feature | Value | SHAP log-odds | Direction |
|---:|---|---:|---:|---|
| 1 | ct_eq_value | 1500 | -1.330074 | toward_t |
| 2 | eq_value_diff_ct | -19800 | -1.315732 | toward_t |
| 3 | ct_armor | 0 | -0.239705 | toward_t |
| 4 | grenade_diff_ct | -17 | -0.180204 | toward_t |
| 5 | rifle_diff_ct | -3 | -0.099224 | toward_t |

### high_confidence_error

| Rank | Feature | Value | SHAP log-odds | Direction |
|---:|---|---:|---:|---|
| 1 | t_eq_value | 1100 | 1.563771 | toward_ct |
| 2 | eq_value_diff_ct | 18850 | 1.026832 | toward_ct |
| 3 | helmet_diff_ct | 5 | 0.290184 | toward_ct |
| 4 | t_armor | 0 | 0.250481 | toward_ct |
| 5 | ct_eq_value | 19950 | 0.106776 | toward_ct |

The error case shows why the model strongly favored CT from the purchase snapshot;
it does not explain the later combat outcome. Position, aim, utility execution, and
other post-freeze events are outside this model and must not be added as pre-round features.

## Acceptance

- three_importance_methods_created: True
- ct_t_and_error_cases_created: True
- top20_has_no_id_or_future_leakage: True
- all_model_features_pass_schema_audit: True
- shap_reconstructs_probability_within_1e_5: True

External benchmark differences are unchanged because M12 did not retrain the model;
see `external_benchmark_comparison.md` in this report directory.
