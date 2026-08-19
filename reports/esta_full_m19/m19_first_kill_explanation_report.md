# M19 首杀后模型解释与泄漏审计报告

## 阶段结论

阻断验收状态：**passed**；可进入 M20：**True**。
本阶段没有训练、调参、删除特征或改变测试概率，只解释 M17/M18 冻结模型。
模型文件保存 459 棵树，部署使用 early stopping 选中的 409 棵树。
TreeSHAP 重建测试概率最大绝对误差为 `0.0000004030`。

## 离正式目标还有多少

`remaining` 是仍需改善量；`margin` 是已经超过目标的余量。

| 目标 | 当前 | 通过线 | 方向 | Remaining | Margin | 通过 |
|---|---:|---:|---|---:|---:|---|
| Test Accuracy | 0.744125 | 0.700000 | higher | 0.000000 | 0.044125 | True |
| Test AUC | 0.809837 | 0.780000 | higher | 0.000000 | 0.029837 | True |
| Test Log Loss | 0.523146 | 0.550000 | lower | 0.000000 | 0.026854 | True |
| Test Brier | 0.175656 | 0.185000 | lower | 0.000000 | 0.009344 | True |
| Test ECE10 | 0.015450 | 0.030000 | lower | 0.000000 | 0.014550 | True |
| AUC 95% CI lower | 0.797731 | 0.790000 | higher | 0.000000 | 0.007731 | True |
| Log Loss 95% CI upper | 0.536146 | 0.540000 | lower | 0.000000 | 0.003854 | True |
| LAN-online absolute AUC gap | 0.010276 | 0.040000 | lower | 0.000000 | 0.029724 | True |
| Large-map minimum AUC | 0.783901 | 0.770000 | higher | 0.000000 | 0.013901 | True |
| Large-map minimum AUC CI lower | 0.750719 | 0.700000 | higher | 0.000000 | 0.050719 | True |

十项正式目标通过 10/10；仍需改善的正式目标数为 0。
这表示首杀后 XGBoost 的当前统计验收目标已经达到；它不表示实时胜率、LightGBM 对照或整个课题已经结束。

从项目模块看，M19 通过后，首杀后 XGBoost 还剩 M20 单条预测接口和 M21 最终验收两个模块；之后才进入 LightGBM 同数据对照和实时胜率。

## 原始特征重要性

分组 Permutation 会把一个原始特征的全部独热列用同一个排列一起打乱。
Gain 和 SHAP 则把对应编码列的值聚合回原始特征。

| 原始特征 | 时点组 | Gain 排名 | 分组 Permutation 排名 | SHAP 排名 | AUC 下降 | Mean abs SHAP |
|---|---|---:|---:|---:|---:|---:|
| `first_kill_advantage_ct` | first_kill_event | 1 | 1 | 1 | 0.138412 | 0.863187 |
| `eq_value_diff_ct` | purchase_end | 2 | 2 | 2 | 0.024478 | 0.331865 |
| `helmet_diff_ct` | purchase_end | 3 | 4 | 4 | 0.002118 | 0.101973 |
| `armor_diff_ct` | purchase_end | 4 | 6 | 7 | 0.001621 | 0.048194 |
| `ct_eq_value` | purchase_end | 11 | 3 | 3 | 0.005294 | 0.108094 |
| `map_name` | purchase_end | 9 | 7 | 6 | 0.001613 | 0.071789 |
| `ct_armor` | purchase_end | 6 | 9 | 8 | 0.000564 | 0.043424 |
| `t_eq_value` | purchase_end | 13 | 5 | 5 | 0.002012 | 0.076547 |
| `rifle_diff_ct` | purchase_end | 5 | 12 | 9 | 0.000366 | 0.041563 |
| `ct_m4a1_s` | purchase_end | 14 | 8 | 10 | 0.000844 | 0.038268 |
| `score_diff_ct` | purchase_end | 19 | 10 | 11 | 0.000547 | 0.036771 |
| `first_kill_time` | first_kill_event | 20 | 11 | 13 | 0.000397 | 0.030274 |
| `grenade_diff_ct` | purchase_end | 8 | 27 | 12 | 0.000000 | 0.034277 |
| `smg_diff_ct` | purchase_end | 16 | 17 | 14 | 0.000068 | 0.025949 |
| `ct_grenades` | purchase_end | 17 | 13 | 23 | 0.000210 | 0.004422 |

编码列三种排名的 Spearman 相关系数：Gain-Permutation `0.486`，Gain-SHAP `0.958`，Permutation-SHAP `0.524`。
相关特征会分摊重要性，负 permutation 值也被保留；三种方法都不是因果证明。

## 购买信息与首杀信息

| 特征组 | 编码列 | AUC 下降均值 | 标准差 |
|---|---:|---:|---:|
| purchase_end | 43 | 0.142717 | 0.007165 |
| first_kill_event | 39 | 0.136379 | 0.005837 |

四个首杀事件原始特征：

| 特征 | Gain 排名 | Permutation 排名 | SHAP 排名 | AUC 下降 |
|---|---:|---:|---:|---:|
| `first_kill_advantage_ct` | 1 | 1 | 1 | 0.138412 |
| `first_kill_time` | 20 | 11 | 13 | 0.000397 |
| `first_kill_headshot` | 22 | 14 | 17 | 0.000129 |
| `first_kill_weapon` | 10 | 36 | 15 | -0.000029 |

## 泄漏审计

82 个编码列全部追溯到 40 个允许原始特征。全部特征失败数 `0`，TreeSHAP 前 20 失败数 `0`。
首杀字段只对“首杀刚发生后”合法，不可复制到购买结束模型。ID、标签、后续击杀、血量/伤害、下包和回合结束字段都没有进入模型。

| SHAP 排名 | 编码列 | 原始特征 | 时点 | 结果 |
|---:|---|---|---|---|
| 1 | `first_kill_advantage_ct` | `first_kill_advantage_ct` | first_valid_enemy_kill | pass |
| 2 | `eq_value_diff_ct` | `eq_value_diff_ct` | purchase_end_pre_combat | pass |
| 3 | `ct_eq_value` | `ct_eq_value` | purchase_end_pre_combat | pass |
| 4 | `helmet_diff_ct` | `helmet_diff_ct` | purchase_end_pre_combat | pass |
| 5 | `t_eq_value` | `t_eq_value` | purchase_end_pre_combat | pass |
| 6 | `armor_diff_ct` | `armor_diff_ct` | purchase_end_pre_combat | pass |
| 7 | `ct_armor` | `ct_armor` | purchase_end_pre_combat | pass |
| 8 | `rifle_diff_ct` | `rifle_diff_ct` | purchase_end_pre_combat | pass |
| 9 | `ct_m4a1_s` | `ct_m4a1_s` | purchase_end_pre_combat | pass |
| 10 | `score_diff_ct` | `score_diff_ct` | purchase_end_pre_combat | pass |
| 11 | `map_name_de_inferno` | `map_name` | purchase_end_pre_combat | pass |
| 12 | `grenade_diff_ct` | `grenade_diff_ct` | purchase_end_pre_combat | pass |
| 13 | `first_kill_time` | `first_kill_time` | first_valid_enemy_kill | pass |
| 14 | `smg_diff_ct` | `smg_diff_ct` | purchase_end_pre_combat | pass |
| 15 | `map_name_de_overpass` | `map_name` | purchase_end_pre_combat | pass |
| 16 | `cash_diff_ct` | `cash_diff_ct` | purchase_end_pre_combat | pass |
| 17 | `map_name_de_dust2` | `map_name` | purchase_end_pre_combat | pass |
| 18 | `first_kill_headshot` | `first_kill_headshot` | first_valid_enemy_kill | pass |
| 19 | `t_armor` | `t_armor` | purchase_end_pre_combat | pass |
| 20 | `first_kill_weapon_AWP` | `first_kill_weapon` | first_valid_enemy_kill | pass |

## 三个回合案例

| 案例 | 主键 | 地图 | 实际 | CT 概率 | 正确 |
|---|---|---|---|---:|---|
| ct_high_probability | `63e0d98d-fd95-4698-9d12-808508005ee2 / online:1f4b1506-6f45-43ee-8b4c-82961eb4bd5b / online:1f4b1506-6f45-43ee-8b4c-82961eb4bd5b_8` | de_vertigo | CT | 0.992547 | True |
| t_high_probability | `c7f0e145-6303-40e6-9b79-ab0f825db57d / online:178eb7eb-2dd3-4394-9a5e-232c52ae0cdb / online:178eb7eb-2dd3-4394-9a5e-232c52ae0cdb_20` | de_nuke | T | 0.012751 | True |
| high_confidence_error | `63e0d98d-fd95-4698-9d12-808508005ee2 / online:1f4b1506-6f45-43ee-8b4c-82961eb4bd5b / online:1f4b1506-6f45-43ee-8b4c-82961eb4bd5b_2` | de_vertigo | T | 0.989381 | False |

### ct_high_probability

| 排名 | 编码列 | 值 | SHAP log-odds | 方向 |
|---:|---|---:|---:|---|
| 1 | `t_eq_value` | 1000.0 | 1.562062 | toward_ct |
| 2 | `first_kill_advantage_ct` | 1.0 | 0.865008 | toward_ct |
| 3 | `eq_value_diff_ct` | 25500.0 | 0.633398 | toward_ct |
| 4 | `helmet_diff_ct` | 4.0 | 0.478805 | toward_ct |
| 5 | `armor_diff_ct` | 5.0 | 0.356384 | toward_ct |

### t_high_probability

| 排名 | 编码列 | 值 | SHAP log-odds | 方向 |
|---:|---|---:|---:|---|
| 1 | `ct_eq_value` | 1200.0 | -1.352677 | toward_t |
| 2 | `first_kill_advantage_ct` | -1.0 | -1.038966 | toward_t |
| 3 | `eq_value_diff_ct` | -21450.0 | -0.930664 | toward_t |
| 4 | `ct_armor` | 0.0 | -0.392963 | toward_t |
| 5 | `armor_diff_ct` | -5.0 | -0.185167 | toward_t |

### high_confidence_error

| 排名 | 编码列 | 值 | SHAP log-odds | 方向 |
|---:|---|---:|---:|---|
| 1 | `t_eq_value` | 1100.0 | 1.525856 | toward_ct |
| 2 | `first_kill_advantage_ct` | 1.0 | 0.829556 | toward_ct |
| 3 | `eq_value_diff_ct` | 18850.0 | 0.628953 | toward_ct |
| 4 | `helmet_diff_ct` | 5.0 | 0.478805 | toward_ct |
| 5 | `armor_diff_ct` | 5.0 | 0.356915 | toward_ct |

## XGBoost 与逻辑回归

M16 没有为“XGBoost 必须领先逻辑回归”设置正式通过线，以下只报告差值。
`performance_advantage_xgboost` 为正表示 XGBoost 更好。

| 指标 | XGBoost | 逻辑回归 | 原始差 | 性能优势 |
|---|---:|---:|---:|---:|
| accuracy | 0.744125 | 0.743405 | +0.000719 | +0.000719 |
| auc | 0.809837 | 0.809059 | +0.000778 | +0.000778 |
| log_loss | 0.523146 | 0.526642 | -0.003496 | +0.003496 |
| brier_score | 0.175656 | 0.176070 | -0.000414 | +0.000414 |
| ece10 | 0.015450 | 0.015017 | +0.000433 | -0.000433 |

## 别人的指标与差值

差值为“本项目 - 外部报告”。Accuracy/AUC 同时显示百分点；低值更好的指标不能只看差值正负。

| 可比性 | 当前模型 | 外部来源 | 指标 | 当前 | 外部 | 差值 |
|---|---|---|---|---:|---:|---:|
| closest_task | `logistic_regression` | CS156 - Round-Win Probability in CS2 via Economic Asymmetry | accuracy | 0.743405 | 0.682400 | +6.10 百分点 |
| closest_task | `logistic_regression` | CS156 - Round-Win Probability in CS2 via Economic Asymmetry | auc | 0.809059 | 0.760000 | +4.91 百分点 |
| not_comparable | `xgboost_tuned` | Valuing Player Actions in Counter-Strike: Global Offensive | auc | 0.809837 | 0.791300 | +1.85 百分点 |
| not_comparable | `xgboost_tuned` | Valuing Player Actions in Counter-Strike: Global Offensive | log_loss | 0.523146 | 0.535300 | -0.012154 |
| not_comparable | `xgboost_tuned` | Valuing Player Actions in Counter-Strike: Global Offensive | brier_score | 0.175656 | 0.184200 | -0.008544 |
| partial | `xgboost_tuned` | Predicting the outcome of a round in CS:GO using a DNN | accuracy | 0.744125 | 0.679220 | +6.49 百分点 |
| partial | `xgboost_tuned` | Predicting the outcome of a round in CS:GO using a DNN | log_loss | 0.523146 | 0.567860 | -0.044714 |

外部数据、预测时点、特征和切分不同，不能把这些差值解释为算法同场排名。

## 下一阶段

M20 建立首杀后 JSON/CSV 单条预测接口和输入一致性校验；M21 做首杀后 XGBoost 最终验收。随后进行 LightGBM 同数据对照，再进入实时胜率。

运行命令：

```powershell
.\scripts\run_first_kill_explanation.ps1
```
