import unittest

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from src.csdemo.m12_explanation import (
    audit_model_features,
    build_importance_comparison,
    build_case_explanations,
    deployment_tree_count,
    gain_importance,
    permutation_auc_importance,
    select_explanation_cases,
    shap_importance,
    tree_shap_contributions,
)


class M12ExplanationTests(unittest.TestCase):
    def test_tree_shap_reconstructs_early_stopped_model_probability(self) -> None:
        rng = np.random.default_rng(7)
        x_train = pd.DataFrame(rng.normal(size=(160, 3)), columns=["a", "b", "c"])
        y_train = (x_train["a"] + 0.4 * x_train["b"] > 0).astype(int)
        x_val = pd.DataFrame(rng.normal(size=(80, 3)), columns=x_train.columns)
        y_val = 1 - (x_val["a"] + 0.4 * x_val["b"] > 0).astype(int)
        model = XGBClassifier(
            n_estimators=60,
            max_depth=2,
            learning_rate=0.15,
            objective="binary:logistic",
            eval_metric="logloss",
            early_stopping_rounds=4,
            random_state=7,
            n_jobs=1,
        )
        model.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)
        bundle = {
            "model": model,
            "columns": list(x_train.columns),
            "best_iteration": model.best_iteration,
        }

        shap_values, base_values = tree_shap_contributions(bundle, x_val.iloc[:12])
        reconstructed = 1.0 / (
            1.0 + np.exp(-(base_values + shap_values.sum(axis=1).to_numpy()))
        )
        deployed = model.predict_proba(x_val.iloc[:12])[:, 1]

        self.assertLess(model.best_iteration + 1, len(model.get_booster().get_dump()))
        self.assertEqual(deployment_tree_count(bundle), model.best_iteration + 1)
        np.testing.assert_allclose(reconstructed, deployed, atol=1e-6)

        gain = gain_importance(bundle)
        self.assertEqual(gain["deployment_tree_count"].unique().tolist(), [model.best_iteration + 1])
        self.assertAlmostEqual(gain["gain_normalized"].sum(), 1.0)

    def test_feature_audit_allows_pre_round_defuse_kits_and_flags_leakage(self) -> None:
        audit = audit_model_features(
            [
                "ct_defuse_kits",
                "map_name_de_nuke",
                "series_id",
                "first_kill_is_ct",
            ]
        ).set_index("feature")

        self.assertEqual(audit.loc["ct_defuse_kits", "audit_result"], "pass")
        self.assertEqual(audit.loc["map_name_de_nuke", "audit_result"], "pass")
        self.assertEqual(audit.loc["series_id", "reason"], "identifier")
        self.assertEqual(audit.loc["first_kill_is_ct", "reason"], "future_information")

    def test_case_selection_returns_ct_t_and_wrong_examples(self) -> None:
        predictions = pd.DataFrame(
            {
                "round_id": ["ct", "t", "wrong", "other"],
                "y_true": [1, 0, 0, 1],
                "ct_win_probability": [0.91, 0.08, 0.86, 0.55],
            }
        )

        selected = select_explanation_cases(predictions).set_index("case_type")

        self.assertEqual(
            set(selected.index),
            {"ct_high_probability", "t_high_probability", "high_confidence_error"},
        )
        self.assertEqual(selected.loc["ct_high_probability", "round_id"], "ct")
        self.assertEqual(selected.loc["t_high_probability", "round_id"], "t")
        self.assertEqual(selected.loc["high_confidence_error", "round_id"], "wrong")

    def test_shap_importance_uses_mean_absolute_contribution(self) -> None:
        shap_values = pd.DataFrame(
            {
                "small": [0.1, -0.1, 0.1],
                "large": [0.8, -0.6, 0.7],
            }
        )

        importance = shap_importance(shap_values)

        self.assertEqual(importance.iloc[0]["feature"], "large")
        self.assertAlmostEqual(importance.iloc[0]["mean_abs_shap"], 0.7)

    def test_permutation_auc_importance_ranks_signal_above_noise(self) -> None:
        rng = np.random.default_rng(11)
        x = pd.DataFrame(
            {"signal": rng.normal(size=400), "noise": rng.normal(size=400)}
        )
        y = (x["signal"] > 0).astype(int)
        model = LogisticRegression().fit(x, y)

        importance = permutation_auc_importance(
            model, x, y, n_repeats=5, seed=11
        ).set_index("feature")

        self.assertGreater(
            importance.loc["signal", "auc_decrease_mean"],
            importance.loc["noise", "auc_decrease_mean"],
        )

    def test_case_explanations_keep_top_contributions_and_probability(self) -> None:
        predictions = pd.DataFrame(
            {
                "round_id": ["ct", "t", "wrong", "other"],
                "y_true": [1, 0, 0, 1],
                "ct_win_probability": [0.91, 0.08, 0.86, 0.55],
            }
        )
        cases = select_explanation_cases(predictions)
        x = pd.DataFrame(
            {"strong": [2.0, -2.0, 1.5, 0.1], "small": [0.2, -0.1, 0.3, 0.0]}
        )
        shap_values = pd.DataFrame(
            {"strong": [1.7, -1.8, 1.4, 0.1], "small": [0.2, -0.1, 0.3, 0.0]}
        )
        base_values = np.zeros(len(x))

        explanations = build_case_explanations(
            cases, x, shap_values, base_values, top_n=2
        )

        self.assertEqual(len(explanations), 6)
        self.assertTrue((explanations.groupby("case_type")["contribution_rank"].max() == 2).all())
        self.assertEqual(
            explanations[explanations["feature"].eq("strong")]["direction"].tolist(),
            ["toward_ct", "toward_t", "toward_ct"],
        )
        self.assertTrue(explanations["reconstructed_ct_probability"].between(0, 1).all())

    def test_importance_comparison_keeps_each_methods_native_units(self) -> None:
        gain = pd.DataFrame(
            {
                "gain_rank": [1, 2],
                "feature": ["economy", "score"],
                "gain_normalized": [0.7, 0.3],
            }
        )
        permutation = pd.DataFrame(
            {
                "permutation_rank": [1, 2],
                "feature": ["score", "economy"],
                "auc_decrease_mean": [0.08, 0.04],
                "auc_decrease_std": [0.01, 0.02],
            }
        )
        shap = pd.DataFrame(
            {
                "shap_rank": [1, 2],
                "feature": ["economy", "score"],
                "mean_abs_shap": [0.6, 0.2],
            }
        )

        comparison = build_importance_comparison(gain, permutation, shap).set_index(
            "feature"
        )

        self.assertAlmostEqual(comparison.loc["economy", "gain_normalized"], 0.7)
        self.assertAlmostEqual(comparison.loc["score", "auc_decrease_mean"], 0.08)
        self.assertAlmostEqual(comparison.loc["economy", "mean_abs_shap"], 0.6)
        self.assertAlmostEqual(comparison.loc["economy", "mean_rank"], 4 / 3)


if __name__ == "__main__":
    unittest.main()
