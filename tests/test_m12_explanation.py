import unittest

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from src.csdemo.m12_explanation import (
    audit_model_features,
    deployment_tree_count,
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


if __name__ == "__main__":
    unittest.main()
