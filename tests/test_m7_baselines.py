import unittest

import numpy as np
import pandas as pd

from src.csdemo.m7_baselines import make_constant_model, make_logistic_model
from src.csdemo.metrics import probability_metrics


class M7BaselineTests(unittest.TestCase):
    def test_probability_metrics_include_calibration_metrics(self) -> None:
        y_true = pd.Series([0, 0, 1, 1])
        probability = np.array([0.1, 0.2, 0.8, 0.9])

        metrics = probability_metrics(y_true, probability, n_bins=10)

        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["auc"], 1.0)
        self.assertAlmostEqual(metrics["brier_score"], 0.025)
        self.assertAlmostEqual(metrics["ece10"], 0.15)

    def test_constant_model_predicts_training_ct_win_rate(self) -> None:
        x_train = pd.DataFrame({"feature": [0, 1, 2, 3]})
        y_train = pd.Series([1, 0, 1, 1])
        model = make_constant_model().fit(x_train, y_train)

        probability = model.predict_proba(pd.DataFrame({"feature": [8, 9]}))[:, 1]

        np.testing.assert_allclose(probability, [0.75, 0.75])

    def test_logistic_model_is_scaled_and_reproducible(self) -> None:
        model = make_logistic_model()

        self.assertEqual(list(model.named_steps), ["scaler", "classifier"])
        self.assertEqual(model.named_steps["classifier"].random_state, 42)
        self.assertGreaterEqual(model.named_steps["classifier"].max_iter, 2000)


if __name__ == "__main__":
    unittest.main()
