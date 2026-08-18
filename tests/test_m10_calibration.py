import unittest

import numpy as np
import pandas as pd

from src.csdemo.m10_calibration import (
    IdentityCalibrator,
    SigmoidCalibrator,
    group_fold_assignments,
    select_calibration_method,
)


class M10CalibrationTests(unittest.TestCase):
    def test_persisted_calibrators_use_a_stable_import_module(self) -> None:
        self.assertEqual(IdentityCalibrator.__module__, "src.csdemo.calibration")

    def test_identity_calibrator_does_not_change_probabilities(self) -> None:
        probability = np.array([0.05, 0.25, 0.75, 0.95])
        calibrated = IdentityCalibrator().fit(probability, [0, 0, 1, 1]).predict(
            probability
        )

        np.testing.assert_allclose(calibrated, probability)

    def test_sigmoid_calibrator_outputs_monotonic_valid_probabilities(self) -> None:
        probability = np.array([0.05, 0.20, 0.40, 0.60, 0.80, 0.95])
        labels = np.array([0, 0, 0, 1, 1, 1])
        calibrator = SigmoidCalibrator().fit(probability, labels)

        calibrated = calibrator.predict(probability)

        self.assertTrue(((calibrated >= 0) & (calibrated <= 1)).all())
        self.assertTrue((np.diff(calibrated) >= 0).all())

    def test_group_fold_assignments_keep_each_series_in_one_fold(self) -> None:
        groups = np.repeat(["s1", "s2", "s3", "s4", "s5", "s6"], 3)

        folds = group_fold_assignments(groups, n_splits=3)

        table = pd.DataFrame({"series_id": groups, "fold": folds})
        self.assertTrue((table.groupby("series_id")["fold"].nunique() == 1).all())
        self.assertEqual(set(folds), {0, 1, 2})

    def test_method_selection_uses_lowest_validation_oof_log_loss(self) -> None:
        comparison = pd.DataFrame(
            {
                "method": ["uncalibrated", "sigmoid", "isotonic"],
                "log_loss": [0.600, 0.590, 0.595],
                "brier_score": [0.210, 0.205, 0.204],
            }
        )

        self.assertEqual(select_calibration_method(comparison), "sigmoid")


if __name__ == "__main__":
    unittest.main()
