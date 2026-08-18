from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from .config import RANDOM_STATE


def probability_array(probability) -> np.ndarray:
    values = np.asarray(probability, dtype=float).reshape(-1)
    if len(values) == 0:
        raise ValueError("probability must not be empty")
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("probability values must be finite and between 0 and 1")
    return values


def logit_probability(probability) -> np.ndarray:
    values = np.clip(probability_array(probability), 1e-6, 1 - 1e-6)
    return np.log(values / (1 - values)).reshape(-1, 1)


class IdentityCalibrator:
    def fit(self, probability, y_true):
        values = probability_array(probability)
        if len(np.asarray(y_true).reshape(-1)) != len(values):
            raise ValueError("probability and y_true must have the same length")
        return self

    def predict(self, probability) -> np.ndarray:
        return probability_array(probability).copy()


class SigmoidCalibrator:
    def __init__(self) -> None:
        self.model = LogisticRegression(
            C=1_000_000,
            solver="lbfgs",
            random_state=RANDOM_STATE,
        )

    def fit(self, probability, y_true):
        self.model.fit(logit_probability(probability), np.asarray(y_true, dtype=int))
        return self

    def predict(self, probability) -> np.ndarray:
        return self.model.predict_proba(logit_probability(probability))[:, 1]


class IsotonicCalibrator:
    def __init__(self) -> None:
        self.model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)

    def fit(self, probability, y_true):
        self.model.fit(probability_array(probability), np.asarray(y_true, dtype=int))
        return self

    def predict(self, probability) -> np.ndarray:
        return np.asarray(
            self.model.predict(probability_array(probability)), dtype=float
        )


def make_calibrator(method: str):
    if method == "uncalibrated":
        return IdentityCalibrator()
    if method == "sigmoid":
        return SigmoidCalibrator()
    if method == "isotonic":
        return IsotonicCalibrator()
    raise ValueError(f"Unknown calibration method: {method}")
