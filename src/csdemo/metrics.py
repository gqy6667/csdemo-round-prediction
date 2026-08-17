from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score


def expected_calibration_error(
    y_true, probability, *, n_bins: int = 10
) -> float:
    if n_bins < 1:
        raise ValueError("n_bins must be at least 1")

    y = np.asarray(y_true, dtype=int).reshape(-1)
    proba = np.asarray(probability, dtype=float).reshape(-1)
    if len(y) == 0 or len(y) != len(proba):
        raise ValueError("y_true and probability must have the same non-zero length")
    if not np.isfinite(proba).all() or ((proba < 0) | (proba > 1)).any():
        raise ValueError("probability values must be finite and between 0 and 1")

    bin_ids = np.minimum((proba * n_bins).astype(int), n_bins - 1)
    error = 0.0
    for bin_id in range(n_bins):
        mask = bin_ids == bin_id
        if mask.any():
            error += mask.mean() * abs(proba[mask].mean() - y[mask].mean())
    return float(error)


def probability_metrics(y_true, probability, *, n_bins: int = 10) -> dict[str, float]:
    y = np.asarray(y_true, dtype=int).reshape(-1)
    proba = np.asarray(probability, dtype=float).reshape(-1)
    pred = (proba >= 0.5).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y, pred)),
        "log_loss": float(log_loss(y, proba, labels=[0, 1])),
        "brier_score": float(brier_score_loss(y, proba)),
        "ece10": expected_calibration_error(y, proba, n_bins=n_bins),
    }
    metrics["auc"] = float(roc_auc_score(y, proba)) if np.unique(y).size == 2 else float("nan")
    return metrics
