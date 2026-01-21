from __future__ import annotations

from typing import Iterable

import numpy as np

DIMENSIONS = {
    "valence": 0,
    "arousal": 1,
    "dominance": 2,
    "liking": 3,
}


def build_labels(
    labels: np.ndarray,
    dimension: str,
    mode: str,
    threshold: float,
    bins: Iterable[float],
    train_trial_ids: Iterable[int] | None = None,
) -> np.ndarray:
    idx = DIMENSIONS[dimension]
    values = labels[:, idx]

    if mode == "binary":
        return (values > threshold).astype(int)
    if mode == "three_class":
        bin_edges = np.asarray(list(bins))
        if bin_edges.size < 2:
            raise ValueError("three_class mode requires at least two bin edges.")
        edges = bin_edges[:2]
        return np.digitize(values, edges, right=False)
    if mode == "subject_median":
        median = float(np.median(values))
        return (values > median).astype(int)
    if mode == "subject_mean":
        if train_trial_ids is None:
            raise ValueError("subject_mean requires train_trial_ids.")
        train_ids = np.asarray(list(train_trial_ids), dtype=int)
        if train_ids.size == 0:
            raise ValueError("subject_mean requires non-empty train_trial_ids.")
        mean = float(np.mean(values[train_ids]))
        return (values > mean).astype(int)
    if mode == "subject_tertile":
        min_val = float(np.min(values))
        max_val = float(np.max(values))
        if max_val <= min_val:
            return np.zeros_like(values, dtype=int)
        step = (max_val - min_val) / 3.0
        edges = np.array([min_val + step, min_val + 2 * step])
        return np.digitize(values, edges, right=False)
    if mode == "quartile":
        bin_edges = np.asarray(list(bins))
        return np.digitize(values, bin_edges, right=False)
    if mode == "regression":
        return values.astype(float)
    raise ValueError(f"Unknown label mode: {mode}")


def build_quadrant_labels(
    labels: np.ndarray, threshold: float = 5.0
) -> np.ndarray:
    valence = labels[:, DIMENSIONS["valence"]] > threshold
    arousal = labels[:, DIMENSIONS["arousal"]] > threshold
    return (valence.astype(int) * 2) + arousal.astype(int)
