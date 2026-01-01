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
) -> np.ndarray:
    idx = DIMENSIONS[dimension]
    values = labels[:, idx]

    if mode == "binary":
        return (values > threshold).astype(int)
    if mode == "quartile":
        bin_edges = np.asarray(list(bins))
        return np.digitize(values, bin_edges, right=False)
    raise ValueError(f"Unknown label mode: {mode}")


def build_quadrant_labels(
    labels: np.ndarray, threshold: float = 5.0
) -> np.ndarray:
    valence = labels[:, DIMENSIONS["valence"]] > threshold
    arousal = labels[:, DIMENSIONS["arousal"]] > threshold
    return (valence.astype(int) * 2) + arousal.astype(int)
