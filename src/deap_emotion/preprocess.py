from __future__ import annotations

import numpy as np


def baseline_correct(eeg: np.ndarray, baseline_samples: int) -> np.ndarray:
    """Subtract the per-channel baseline mean from each trial."""
    if baseline_samples <= 0:
        return eeg
    baseline = eeg[:, :, :baseline_samples].mean(axis=2, keepdims=True)
    return eeg - baseline
