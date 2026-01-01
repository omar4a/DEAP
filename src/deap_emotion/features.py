from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np

BandSpec = Tuple[str, float, float]


def _bandpower_fft(
    eeg: np.ndarray,
    sfreq: int,
    bands: Iterable[BandSpec],
    mode: str,
) -> np.ndarray:
    """Compute per-channel bandpower features for each trial."""
    n_trials, n_channels, n_samples = eeg.shape
    fft = np.fft.rfft(eeg, axis=-1)
    psd = (np.abs(fft) ** 2) / n_samples
    freqs = np.fft.rfftfreq(n_samples, d=1.0 / sfreq)

    band_features = []
    for _, low, high in bands:
        mask = (freqs >= low) & (freqs < high)
        power = psd[:, :, mask].mean(axis=-1)
        band_features.append(power)

    features = np.stack(band_features, axis=-1)
    if mode == "bandpower_log":
        features = np.log(features + 1e-8)
    elif mode == "bandpower":
        pass
    else:
        raise ValueError(f"Unknown feature mode: {mode}")
    return features


def _flatten(features: np.ndarray) -> np.ndarray:
    return features.reshape(features.shape[0], -1)


def extract_features(
    eeg: np.ndarray,
    sfreq: int,
    bands: Iterable[BandSpec],
    mode: str,
    window_samples: int | None = None,
    step_samples: int | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return features and group ids (trial index per sample)."""
    n_trials, _, n_samples = eeg.shape
    if window_samples is None:
        feats = _bandpower_fft(eeg, sfreq, bands, mode)
        return _flatten(feats), np.arange(n_trials)

    if step_samples is None:
        step_samples = window_samples

    feature_list = []
    group_list = []
    for trial_idx in range(n_trials):
        start = 0
        while start + window_samples <= n_samples:
            window = eeg[trial_idx : trial_idx + 1, :, start : start + window_samples]
            feats = _bandpower_fft(window, sfreq, bands, mode)
            feature_list.append(_flatten(feats))
            group_list.append(trial_idx)
            start += step_samples

    features = np.concatenate(feature_list, axis=0)
    groups = np.asarray(group_list, dtype=int)
    return features, groups
