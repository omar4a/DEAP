"""Feature extraction module for EEG emotion recognition.

Includes features from high-accuracy papers:
- Paper 13: Differential Entropy (DE), Higuchi's Fractal Dimension (HFD)
- Paper 14: Statistical features (mean, std, skewness, kurtosis), band power
- Paper 11: Band power with feature selection
"""
from __future__ import annotations

from typing import Iterable, Sequence, Tuple

import numpy as np
from scipy.signal import hilbert
from scipy.stats import skew, kurtosis

BandSpec = Tuple[str, float, float]
FeatureSet = Sequence[str]

ALLOWED_FEATURES = {
    "bandpower",
    "rel_bandpower",
    "psd",
    "de",           # Gaussian-assumption DE (old method)
    "de_histogram", # Histogram-based DE (Paper 13 exact)
    "asymmetry",
    "connectivity",
    "connectivity_pcc",
    "connectivity_plv",
    "riemann",
    # New features from papers
    "statistical",  # mean, std, skewness, kurtosis (Paper 14)
    "hjorth",       # activity, mobility, complexity (Paper 14)
    "hfd",          # Higuchi's Fractal Dimension (Paper 13)
}


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


def _psd_fft(eeg: np.ndarray, sfreq: int) -> Tuple[np.ndarray, np.ndarray]:
    n_samples = eeg.shape[-1]
    fft = np.fft.rfft(eeg, axis=-1)
    psd = (np.abs(fft) ** 2) / n_samples
    freqs = np.fft.rfftfreq(n_samples, d=1.0 / sfreq)
    return freqs, psd


def _bandpower_from_psd(
    psd: np.ndarray, freqs: np.ndarray, bands: Iterable[BandSpec]
) -> np.ndarray:
    band_features = []
    for _, low, high in bands:
        mask = (freqs >= low) & (freqs < high)
        if not np.any(mask):
            power = np.zeros(psd.shape[:-1])
        else:
            power = psd[..., mask].mean(axis=-1)
        band_features.append(power)
    return np.stack(band_features, axis=-1)


def _psd_bins(
    psd: np.ndarray, freqs: np.ndarray, bands: Iterable[BandSpec]
) -> np.ndarray:
    bin_features = []
    for _, low, high in bands:
        mask = (freqs >= low) & (freqs < high)
        if not np.any(mask):
            power = np.zeros(psd.shape[:-1])
        else:
            power = psd[..., mask].mean(axis=-1)
        bin_features.append(power)
    return np.stack(bin_features, axis=-1)


def _differential_entropy_gaussian(bandpower: np.ndarray) -> np.ndarray:
    """Gaussian assumption DE (old method)."""
    return 0.5 * np.log(2 * np.pi * np.e * bandpower + 1e-8)


def _differential_entropy_histogram(eeg: np.ndarray, bins: int = 100) -> np.ndarray:
    """Histogram-based Differential Entropy per channel (Paper 13 exact).

    Formula: h(X) = -Σ p(xi) × log(p(xi)) × Δx

    Paper 13 achieved 89% valence, 88% arousal with this method.

    Args:
        eeg: Shape [trials, channels, samples]
        bins: Number of histogram bins

    Returns:
        Features with shape [trials, channels, 1]
    """
    n_trials, n_channels, _ = eeg.shape
    features = np.zeros((n_trials, n_channels, 1))

    for trial_idx in range(n_trials):
        for ch_idx in range(n_channels):
            signal = eeg[trial_idx, ch_idx]
            hist, bin_edges = np.histogram(signal, bins=bins, density=True)
            bin_width = bin_edges[1] - bin_edges[0]
            # Avoid log(0)
            hist = hist[hist > 0]
            if len(hist) > 0 and bin_width > 0:
                features[trial_idx, ch_idx, 0] = -np.sum(hist * np.log(hist) * bin_width)
            else:
                features[trial_idx, ch_idx, 0] = 0.0

    return features


def _differential_entropy(bandpower: np.ndarray) -> np.ndarray:
    """DE for backward compatibility (Gaussian assumption)."""
    return _differential_entropy_gaussian(bandpower)


def _relative_bandpower(bandpower: np.ndarray) -> np.ndarray:
    total = bandpower.sum(axis=-1, keepdims=True)
    return bandpower / (total + 1e-8)


def _asymmetry_features(
    features: np.ndarray, asymmetry_pairs: Sequence[Tuple[int, int]]
) -> np.ndarray:
    if not asymmetry_pairs:
        return np.empty((features.shape[0], 0, features.shape[2] * 2))
    left_idx = [pair[0] for pair in asymmetry_pairs]
    right_idx = [pair[1] for pair in asymmetry_pairs]
    left = features[:, left_idx, :]
    right = features[:, right_idx, :]
    diff = left - right
    ratio = left / (right + 1e-8)
    return np.concatenate([diff, ratio], axis=2)


def _band_signal_from_fft(
    fft: np.ndarray,
    freqs: np.ndarray,
    low: float,
    high: float,
    n_samples: int,
) -> np.ndarray:
    mask = (freqs >= low) & (freqs < high)
    if not np.any(mask):
        return np.zeros((fft.shape[0], fft.shape[1], n_samples))
    filtered_fft = fft * mask
    return np.fft.irfft(filtered_fft, n=n_samples, axis=-1)


def _connectivity_pcc_summary(
    eeg: np.ndarray,
    fft: np.ndarray,
    freqs: np.ndarray,
    bands: Iterable[BandSpec],
) -> np.ndarray:
    n_trials, n_channels, n_samples = eeg.shape
    upper_idx = np.triu_indices(n_channels, k=1)
    summaries = []
    for _, low, high in bands:
        band_signal = _band_signal_from_fft(fft, freqs, low, high, n_samples)
        band_stats = np.zeros((n_trials, 3))
        for trial_idx in range(n_trials):
            corr = np.corrcoef(band_signal[trial_idx])
            if corr.shape[0] < 2:
                continue
            tri = corr[upper_idx]
            tri = np.nan_to_num(tri, nan=0.0, posinf=0.0, neginf=0.0)
            if tri.size:
                band_stats[trial_idx, 0] = float(np.mean(tri))
                band_stats[trial_idx, 1] = float(np.std(tri))
                band_stats[trial_idx, 2] = float(np.mean(np.abs(tri)))
        summaries.append(band_stats)
    return np.concatenate(summaries, axis=1)


def _connectivity_plv_summary(
    eeg: np.ndarray,
    fft: np.ndarray,
    freqs: np.ndarray,
    bands: Iterable[BandSpec],
) -> np.ndarray:
    n_trials, n_channels, n_samples = eeg.shape
    upper_idx = np.triu_indices(n_channels, k=1)
    summaries = []
    for _, low, high in bands:
        band_signal = _band_signal_from_fft(fft, freqs, low, high, n_samples)
        band_stats = np.zeros((n_trials, 3))
        for trial_idx in range(n_trials):
            analytic = hilbert(band_signal[trial_idx], axis=-1)
            phases = np.angle(analytic)
            plv_vals = []
            for i, j in zip(upper_idx[0], upper_idx[1]):
                phase_diff = phases[i] - phases[j]
                plv = np.abs(np.mean(np.exp(1j * phase_diff)))
                plv_vals.append(plv)
            if plv_vals:
                values = np.asarray(plv_vals, dtype=float)
                band_stats[trial_idx, 0] = float(np.mean(values))
                band_stats[trial_idx, 1] = float(np.std(values))
                band_stats[trial_idx, 2] = float(np.mean(np.abs(values)))
        summaries.append(band_stats)
    return np.concatenate(summaries, axis=1)


def _covariance_features(eeg: np.ndarray, epsilon: float) -> np.ndarray:
    n_trials, n_channels, n_samples = eeg.shape
    tri = np.triu_indices(n_channels)
    features = np.zeros((n_trials, len(tri[0])), dtype=float)
    for idx in range(n_trials):
        signal = eeg[idx]
        signal = signal - signal.mean(axis=1, keepdims=True)
        cov = (signal @ signal.T) / max(n_samples - 1, 1)
        cov = cov + np.eye(n_channels) * epsilon
        features[idx] = cov[tri]
    return features


# =============================================================================
# NEW FEATURES FROM HIGH-ACCURACY PAPERS
# =============================================================================


def _statistical_features(eeg: np.ndarray) -> np.ndarray:
    """Compute statistical features per channel (Paper 14).

    Features: mean, std, skewness, kurtosis
    Achieved 91% valence, 89% arousal in Paper 14.

    Args:
        eeg: Shape [trials, channels, samples]

    Returns:
        Features with shape [trials, channels, 4]
    """
    n_trials, n_channels, _ = eeg.shape
    features = np.zeros((n_trials, n_channels, 4))

    features[:, :, 0] = np.mean(eeg, axis=-1)
    features[:, :, 1] = np.std(eeg, axis=-1)
    features[:, :, 2] = skew(eeg, axis=-1)
    features[:, :, 3] = kurtosis(eeg, axis=-1)

    return features


def _hjorth_parameters(eeg: np.ndarray) -> np.ndarray:
    """Compute Hjorth parameters per channel (Paper 14).

    Features:
    - Activity: variance of the signal (signal power)
    - Mobility: std of first derivative / std of signal
    - Complexity: mobility of first derivative / mobility of signal

    Args:
        eeg: Shape [trials, channels, samples]

    Returns:
        Features with shape [trials, channels, 3]
    """
    n_trials, n_channels, _ = eeg.shape
    features = np.zeros((n_trials, n_channels, 3))

    # First derivative (difference)
    diff1 = np.diff(eeg, axis=-1)
    # Second derivative
    diff2 = np.diff(diff1, axis=-1)

    # Activity: variance of signal
    var_signal = np.var(eeg, axis=-1)
    features[:, :, 0] = var_signal

    # Mobility: sqrt(var(diff1) / var(signal))
    var_diff1 = np.var(diff1, axis=-1)
    mobility = np.sqrt(var_diff1 / (var_signal + 1e-8))
    features[:, :, 1] = mobility

    # Complexity: mobility(diff1) / mobility(signal)
    var_diff2 = np.var(diff2, axis=-1)
    mobility_diff1 = np.sqrt(var_diff2 / (var_diff1 + 1e-8))
    features[:, :, 2] = mobility_diff1 / (mobility + 1e-8)

    return features


def _higuchi_fd_single(signal: np.ndarray, kmax: int = 10) -> float:
    """Compute Higuchi's Fractal Dimension for a single 1D signal.

    Based on Paper 13 which achieved 89% valence, 88% arousal.

    Args:
        signal: 1D array of signal values
        kmax: Maximum k value (default 10)

    Returns:
        Fractal dimension value
    """
    n = len(signal)
    if n < kmax + 1:
        kmax = max(1, n - 1)

    # Length of the curve for each k
    lk = np.zeros(kmax)

    for k in range(1, kmax + 1):
        # Number of subsequences for this k
        lm_sum = 0.0
        for m in range(1, k + 1):
            # Indices for this subsequence
            idx = np.arange(m - 1, n, k)
            if len(idx) < 2:
                continue

            # Compute length for this subsequence
            lm = np.sum(np.abs(np.diff(signal[idx])))
            # Normalize
            lm = lm * (n - 1) / (k * ((n - m) // k) * k)
            lm_sum += lm

        lk[k - 1] = lm_sum / k

    # Fit line to log-log plot
    lk = lk[lk > 0]  # Remove zeros
    if len(lk) < 2:
        return 1.0  # Default value

    x = np.log(1.0 / np.arange(1, len(lk) + 1))
    y = np.log(lk)

    # Linear regression
    slope, _ = np.polyfit(x, y, 1)
    return slope


def _higuchi_fd(eeg: np.ndarray, kmax: int = 10) -> np.ndarray:
    """Compute Higuchi's Fractal Dimension per channel (Paper 13).

    Achieved 89% valence, 88% arousal in Paper 13.

    Args:
        eeg: Shape [trials, channels, samples]
        kmax: Maximum k value for HFD computation

    Returns:
        Features with shape [trials, channels, 1]
    """
    n_trials, n_channels, _ = eeg.shape
    features = np.zeros((n_trials, n_channels, 1))

    for trial_idx in range(n_trials):
        for ch_idx in range(n_channels):
            features[trial_idx, ch_idx, 0] = _higuchi_fd_single(
                eeg[trial_idx, ch_idx], kmax
            )

    return features


def _normalize_feature_set(feature_set: FeatureSet) -> Tuple[str, ...]:
    normalized = tuple(name.lower() for name in feature_set)
    unknown = set(normalized) - ALLOWED_FEATURES
    if unknown:
        raise ValueError(f"Unknown feature names: {sorted(unknown)}")
    return normalized


def _extract_feature_matrix(
    eeg: np.ndarray,
    sfreq: int,
    bands: Iterable[BandSpec],
    mode: str,
    feature_set: FeatureSet,
    psd_bands: Iterable[BandSpec],
    asymmetry_pairs: Sequence[Tuple[int, int]],
    riemann_epsilon: float,
) -> np.ndarray:
    n_samples = eeg.shape[-1]
    fft = np.fft.rfft(eeg, axis=-1)
    psd = (np.abs(fft) ** 2) / n_samples
    freqs = np.fft.rfftfreq(n_samples, d=1.0 / sfreq)

    needs_bandpower = any(
        name in feature_set for name in ("bandpower", "rel_bandpower", "de", "asymmetry")
    )
    bandpower = None
    bandpower_log = None
    de_features = None
    if needs_bandpower:
        bandpower = _bandpower_from_psd(psd, freqs, bands)
        if mode == "bandpower_log":
            bandpower_log = np.log(bandpower + 1e-8)
        elif mode == "bandpower":
            bandpower_log = None
        else:
            raise ValueError(f"Unknown feature mode: {mode}")
    if "de" in feature_set or "asymmetry" in feature_set:
        if bandpower is None:
            raise ValueError("Bandpower required for DE features.")
        de_features = _differential_entropy(bandpower)

    feature_blocks = []
    if "riemann" in feature_set:
        feature_blocks.append(_covariance_features(eeg, riemann_epsilon))
    if "bandpower" in feature_set:
        if mode == "bandpower_log":
            feature_blocks.append(bandpower_log)
        else:
            feature_blocks.append(bandpower)
    if "rel_bandpower" in feature_set:
        if bandpower is None:
            raise ValueError("Bandpower required for relative bandpower features.")
        feature_blocks.append(_relative_bandpower(bandpower))
    if "psd" in feature_set:
        feature_blocks.append(_psd_bins(psd, freqs, psd_bands))
    if "de" in feature_set:
        if de_features is None:
            raise ValueError("Bandpower required for DE features.")
        feature_blocks.append(de_features)
    if "de_histogram" in feature_set:
        feature_blocks.append(_differential_entropy_histogram(eeg))
    if "asymmetry" in feature_set:
        if de_features is None:
            raise ValueError("DE required for asymmetry features.")
        feature_blocks.append(_asymmetry_features(de_features, asymmetry_pairs))
    if "connectivity" in feature_set or "connectivity_pcc" in feature_set:
        feature_blocks.append(_connectivity_pcc_summary(eeg, fft, freqs, bands))
    if "connectivity_plv" in feature_set:
        feature_blocks.append(_connectivity_plv_summary(eeg, fft, freqs, bands))

    # New features from high-accuracy papers
    if "statistical" in feature_set:
        feature_blocks.append(_statistical_features(eeg))
    if "hjorth" in feature_set:
        feature_blocks.append(_hjorth_parameters(eeg))
    if "hfd" in feature_set:
        feature_blocks.append(_higuchi_fd(eeg))

    if not feature_blocks:
        raise ValueError("No features selected.")

    flat_blocks = [block.reshape(block.shape[0], -1) for block in feature_blocks]
    return np.concatenate(flat_blocks, axis=1)


def extract_features(
    eeg: np.ndarray,
    sfreq: int,
    bands: Iterable[BandSpec],
    mode: str,
    window_samples: int | None = None,
    step_samples: int | None = None,
    feature_set: FeatureSet | None = None,
    psd_bands: Iterable[BandSpec] | None = None,
    asymmetry_pairs: Sequence[Tuple[int, int]] | None = None,
    riemann_epsilon: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return features and group ids (trial index per sample)."""
    n_trials, _, n_samples = eeg.shape
    if feature_set is not None:
        feature_set = _normalize_feature_set(feature_set)
        if "psd" in feature_set and psd_bands is None:
            raise ValueError("PSD bands must be provided when using PSD features.")
        if asymmetry_pairs is None:
            asymmetry_pairs = []
    if window_samples is None:
        if feature_set is None:
            feats = _bandpower_fft(eeg, sfreq, bands, mode)
            return _flatten(feats), np.arange(n_trials)
        else:
            feats = _extract_feature_matrix(
                eeg,
                sfreq,
                bands,
                mode,
                feature_set,
                psd_bands,
                asymmetry_pairs,
                riemann_epsilon,
            )
            return feats, np.arange(n_trials)

    if step_samples is None:
        step_samples = window_samples

    feature_list = []
    group_list = []
    for trial_idx in range(n_trials):
        start = 0
        while start + window_samples <= n_samples:
            window = eeg[trial_idx : trial_idx + 1, :, start : start + window_samples]
            if feature_set is None:
                feats = _bandpower_fft(window, sfreq, bands, mode)
                feature_list.append(_flatten(feats))
            else:
                feats = _extract_feature_matrix(
                    window,
                    sfreq,
                    bands,
                    mode,
                    feature_set,
                    psd_bands,
                    asymmetry_pairs,
                    riemann_epsilon,
                )
                feature_list.append(feats)
            group_list.append(trial_idx)
            start += step_samples

    features = np.concatenate(feature_list, axis=0)
    groups = np.asarray(group_list, dtype=int)
    return features, groups
