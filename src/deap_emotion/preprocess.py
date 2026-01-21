"""Preprocessing module for EEG data.

Based on Paper 13 methodology:
- Bandpass filter: 4-45 Hz (Butterworth 4th order)
- Notch filter: 50 Hz (to remove power line interference)
- Baseline correction
- Segmentation (splits 60s trials into smaller windows)
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch


def baseline_correct(eeg: np.ndarray, baseline_samples: int) -> np.ndarray:
    """Subtract the per-channel baseline mean from each trial."""
    if baseline_samples <= 0:
        return eeg
    baseline = eeg[:, :, :baseline_samples].mean(axis=2, keepdims=True)
    return eeg - baseline


def butter_bandpass(
    lowcut: float,
    highcut: float,
    fs: float,
    order: int = 4,
) -> Tuple[np.ndarray, np.ndarray]:
    """Design a Butterworth bandpass filter."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return b, a


def notch_filter(
    freq: float,
    fs: float,
    quality: float = 30.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Design a notch filter to remove power line interference."""
    b, a = iirnotch(freq, quality, fs)
    return b, a


def apply_bandpass(
    eeg: np.ndarray,
    lowcut: float,
    highcut: float,
    fs: float,
    order: int = 4,
) -> np.ndarray:
    """Apply bandpass filter to EEG data.

    Args:
        eeg: EEG data with shape [trials, channels, samples] or [channels, samples]
        lowcut: Low cutoff frequency (Hz)
        highcut: High cutoff frequency (Hz)
        fs: Sampling frequency (Hz)
        order: Filter order (default 4, as in Paper 13)

    Returns:
        Filtered EEG data with same shape as input
    """
    b, a = butter_bandpass(lowcut, highcut, fs, order)
    return filtfilt(b, a, eeg, axis=-1)


def apply_notch(
    eeg: np.ndarray,
    freq: float,
    fs: float,
    quality: float = 30.0,
) -> np.ndarray:
    """Apply notch filter to remove power line interference.

    Args:
        eeg: EEG data with shape [trials, channels, samples] or [channels, samples]
        freq: Frequency to remove (typically 50 Hz or 60 Hz)
        fs: Sampling frequency (Hz)
        quality: Quality factor (higher = narrower notch)

    Returns:
        Filtered EEG data with same shape as input
    """
    b, a = notch_filter(freq, fs, quality)
    return filtfilt(b, a, eeg, axis=-1)


def preprocess_eeg(
    eeg: np.ndarray,
    fs: float,
    bandpass_low: float = 4.0,
    bandpass_high: float = 45.0,
    bandpass_order: int = 4,
    notch_freq: float | None = 50.0,
    notch_quality: float = 30.0,
    baseline_samples: int | None = None,
) -> np.ndarray:
    """Full preprocessing pipeline for EEG data.

    Based on Paper 13 methodology:
    1. Bandpass filter 4-45 Hz (Butterworth 4th order)
    2. Notch filter 50 Hz (optional, for power line interference)
    3. Baseline correction (optional, removes pre-stimulus baseline)

    Args:
        eeg: EEG data with shape [trials, channels, samples]
        fs: Sampling frequency (Hz)
        bandpass_low: Low cutoff for bandpass (default 4 Hz)
        bandpass_high: High cutoff for bandpass (default 45 Hz)
        bandpass_order: Butterworth filter order (default 4)
        notch_freq: Notch filter frequency (default 50 Hz, None to skip)
        notch_quality: Notch filter quality factor
        baseline_samples: Number of samples for baseline correction (None to skip)

    Returns:
        Preprocessed EEG data with shape [trials, channels, post_baseline_samples]
    """
    # Step 1: Bandpass filter
    filtered = apply_bandpass(eeg, bandpass_low, bandpass_high, fs, bandpass_order)

    # Step 2: Notch filter (optional)
    if notch_freq is not None:
        filtered = apply_notch(filtered, notch_freq, fs, notch_quality)

    # Step 3: Baseline correction (optional)
    if baseline_samples is not None and baseline_samples > 0:
        baseline = filtered[:, :, :baseline_samples].mean(axis=-1, keepdims=True)
        filtered = filtered[:, :, baseline_samples:] - baseline

    return filtered


def segment_trials(
    eeg: np.ndarray,
    segment_samples: int,
    overlap_samples: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Segment EEG trials into smaller windows.

    This is a critical technique from Paper 13 that increases sample count
    from ~1280 (32 subjects × 40 trials) to ~25,600 (with 20 segments per trial).

    Args:
        eeg: EEG data with shape [trials, channels, samples]
        segment_samples: Number of samples per segment
        overlap_samples: Number of overlapping samples between segments

    Returns:
        segments: Segmented data with shape [n_segments, channels, segment_samples]
        trial_ids: Trial index for each segment (for GroupKFold)
    """
    n_trials, n_channels, n_samples = eeg.shape
    step = segment_samples - overlap_samples

    if step <= 0:
        raise ValueError("overlap_samples must be less than segment_samples")

    segments = []
    trial_ids = []

    for trial_idx in range(n_trials):
        start = 0
        while start + segment_samples <= n_samples:
            segment = eeg[trial_idx, :, start:start + segment_samples]
            segments.append(segment)
            trial_ids.append(trial_idx)
            start += step

    return np.array(segments), np.array(trial_ids)