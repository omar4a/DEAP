"""Differential Entropy (DE) feature extraction from DEAP .dat files.

DE for a segment assumed Gaussian: DE = 0.5 * log(2*pi*e * var(x))
Applied per channel, per 1-second window, per frequency band.

Output per subject: de_features (40, 60, 32, 5)  [trials, time, channels, bands]
                    labels      (40, 2)            [valence, arousal] — binary
"""
import os
import pickle

import numpy as np
import scipy.signal

import config

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(BASE_DIR, 'results', 'eegnet_de')


# ── Data directory discovery ──────────────────────────────────────────────────

def _find_data_dir() -> str:
    """Search common locations for DEAP .dat files."""
    candidates = [
        os.path.join(BASE_DIR, '..', 'eeg_dl_experiment', 'data'),
        os.path.join(BASE_DIR, 'data'),
        os.path.join(BASE_DIR, '..', 'archive'),
    ]
    for d in candidates:
        d = os.path.normpath(d)
        if os.path.isdir(d) and any(f.endswith('.dat') for f in os.listdir(d)):
            return d
    raise FileNotFoundError(
        f"No DEAP .dat files found. Searched: {[os.path.normpath(c) for c in candidates]}"
    )


# ── Signal processing helpers ─────────────────────────────────────────────────

def _bandpass(data: np.ndarray, lowcut: float, highcut: float,
              fs: int = 128, order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth bandpass filter, applied along last axis."""
    sos = scipy.signal.butter(order, [lowcut, highcut], btype='band',
                              fs=fs, output='sos')
    return scipy.signal.sosfiltfilt(sos, data, axis=-1)


def _compute_de(segment: np.ndarray) -> np.ndarray:
    """DE per channel for one segment.

    Args:
        segment: (n_channels, n_samples)
    Returns:
        de: (n_channels,)   — DE = 0.5 * log(2*pi*e * var)
    """
    var = np.var(segment, axis=-1)          # (n_channels,)
    return 0.5 * np.log(2.0 * np.pi * np.e * var)


# ── Main extraction function ──────────────────────────────────────────────────

def extract_de_features(subject_id: int, data_dir: str = None,
                        force_recompute: bool = False) -> dict:
    """Load DEAP .dat, extract DE features, normalise, binarise labels, cache.

    Returns:
        {
          'de_features': float32 array (40, 60, 32, 5)
                         axes: [trial, time_step, channel, band]
          'labels':      int64  array (40, 2)
                         axes: [trial, (valence, arousal)]
        }
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    feat_path  = os.path.join(CACHE_DIR, f'de_features_s{subject_id:02d}.npy')
    label_path = os.path.join(CACHE_DIR, f'labels_s{subject_id:02d}.npy')

    if not force_recompute and os.path.exists(feat_path) and os.path.exists(label_path):
        return {
            'de_features': np.load(feat_path),
            'labels':      np.load(label_path),
        }

    if data_dir is None:
        data_dir = _find_data_dir()

    # ── Load raw data ──
    fname = os.path.join(data_dir, f's{subject_id:02d}.dat')
    with open(fname, 'rb') as f:
        subject = pickle.load(f, encoding='latin1')

    data       = subject['data']    # (40, 40, 8064)
    raw_labels = subject['labels']  # (40, 4)

    # Keep EEG channels only
    data = data[:, :config.N_CHANNELS, :]   # (40, 32, 8064)

    # Crop 3-second baseline (384 samples @ 128 Hz)
    data = data[:, :, 384:]                 # (40, 32, 7680)

    n_trials  = data.shape[0]                               # 40
    n_windows = data.shape[-1] // config.DE_WINDOW_SAMPLES  # 60
    bands     = list(config.FREQ_BANDS.values())            # 5 × (low, high)

    de_all = np.zeros(
        (n_trials, n_windows, config.N_CHANNELS, config.N_BANDS),
        dtype=np.float32
    )

    for t in range(n_trials):
        trial = data[t]  # (32, 7680)
        for b_idx, (low, high) in enumerate(bands):
            filtered = _bandpass(trial, low, high)  # (32, 7680)
            for w in range(n_windows):
                start = w * config.DE_WINDOW_SAMPLES
                seg   = filtered[:, start:start + config.DE_WINDOW_SAMPLES]  # (32, 128)
                de_all[t, w, :, b_idx] = _compute_de(seg)                    # (32,)

    # ── Per-subject z-score over (trials × time) for each (channel, band) ──
    # de_all: (40, 60, 32, 5)  →  mean/std over axes (0, 1)  →  shape (32, 5)
    mean = de_all.mean(axis=(0, 1), keepdims=True)   # (1, 1, 32, 5)
    std  = de_all.std( axis=(0, 1), keepdims=True)   # (1, 1, 32, 5)
    std  = np.where(std < 1e-8, 1e-8, std)
    de_all = ((de_all - mean) / std).astype(np.float32)

    # ── Binary labels: threshold at 5.0 (strict >) ──
    labels = np.stack([
        (raw_labels[:, 0] > config.LABEL_THRESHOLD).astype(np.int64),  # valence
        (raw_labels[:, 1] > config.LABEL_THRESHOLD).astype(np.int64),  # arousal
    ], axis=1)   # (40, 2)

    np.save(feat_path,  de_all)
    np.save(label_path, labels)

    print(f"  s{subject_id:02d}: DE features {de_all.shape}, "
          f"labels {labels.shape}  "
          f"[val pos={labels[:,0].mean():.2f}, aro pos={labels[:,1].mean():.2f}]")

    return {'de_features': de_all, 'labels': labels}
