"""DEAP data loading, preprocessing, and DataLoader creation."""
import os
import json
import pickle

import numpy as np
import scipy.signal
import torch
from torch.utils.data import Dataset, DataLoader

import config


class EEGDataset(Dataset):
    """Simple wrapper for EEG windows and labels."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X).float()    # (N, 32, 512)
        self.y = torch.from_numpy(y).long()     # (N, 2)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def bandpass_filter(data: np.ndarray, lowcut: float = 4.0, highcut: float = 45.0,
                    fs: int = 128, order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth bandpass filter applied along the last axis."""
    sos = scipy.signal.butter(order, [lowcut, highcut], btype='band', fs=fs, output='sos')
    return scipy.signal.sosfiltfilt(sos, data, axis=-1)


def create_windows(data: np.ndarray, window_size: int, step_size: int):
    """Slide a window over the time axis of each trial.

    Args:
        data: (n_trials, n_channels, n_samples)
        window_size: samples per window
        step_size: step between windows

    Returns:
        windows: (n_windows, n_channels, window_size)  float32
        trial_indices: (n_windows,) int — which trial each window came from
    """
    windows = []
    trial_indices = []
    n_samples = data.shape[-1]
    for t_idx in range(data.shape[0]):
        for start in range(0, n_samples - window_size + 1, step_size):
            windows.append(data[t_idx, :, start:start + window_size])
            trial_indices.append(t_idx)
    return np.array(windows, dtype=np.float32), np.array(trial_indices, dtype=np.int64)


def load_subject(subject_id: int, data_dir: str = None) -> dict:
    """Load, filter, normalise, window, and split one DEAP subject.

    Critical guarantees:
    - Split is by trial index (not by window). Windows from the same trial
      are never in both train and test.
    - Z-score stats and label medians come from training trials only.

    Returns dict with keys: X_train, y_train, X_val, y_val, X_test, y_test.
    """
    if data_dir is None:
        data_dir = config.DATA_DIR

    fname = os.path.join(data_dir, f's{subject_id:02d}.dat')
    with open(fname, 'rb') as f:
        subject = pickle.load(f, encoding='latin1')

    data = subject['data']      # (40, 40, 8064)
    labels = subject['labels']  # (40, 4)

    # --- Keep first 32 EEG channels ---
    data = data[:, :config.N_CHANNELS, :]   # (40, 32, 8064)

    # --- Crop baseline (first 3 s = 384 samples at 128 Hz) ---
    baseline_samples = config.BASELINE_SEC * config.SFREQ  # 384
    data = data[:, :, baseline_samples:]    # (40, 32, 7680)
    expected = config.TRIAL_LENGTH_SEC * config.SFREQ
    assert data.shape[-1] == expected, (
        f"s{subject_id:02d}: expected {expected} samples after baseline crop, "
        f"got {data.shape[-1]}"
    )

    # --- Fixed deterministic trial split (no shuffling) ---
    train_idx = list(range(config.TRAIN_TRIALS))                                              # 0–27
    val_idx   = list(range(config.TRAIN_TRIALS, config.TRAIN_TRIALS + config.VAL_TRIALS))    # 28–33
    test_idx  = list(range(config.TRAIN_TRIALS + config.VAL_TRIALS, config.N_TRIALS))        # 34–39

    data_train = data[train_idx]            # (28, 32, 7680)
    data_val   = data[val_idx]              # (6,  32, 7680)
    data_test  = data[test_idx]             # (6,  32, 7680)

    labels_train = labels[train_idx, :2]    # (28, 2) — valence, arousal
    labels_val   = labels[val_idx,   :2]    # (6,  2)
    labels_test  = labels[test_idx,  :2]    # (6,  2)

    # --- Bandpass filter (4–45 Hz) ---
    data_train = bandpass_filter(data_train)
    data_val   = bandpass_filter(data_val)
    data_test  = bandpass_filter(data_test)

    # --- Per-channel z-score (statistics from training trials only) ---
    # mean/std over (trials, time) axes → shape (1, 32, 1)
    mean = data_train.mean(axis=(0, 2), keepdims=True)
    std  = data_train.std(axis=(0, 2),  keepdims=True)
    std  = np.where(std < 1e-8, 1e-8, std)

    data_train = (data_train - mean) / std
    data_val   = (data_val   - mean) / std
    data_test  = (data_test  - mean) / std

    # --- Windowing ---
    X_train, win_idx_train = create_windows(data_train, config.N_SAMPLES, config.STEP_SAMPLES)
    X_val,   win_idx_val   = create_windows(data_val,   config.N_SAMPLES, config.STEP_SAMPLES)
    X_test,  win_idx_test  = create_windows(data_test,  config.N_SAMPLES, config.STEP_SAMPLES)

    # Each window inherits the continuous label of its trial
    y_train_cont = labels_train[win_idx_train]   # (N_train_windows, 2)
    y_val_cont   = labels_val[win_idx_val]
    y_test_cont  = labels_test[win_idx_test]

    # --- Label binarisation: median from training windows only ---
    val_median = np.median(y_train_cont[:, 0])
    aro_median = np.median(y_train_cont[:, 1])

    def binarise(y_cont: np.ndarray) -> np.ndarray:
        y_bin = np.zeros(y_cont.shape, dtype=np.int64)
        y_bin[:, 0] = (y_cont[:, 0] >= val_median).astype(np.int64)
        y_bin[:, 1] = (y_cont[:, 1] >= aro_median).astype(np.int64)
        return y_bin

    y_train = binarise(y_train_cont)
    y_val   = binarise(y_val_cont)
    y_test  = binarise(y_test_cont)

    return {
        'X_train': X_train,   # (N, 32, 512) float32
        'y_train': y_train,   # (N, 2)       int64
        'X_val':   X_val,
        'y_val':   y_val,
        'X_test':  X_test,
        'y_test':  y_test,
    }


def save_splits(splits_dir: str = None) -> None:
    """Save the fixed trial split indices to splits/subject_splits.json."""
    if splits_dir is None:
        splits_dir = config.SPLITS_DIR

    os.makedirs(splits_dir, exist_ok=True)
    splits_path = os.path.join(splits_dir, 'subject_splits.json')

    if os.path.exists(splits_path):
        return  # already saved

    splits = {}
    for sid in range(1, config.N_SUBJECTS + 1):
        splits[f'subject_{sid:02d}'] = {
            'train': list(range(config.TRAIN_TRIALS)),
            'val':   list(range(config.TRAIN_TRIALS, config.TRAIN_TRIALS + config.VAL_TRIALS)),
            'test':  list(range(config.TRAIN_TRIALS + config.VAL_TRIALS, config.N_TRIALS)),
        }

    with open(splits_path, 'w') as f:
        json.dump(splits, f, indent=2)
    print(f"Saved trial splits to {splits_path}")


def get_dataloaders(subject_data: dict, batch_size: int = None):
    """Return (train_dl, val_dl, test_dl) DataLoaders for one subject."""
    if batch_size is None:
        batch_size = config.BATCH_SIZE

    train_ds = EEGDataset(subject_data['X_train'], subject_data['y_train'])
    val_ds   = EEGDataset(subject_data['X_val'],   subject_data['y_val'])
    test_ds  = EEGDataset(subject_data['X_test'],  subject_data['y_test'])

    kw = dict(num_workers=config.NUM_WORKERS, pin_memory=True)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  **kw)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, **kw)
    test_dl  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, **kw)

    return train_dl, val_dl, test_dl


def prepare_all_subjects(data_dir: str = None, subjects=None) -> dict:
    """Load data for all (or specified) subjects. Returns {subject_id: subject_data}."""
    if data_dir is None:
        data_dir = config.DATA_DIR
    if subjects is None:
        subjects = range(1, config.N_SUBJECTS + 1)

    all_data = {}
    for sid in subjects:
        print(f"  Loading subject {sid:02d}...")
        try:
            all_data[sid] = load_subject(sid, data_dir)
        except FileNotFoundError:
            print(f"  Warning: s{sid:02d}.dat not found, skipping.")
    return all_data
