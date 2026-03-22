"""Data loading and grouped trial/window preparation for DEAP EEGNet experiments."""

from __future__ import annotations

import os
import pickle
from dataclasses import dataclass

import numpy as np
import scipy.signal
import torch
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from torch.utils.data import DataLoader, Dataset

try:
    from . import config
except ImportError:
    import config


@dataclass(frozen=True)
class SubjectData:
    """Full DEAP recording for one subject before fold-specific preprocessing."""

    subject_id: int
    eeg: np.ndarray
    ratings: np.ndarray


class WindowDataset(Dataset):
    """Window-level dataset with inherited trial labels."""

    def __init__(self, windows: np.ndarray, labels: np.ndarray):
        self.windows = torch.from_numpy(windows).float()
        self.labels = torch.from_numpy(labels).long()

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int):
        return self.windows[index], self.labels[index]


def find_data_dir() -> str:
    """Return the first directory that contains DEAP .dat files."""
    for candidate in config.DATA_DIR_CANDIDATES:
        if os.path.isdir(candidate) and any(name.endswith(".dat") for name in os.listdir(candidate)):
            return candidate
    raise FileNotFoundError(
        f"No DEAP .dat files found. Searched: {config.DATA_DIR_CANDIDATES}"
    )


def load_subject(subject_id: int, data_dir: str | None = None) -> SubjectData:
    """Load one DEAP subject from the official .dat format."""
    if data_dir is None:
        data_dir = find_data_dir()

    path = os.path.join(data_dir, f"s{subject_id:02d}.dat")
    with open(path, "rb") as handle:
        payload = pickle.load(handle, encoding="latin1")

    eeg = payload["data"][:, : config.N_CHANNELS, :].astype(np.float32)
    ratings = payload["labels"].astype(np.float32)

    expected_shape = (config.N_TRIALS, config.N_CHANNELS, config.FULL_TRIAL_SAMPLES)
    if eeg.shape != expected_shape:
        raise ValueError(f"s{subject_id:02d}: expected {expected_shape}, got {eeg.shape}")

    return SubjectData(subject_id=subject_id, eeg=eeg, ratings=ratings)


def get_target_labels(ratings: np.ndarray, target: str) -> np.ndarray:
    """Return standard DEAP binary labels for one target dimension."""
    if target not in config.TARGETS:
        raise ValueError(f"Unknown target: {target}")
    dim = 0 if target == "valence" else 1
    return (ratings[:, dim] >= config.LABEL_THRESHOLD).astype(np.int64)

def filter_neutral_trials(ratings: np.ndarray, target: str) -> np.ndarray:
    """Return indices of trials that are outside the neutral bounds."""
    if not getattr(config, "FILTER_NEUTRAL_TRIALS", False):
        return np.arange(len(ratings))
        
    dim = 0 if target == "valence" else 1
    trial_ratings = ratings[:, dim]
    
    lower_bound, upper_bound = getattr(config, "NEUTRAL_TRIAL_BOUNDS", (4.0, 6.0))
    valid_mask = (trial_ratings <= lower_bound) | (trial_ratings >= upper_bound)
    return np.where(valid_mask)[0]


def build_outer_splits(trial_ids: np.ndarray | None = None) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return deterministic 5-fold grouped outer splits over the valid trials."""
    if trial_ids is None:
        trial_ids = np.arange(config.N_TRIALS)
        
    # Standard GroupKFold cannot split fewer groups than n_splits
    n_splits = min(config.OUTER_FOLDS, len(trial_ids))
    if n_splits < 2:
        return []
        
    splitter = GroupKFold(n_splits=n_splits)
    splits = []
    for train_idx, test_idx in splitter.split(trial_ids, groups=trial_ids):
        splits.append((trial_ids[train_idx], trial_ids[test_idx]))
    return splits


def build_inner_split(train_trial_ids: np.ndarray, random_state: int) -> tuple[np.ndarray, np.ndarray]:
    """Split 32 outer-train trials into 24 train and 8 validation trials."""
    groups = train_trial_ids
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=config.INNER_VAL_TRIALS,
        random_state=random_state,
    )
    local_train_idx, local_val_idx = next(splitter.split(train_trial_ids, groups=groups))
    return train_trial_ids[local_train_idx], train_trial_ids[local_val_idx]


def _apply_common_average_reference(data: np.ndarray) -> np.ndarray:
    return data - data.mean(axis=1, keepdims=True)


def _apply_bandpass(data: np.ndarray) -> np.ndarray:
    sos = scipy.signal.butter(
        config.BANDPASS_ORDER,
        [config.BANDPASS_LOWCUT, config.BANDPASS_HIGHCUT],
        btype="bandpass",
        fs=config.SFREQ,
        output="sos",
    )
    return scipy.signal.sosfilt(sos, data, axis=-1).astype(np.float32)


def _apply_notch(data: np.ndarray) -> np.ndarray:
    if config.NOTCH_FREQ is None:
        return data
    b, a = scipy.signal.iirnotch(config.NOTCH_FREQ, config.NOTCH_Q, fs=config.SFREQ)
    return scipy.signal.lfilter(b, a, data, axis=-1).astype(np.float32)


def preprocess_trials(eeg: np.ndarray, train_trial_ids: np.ndarray) -> tuple[np.ndarray, dict]:
    """Apply causal preprocessing and train-only normalization to all trials."""
    processed = eeg.astype(np.float32, copy=True)

    baseline_samples = config.BASELINE_SEC * config.SFREQ
    baseline_mean = processed[:, :, :baseline_samples].mean(axis=-1, keepdims=True)
    processed = processed - baseline_mean

    if config.USE_COMMON_AVERAGE_REFERENCE:
        processed = _apply_common_average_reference(processed)

    processed = _apply_bandpass(processed)
    processed = _apply_notch(processed)

    processed = processed[:, :, baseline_samples:]

    mean = processed[train_trial_ids].mean(axis=(0, 2), keepdims=True)
    std = processed[train_trial_ids].std(axis=(0, 2), keepdims=True)
    std = np.where(std < 1e-8, 1e-8, std)

    processed = ((processed - mean) / std).astype(np.float32)

    stats = {
        "baseline_subtraction": True,
        "mean_shape": list(mean.shape),
        "std_shape": list(std.shape),
    }
    return processed, stats


def create_windows(processed_trials: np.ndarray, trial_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create overlapping 2 s windows with 1 s stride from the selected trials."""
    windows = []
    parent_trials = []
    start_samples = []

    for trial_id in trial_ids:
        signal = processed_trials[trial_id]
        for start in range(0, config.VIDEO_SAMPLES - config.WINDOW_SAMPLES + 1, config.STRIDE_SAMPLES):
            windows.append(signal[:, start : start + config.WINDOW_SAMPLES])
            parent_trials.append(trial_id)
            start_samples.append(start)

    return (
        np.asarray(windows, dtype=np.float32),
        np.asarray(parent_trials, dtype=np.int64),
        np.asarray(start_samples, dtype=np.int64),
    )


def make_window_split(
    processed_trials: np.ndarray,
    trial_ids: np.ndarray,
    trial_labels: np.ndarray,
) -> dict:
    """Create one fold split with windows, labels, and trial metadata."""
    windows, parent_trials, start_samples = create_windows(processed_trials, trial_ids)
    labels = trial_labels[parent_trials]
    return {
        "windows": windows,
        "labels": labels.astype(np.int64),
        "parent_trials": parent_trials,
        "start_samples": start_samples,
    }


def build_dataloader(split: dict, shuffle: bool, batch_size: int | None = None) -> DataLoader:
    """Wrap a split into a DataLoader."""
    if batch_size is None:
        batch_size = config.BATCH_SIZE
    dataset = WindowDataset(split["windows"], split["labels"])
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=config.NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )
