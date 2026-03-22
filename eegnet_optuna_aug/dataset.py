import numpy as np
import scipy.signal
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
import random
import sys

import config

class EEGAugmentDataset(Dataset):
    def __init__(self, windows: np.ndarray, labels: np.ndarray, is_train: bool = False):
        self.windows = torch.tensor(windows, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.is_train = is_train
        
    def __len__(self):
        return len(self.windows)
        
    def __getitem__(self, idx):
        x = self.windows[idx].clone()
        y = self.labels[idx]
        
        if self.is_train:
            # Random Gaussian Noise
            if random.random() < config.AUG_PROBABILITY:
                noise = torch.randn_like(x) * config.AUG_GAUSSIAN_NOISE_STD
                x = x + noise
                
            # Random Amplitude Scaling
            if random.random() < config.AUG_PROBABILITY:
                scale = random.uniform(config.AUG_SCALE_MIN, config.AUG_SCALE_MAX)
                x = x * scale
                
            # Random Time-Shifting
            if random.random() < config.AUG_PROBABILITY:
                max_shift = int(config.AUG_SHIFT_MAX_SEC * config.SFREQ)
                shift = random.randint(-max_shift, max_shift)
                x = torch.roll(x, shifts=shift, dims=-1)
                
        return x, y

def load_subject(subject_id: int):
    sys.path.append(str(config.BASE_DIR.parent))
    from eeg_eegnet_grouped_experiment.data import load_subject as orig_load_subject
    return orig_load_subject(subject_id)

def get_target_labels(ratings: np.ndarray, target: str) -> np.ndarray:
    dim = 0 if target == "valence" else 1
    return (ratings[:, dim] >= config.LABEL_THRESHOLD).astype(np.int64)

def filter_neutral_trials(ratings: np.ndarray, target: str) -> np.ndarray:
    if not getattr(config, "FILTER_NEUTRAL_TRIALS", False):
        return np.arange(len(ratings))
    dim = 0 if target == "valence" else 1
    trial_ratings = ratings[:, dim]
    lower_bound, upper_bound = getattr(config, "NEUTRAL_TRIAL_BOUNDS", (4.0, 6.0))
    valid_mask = (trial_ratings <= lower_bound) | (trial_ratings >= upper_bound)
    return np.where(valid_mask)[0]

def build_outer_splits(trial_ids: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    n_splits = min(config.OUTER_FOLDS, len(trial_ids))
    if n_splits < 2: return []
    splitter = GroupKFold(n_splits=n_splits)
    splits = []
    for train_idx, test_idx in splitter.split(trial_ids, groups=trial_ids):
        splits.append((trial_ids[train_idx], trial_ids[test_idx]))
    return splits

def build_inner_split(train_trial_ids: np.ndarray, random_state: int) -> tuple[np.ndarray, np.ndarray]:
    groups = train_trial_ids
    splitter = GroupShuffleSplit(n_splits=1, test_size=config.INNER_VAL_TRIALS, random_state=random_state)
    local_train_idx, local_val_idx = next(splitter.split(train_trial_ids, groups=groups))
    return train_trial_ids[local_train_idx], train_trial_ids[local_val_idx]

def preprocess_trials(eeg: np.ndarray, train_trial_ids: np.ndarray) -> tuple[np.ndarray, dict]:
    processed = eeg.astype(np.float32, copy=True)
    baseline_samples = int(config.BASELINE_SEC * config.SFREQ)
    baseline_mean = processed[:, :, :baseline_samples].mean(axis=-1, keepdims=True)
    processed = processed - baseline_mean

    if config.USE_COMMON_AVERAGE_REFERENCE:
        processed = processed - processed.mean(axis=1, keepdims=True)

    sos = scipy.signal.butter(config.BANDPASS_ORDER, [config.BANDPASS_LOWCUT, config.BANDPASS_HIGHCUT], btype="bandpass", fs=config.SFREQ, output="sos")
    processed = scipy.signal.sosfilt(sos, processed, axis=-1).astype(np.float32)

    if config.NOTCH_FREQ is not None:
        b, a = scipy.signal.iirnotch(config.NOTCH_FREQ, config.NOTCH_Q, fs=config.SFREQ)
        processed = scipy.signal.lfilter(b, a, processed, axis=-1).astype(np.float32)

    processed = processed[:, :, baseline_samples:]

    mean = processed[train_trial_ids].mean(axis=(0, 2), keepdims=True)
    std = processed[train_trial_ids].std(axis=(0, 2), keepdims=True)
    std = np.where(std < 1e-8, 1e-8, std)

    processed = ((processed - mean) / std).astype(np.float32)
    norm_stats = {"mean": float(mean[0, 0, 0]), "std": float(std[0, 0, 0])}
    return processed, norm_stats

def make_window_split(eeg: np.ndarray, split_trial_ids: np.ndarray, all_trial_labels: np.ndarray, window_samples: int, stride_samples: int):
    windows_list = []
    labels_list = []
    parent_trials_list = []
    
    for tid in split_trial_ids:
        trial_data = eeg[tid]
        label = all_trial_labels[tid]
        n_samples = trial_data.shape[1]
        n_windows = ((n_samples - window_samples) // stride_samples) + 1
        for w in range(n_windows):
            start = w * stride_samples
            end = start + window_samples
            windows_list.append(trial_data[:, start:end])
            labels_list.append(label)
            parent_trials_list.append(tid)
            
    return {
        "windows": np.stack(windows_list) if len(windows_list) > 0 else np.array([]),
        "labels": np.array(labels_list),
        "parent_trials": np.array(parent_trials_list)
    }

def build_dataloader(split_data: dict, batch_size: int, is_train: bool) -> DataLoader:
    dataset = EEGAugmentDataset(
        windows=split_data["windows"], 
        labels=split_data["labels"], 
        is_train=is_train
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=is_train, num_workers=0)
