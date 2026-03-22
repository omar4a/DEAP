"""Configuration for leakage-safe subject-dependent EEGNet experiments on DEAP."""

from __future__ import annotations

import os


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
DEBUG_RESULTS_DIR = os.path.join(BASE_DIR, "results_debug")

DATA_DIR_CANDIDATES = [
    os.path.normpath(os.path.join(BASE_DIR, "..", "archive")),
    os.path.normpath(os.path.join(BASE_DIR, "..", "eeg_dl_experiment", "data")),
    os.path.normpath(os.path.join(BASE_DIR, "data")),
]

SEED = 42

# DEAP signal
N_CHANNELS = 32
SFREQ = 128
N_SUBJECTS = 32
N_TRIALS = 40
BASELINE_SEC = 3
VIDEO_SEC = 60
FULL_TRIAL_SEC = BASELINE_SEC + VIDEO_SEC
FULL_TRIAL_SAMPLES = FULL_TRIAL_SEC * SFREQ
VIDEO_SAMPLES = VIDEO_SEC * SFREQ

# Labels
LABEL_THRESHOLD = 5.0
TARGETS = ("valence", "arousal")

# Filtering Logic
FILTER_NEUTRAL_TRIALS = True
NEUTRAL_TRIAL_BOUNDS = (4.0, 6.0)
EXCLUDE_SUBJECTS_VALENCE = [1, 2, 3, 7, 11, 12, 17, 21]
EXCLUDE_SUBJECTS_AROUSAL = [5, 6, 7, 9, 15, 22, 28, 30]

# Preprocessing
USE_COMMON_AVERAGE_REFERENCE = True
BANDPASS_LOWCUT = 4.0
BANDPASS_HIGHCUT = 45.0
BANDPASS_ORDER = 4
NOTCH_FREQ = None
NOTCH_Q = 30.0

# Windowing
WINDOW_SEC = 4.0
STRIDE_SEC = 2.0
WINDOW_SAMPLES = int(WINDOW_SEC * SFREQ)
STRIDE_SAMPLES = int(STRIDE_SEC * SFREQ)
WINDOWS_PER_TRIAL = ((VIDEO_SAMPLES - WINDOW_SAMPLES) // STRIDE_SAMPLES) + 1

# Grouped evaluation
OUTER_FOLDS = 5
INNER_VAL_TRIALS = 8

# Training
BATCH_SIZE = 64
EPOCHS = 100
LR = 1e-3
WEIGHT_DECAY = 1e-4
EARLY_STOPPING_PATIENCE = 15
NUM_WORKERS = 0
MODEL_SELECTION_METRIC = "video_macro_f1"

# EEGNet
TEMPORAL_KERNEL = 64
EEGNET_F1 = 8
EEGNET_D = 2
EEGNET_F2 = 16
DROPOUT = 0.5

# Repeated runs
DEFAULT_SEEDS = (42, 52, 62)
