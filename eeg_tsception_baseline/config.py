import sys
from pathlib import Path

# Dynamic Environment Detection & Pathing
if 'google.colab' in sys.modules:
    # Google Colab (Requires mounting drive beforehand)
    BASE_DIR = Path('/content/drive/MyDrive/Computers/My Laptop/DEAP/eeg_tsception_baseline')
else:
    # Windows Local Fallback
    BASE_DIR = Path(__file__).resolve().parent

BASE_DIR.mkdir(parents=True, exist_ok=True)

# Dataset Info
SFREQ = 128
CHANNELS = 32
VIDEO_SAMPLES = 63 * SFREQ
BASELINE_SEC = 3.0

# Labels
LABEL_THRESHOLD = 5.0
TARGETS = ("valence", "arousal")

# Filtering Logic (STRICT BASELINE - NO FILTERING)
FILTER_NEUTRAL_TRIALS = False 
NEUTRAL_TRIAL_BOUNDS = (4.0, 6.0)
EXCLUDE_SUBJECTS_VALENCE = []
EXCLUDE_SUBJECTS_AROUSAL = []

# Preprocessing
USE_COMMON_AVERAGE_REFERENCE = True
BANDPASS_LOWCUT = 4.0
BANDPASS_HIGHCUT = 45.0
BANDPASS_ORDER = 4
NOTCH_FREQ = 50.0
NOTCH_Q = 30.0

# Cross-Validation
OUTER_FOLDS = 5
INNER_VAL_TRIALS = 8

# Data Augmentation (STRICT BASELINE - NO AUGMENTATION)
AUG_PROBABILITY = 0.0
AUG_GAUSSIAN_NOISE_STD = 0.05
AUG_SCALE_MIN = 0.8
AUG_SCALE_MAX = 1.2
AUG_SHIFT_MAX_SEC = 0.2

# Tsception Specifics (Ding et al. 2020 framework metrics)
LEARNING_RATE = 0.001
DROPOUT = 0.3
BATCH_SIZE = 64
WINDOW_SEC = 2.0  # Kept at 2.0s to match original EEGNet benchmark exactly
NUM_T = 9         # Temporal kernels (3 scales * 9)
NUM_S = 6         # Spatial kernels (2 scales * 6)

# Model Selection
MODEL_SELECTION_METRIC = "video_macro_f1"
EARLY_STOPPING_PATIENCE = 15
