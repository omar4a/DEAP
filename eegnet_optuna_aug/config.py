import sys
from pathlib import Path

# Dynamic Environment Detection & Pathing
if 'google.colab' in sys.modules:
    # Google Colab (Requires mounting drive beforehand)
    BASE_DIR = Path('/content/drive/MyDrive/Computers/My Laptop/DEAP/eegnet_optuna_aug')
else:
    # Windows Local Fallback
    BASE_DIR = Path(__file__).resolve().parent

BASE_DIR.mkdir(parents=True, exist_ok=True)

SFREQ = 128
CHANNELS = 32
VIDEO_SAMPLES = 63 * SFREQ
BASELINE_SEC = 3.0

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
NOTCH_FREQ = 50.0
NOTCH_Q = 30.0

# Cross-Validation
OUTER_FOLDS = 5
INNER_VAL_TRIALS = 8

# Data Augmentation (Dynamic)
AUG_GAUSSIAN_NOISE_STD = 0.05
AUG_SCALE_MIN = 0.8
AUG_SCALE_MAX = 1.2
AUG_SHIFT_MAX_SEC = 0.2
AUG_PROBABILITY = 0.3

# Optuna Specifics
N_OPTUNA_TRIALS = 50
OPTUNA_DB_URI = f"sqlite:///{BASE_DIR / 'optuna_study.db'}"

# Model Selection
MODEL_SELECTION_METRIC = "video_macro_f1"
EARLY_STOPPING_PATIENCE = 15
