"""Single source of truth for all hyperparameters and settings."""
import os
import platform

SEED = 42

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', 'archive')
SPLITS_DIR = os.path.join(BASE_DIR, 'splits')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

# DEAP signal
N_CHANNELS = 32         # EEG channels only (drop last 8 peripheral)
SFREQ = 128             # Hz
N_SUBJECTS = 32
N_TRIALS = 40
TRIAL_LENGTH_SEC = 60
BASELINE_SEC = 3

# Windowing
WINDOW_SEC = 4.0
STEP_SEC = 0.5
N_SAMPLES = int(WINDOW_SEC * SFREQ)    # = 512
STEP_SAMPLES = int(STEP_SEC * SFREQ)   # = 64

# Labels
LABEL_DIMS = ['valence', 'arousal']

# Subject-dependent split: first 28 train / next 6 val / last 6 test (by trial index)
TRAIN_TRIALS = 28
VAL_TRIALS = 6
TEST_TRIALS = 6
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15

# Training
BATCH_SIZE = 64
EPOCHS = 100
LR = 1e-3
WEIGHT_DECAY = 1e-4
EARLY_STOPPING_PATIENCE = 15
SCHEDULER = 'cosine'

# Loss weighting
VALENCE_WEIGHT = 0.5
AROUSAL_WEIGHT = 0.5

# Shared model capacity settings (applied to all models for fairness)
DROPOUT = 0.25

# EEGNet
EEGNET_F1 = 8
EEGNET_D  = 2
EEGNET_F2 = 16

# EEGNet + Transformer
D_MODEL              = 64
N_HEADS              = 4
N_TRANSFORMER_LAYERS = 2
TRANSFORMER_FFN_DIM  = 256
TRANSFORMER_DROPOUT  = 0.1

# ConvNeXt-EEG (deliberately small)
CONVNEXT_DIMS   = [32, 64, 128]
CONVNEXT_BLOCKS = [2, 2, 2]

# DataLoader workers: use 0 on Windows to avoid multiprocessing issues
NUM_WORKERS = 0 if platform.system() == 'Windows' else 2
