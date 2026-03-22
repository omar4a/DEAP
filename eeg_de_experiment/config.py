"""Single source of truth for the DE + EEGNet experiment."""

SEED = 42

# DEAP signal
N_CHANNELS = 32
SFREQ      = 128
N_SUBJECTS = 32
N_TRIALS   = 40

# Frequency bands for DE extraction (Hz)
FREQ_BANDS = {
    'delta': (1,  3),
    'theta': (4,  7),
    'alpha': (8,  12),
    'beta':  (13, 30),
    'gamma': (31, 45),
}
N_BANDS = 5

# DE windowing — non-overlapping 1-second windows
DE_WINDOW_SEC     = 1.0
DE_WINDOW_SAMPLES = 128          # 1 s × 128 Hz
# Each 60 s trial → 60 DE vectors  → per-trial DE shape: (60, 32, 5)

# Classification input reshaping:
#   (60, 32, 5) → permute → (32, 5, 60) → reshape → (32, 300)
#   Then unsqueeze → EEGNet input: (B, 1, 32, 300)
#   First 60 cols = delta, next 60 = theta, ..., last 60 = gamma
DE_TIME_STEPS  = 60              # windows per trial
DE_SEQ_LEN     = N_BANDS * DE_TIME_STEPS   # 300

# Labels — fixed threshold (not median split)
LABEL_DIMS      = ['valence', 'arousal']
LABEL_THRESHOLD = 5.0            # > 5.0 → high (1), else low (0)

# Cross-validation
CV_FOLDS = 10   # tenfold over trials
# Each fold: ~36 train trials, 4 test trials
# Within train: last 4 → val, remaining 32 → actual train

# Training
BATCH_SIZE              = 32     # == train set size → full-batch GD
EPOCHS                  = 150
LR                      = 1e-3
WEIGHT_DECAY            = 1e-4
EARLY_STOPPING_PATIENCE = 20
SCHEDULER               = 'cosine'

# Loss
VALENCE_WEIGHT = 0.5
AROUSAL_WEIGHT = 0.5

# EEGNet-DE parameters
EEGNET_F1 = 8
EEGNET_D  = 2
EEGNET_F2 = 16
DROPOUT   = 0.5     # higher than raw-EEG version — less data per fold
