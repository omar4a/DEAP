# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Goal

Build a classical ML pipeline to achieve the best arousal/valence recognition accuracies on the DEAP dataset in the simplest way possible.

## Paper-Based Pipeline (NEW)

Based on high-accuracy papers (13, 14), the pipeline now includes:

### Preprocessing (Paper 13)
- Bandpass filter: 4-45 Hz (Butterworth 4th order)
- Notch filter: 50 Hz (power line interference removal)
- Baseline correction

### Features
| Feature | Paper | Reported Accuracy |
|---------|-------|-------------------|
| Differential Entropy (DE) | 13 | 89% valence, 88% arousal |
| Higuchi Fractal Dimension (HFD) | 13 | 89% valence, 88% arousal |
| Statistical (mean, std, skew, kurtosis) | 14 | 91% valence, 89% arousal |
| Hjorth (activity, mobility, complexity) | 14 | 69-70% |
| Band power | 11, 14 | 64-89% |

### Models
- XGBoost (best in papers)
- Random Forest
- CatBoost
- Soft voting ensemble

## Commands

```bash
# Run tests
pytest -q

# Paper-based experiment (recommended)
python run_experiments.py --models xgb rf \
    --feature-set de statistical hfd hjorth bandpower \
    --window-seconds 3.0 --step-seconds 3.0 \
    --label-mode binary --label-threshold 5.0 \
    --progress --checkpoint reports/paper_exp/checkpoint.csv \
    --output-dir reports/paper_exp

# Quick test with subset
python run_experiments.py --models xgb --subjects 1 2 3 \
    --feature-set de statistical hfd \
    --outer-repeats 2 --inner-folds 2 --progress

# Grid-search experiments
python run_experiments.py --models logreg rbf_svm --outer-folds 5 --inner-folds 3 --scoring f1_macro

# Resume interrupted experiment
python run_experiments.py --models xgb \
    --checkpoint reports/paper_exp/checkpoint.csv \
    --output-dir reports/paper_exp --progress

# Simple evaluation (legacy)
python run_pipeline.py --mode both --output reports/metrics.json
```

## Architecture

### Data Flow
```
DEAP files (archive/*.mat) → load_dataset() → SubjectData
    → bandpass/notch filter → baseline correction
    → feature extraction → labeling
    → model fitting/evaluation → metrics
```

### Key Components

- **`config.py`**: Frozen dataclass `Config` with all parameters (channels, bands, features, labels, classifier, preprocessing). Central source of truth.

- **`preprocess.py`**: Bandpass filter (4-45 Hz), notch filter (50 Hz), baseline correction, segmentation.

- **`data.py`**: Loads DEAP files (.mat/.npz/.dat). Returns `SubjectData(subject_id, eeg[trials×channels×samples], labels)`.

- **`features.py`**: Feature extraction. Types:
  - **Frequency**: `bandpower`, `rel_bandpower`, `de` (differential entropy), `psd`
  - **Time-domain**: `statistical`, `hjorth`, `hfd` (Higuchi FD)
  - **Spatial**: `asymmetry`, `connectivity_pcc`, `connectivity_plv`, `riemann`

- **`labels.py`**: `build_labels()` handles 7 modes: `binary`, `three_class`, `quartile`, `subject_mean`, `subject_median`, `subject_tertile`, `regression`.

- **`eval.py`**: Two protocols:
  - `run_subject_dependent()`: GroupKFold within each subject
  - `run_cross_subject()`: Leave-one-subject-out

- **`experiment.py`**: Grid-search with nested CV. Pausable/resumable via checkpoint. Outputs to `reports/`.

- **`model_registry.py`**: `ModelSpec` dataclass with estimator definitions and hyperparameter grids. Models: logreg, linear_svm, rbf_svm, mlp, rf, xgb, lgbm, catboost, hgb, soft_vote.

### Caching

Features are cached in `cache/` using hash of feature-relevant config (including preprocessing params). Disable with `--no-cache` or `use_cache=False`.

### Checkpointing

Experiments are pausable and resumable:
- Use `--checkpoint path/to/checkpoint.csv` to enable
- Progress saved after each fold
- Live summary written to `summary_live.txt`
- Resume automatically on rerun (use `--no-resume` to start fresh)

## Shape Conventions

- EEG: `[trials, channels, samples]`
- Features: `[samples, n_features]`
- Labels/Groups: `[samples]`

## Data Setup

Place DEAP files in `archive/` directory with naming `s01.mat`, `s02.mat`, etc.
