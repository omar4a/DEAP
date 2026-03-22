# DEAP Emotion Recognition Workspace

Research code for EEG-based emotion recognition on the DEAP dataset. The repository combines a configurable classical machine-learning pipeline with several deep-learning experiment branches, evaluation scripts, and report assets used in the graduation project.

## What Is In The Repo

- `src/deap_emotion/`: preprocessing, feature extraction, labeling, metrics, and experiment orchestration
- `run_pipeline.py`: quick baseline evaluation for subject-dependent and cross-subject runs
- `run_experiments.py`: configurable experiment runner with model search and report generation
- `eeg_*`: model-specific deep-learning experiments such as EEGNet, EEGNet+Transformer, ConvNeXt-EEG, TSception, and DE-based variants
- `reports/`: generated summaries, paper notes, and diagram sources
- `tests/`: smoke and unit tests for the core pipeline

## Quick Start

Install the root dependencies:

```bash
pip install -r requirements.txt
```

Place the DEAP subject `.dat` files in `archive/`.

Run the baseline pipeline:

```bash
python run_pipeline.py --mode both --output reports/metrics.json
```

Run a configurable experiment sweep:

```bash
python run_experiments.py --models xgb rbf_svm --outer-repeats 5 --inner-folds 4 --scoring f1_macro
```

Run the test suite:

```bash
pytest -q
```

## Notes

- The main pipeline supports multiple label schemes, configurable feature sets, feature selection, and trial-level aggregation.
- Deep-learning folders ship with their own training scripts and, in some cases, separate dependency files.
- Large datasets, caches, and generated training outputs are intentionally kept out of version control.
