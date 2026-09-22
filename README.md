# DEAP Emotion Recognition Workspace

Research code for EEG-based emotion recognition on the DEAP dataset (32 subjects, 40 music-video trials each).

**The headline finding:** a published paper reported **88–89 %** valence/arousal accuracy on DEAP using DE-histogram + Higuchi fractal-dimension features. With the same features and honest, subject-aware evaluation, this pipeline reproduces **~67–72 %**, and cross-subject performance without calibration is close to chance. `run_paper13_leaky_baseline.py` is a deliberately leaky pooled-CV version kept for comparison. It shows exactly how segment-level splitting lets the same subject's data appear in both train and test. The repository combines a configurable classical machine-learning pipeline with several deep-learning experiment branches, evaluation scripts, and report assets used in the graduation project.

## What Is In The Repo

- `src/deap_emotion/`: preprocessing, feature extraction, labeling, metrics, and experiment orchestration
- `run_pipeline.py`: quick baseline evaluation for subject-dependent and cross-subject runs
- `run_experiments.py`: configurable experiment runner with model search and report generation
- `eeg_*`: model-specific deep-learning experiments such as EEGNet, EEGNet+Transformer, ConvNeXt-EEG, TSception, and DE-based variants
- `reports/`: generated summaries, paper notes, and diagram sources
- `tests/`: smoke and unit tests for the core pipeline

## Results

| Evaluation | Model | Valence | Arousal | Source |
|---|---|---|---|---|
| Reported by the replicated paper | — | 89 % acc | 88 % acc | — |
| Subject-dependent, trial-level, 5 × (80/20) with 4-fold inner grid search | XGBoost | 66.7 % acc (F1-macro 0.53) | 66.3 % acc | `reports/paper13/summary.txt` |
| Proper CV (best configuration) | XGBoost | 69.6 % acc | 71.8 % acc | `reports/paper13_LEAKAGE_POOLED_CV/` |
| Baseline pipeline, subject-dependent | SVM | 64.5 % acc (F1 0.60) | 59.6 % acc (F1 0.53) | `reports/metrics.json` |
| Baseline pipeline, **cross-subject** | SVM | 48.1 % acc | 49.3 % acc | `reports/metrics.json` |

Per-subject accuracy ranges from **47.5 % to 82.5 %**. That spread is why the downstream app ([eeg-emotion-recognition](https://github.com/omar4a/eeg-emotion-recognition)) calibrates per user instead of shipping one global model.

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
