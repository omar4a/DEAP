# DEAP Emotion Pipeline (Approach 3)

Simple, fast EEG emotion recognition for DEAP using bandpower features and classical models.

## Setup

```bash
pip install -r requirements.txt
```

## Run evaluation

```bash
python run_pipeline.py --mode both --output reports/metrics.json
```

## Grid-search experiments

```bash
python run_experiments.py --models logreg rbf_svm --outer-folds 5 --inner-folds 3 --scoring f1_macro
```

Use a smaller run while iterating:

```bash
python run_experiments.py --models logreg --subjects 1 2 3 --trials 0 1 2 3 --outer-folds 2 --inner-folds 2
```

Outputs:

- `reports/experiment/experiment_report.json`
- `reports/experiment/summary.csv`

## Configuration

- Label modes: `--label-mode binary|three_class|quartile|regression|subject_median|subject_tertile`
- Label bins (three_class/quartile): `--label-bins 3 6` or `--label-bins 3 5 7`
- Windowed features: `--window-seconds 2 --step-seconds 1`
- Feature sets: `--feature-set bandpower psd de asymmetry`
- PSD bins: `--psd-min-hz 1 --psd-max-hz 45 --psd-bin-width 2`
- PCA: `--pca-variance 0.95`
- Per-subject normalization: `--per-subject-norm`

## Tests

```bash
pytest -q
```
