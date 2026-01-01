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

## Configuration

- Binary vs 4-class labels: `--label-mode binary|quartile`
- Windowed features: `--window-seconds 2 --step-seconds 1`
- Classifier: `--classifier logreg|linear_svm`

## Tests

```bash
pytest -q
```
