# Grouped EEGNet DEAP Experiment

This directory contains a leakage-safe, subject-dependent EEGNet experiment for DEAP that follows the requested protocol:

- EEG only, all 32 channels
- standard DEAP binary labels: high if rating `>= 5`, else low
- causal preprocessing: common average reference, `4-45 Hz` band-pass, optional notch
- baseline removed; only the `60 s` video EEG is windowed
- `2 s` windows with `1 s` stride
- outer `5`-fold grouped cross-validation by trial/video
- inner grouped validation split of `24` train videos and `8` validation videos
- train on windows, evaluate on video-level predictions only
- per-video aggregation by mean probability across windows
- repeated across multiple random seeds

## Directory

- [config.py](/c:/Omar/Education/GP/DEAP/eeg_eegnet_grouped_experiment/config.py): protocol and model constants
- [data.py](/c:/Omar/Education/GP/DEAP/eeg_eegnet_grouped_experiment/data.py): DEAP loading, causal preprocessing, grouped splits, windowing
- [model.py](/c:/Omar/Education/GP/DEAP/eeg_eegnet_grouped_experiment/model.py): compact EEGNet binary classifier
- [train.py](/c:/Omar/Education/GP/DEAP/eeg_eegnet_grouped_experiment/train.py): training, early stopping, evaluation, summaries

## Data discovery

The loader searches these locations for official DEAP `.dat` files:

- `../archive`
- `../eeg_dl_experiment/data`
- `./data`

## Run

Install dependencies:

```powershell
pip install -r eeg_eegnet_grouped_experiment/requirements.txt
```

Quick smoke test:

```powershell
python eeg_eegnet_grouped_experiment/train.py --debug
```

`--debug` writes into `eeg_eegnet_grouped_experiment/results_debug` so it does not contaminate full runs.

Full experiment with the default seeds (`42 52 62`) and both targets:

```powershell
python eeg_eegnet_grouped_experiment/train.py
```

Example with a notch filter and a custom output directory:

```powershell
python eeg_eegnet_grouped_experiment/train.py --notch-freq 50 --results-dir eeg_eegnet_grouped_experiment/results_notch50
```

## Outputs

Top-level outputs:

- `run_config.json`
- `fold_results.csv`
- `video_predictions.csv`
- `subject_seed_summary.csv`
- `subject_summary.csv`
- `overall_summary.csv`
- `confusion_matrix_valence.csv`
- `confusion_matrix_arousal.csv`

Per fold output:

- `best_model.pth`
- `training_metrics.csv`
- `test_video_predictions.csv`
- `fold_summary.json`
- `split.json`

## Evaluation

Each held-out video produces `59` overlapping windows:

- first prediction at `t = 2 s`
- then every `1 s`
- each prediction uses only the most recent `2 s`

For evaluation, all per-window high-class probabilities from a video are averaged into one video-level score. The final prediction is high if the mean probability is `>= 0.5`.

Reported metrics are video-level:

- accuracy
- macro-F1
- ROC-AUC
- confusion matrix
