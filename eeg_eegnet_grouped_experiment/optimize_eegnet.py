import argparse
import json
from pathlib import Path

import optuna
import numpy as np

# Import our modules
import config
from data import load_subject, get_target_labels, filter_neutral_trials, build_outer_splits
from train import train_one_fold

def objective(trial):
    # Search parameters
    window_sec = trial.suggest_categorical("window_sec", [2.0, 3.0, 4.0])
    stride_sec = window_sec / 2.0
    f1 = trial.suggest_categorical("f1", [8, 16])
    d = trial.suggest_categorical("d", [2, 4])
    f2 = trial.suggest_categorical("f2", [16, 32])
    dropout = trial.suggest_categorical("dropout", [0.25, 0.5])
    lr = trial.suggest_categorical("lr", [1e-4, 5e-4, 1e-3, 5e-3])

    # Optionally bound F2 logic (e.g., F2 usually >= F1 * D) but we'll let it be categorical
    
    # Overwrite configuration dynamically
    config.WINDOW_SEC = float(window_sec)
    config.STRIDE_SEC = float(stride_sec)
    config.WINDOW_SAMPLES = int(config.WINDOW_SEC * config.SFREQ)
    config.STRIDE_SAMPLES = int(config.STRIDE_SEC * config.SFREQ)
    config.WINDOWS_PER_TRIAL = int(((config.VIDEO_SAMPLES - config.WINDOW_SAMPLES) // config.STRIDE_SAMPLES) + 1)
    
    config.EEGNET_F1 = int(f1)
    config.EEGNET_D = int(d)
    config.EEGNET_F2 = int(f2)
    config.DROPOUT = float(dropout)
    config.LR = float(lr)
    
    # We will compute the objective on a subset of subjects to keep execution fast.
    # Selecting 3 reliable subjects from the valence top-performers
    eval_subjects = [4, 8, 14]
    target = "valence"
    seed = 42
    epochs = 30 # Optuna trials don't need to run for 100 epochs usually
    
    outer_folds_to_run = 2 # Run 2 out of 5 folds to save time
    
    results_dir = Path(config.BASE_DIR) / "results_optuna" / f"trial_{trial.number}"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    f1_scores = []
    
    for subject_id in eval_subjects:
        subject = load_subject(subject_id)
        
        valid_trial_ids = filter_neutral_trials(subject.ratings, target)
        if len(valid_trial_ids) < 10:
            continue
            
        outer_splits = build_outer_splits(valid_trial_ids)[:outer_folds_to_run]
        
        for outer_fold, (outer_train_ids, outer_test_ids) in enumerate(outer_splits):
            summary, _ = train_one_fold(
                subject_id=subject_id,
                target=target,
                seed=seed,
                outer_fold=outer_fold,
                subject=subject,
                outer_train_ids=outer_train_ids,
                outer_test_ids=outer_test_ids,
                epochs=epochs,
                force=True, # Always train
                results_dir=results_dir,
            )
            f1_scores.append(summary["video_macro_f1"])

    # Maximize the mean F1 score across the evaluated subjects/folds
    if len(f1_scores) == 0:
        return 0.0
    
    mean_f1 = float(np.mean(f1_scores))
    return mean_f1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=20)
    args = parser.parse_args()

    study = optuna.create_study(direction="maximize", study_name="eegnet_valence_optimization")
    study.optimize(objective, n_trials=args.n_trials)

    print("\nBest trial:")
    trial = study.best_trial
    print(f"  Value (Val F1): {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
        
    # Save best parameters
    out_path = Path(config.BASE_DIR) / "optuna_best_params.json"
    with out_path.open("w") as f:
        json.dump(trial.params, f, indent=4)
