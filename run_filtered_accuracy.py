import sys
import os
import numpy as np
from pathlib import Path
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

sys.path.append(str(Path(__file__).resolve().parent / "src"))
from deap_emotion.config import Config
from deap_emotion.data import load_dataset
from deap_emotion.features import extract_features
from deap_emotion.labels import DIMENSIONS

def run_filtered_evaluation():
    config = Config(
        feature_set=("de_histogram",),
        window_seconds=3.0,
        step_seconds=3.0,
        per_subject_normalize=True,
    )
    
    subjects = [1, 2, 3, 4, 5]
    dataset = load_dataset(
        data_dir=config.data_dir, 
        eeg_channels=config.eeg_channels, 
        subjects=subjects
    )
    
    metrics = {"valence": [], "arousal": []}
    
    for subject in dataset:
        print(f"\nProcessing Subject {subject.subject_id}...")
        
        # 1. Manually run preprocessing
        from deap_emotion.preprocess import preprocess_eeg
        eeg_prep = preprocess_eeg(
            subject.eeg,
            fs=config.sfreq,
            bandpass_low=config.bandpass_low,
            bandpass_high=config.bandpass_high,
            bandpass_order=config.bandpass_order,
            notch_freq=config.notch_freq,
        )
        
        if config.per_subject_normalize:
            # Z-score normalization per subject
            mean = eeg_prep.mean(axis=-1, keepdims=True)
            std = eeg_prep.std(axis=-1, keepdims=True)
            eeg_prep = (eeg_prep - mean) / (std + 1e-8)
            
        # 2. Extract Features
        X, groups = extract_features(
            eeg_prep,
            sfreq=config.sfreq,
            bands=config.bands,
            mode=config.feature_mode,
            window_samples=config.window_samples(),
            step_samples=config.step_samples(),
            feature_set=config.feature_set,
        )
        
        for target in ["valence", "arousal"]:
            dim_idx = DIMENSIONS[target]
            
            # y_ratings comes from subject.labels (trials x 4)
            # We need to map it to window level using `groups`
            target_ratings = subject.labels[:, dim_idx]
            window_ratings = target_ratings[groups]
            
            # 3. Filter out neutral trials (ratings between 4.0 and 6.0)
            valid_mask = (window_ratings <= 4.0) | (window_ratings >= 6.0)
            
            X_filtered = X[valid_mask]
            y_ratings_filtered = window_ratings[valid_mask]
            groups_filtered = groups[valid_mask]
            
            if len(np.unique(groups_filtered)) < 5:
                print(f"Skipping Subject {subject.subject_id} {target}: Not enough valid trials.")
                continue
                
            y_binary = (y_ratings_filtered >= 6.0).astype(int)
            
            if len(np.unique(y_binary)) < 2:
                print(f"Skipping Subject {subject.subject_id} {target}: Only one class present after filtering.")
                continue
            
            # 4. Cross Validation
            gkf = GroupKFold(n_splits=5)
            fold_accs = []
            
            for train_idx, test_idx in gkf.split(X_filtered, y_binary, groups_filtered):
                X_train, y_train = X_filtered[train_idx], y_binary[train_idx]
                X_test = X_filtered[test_idx]
                
                clf = XGBClassifier(
                    n_estimators=100, 
                    max_depth=4, 
                    learning_rate=0.1,
                    eval_metric="logloss",
                )
                clf.fit(X_train, y_train)
                
                y_pred = clf.predict(X_test)
                
                # Video level aggregation
                unique_trials = np.unique(groups_filtered[test_idx])
                trial_preds = []
                trial_trues = []
                
                for trial in unique_trials:
                    trial_mask = (groups_filtered[test_idx] == trial)
                    pred_label = 1 if np.mean(y_pred[trial_mask]) >= 0.5 else 0
                    true_label = y_binary[test_idx][trial_mask][0]
                    
                    trial_preds.append(pred_label)
                    trial_trues.append(true_label)
                
                fold_accs.append(accuracy_score(trial_trues, trial_preds))
                
            subj_acc = np.mean(fold_accs)
            print(f"  {target.capitalize()} Accuracy: {subj_acc:.2%}")
            metrics[target].append(subj_acc)
            
    print("\n=== FINAL RESULTS (Filtered Extreme Trials - DE Features) ===")
    for target in ["valence", "arousal"]:
        mean_acc = np.mean(metrics[target]) if metrics[target] else 0
        print(f"Mean {target.capitalize()}: {mean_acc:.2%} (over {len(metrics[target])} subjects)")

if __name__ == "__main__":
    run_filtered_evaluation()
