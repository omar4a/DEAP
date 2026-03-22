import sys
import os
import numpy as np
from pathlib import Path
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

sys.path.append(str(Path(__file__).resolve().parent / "src"))
from deap_emotion.config import Config
from deap_emotion.data import load_dataset
from deap_emotion.features import extract_features
from deap_emotion.labels import DIMENSIONS

def quick_ml_benchmark():
    config = Config(
        feature_set=("bandpower",),
        window_seconds=3.0,
        step_seconds=3.0,
        per_subject_normalize=True,
    )
    
    subjects = list(range(1, 33)) # 1 to 32
    print("Loading dataset for all subjects...")
    dataset = load_dataset(
        data_dir=config.data_dir, 
        eeg_channels=config.eeg_channels, 
        subjects=subjects
    )
    
    metrics = {"valence": {}, "arousal": {}}
    
    print("Extracting features and evaluating...")
    for subject in dataset:
        
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
        
        # We need to flatten X for Logistic regression: [samples, features]
        # extract_features already flattens it if needed, but let's check
        if len(X.shape) > 2:
            X = X.reshape(X.shape[0], -1)
            
        for target in ["valence", "arousal"]:
            dim_idx = DIMENSIONS[target]
            
            target_ratings = subject.labels[:, dim_idx]
            window_ratings = target_ratings[groups]
            
            # Standard binary label thresholding at 5.0
            y_binary = (window_ratings >= 5.0).astype(int)
            
            # 4. Cross Validation
            gkf = GroupKFold(n_splits=5)
            fold_accs = []
            
            # If only one class is present after binarization, acc is 100% or ill-defined
            if len(np.unique(y_binary)) < 2:
               metrics[target][subject.subject_id] = 1.0
               continue
               
            for train_idx, test_idx in gkf.split(X, y_binary, groups):
                X_train, y_train = X[train_idx], y_binary[train_idx]
                X_test = X[test_idx]
                
                clf = LogisticRegression(max_iter=1000, n_jobs=-1)
                clf.fit(X_train, y_train)
                
                y_pred = clf.predict(X_test)
                
                # Video level aggregation
                unique_trials = np.unique(groups[test_idx])
                trial_preds = []
                trial_trues = []
                
                for trial in unique_trials:
                    trial_mask = (groups[test_idx] == trial)
                    pred_label = 1 if np.mean(y_pred[trial_mask]) >= 0.5 else 0
                    true_label = y_binary[test_idx][trial_mask][0]
                    
                    trial_preds.append(pred_label)
                    trial_trues.append(true_label)
                
                fold_accs.append(accuracy_score(trial_trues, trial_preds))
                
            subj_acc = np.mean(fold_accs)
            metrics[target][subject.subject_id] = subj_acc

    # Output bottom 8 subjects for ML
    for target in ["valence", "arousal"]:
        print(f"\n=== Classical ML - {target.capitalize()} ===")
        subj_accs = metrics[target]
        sorted_accs = sorted(subj_accs.items(), key=lambda x: x[1])
        bottom_quarter = sorted_accs[:8]
        top_three_quarters = sorted_accs[8:]
        
        bottom_subs = [s for s, a in bottom_quarter]
        bottom_mean = np.mean([a for s, a in bottom_quarter])
        full_mean = np.mean([a for s, a in sorted_accs])
        new_mean = np.mean([a for s, a in top_three_quarters])
        
        print(f"Original Mean Accuracy (32 subjects): {full_mean:.2%}")
        print(f"Bottom 25% Subjects (8 subjects): {bottom_subs}")
        print(f"Bottom 25% Mean Accuracy: {bottom_mean:.2%}")
        print(f"New Mean Accuracy (Top 24 subjects): {new_mean:.2%}")
        print(f"Absolute Jump: {new_mean - full_mean:.2%}")

if __name__ == "__main__":
    quick_ml_benchmark()
