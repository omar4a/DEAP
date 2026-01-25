"""Paper 13 Replication - POOLED CV (DATA LEAKAGE).

!!! WARNING: THIS EXPERIMENT HAS DATA LEAKAGE !!!
!!! DO NOT USE THESE RESULTS FOR SCIENTIFIC CLAIMS !!!

This replicates Paper 13's EXACT methodology which pools all subjects
together before CV, allowing segments from the same subject to appear
in both train and test sets. This artificially inflates accuracy.

Purpose: Verify that pooled CV explains Paper 13's inflated 88-89% accuracy.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import numpy as np
from sklearn.model_selection import StratifiedKFold

from deap_emotion.config import Config
from deap_emotion.data import load_dataset
from deap_emotion.features import extract_features
from deap_emotion.labels import build_labels
from deap_emotion.model_registry import build_model
from deap_emotion.preprocess import baseline_correct, apply_bandpass, apply_notch


def run_pooled_experiment(config: Config, subjects: list | None, n_splits: int, progress: bool):
    """
    Run Paper 13 experiment with POOLED CV (data leakage).

    WARNING: This allows same-subject data in train and test!
    """

    # Output directory - clearly flagged
    output_dir = Path("reports/paper13_LEAKAGE_POOLED_CV")
    output_dir.mkdir(parents=True, exist_ok=True)
    live_file = output_dir / "summary_live.txt"

    dataset = load_dataset(config.data_dir, config.eeg_channels, subjects=subjects)

    # Collect ALL features from ALL subjects into one pool
    all_features = []
    all_labels_valence = []
    all_labels_arousal = []
    all_subject_ids = []  # Track which subject each sample came from

    total_subjects = len(dataset)

    for subj_idx, subj in enumerate(dataset):
        if progress:
            print(f"[Subject {subj.subject_id}] ({subj_idx+1}/{total_subjects}) Extracting features...")

        # Preprocess EEG
        eeg = subj.eeg
        if config.apply_filtering:
            eeg = apply_bandpass(eeg, config.bandpass_low, config.bandpass_high,
                                 config.sfreq, config.bandpass_order)
            if config.notch_freq:
                eeg = apply_notch(eeg, config.notch_freq, config.sfreq)

        # Baseline correction
        eeg = baseline_correct(eeg, config.baseline_samples())

        # Extract features
        features, trial_groups = extract_features(
            eeg,
            config.sfreq,
            config.band_edges(),
            config.feature_mode,
            window_samples=config.window_samples(),
            step_samples=config.step_samples(),
            feature_set=config.feature_set,
        )

        # Build labels for both dimensions
        labels_v = build_labels(
            subj.labels, dimension="valence", mode=config.label_mode,
            threshold=config.label_threshold, bins=config.label_bins,
        )
        labels_a = build_labels(
            subj.labels, dimension="arousal", mode=config.label_mode,
            threshold=config.label_threshold, bins=config.label_bins,
        )

        # Expand labels to match segmented features
        if len(labels_v) != len(features):
            labels_v = labels_v[trial_groups]
            labels_a = labels_a[trial_groups]

        # Add to pool
        all_features.append(features)
        all_labels_valence.append(labels_v)
        all_labels_arousal.append(labels_a)
        all_subject_ids.append(np.full(len(features), subj.subject_id))

    # Concatenate all subjects into one big pool
    X = np.concatenate(all_features, axis=0)
    y_valence = np.concatenate(all_labels_valence, axis=0)
    y_arousal = np.concatenate(all_labels_arousal, axis=0)
    subject_ids = np.concatenate(all_subject_ids, axis=0)

    if progress:
        print(f"\n*** POOLED DATA: {len(X)} samples from {total_subjects} subjects ***")
        print("*** WARNING: DATA LEAKAGE - Same subjects in train/test! ***\n")

    results = {"valence": [], "arousal": []}

    for dim, y in [("valence", y_valence), ("arousal", y_arousal)]:
        if progress:
            print(f"[{dim}] Running {n_splits}-fold CV on POOLED data (LEAKAGE!)...")

        # 5-fold stratified CV on the POOLED data (Paper 13 methodology)
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        fold_accs = []
        leakage_stats = []

        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Calculate leakage: how many subjects appear in BOTH train and test?
            train_subjects = set(subject_ids[train_idx])
            test_subjects = set(subject_ids[test_idx])
            leaked_subjects = train_subjects & test_subjects
            leakage_pct = len(leaked_subjects) / len(test_subjects) * 100

            leakage_stats.append(leakage_pct)

            # Build model with DEFAULT parameters (same as FAST experiment)
            model = build_model("xgb", params={"random_state": 42})

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            acc = np.mean(y_pred == y_test)
            fold_accs.append(acc)

            if progress:
                print(f"  Fold {fold_idx+1}: {acc*100:.1f}% (leakage: {leakage_pct:.0f}% of test subjects in train)")

        mean_acc = np.mean(fold_accs)
        std_acc = np.std(fold_accs)
        mean_leakage = np.mean(leakage_stats)

        results[dim] = {
            "accuracy": mean_acc,
            "std": std_acc,
            "fold_accs": fold_accs,
            "mean_leakage_pct": mean_leakage,
        }

        if progress:
            print(f"  {dim} FINAL: {mean_acc*100:.1f}% ± {std_acc*100:.1f}% (avg leakage: {mean_leakage:.0f}%)\n")

        # Update live summary
        with open(live_file, "w") as f:
            f.write("!!! WARNING: DATA LEAKAGE EXPERIMENT !!!\n")
            f.write("!!! DO NOT USE FOR SCIENTIFIC CLAIMS !!!\n")
            f.write("=" * 50 + "\n\n")
            f.write("Paper 13 POOLED CV Replication\n")
            f.write("Purpose: Verify that pooled CV explains inflated accuracy\n\n")
            f.write("Results:\n")
            for d in ["valence", "arousal"]:
                if d in results and results[d]:
                    r = results[d]
                    f.write(f"  {d}: {r['accuracy']*100:.1f}% ± {r['std']*100:.1f}%\n")
                    f.write(f"    (avg {r['mean_leakage_pct']:.0f}% of test subjects leaked into train)\n")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Paper 13 POOLED CV (DATA LEAKAGE) - DO NOT USE FOR SCIENCE"
    )
    parser.add_argument("--subjects", nargs="*", type=int, default=None)
    parser.add_argument("--n-splits", type=int, default=5, help="Number of CV folds")
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    # Paper 13 exact configuration
    config = Config(
        data_dir=Path("archive"),
        bandpass_low=4.0,
        bandpass_high=45.0,
        bandpass_order=4,
        notch_freq=50.0,
        apply_filtering=True,
        window_seconds=3.0,
        step_seconds=3.0,
        feature_set=("de_histogram", "hfd"),
        label_mode="binary",
        label_threshold=4.5,
        feature_selection="none",
        use_cache=not args.no_cache,
    )

    print("!" * 60)
    print("!!! WARNING: DATA LEAKAGE EXPERIMENT !!!")
    print("!!! DO NOT USE THESE RESULTS FOR SCIENTIFIC CLAIMS !!!")
    print("!" * 60)
    print()
    print("Paper 13 POOLED CV Replication")
    print("=" * 60)
    print(f"Features: {config.feature_set}")
    print(f"Window: {config.window_seconds}s, Step: {config.step_seconds}s")
    print(f"Label: {config.label_mode}, threshold={config.label_threshold}")
    print(f"CV: {args.n_splits}-fold stratified on POOLED data")
    print(f"XGBoost: DEFAULT parameters")
    print()
    print("METHODOLOGY: All subjects pooled, then CV applied")
    print("LEAKAGE: Same subject segments can be in train AND test!")
    print("=" * 60)

    results = run_pooled_experiment(config, args.subjects, args.n_splits, args.progress)

    # Write final results
    output_dir = Path("reports/paper13_LEAKAGE_POOLED_CV")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("FINAL RESULTS (WITH DATA LEAKAGE)")
    print("=" * 60)

    for dim in ["valence", "arousal"]:
        r = results[dim]
        print(f"{dim}: {r['accuracy']*100:.1f}% ± {r['std']*100:.1f}%")
        print(f"  Avg leakage: {r['mean_leakage_pct']:.0f}% of test subjects in train")

    # Write detailed results
    with open(output_dir / "results_LEAKAGE.txt", "w") as f:
        f.write("!!! WARNING: DATA LEAKAGE EXPERIMENT !!!\n")
        f.write("!!! DO NOT USE FOR SCIENTIFIC CLAIMS !!!\n\n")
        f.write("Paper 13 Pooled CV Replication\n")
        f.write("=" * 50 + "\n\n")
        f.write("Purpose: Verify that pooled CV explains Paper 13's 88-89% accuracy\n\n")
        f.write("Configuration:\n")
        f.write(f"  Features: {config.feature_set}\n")
        f.write(f"  Window: {config.window_seconds}s\n")
        f.write(f"  Threshold: {config.label_threshold}\n")
        f.write(f"  CV: {args.n_splits}-fold on pooled data\n\n")
        f.write("Results:\n")
        for dim in ["valence", "arousal"]:
            r = results[dim]
            f.write(f"\n{dim}:\n")
            f.write(f"  Accuracy: {r['accuracy']*100:.1f}% ± {r['std']*100:.1f}%\n")
            f.write(f"  Fold accuracies: {[f'{a*100:.1f}%' for a in r['fold_accs']]}\n")
            f.write(f"  Avg leakage: {r['mean_leakage_pct']:.0f}% of test subjects in train\n")

        f.write("\n\nComparison:\n")
        f.write("  Paper 13 reported: 89% valence, 88% arousal\n")
        f.write("  Our proper CV:     69.6% valence, 71.8% arousal\n")
        f.write("  This leaky CV:     See results above\n")

    print(f"\nResults written to {output_dir}")
    print("\n" + "!" * 60)
    print("!!! REMINDER: THESE RESULTS HAVE DATA LEAKAGE !!!")
    print("!" * 60)


if __name__ == "__main__":
    main()
