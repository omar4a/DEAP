"""Paper 13 Replication - FAST (No Grid Search).

Uses default XGBoost parameters exactly as specified in Paper 13 Table 7.
No hyperparameter tuning - just defaults.
Expected to be ~128x faster than grid search version.
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


def run_fast_experiment(config: Config, subjects: list | None, n_splits: int, progress: bool):
    """Run Paper 13 experiment with default XGBoost (no grid search)."""

    results = {"valence": [], "arousal": []}

    # Live summary file
    output_dir = Path("reports/paper13_fast")
    output_dir.mkdir(parents=True, exist_ok=True)
    live_file = output_dir / "summary_live.txt"

    dataset = load_dataset(config.data_dir, config.eeg_channels, subjects=subjects)

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

        for dim in ["valence", "arousal"]:
            # Build labels
            labels = build_labels(
                subj.labels,
                dimension=dim,
                mode=config.label_mode,
                threshold=config.label_threshold,
                bins=config.label_bins,
            )

            # Expand labels to match segmented features
            if len(labels) != len(features):
                labels = labels[trial_groups]

            # 5-fold stratified CV (Paper 13 setup)
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

            fold_accs = []
            for fold_idx, (train_idx, test_idx) in enumerate(skf.split(features, labels)):
                X_train, X_test = features[train_idx], features[test_idx]
                y_train, y_test = labels[train_idx], labels[test_idx]

                # Build model with DEFAULT parameters (Paper 13 Table 7)
                model = build_model("xgb", params={"random_state": 42})

                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                acc = np.mean(y_pred == y_test)
                fold_accs.append(acc)

            mean_acc = np.mean(fold_accs)
            results[dim].append({
                "subject": subj.subject_id,
                "accuracy": mean_acc,
                "fold_accs": fold_accs,
            })

            if progress:
                print(f"  [{dim}] Subject {subj.subject_id}: {mean_acc*100:.1f}%")

        # Update live summary
        with open(live_file, "w") as f:
            f.write("Paper 13 FAST - Live Progress\n")
            f.write("=" * 40 + "\n")
            f.write(f"Completed: {subj_idx+1}/{total_subjects} subjects\n\n")
            for dim in ["valence", "arousal"]:
                if results[dim]:
                    accs = [r["accuracy"] for r in results[dim]]
                    f.write(f"{dim}: {np.mean(accs)*100:.1f}% ± {np.std(accs)*100:.1f}% (n={len(accs)})\n")

    return results


def main():
    parser = argparse.ArgumentParser(description="Paper 13 FAST replication (no grid search)")
    parser.add_argument("--subjects", nargs="*", type=int, default=None)
    parser.add_argument("--n-splits", type=int, default=5, help="Number of CV folds")
    parser.add_argument("--output-dir", type=Path, default=Path("reports/paper13_fast"))
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

    print("=" * 60)
    print("Paper 13 FAST Replication (No Grid Search)")
    print("=" * 60)
    print(f"Features: {config.feature_set}")
    print(f"Window: {config.window_seconds}s, Step: {config.step_seconds}s")
    print(f"Label: {config.label_mode}, threshold={config.label_threshold}")
    print(f"Preprocessing: bandpass {config.bandpass_low}-{config.bandpass_high}Hz, notch {config.notch_freq}Hz")
    print(f"CV: {args.n_splits}-fold stratified (within-subject)")
    print(f"XGBoost: DEFAULT parameters (Paper 13 Table 7)")
    print("=" * 60)

    results = run_fast_experiment(config, args.subjects, args.n_splits, args.progress)

    # Write results
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)

    for dim in ["valence", "arousal"]:
        accs = [r["accuracy"] for r in results[dim]]
        mean_acc = np.mean(accs)
        std_acc = np.std(accs)
        print(f"{dim}: {mean_acc*100:.1f}% ± {std_acc*100:.1f}%")

        # Write per-subject results
        with open(args.output_dir / f"{dim}_results.txt", "w") as f:
            f.write(f"Paper 13 FAST - {dim}\n")
            f.write(f"Mean: {mean_acc*100:.2f}% ± {std_acc*100:.2f}%\n\n")
            for r in results[dim]:
                f.write(f"Subject {r['subject']}: {r['accuracy']*100:.1f}%\n")
                f.write(f"  Folds: {[f'{a*100:.1f}%' for a in r['fold_accs']]}\n")

    # Final summary
    with open(args.output_dir / "summary_live.txt", "w") as f:
        f.write("Paper 13 FAST - FINAL RESULTS\n")
        f.write("=" * 40 + "\n")
        f.write("XGBoost with DEFAULT parameters\n")
        f.write("5-fold stratified CV (within-subject)\n\n")
        for dim in ["valence", "arousal"]:
            accs = [r["accuracy"] for r in results[dim]]
            f.write(f"{dim}: {np.mean(accs)*100:.1f}% ± {np.std(accs)*100:.1f}%\n")

    print(f"\nResults written to {args.output_dir}")


if __name__ == "__main__":
    main()
