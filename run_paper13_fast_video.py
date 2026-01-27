"""Paper 13 Replication - FAST (No Grid Search, Video-Level).

Uses default XGBoost parameters exactly as specified in Paper 13 Table 7.
No hyperparameter tuning - just defaults.
Expected to be ~128x faster than grid search version.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import numpy as np
from sklearn.model_selection import GroupKFold

from deap_emotion.config import Config
from deap_emotion.data import load_dataset
from deap_emotion.features import extract_features
from deap_emotion.labels import build_labels
from deap_emotion.model_registry import build_model
from deap_emotion.preprocess import baseline_correct, apply_bandpass, apply_notch


def _video_soft_vote_accuracy(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    groups: np.ndarray,
) -> float:
    """Soft vote (mean probability) over 3s windows for each 60s video."""
    trial_ids = np.unique(groups)
    if trial_ids.size == 0:
        return 0.0
    correct = 0
    for trial_id in trial_ids:
        idx = groups == trial_id
        if not np.any(idx):
            continue
        true_label = int(y_true[idx][0])
        probs = y_prob[idx]
        if probs.ndim == 2:
            probs = probs[:, 1]
        mean_prob = float(np.mean(probs))
        pred_label = int(mean_prob >= 0.5)
        correct += int(pred_label == true_label)
    return correct / float(trial_ids.size)


def run_fast_experiment(
    config: Config,
    subjects: list | None,
    n_splits: int,
    progress: bool,
    output_dir: Path,
):
    """Run Paper 13 experiment with default XGBoost (video-level soft vote)."""

    results = {"valence": [], "arousal": []}

    # Live summary file
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

        unique_groups = np.unique(trial_groups)
        if unique_groups.size != len(subj.labels):
            raise ValueError(
                f"trial_groups has {unique_groups.size} unique ids, "
                f"but labels has {len(subj.labels)} trials"
            )
        if not np.array_equal(unique_groups, np.arange(unique_groups.size)):
            group_map = {g: i for i, g in enumerate(unique_groups)}
            trial_groups = np.array([group_map[g] for g in trial_groups], dtype=int)
            if progress:
                print(f"[Subject {subj.subject_id}] Remapped trial_groups to 0..{unique_groups.size-1}")

        for dim in ["valence", "arousal"]:
            # Build labels
            labels = build_labels(
                subj.labels,
                dimension=dim,
                mode=config.label_mode,
                threshold=config.label_threshold,
                bins=config.label_bins,
            )

            # Expand labels to match segmented features (windows)
            window_labels = labels if len(labels) == len(features) else labels[trial_groups]

            # 5-fold grouped CV by video (no leakage across 60s trials)
            gkf = GroupKFold(n_splits=n_splits)

            fold_accs = []
            for fold_idx, (train_idx, test_idx) in enumerate(
                gkf.split(features, window_labels, groups=trial_groups)
            ):
                X_train, X_test = features[train_idx], features[test_idx]
                y_train, y_test = window_labels[train_idx], window_labels[test_idx]

                # Build model with DEFAULT parameters (Paper 13 Table 7)
                model = build_model("xgb", params={"random_state": 42})

                model.fit(X_train, y_train)
                y_prob = model.predict_proba(X_test)

                video_acc = _video_soft_vote_accuracy(
                    y_test,
                    y_prob,
                    trial_groups[test_idx],
                )
                fold_accs.append(video_acc)

            mean_acc = np.mean(fold_accs)
            results[dim].append({
                "subject": subj.subject_id,
                "accuracy": mean_acc,
                "fold_accs": fold_accs,
            })

            if progress:
                print(f"  [{dim}] Subject {subj.subject_id}: {mean_acc*100:.1f}% (video-level)")

        # Update live summary
        with open(live_file, "w") as f:
            f.write("Paper 13 FAST - Live Progress (Video-Level, Soft Vote)\n")
            f.write("=" * 40 + "\n")
            f.write(f"Completed: {subj_idx+1}/{total_subjects} subjects\n\n")
            for dim in ["valence", "arousal"]:
                if results[dim]:
                    accs = [r["accuracy"] for r in results[dim]]
                    f.write(f"{dim}: {np.mean(accs)*100:.1f}% +/- {np.std(accs)*100:.1f}% (n={len(accs)})\n")

    return results


def main():
    parser = argparse.ArgumentParser(description="Paper 13 FAST video-level (soft vote)")
    parser.add_argument("--subjects", nargs="*", type=int, default=None)
    parser.add_argument("--n-splits", type=int, default=5, help="Number of CV folds")
    parser.add_argument("--output-dir", type=Path, default=Path("reports/video-level/paper13_fast"))
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
    print("Paper 13 FAST Replication (Video-Level Soft Vote)")
    print("=" * 60)
    print(f"Features: {config.feature_set}")
    print(f"Window: {config.window_seconds}s, Step: {config.step_seconds}s")
    print(f"Label: {config.label_mode}, threshold={config.label_threshold}")
    print(f"Preprocessing: bandpass {config.bandpass_low}-{config.bandpass_high}Hz, notch {config.notch_freq}Hz")
    print(f"CV: {args.n_splits}-fold GROUPED by video (within-subject, no leakage)")
    print(f"XGBoost: DEFAULT parameters (Paper 13 Table 7)")
    print("=" * 60)

    results = run_fast_experiment(config, args.subjects, args.n_splits, args.progress, args.output_dir)

    # Write results
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("FINAL RESULTS (Video-Level, Soft Vote)")
    print("=" * 60)

    for dim in ["valence", "arousal"]:
        accs = [r["accuracy"] for r in results[dim]]
        mean_acc = np.mean(accs)
        std_acc = np.std(accs)
        print(f"{dim}: {mean_acc*100:.1f}% +/- {std_acc*100:.1f}% (video-level, soft vote)")

        # Write per-subject results
        with open(args.output_dir / f"{dim}_results.txt", "w") as f:
            f.write(f"Paper 13 FAST - {dim}\n")
            f.write(f"Mean: {mean_acc*100:.2f}% +/- {std_acc*100:.2f}% (video-level, soft vote)\n\n")
            for r in results[dim]:
                f.write(f"Subject {r['subject']}: {r['accuracy']*100:.1f}%\n")
                f.write(f"  Folds: {[f'{a*100:.1f}%' for a in r['fold_accs']]}\n")

    # Final summary
    with open(args.output_dir / "summary_live.txt", "w") as f:
        f.write("Paper 13 FAST - FINAL RESULTS (Video-Level, Soft Vote)\n")
        f.write("=" * 40 + "\n")
        f.write("XGBoost with DEFAULT parameters\n")
        f.write("5-fold GROUPED CV by video (within-subject, no leakage)\n")
        f.write("Aggregation: mean probability (soft vote)\n\n")
        for dim in ["valence", "arousal"]:
            accs = [r["accuracy"] for r in results[dim]]
            f.write(f"{dim}: {np.mean(accs)*100:.1f}% +/- {np.std(accs)*100:.1f}%\n")

    print(f"\nResults written to {args.output_dir}")


if __name__ == "__main__":
    main()
