"""Paper 13 Replication Script.

Exact configuration from Paper 13:
- Bandpass: 4-45 Hz (Butterworth 4th order)
- Notch: 50 Hz
- Segments: 3 seconds, no overlap
- Features: Histogram-based DE + Higuchi's Fractal Dimension
- Labels: Binary, threshold 4.5
- Classifier: XGBoost
- CV: 5-fold, 80/20 split

Expected results: ~89% valence, ~88% arousal
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from deap_emotion.config import Config
from deap_emotion.experiment import (
    ExperimentSettings,
    run_experiments,
    write_report,
    write_summary_csv,
    write_summary_txt,
)


def main():
    parser = argparse.ArgumentParser(description="Paper 13 replication experiment")
    parser.add_argument("--subjects", nargs="*", type=int, default=None,
                        help="Subject IDs to process (default: all)")
    parser.add_argument("--outer-repeats", type=int, default=5,
                        help="Number of outer CV repeats")
    parser.add_argument("--inner-folds", type=int, default=4,
                        help="Number of inner CV folds for hyperparameter tuning")
    parser.add_argument("--output-dir", type=Path, default=Path("reports/paper13"),
                        help="Output directory for results")
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="Checkpoint file for resume capability")
    parser.add_argument("--progress", action="store_true",
                        help="Show progress output")
    parser.add_argument("--no-cache", action="store_true",
                        help="Disable feature caching")
    args = parser.parse_args()

    # Paper 13 exact configuration
    config = Config(
        data_dir=Path("archive"),
        # Preprocessing (Paper 13)
        bandpass_low=4.0,
        bandpass_high=45.0,
        bandpass_order=4,
        notch_freq=50.0,
        apply_filtering=True,
        # Segmentation (Paper 13: 3s, no overlap)
        window_seconds=3.0,
        step_seconds=3.0,
        # Features (Paper 13: DE + HFD)
        feature_set=("de_histogram", "hfd"),
        feature_mode="bandpower",
        # Labels (Paper 13: binary, threshold 4.5)
        label_mode="binary",
        label_threshold=4.5,
        # No feature selection for Paper 13 replication
        feature_selection="none",
        use_cache=not args.no_cache,
    )

    settings = ExperimentSettings(
        task="classification",
        dimensions=("valence", "arousal"),
        outer_repeats=args.outer_repeats,
        train_ratio=0.8,
        inner_folds=args.inner_folds,
        scoring="f1_macro",
        progress=args.progress,
        checkpoint_path=args.checkpoint,
        resume=True,
    )

    print("=" * 60)
    print("Paper 13 Replication Experiment")
    print("=" * 60)
    print(f"Features: {config.feature_set}")
    print(f"Window: {config.window_seconds}s, Step: {config.step_seconds}s")
    print(f"Label: {config.label_mode}, threshold={config.label_threshold}")
    print(f"Preprocessing: bandpass {config.bandpass_low}-{config.bandpass_high}Hz, notch {config.notch_freq}Hz")
    print(f"Subjects: {args.subjects or 'all'}")
    print(f"CV: {args.outer_repeats} repeats, {args.inner_folds} inner folds")
    print("=" * 60)

    report = run_experiments(
        config,
        settings,
        model_names=["xgb"],  # Paper 13 uses XGBoost
        subjects=args.subjects,
    )

    # Write outputs
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = write_report(report, args.output_dir)
    csv_path = write_summary_csv(report, args.output_dir)
    txt_path = write_summary_txt(report, args.output_dir)

    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)

    # Print summary
    for model_name, model_data in report.get("models", {}).items():
        sd = model_data.get("subject_dependent", {})
        for dim, payload in sd.get("dimensions", {}).items():
            acc = payload.get("overall_mean_trial_accuracy", 0)
            std = payload.get("overall_std_trial_accuracy", 0)
            print(f"{dim}: {acc*100:.1f}% ± {std*100:.1f}%")

    print(f"\nWrote: {report_path}")
    print(f"Wrote: {csv_path}")
    print(f"Wrote: {txt_path}")


if __name__ == "__main__":
    main()
