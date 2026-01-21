"""CLI for DEAP emotion recognition experiments."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

from .config import Config
from .experiment import (
    ExperimentSettings,
    run_experiments,
    write_report,
    write_summary_csv,
    write_summary_txt,
)


def _load_grid(path: Path) -> Dict[str, dict]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def build_parser() -> argparse.ArgumentParser:
    defaults = Config()
    parser = argparse.ArgumentParser(description="DEAP experiment runner.")

    # Data options
    parser.add_argument("--data-dir", type=Path, default=defaults.data_dir)
    parser.add_argument("--subjects", nargs="*", type=int, default=None)
    parser.add_argument("--trials", nargs="*", type=int, default=None)

    # Label options
    parser.add_argument(
        "--label-mode",
        choices=["binary", "three_class", "quartile", "regression",
                 "subject_median", "subject_tertile", "subject_mean"],
        default=defaults.label_mode,
    )
    parser.add_argument("--label-threshold", type=float, default=defaults.label_threshold)
    parser.add_argument("--label-bins", nargs="+", type=float, default=None)

    # Preprocessing options (Paper 13)
    parser.add_argument("--bandpass-low", type=float, default=defaults.bandpass_low)
    parser.add_argument("--bandpass-high", type=float, default=defaults.bandpass_high)
    parser.add_argument("--bandpass-order", type=int, default=defaults.bandpass_order)
    parser.add_argument("--notch-freq", type=float, default=defaults.notch_freq,
                        help="Notch filter Hz (0 to disable)")
    parser.add_argument("--no-filtering", action="store_true")

    # Feature options
    parser.add_argument("--window-seconds", type=float, default=defaults.window_seconds)
    parser.add_argument("--step-seconds", type=float, default=defaults.step_seconds)
    parser.add_argument("--feature-mode", default=defaults.feature_mode)
    parser.add_argument(
        "--feature-set", nargs="+",
        choices=["bandpower", "rel_bandpower", "psd", "de", "de_histogram",
                 "asymmetry", "connectivity", "connectivity_pcc", "connectivity_plv",
                 "riemann", "statistical", "hjorth", "hfd"],
        default=None,
    )
    parser.add_argument("--feature-selection", choices=["none", "mi", "pca", "l1"],
                        default=defaults.feature_selection)
    parser.add_argument("--mi-k", type=int, default=defaults.mi_k)
    parser.add_argument("--pca-variance", type=float, default=None)
    parser.add_argument("--scaler", choices=["standard", "robust"], default=defaults.scaler)

    # Model options
    parser.add_argument("--models", nargs="+", default=["xgb"])
    parser.add_argument("--grid", type=Path, default=Path("configs/grid_default.json"))

    # CV options
    parser.add_argument("--outer-repeats", type=int, default=5)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--inner-folds", type=int, default=4)
    parser.add_argument("--search-strategy", choices=["grid", "random"], default="grid")
    parser.add_argument("--search-trials", type=int, default=None)
    parser.add_argument("--scoring", default="f1_macro")

    # Task options
    parser.add_argument("--task", choices=["classification", "regression"], default="classification")
    parser.add_argument("--dimensions", nargs="+", default=["valence", "arousal"])
    parser.add_argument("--trial-aggregation", choices=["majority", "mean_prob"],
                        default=defaults.trial_aggregation)

    # Output options
    parser.add_argument("--output-dir", type=Path, default=Path("reports/experiment"))
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--progress-file", type=Path, default=None)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    defaults = Config()
    label_bins = tuple(args.label_bins) if args.label_bins else defaults.label_bins
    feature_set = tuple(args.feature_set) if args.feature_set else defaults.feature_set
    notch_freq = None if args.notch_freq == 0 else args.notch_freq

    config = Config(
        data_dir=args.data_dir,
        label_mode=args.label_mode,
        label_threshold=args.label_threshold,
        label_bins=label_bins,
        window_seconds=args.window_seconds,
        step_seconds=args.step_seconds,
        feature_mode=args.feature_mode,
        feature_set=feature_set,
        pca_variance=args.pca_variance,
        feature_selection=args.feature_selection,
        mi_k=args.mi_k,
        scaler=args.scaler,
        trial_aggregation=args.trial_aggregation,
        use_cache=not args.no_cache,
        bandpass_low=args.bandpass_low,
        bandpass_high=args.bandpass_high,
        bandpass_order=args.bandpass_order,
        notch_freq=notch_freq,
        apply_filtering=not args.no_filtering,
    )

    settings = ExperimentSettings(
        task=args.task,
        dimensions=tuple(args.dimensions),
        outer_repeats=args.outer_repeats,
        train_ratio=args.train_ratio,
        inner_folds=args.inner_folds,
        search_strategy=args.search_strategy,
        search_trials=args.search_trials,
        scoring=args.scoring,
        use_cache=not args.no_cache,
        progress=args.progress,
        progress_path=args.progress_file,
        checkpoint_path=args.checkpoint,
        resume=not args.no_resume,
    )

    grid_overrides = _load_grid(args.grid)

    report = run_experiments(
        config,
        settings,
        model_names=args.models,
        grid_overrides=grid_overrides,
        subjects=args.subjects,
        trial_ids=args.trials,
    )

    report_path = write_report(report, args.output_dir)
    csv_path = write_summary_csv(report, args.output_dir)
    txt_path = write_summary_txt(report, args.output_dir)
    print(f"Wrote {report_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {txt_path}")


if __name__ == "__main__":
    main()
