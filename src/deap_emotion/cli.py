from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import Config
from .eval import metrics_to_dict, run_cross_subject, run_subject_dependent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DEAP emotion recognition (approach 3).")
    parser.add_argument("--data-dir", type=Path, default=Path("archive"))
    parser.add_argument("--label-mode", choices=["binary", "quartile"], default="binary")
    parser.add_argument("--label-threshold", type=float, default=5.0)
    parser.add_argument("--window-seconds", type=float, default=None)
    parser.add_argument("--step-seconds", type=float, default=None)
    parser.add_argument("--classifier", choices=["logreg", "linear_svm"], default="logreg")
    parser.add_argument("--subject-folds", type=int, default=5)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("reports/metrics.json"))
    parser.add_argument("--mode", choices=["subject", "cross", "both"], default="both")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    config = Config(
        data_dir=args.data_dir,
        label_mode=args.label_mode,
        label_threshold=args.label_threshold,
        window_seconds=args.window_seconds,
        step_seconds=args.step_seconds,
        classifier=args.classifier,
        subject_folds=args.subject_folds,
        use_cache=not args.no_cache,
    )

    output = {}
    if args.mode in ("subject", "both"):
        output["subject_dependent"] = metrics_to_dict(run_subject_dependent(config))
    if args.mode in ("cross", "both"):
        output["cross_subject"] = metrics_to_dict(run_cross_subject(config))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2))
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
