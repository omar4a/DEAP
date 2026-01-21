import json
from pathlib import Path


def load_report(path: Path) -> dict:
    return json.loads(path.read_text())


def compare_reports(old_path: Path, new_path: Path) -> None:
    old = load_report(old_path)
    new = load_report(new_path)

    for model_name, model_data in new.get("models", {}).items():
        old_model = old.get("models", {}).get(model_name, {})
        print(f"== {model_name} ==")
        for mode in ("subject_dependent", "cross_subject"):
            new_mode = model_data.get(mode, {})
            old_mode = old_model.get(mode, {})
            print(f"-- {mode} --")
            for dimension, payload in new_mode.get("dimensions", {}).items():
                new_scores = [f["metrics"]["accuracy"] for f in payload.get("folds", []) if "accuracy" in f["metrics"]]
                old_scores = [
                    f["metrics"]["accuracy"] for f in old_mode.get("dimensions", {}).get(dimension, {}).get("folds", [])
                    if "accuracy" in f["metrics"]
                ]
                if not new_scores or not old_scores:
                    print(f"{dimension}: missing accuracy for comparison")
                    continue
                new_mean = sum(new_scores) / len(new_scores)
                old_mean = sum(old_scores) / len(old_scores)
                delta = new_mean - old_mean
                print(f"{dimension}: {old_mean:.4f} -> {new_mean:.4f} (delta {delta:+.4f})")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compare two experiment reports.")
    parser.add_argument("old_report", type=Path)
    parser.add_argument("new_report", type=Path)
    args = parser.parse_args()

    compare_reports(args.old_report, args.new_report)
