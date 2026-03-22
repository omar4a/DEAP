"""Leakage-safe grouped EEGNet experiment on DEAP."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from torch.optim import AdamW

try:
    from . import config
    from .data import (
        build_dataloader,
        build_inner_split,
        build_outer_splits,
        get_target_labels,
        load_subject,
        make_window_split,
        preprocess_trials,
    )
    from .model import build_model
except ImportError:
    import config
    from data import (
        build_dataloader,
        build_inner_split,
        build_outer_splits,
        get_target_labels,
        load_subject,
        make_window_split,
        preprocess_trials,
    )
    from model import build_model


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def aggregate_video_predictions(
    parent_trials: np.ndarray,
    probs_high: np.ndarray,
    true_window_labels: np.ndarray,
) -> tuple[list[dict], dict]:
    """Aggregate window probabilities into one prediction per video."""
    grouped_probs: dict[int, list[float]] = defaultdict(list)
    grouped_labels: dict[int, int] = {}

    for trial_id, prob_high, label in zip(parent_trials, probs_high, true_window_labels):
        trial_key = int(trial_id)
        grouped_probs[trial_key].append(float(prob_high))
        grouped_labels[trial_key] = int(label)

    rows = []
    y_true = []
    y_pred = []
    y_prob = []

    for trial_id in sorted(grouped_probs):
        mean_prob_high = float(np.mean(grouped_probs[trial_id]))
        true_label = grouped_labels[trial_id]
        pred_label = int(mean_prob_high >= 0.5)

        rows.append(
            {
                "trial_id": trial_id,
                "true_label": true_label,
                "pred_label": pred_label,
                "mean_prob_high": mean_prob_high,
                "n_windows": len(grouped_probs[trial_id]),
            }
        )
        y_true.append(true_label)
        y_pred.append(pred_label)
        y_prob.append(mean_prob_high)

    y_true_arr = np.asarray(y_true, dtype=np.int64)
    y_pred_arr = np.asarray(y_pred, dtype=np.int64)
    y_prob_arr = np.asarray(y_prob, dtype=np.float64)

    metrics = {
        "video_accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
        "video_macro_f1": float(f1_score(y_true_arr, y_pred_arr, average="macro", zero_division=0)),
        "video_auc": (
            float(roc_auc_score(y_true_arr, y_prob_arr))
            if len(np.unique(y_true_arr)) > 1
            else 0.5
        ),
        "confusion_matrix": confusion_matrix(y_true_arr, y_pred_arr, labels=[0, 1]).tolist(),
        "n_videos": int(len(rows)),
    }
    return rows, metrics


def evaluate_loader(
    model: nn.Module,
    loader,
    split: dict,
    criterion: nn.Module,
) -> tuple[float, list[dict], dict]:
    """Evaluate a loader and return window loss plus aggregated video metrics."""
    model.eval()

    losses = []
    probs_high = []

    with torch.no_grad():
        for windows, labels in loader:
            windows = windows.to(DEVICE)
            labels = labels.to(DEVICE)
            logits = model(windows)
            losses.append(float(criterion(logits, labels).item()))
            probs_high.extend(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())

    mean_loss = float(np.mean(losses)) if losses else 0.0
    rows, metrics = aggregate_video_predictions(
        parent_trials=split["parent_trials"],
        probs_high=np.asarray(probs_high, dtype=np.float64),
        true_window_labels=split["labels"],
    )
    return mean_loss, rows, metrics


def _is_better_score(candidate: dict, best: dict | None) -> bool:
    if best is None:
        return True
    if candidate[config.MODEL_SELECTION_METRIC] != best[config.MODEL_SELECTION_METRIC]:
        return candidate[config.MODEL_SELECTION_METRIC] > best[config.MODEL_SELECTION_METRIC]
    if candidate["video_accuracy"] != best["video_accuracy"]:
        return candidate["video_accuracy"] > best["video_accuracy"]
    return candidate["val_loss"] < best["val_loss"]


def build_class_weights(labels: np.ndarray) -> tuple[torch.Tensor, dict]:
    """Build inverse-frequency class weights from the inner-train set."""
    counts = np.bincount(labels, minlength=2).astype(np.float32)
    safe_counts = np.maximum(counts, 1.0)
    weights = counts.sum() / (len(counts) * safe_counts)
    weights_tensor = torch.tensor(weights, dtype=torch.float32, device=DEVICE)
    metadata = {
        "class_counts": {"low": int(counts[0]), "high": int(counts[1])},
        "class_weights": {"low": float(weights[0]), "high": float(weights[1])},
    }
    return weights_tensor, metadata


def train_one_fold(
    subject_id: int,
    target: str,
    seed: int,
    outer_fold: int,
    subject,
    outer_train_ids: np.ndarray,
    outer_test_ids: np.ndarray,
    epochs: int,
    force: bool,
    results_dir: Path,
) -> tuple[dict, list[dict]]:
    """Train and evaluate one subject-target-seed-fold run."""
    fold_dir = (
        results_dir
        / f"subject_{subject_id:02d}"
        / target
        / f"seed_{seed}"
        / f"fold_{outer_fold:02d}"
    )
    fold_dir.mkdir(parents=True, exist_ok=True)

    summary_path = fold_dir / "fold_summary.json"
    preds_path = fold_dir / "test_video_predictions.csv"
    metrics_path = fold_dir / "training_metrics.csv"
    best_path = fold_dir / "best_model.pth"
    split_path = fold_dir / "split.json"

    if not force and summary_path.exists() and preds_path.exists():
        with summary_path.open() as handle:
            summary = json.load(handle)
        with preds_path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        return summary, rows

    inner_train_ids, inner_val_ids = build_inner_split(
        train_trial_ids=outer_train_ids,
        random_state=(seed * 1000) + subject_id * 10 + outer_fold,
    )

    trial_labels = get_target_labels(subject.ratings, target)
    processed_trials, norm_stats = preprocess_trials(subject.eeg, train_trial_ids=inner_train_ids)

    train_split = make_window_split(processed_trials, inner_train_ids, trial_labels)
    val_split = make_window_split(processed_trials, inner_val_ids, trial_labels)
    test_split = make_window_split(processed_trials, outer_test_ids, trial_labels)

    train_loader = build_dataloader(train_split, shuffle=True)
    val_loader = build_dataloader(val_split, shuffle=False)
    test_loader = build_dataloader(test_split, shuffle=False)

    set_seed(seed)
    model = build_model().to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY)
    class_weights, class_weight_metadata = build_class_weights(train_split["labels"])
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    with metrics_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "epoch",
                "train_loss",
                "val_loss",
                "val_video_accuracy",
                "val_video_macro_f1",
                "val_video_auc",
            ]
        )

    best_snapshot = None
    best_epoch = 0
    patience_counter = 0
    stopped_epoch = epochs

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for windows, labels in train_loader:
            windows = windows.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            logits = model(windows)
            loss = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(float(loss.item()))

        train_loss = float(np.mean(train_losses)) if train_losses else 0.0
        val_loss, _, val_metrics = evaluate_loader(model, val_loader, val_split, criterion)

        with metrics_path.open("a", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    epoch,
                    f"{train_loss:.6f}",
                    f"{val_loss:.6f}",
                    f"{val_metrics['video_accuracy']:.6f}",
                    f"{val_metrics['video_macro_f1']:.6f}",
                    f"{val_metrics['video_auc']:.6f}",
                ]
            )

        candidate = {
            config.MODEL_SELECTION_METRIC: val_metrics[config.MODEL_SELECTION_METRIC],
            "video_macro_f1": val_metrics["video_macro_f1"],
            "video_accuracy": val_metrics["video_accuracy"],
            "val_loss": val_loss,
        }
        if _is_better_score(candidate, best_snapshot):
            best_snapshot = candidate
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), best_path)
        else:
            patience_counter += 1

        if epoch == 1 or epoch % 10 == 0:
            print(
                f"        ep {epoch:3d}/{epochs} train={train_loss:.4f} "
                f"val_loss={val_loss:.4f} val_acc={val_metrics['video_accuracy']:.4f} "
                f"val_f1={val_metrics['video_macro_f1']:.4f}"
            )

        if patience_counter >= config.EARLY_STOPPING_PATIENCE:
            stopped_epoch = epoch
            break

    model.load_state_dict(torch.load(best_path, map_location=DEVICE))
    test_loss, test_rows, test_metrics = evaluate_loader(model, test_loader, test_split, criterion)

    with preds_path.open("w", newline="") as handle:
        fieldnames = ["trial_id", "true_label", "pred_label", "mean_prob_high", "n_windows"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(test_rows)

    split_payload = {
        "outer_train_trial_ids": outer_train_ids.tolist(),
        "outer_test_trial_ids": outer_test_ids.tolist(),
        "inner_train_trial_ids": inner_train_ids.tolist(),
        "inner_val_trial_ids": inner_val_ids.tolist(),
        "model_selection_metric": config.MODEL_SELECTION_METRIC,
        **class_weight_metadata,
        "normalization_stats": norm_stats,
    }
    split_path.write_text(json.dumps(split_payload, indent=2))

    summary = {
        "subject_id": subject_id,
        "target": target,
        "seed": seed,
        "outer_fold": outer_fold,
        "best_epoch": best_epoch,
        "stopped_epoch": stopped_epoch,
        "train_trial_count": int(len(inner_train_ids)),
        "val_trial_count": int(len(inner_val_ids)),
        "test_trial_count": int(len(outer_test_ids)),
        "train_window_count": int(len(train_split["windows"])),
        "val_window_count": int(len(val_split["windows"])),
        "test_window_count": int(len(test_split["windows"])),
        "test_loss": test_loss,
        "model_selection_metric": config.MODEL_SELECTION_METRIC,
        **class_weight_metadata,
        **test_metrics,
    }
    summary_path.write_text(json.dumps(summary, indent=2))

    print(
        f"        TEST videos={test_metrics['n_videos']} "
        f"acc={test_metrics['video_accuracy']:.4f} "
        f"f1={test_metrics['video_macro_f1']:.4f}"
    )

    return summary, test_rows


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    """Write a list of dictionaries to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_summaries(fold_rows: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    """Aggregate fold metrics first across folds, then across seeds."""
    per_seed = []
    grouped_seed: dict[tuple[int, str, int], list[dict]] = defaultdict(list)
    for row in fold_rows:
        grouped_seed[(row["subject_id"], row["target"], row["seed"])].append(row)

    for (subject_id, target, seed), rows in sorted(grouped_seed.items()):
        accs = [row["video_accuracy"] for row in rows]
        f1s = [row["video_macro_f1"] for row in rows]
        aucs = [row["video_auc"] for row in rows]
        best_epochs = [row["best_epoch"] for row in rows]
        stops = [row["stopped_epoch"] for row in rows]
        videos = [row["test_trial_count"] for row in rows]
        per_seed.append(
            {
                "subject_id": subject_id,
                "target": target,
                "seed": seed,
                "fold_count": len(rows),
                "video_accuracy_mean": float(np.mean(accs)),
                "video_accuracy_std": float(np.std(accs)),
                "video_macro_f1_mean": float(np.mean(f1s)),
                "video_macro_f1_std": float(np.std(f1s)),
                "video_auc_mean": float(np.mean(aucs)),
                "video_auc_std": float(np.std(aucs)),
                "best_epoch_mean": float(np.mean(best_epochs)),
                "stopped_epoch_mean": float(np.mean(stops)),
                "videos_per_fold": int(videos[0]) if videos else 0,
            }
        )

    per_subject = []
    grouped_subject: dict[tuple[int, str], list[dict]] = defaultdict(list)
    for row in per_seed:
        grouped_subject[(row["subject_id"], row["target"])].append(row)

    for (subject_id, target), rows in sorted(grouped_subject.items()):
        accs = [row["video_accuracy_mean"] for row in rows]
        f1s = [row["video_macro_f1_mean"] for row in rows]
        aucs = [row["video_auc_mean"] for row in rows]
        per_subject.append(
            {
                "subject_id": subject_id,
                "target": target,
                "seed_count": len(rows),
                "video_accuracy_mean": float(np.mean(accs)),
                "video_accuracy_std": float(np.std(accs)),
                "video_macro_f1_mean": float(np.mean(f1s)),
                "video_macro_f1_std": float(np.std(f1s)),
                "video_auc_mean": float(np.mean(aucs)),
                "video_auc_std": float(np.std(aucs)),
            }
        )

    overall = []
    grouped_target: dict[str, list[dict]] = defaultdict(list)
    for row in per_subject:
        grouped_target[row["target"]].append(row)

    for target, rows in sorted(grouped_target.items()):
        accs = [row["video_accuracy_mean"] for row in rows]
        f1s = [row["video_macro_f1_mean"] for row in rows]
        aucs = [row["video_auc_mean"] for row in rows]
        overall.append(
            {
                "target": target,
                "subject_count": len(rows),
                "video_accuracy_mean": float(np.mean(accs)),
                "video_accuracy_std": float(np.std(accs)),
                "video_macro_f1_mean": float(np.mean(f1s)),
                "video_macro_f1_std": float(np.std(f1s)),
                "video_auc_mean": float(np.mean(aucs)),
                "video_auc_std": float(np.std(aucs)),
            }
        )

    return per_seed, per_subject, overall


def write_confusion_matrices(video_rows: list[dict], results_dir: Path) -> None:
    """Write aggregated video-level confusion matrices for each target."""
    grouped_target: dict[str, list[dict]] = defaultdict(list)
    for row in video_rows:
        grouped_target[row["target"]].append(row)

    for target, rows in grouped_target.items():
        y_true = [int(row["true_label"]) for row in rows]
        y_pred = [int(row["pred_label"]) for row in rows]
        matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
        path = results_dir / f"confusion_matrix_{target}.csv"
        with path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["", "pred_low", "pred_high"])
            writer.writerow(["true_low", int(matrix[0, 0]), int(matrix[0, 1])])
            writer.writerow(["true_high", int(matrix[1, 0]), int(matrix[1, 1])])


def run(args) -> None:
    """Execute the grouped EEGNet experiment."""
    if args.notch_freq is not None:
        config.NOTCH_FREQ = args.notch_freq

    results_dir = Path(
        args.results_dir
        or (config.DEBUG_RESULTS_DIR if args.debug else config.RESULTS_DIR)
    )
    results_dir.mkdir(parents=True, exist_ok=True)

    subjects = args.subjects or list(range(1, config.N_SUBJECTS + 1))
    targets = args.targets or list(config.TARGETS)
    seeds = args.seeds or list(config.DEFAULT_SEEDS)
    epochs = args.epochs or config.EPOCHS
    max_outer_folds = args.max_outer_folds or config.OUTER_FOLDS

    if args.debug:
        subjects = subjects[:1]
        targets = targets[:1]
        seeds = seeds[:1]
        epochs = min(epochs, 2)
        max_outer_folds = 1

    run_config = {
        "subjects": subjects,
        "targets": targets,
        "seeds": seeds,
        "epochs": epochs,
        "max_outer_folds": max_outer_folds,
        "window_sec": config.WINDOW_SEC,
        "stride_sec": config.STRIDE_SEC,
        "window_samples": config.WINDOW_SAMPLES,
        "stride_samples": config.STRIDE_SAMPLES,
        "windows_per_trial": config.WINDOWS_PER_TRIAL,
        "outer_folds": config.OUTER_FOLDS,
        "inner_val_trials": config.INNER_VAL_TRIALS,
        "model_selection_metric": config.MODEL_SELECTION_METRIC,
        "device": str(DEVICE),
        "notch_freq": config.NOTCH_FREQ,
    }
    Path(results_dir, "run_config.json").write_text(json.dumps(run_config, indent=2))

    fold_rows = []
    video_rows = []

    print(f"\nDevice: {DEVICE}")
    print(f"Window: {config.WINDOW_SEC:.1f}s, stride: {config.STRIDE_SEC:.1f}s")
    print(f"Windows per trial: {config.WINDOWS_PER_TRIAL}")
    print(f"Seeds: {seeds}")

    for subject_id in subjects:
        print(f"\nSubject {subject_id:02d}")
        subject = load_subject(subject_id)

        for target in targets:
            print(f"  Target: {target}")
            
            if target == "valence" and subject_id in getattr(config, "EXCLUDE_SUBJECTS_VALENCE", []):
                print(f"    Skipping subject {subject_id} for target {target} (exclusion list)")
                continue
            if target == "arousal" and subject_id in getattr(config, "EXCLUDE_SUBJECTS_AROUSAL", []):
                print(f"    Skipping subject {subject_id} for target {target} (exclusion list)")
                continue

            from data import filter_neutral_trials, build_outer_splits
            valid_trial_ids = filter_neutral_trials(subject.ratings, target)
            if len(valid_trial_ids) < 10:
                print(f"    Skipping subject {subject_id} for target {target}: only {len(valid_trial_ids)} valid trials")
                continue
                
            outer_splits = build_outer_splits(valid_trial_ids)[:max_outer_folds]

            for seed in seeds:
                print(f"    Seed: {seed}")

                for outer_fold, (outer_train_ids, outer_test_ids) in enumerate(outer_splits):
                    print(
                        f"      Fold {outer_fold:02d}: outer_train={len(outer_train_ids)} "
                        f"outer_test={len(outer_test_ids)}"
                    )
                    summary, rows = train_one_fold(
                        subject_id=subject_id,
                        target=target,
                        seed=seed,
                        outer_fold=outer_fold,
                        subject=subject,
                        outer_train_ids=outer_train_ids,
                        outer_test_ids=outer_test_ids,
                        epochs=epochs,
                        force=args.force,
                        results_dir=results_dir,
                    )
                    fold_rows.append(summary)
                    for row in rows:
                        video_rows.append(
                            {
                                "subject_id": subject_id,
                                "target": target,
                                "seed": seed,
                                "outer_fold": outer_fold,
                                **row,
                            }
                        )

    fold_fieldnames = [
        "subject_id",
        "target",
        "seed",
        "outer_fold",
        "best_epoch",
        "stopped_epoch",
        "train_trial_count",
        "val_trial_count",
        "test_trial_count",
        "train_window_count",
        "val_window_count",
        "test_window_count",
        "test_loss",
        "model_selection_metric",
        "class_counts",
        "class_weights",
        "video_accuracy",
        "video_macro_f1",
        "video_auc",
        "n_videos",
        "confusion_matrix",
    ]
    write_csv(results_dir / "fold_results.csv", fold_rows, fold_fieldnames)

    video_fieldnames = [
        "subject_id",
        "target",
        "seed",
        "outer_fold",
        "trial_id",
        "true_label",
        "pred_label",
        "mean_prob_high",
        "n_windows",
    ]
    write_csv(results_dir / "video_predictions.csv", video_rows, video_fieldnames)

    per_seed, per_subject, overall = build_summaries(fold_rows)
    write_csv(
        results_dir / "subject_seed_summary.csv",
        per_seed,
        [
            "subject_id",
            "target",
            "seed",
            "fold_count",
            "video_accuracy_mean",
            "video_accuracy_std",
            "video_macro_f1_mean",
            "video_macro_f1_std",
            "video_auc_mean",
            "video_auc_std",
            "best_epoch_mean",
            "stopped_epoch_mean",
            "videos_per_fold",
        ],
    )
    write_csv(
        results_dir / "subject_summary.csv",
        per_subject,
        [
            "subject_id",
            "target",
            "seed_count",
            "video_accuracy_mean",
            "video_accuracy_std",
            "video_macro_f1_mean",
            "video_macro_f1_std",
            "video_auc_mean",
            "video_auc_std",
        ],
    )
    write_csv(
        results_dir / "overall_summary.csv",
        overall,
        [
            "target",
            "subject_count",
            "video_accuracy_mean",
            "video_accuracy_std",
            "video_macro_f1_mean",
            "video_macro_f1_std",
            "video_auc_mean",
            "video_auc_std",
        ],
    )
    write_confusion_matrices(video_rows, results_dir)

    print("\nOverall summary")
    for row in overall:
        print(
            f"  {row['target']}: acc={row['video_accuracy_mean']:.4f} +/- {row['video_accuracy_std']:.4f} "
            f"f1={row['video_macro_f1_mean']:.4f} +/- {row['video_macro_f1_std']:.4f}"
        )
    print(f"\nOutputs written to {results_dir}")


def parse_args():
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Leakage-safe subject-dependent EEGNet experiment on DEAP"
    )
    parser.add_argument("--subjects", nargs="+", type=int, default=None)
    parser.add_argument("--targets", nargs="+", choices=config.TARGETS, default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-outer-folds", type=int, default=None)
    parser.add_argument("--results-dir", type=str, default=None)
    parser.add_argument("--notch-freq", type=float, default=None)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
