import argparse
import csv
import json
from pathlib import Path
import random
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import torch
import sys
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt

import config
from dataset import (
    load_subject, get_target_labels, filter_neutral_trials,
    build_outer_splits, build_inner_split, preprocess_trials,
    make_window_split, build_dataloader
)

from model import EEGNetBinary, Tsception

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def aggregate_video_predictions(parent_trials, probs_high, true_window_labels):
    grouped_probs = defaultdict(list)
    grouped_labels = {}
    for tid, prob, lbl in zip(parent_trials, probs_high, true_window_labels):
        grouped_probs[int(tid)].append(float(prob))
        grouped_labels[int(tid)] = int(lbl)

    rows, y_true, y_pred, y_prob = [], [], [], []
    for tid in sorted(grouped_probs):
        mean_prob = float(np.mean(grouped_probs[tid]))
        true_lbl = grouped_labels[tid]
        pred_lbl = int(mean_prob >= 0.5)

        rows.append({
            "trial_id": tid, "true_label": true_lbl, 
            "pred_label": pred_lbl, "mean_prob_high": mean_prob,
            "n_windows": len(grouped_probs[tid])
        })
        y_true.append(true_lbl)
        y_pred.append(pred_lbl)
        y_prob.append(mean_prob)

    y_true_arr = np.array(y_true, dtype=np.int64)
    y_pred_arr = np.array(y_pred, dtype=np.int64)
    y_prob_arr = np.array(y_prob, dtype=np.float64)

    metrics = {
        "video_accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
        "video_macro_f1": float(f1_score(y_true_arr, y_pred_arr, average="macro", zero_division=0)),
        "video_auc": float(roc_auc_score(y_true_arr, y_prob_arr)) if len(np.unique(y_true_arr)) > 1 else 0.5,
        "n_videos": len(rows),
    }
    return rows, metrics

def evaluate_loader(model, loader, split, criterion):
    model.eval()
    losses, probs_high = [], []
    with torch.no_grad():
        for windows, labels in loader:
            windows = windows.to(DEVICE)
            labels = labels.to(DEVICE)
            logits = model(windows)
            losses.append(float(criterion(logits, labels).item()))
            probs_high.extend(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())

    mean_loss = float(np.mean(losses)) if losses else 0.0
    rows, metrics = aggregate_video_predictions(
        split["parent_trials"], np.array(probs_high, dtype=np.float64), split["labels"]
    )
    return mean_loss, rows, metrics

def build_class_weights(labels: np.ndarray):
    counts = np.bincount(labels, minlength=2).astype(np.float32)
    safe_counts = np.maximum(counts, 1.0)
    weights = counts.sum() / (len(counts) * safe_counts)
    return torch.tensor(weights, dtype=torch.float32, device=DEVICE)

def _is_better_score(candidate: dict, best: dict | None) -> bool:
    if best is None: return True
    if candidate[config.MODEL_SELECTION_METRIC] != best[config.MODEL_SELECTION_METRIC]:
        return candidate[config.MODEL_SELECTION_METRIC] > best[config.MODEL_SELECTION_METRIC]
    return candidate["val_loss"] < best["val_loss"]

def train_one_fold(
    subject_id: int, target: str, seed: int, outer_fold: int, subject,
    outer_train_ids: np.ndarray, outer_test_ids: np.ndarray,
    epochs: int, results_dir: Path, lr: float, dropout: float,
    window_samples: int, stride_samples: int, f1: int, d: int, f2: int,
    use_tsception: bool = True
) -> tuple[dict, list[dict]]:
    
    fold_dir = results_dir / f"subject_{subject_id:02d}" / target / f"seed_{seed}" / f"fold_{outer_fold:02d}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    summary_path = fold_dir / "fold_summary.json"
    preds_path = fold_dir / "test_video_predictions.csv"
    best_model_path = fold_dir / "best_model.pth"
    checkpoint_path = fold_dir / "checkpoint.pth"

    if summary_path.exists() and preds_path.exists():
        with summary_path.open() as f: summary = json.load(f)
        with preds_path.open() as f: rows = list(csv.DictReader(f))
        return summary, rows

    inner_train_ids, inner_val_ids = build_inner_split(outer_train_ids, (seed * 1000) + subject_id * 10 + outer_fold)
    trial_labels = get_target_labels(subject.ratings, target)
    processed_trials, norm_stats = preprocess_trials(subject.eeg, inner_train_ids)

    train_split = make_window_split(processed_trials, inner_train_ids, trial_labels, window_samples, stride_samples)
    val_split = make_window_split(processed_trials, inner_val_ids, trial_labels, window_samples, stride_samples)
    test_split = make_window_split(processed_trials, outer_test_ids, trial_labels, window_samples, stride_samples)

    train_loader = build_dataloader(train_split, 64, is_train=True)
    val_loader = build_dataloader(val_split, 64, is_train=False)
    test_loader = build_dataloader(test_split, 64, is_train=False)

    set_seed(seed)
    
    # Init Model dynamically
    if use_tsception:
        model = Tsception(
            num_classes=2, 
            input_size=(1, config.CHANNELS, window_samples),
            sampling_rate=config.SFREQ,
            num_T=f1, num_S=f2, hidden=32, dropout_rate=dropout
        ).to(DEVICE)
    else:
        model = EEGNetBinary(
            channels=config.CHANNELS, samples=window_samples,
            f1=f1, depth=d, f2=f2, dropout=dropout,
            temporal_kernel=config.SFREQ // 2
        ).to(DEVICE)
    
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4) # Fixed weight decay
    class_weights = build_class_weights(train_split["labels"])
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    best_snapshot = None
    best_epoch = 0
    patience_counter = 0
    start_epoch = 1
    
    # History for continuous logging
    history = {"epoch": [], "train_loss": [], "val_loss": [], "val_acc": [], "val_f1": []}

    # Checkpoint restoring logic
    if checkpoint_path.exists():
        try:
            checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_epoch = checkpoint["epoch"] + 1
            best_snapshot = checkpoint["best_snapshot"]
            best_epoch = checkpoint["best_epoch"]
            patience_counter = checkpoint["patience_counter"]
            if "history" in checkpoint:
                history = checkpoint["history"]
            print(f"        Resuming from checkpoint at epoch {start_epoch - 1}")
        except Exception as e:
            print(f"        Checkpoint corrupted or failed to load: {e}. Starting from scratch.")

    for epoch in range(start_epoch, epochs + 1):
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

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_metrics["video_accuracy"])
        history["val_f1"].append(val_metrics["video_macro_f1"])

        candidate = {
            config.MODEL_SELECTION_METRIC: val_metrics[config.MODEL_SELECTION_METRIC],
            "video_accuracy": val_metrics["video_accuracy"],
            "val_loss": val_loss,
        }
        
        if _is_better_score(candidate, best_snapshot):
            best_snapshot = candidate
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_counter += 1

        # Checkpoint every epoch
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_snapshot": best_snapshot,
            "best_epoch": best_epoch,
            "patience_counter": patience_counter,
            "history": history,
        }, checkpoint_path)
        
        # Save continuous CSV and Plot
        csv_path = fold_dir / "training_history.csv"
        plot_path = fold_dir / "loss_curve.png"
        
        df = pd.DataFrame(history)
        df.to_csv(csv_path, index=False)
        
        plt.figure(figsize=(10, 5))
        plt.plot(df["epoch"], df["train_loss"], label="Train Loss")
        plt.plot(df["epoch"], df["val_loss"], label="Val Loss")
        plt.title(f"Trial Losses (Sub {subject_id} Fold {outer_fold})")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        plt.savefig(plot_path)
        plt.close()

        if epoch == 1 or epoch % 10 == 0:
            print(
                f"        ep {epoch:3d}/{epochs} train={train_loss:.4f} "
                f"val_loss={val_loss:.4f} val_acc={val_metrics['video_accuracy']:.4f} "
                f"val_f1={val_metrics['video_macro_f1']:.4f}"
            )

        if patience_counter >= config.EARLY_STOPPING_PATIENCE:
            break

    model.load_state_dict(torch.load(best_model_path, map_location=DEVICE, weights_only=False))
    test_loss, test_rows, test_metrics = evaluate_loader(model, test_loader, test_split, criterion)

    summary = {
        "subject_id": subject_id,
        "outer_fold": outer_fold,
        "best_epoch": best_epoch,
        **test_metrics,
    }
    
    with preds_path.open("w", newline="") as h:
        writer = csv.DictWriter(h, fieldnames=["trial_id", "true_label", "pred_label", "mean_prob_high", "n_windows"])
        writer.writeheader()
        writer.writerows(test_rows)
        
    summary_path.write_text(json.dumps(summary, indent=2))
    
    # Cleanup checkpoint post fold clear
    if checkpoint_path.exists():
        checkpoint_path.unlink()
        
    return summary, test_rows


def main(test_run: bool = False):
    window_samples = int(config.WINDOW_SEC * config.SFREQ)
    stride_samples = int((config.WINDOW_SEC / 2.0) * config.SFREQ)
    
    results_dir = Path(config.BASE_DIR) / "results_tsception_baseline"
    
    all_subjects = list(range(1, 33))
    seed = 42
    epochs = 40 # Moderate cap for baseline convergence
    
    target = "valence"
    
    # Strictly empty exclusion list for the baseline
    exclusion_list = getattr(config, f"EXCLUDE_SUBJECTS_{target.upper()}", [])
    
    overall_f1_scores = []
    
    for subject_id in all_subjects:
        if subject_id in exclusion_list:
            continue
            
        subject = load_subject(subject_id)
        
        # NOTE: strictly passing valid_trial_ids without neutral filtering.
        valid_trial_ids = filter_neutral_trials(subject.ratings, target)
        
        if len(valid_trial_ids) < 10:
            print(f"Skipping Subject {subject_id} due to insufficient trials.")
            continue
            
        outer_splits = build_outer_splits(valid_trial_ids)
        
        for outer_fold, (outer_train_ids, outer_test_ids) in enumerate(outer_splits):
            summary, _ = train_one_fold(
                subject_id=subject_id, target=target, seed=seed, 
                outer_fold=outer_fold, subject=subject, 
                outer_train_ids=outer_train_ids, outer_test_ids=outer_test_ids,
                epochs=epochs, results_dir=results_dir, 
                lr=config.LEARNING_RATE, dropout=config.DROPOUT,
                window_samples=window_samples, stride_samples=stride_samples,
                f1=config.NUM_T, d=2, f2=config.NUM_S, use_tsception=True
            )
            overall_f1_scores.append(summary["video_macro_f1"])

            if test_run:
                print("Test run completed 1 fold. Exiting.")
                return

    final_f1 = float(np.mean(overall_f1_scores)) if overall_f1_scores else 0.0
    print(f"\n[BASELINE COMPLETE] Target: {target} | Final Mean Video F1: {final_f1:.4f}")
    
    # Save overall summary
    pd.DataFrame({"target": [target], "video_macro_f1_mean": [final_f1]}).to_csv(results_dir / "overall_summary.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-run", action="store_true", help="Run 1 fold on 1 subject for debugging.")
    args = parser.parse_args()

    main(test_run=args.test_run)
