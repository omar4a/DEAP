"""Training script for EEG DL architectures on DEAP.

Usage:
    python train.py                          # all 32 subjects, all 3 models
    python train.py --models eegnet          # single model
    python train.py --debug                  # subjects 1-3, 20 epochs
    python train.py --subjects 1 2 3         # specific subjects
    python train.py --no-eval               # skip evaluate.py at the end
"""
import os
import sys
import json
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import f1_score, roc_auc_score

# Ensure package imports work when run from any directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from data_loader import load_subject, get_dataloaders, save_splits

# ── Reproducibility ──────────────────────────────────────────────────────────
torch.manual_seed(config.SEED)
np.random.seed(config.SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── Model registry ───────────────────────────────────────────────────────────
def _get_builders(requested: list[str]) -> dict:
    builders = {}
    if 'eegnet' in requested:
        from models.eegnet import build_model
        builders['eegnet'] = build_model
    if 'eegnet_transformer' in requested:
        from models.eegnet_transformer import build_model
        builders['eegnet_transformer'] = build_model
    if 'convnext_eeg' in requested:
        from models.convnext_eeg import build_model
        builders['convnext_eeg'] = build_model
    return builders


# ── Training ─────────────────────────────────────────────────────────────────
def train_model(model: nn.Module, subject_data: dict, model_name: str,
                subject_id: int, epochs: int = None) -> dict:
    """Train one model on one subject. Saves results and returns test metrics.

    Skips if results already exist (allows resuming interrupted runs).
    """
    if epochs is None:
        epochs = config.EPOCHS

    out_dir = os.path.join(config.RESULTS_DIR, model_name, f'subject_{subject_id:02d}')
    os.makedirs(out_dir, exist_ok=True)

    metrics_csv   = os.path.join(out_dir, 'metrics.csv')
    test_json     = os.path.join(out_dir, 'test_metrics.json')
    best_pth      = os.path.join(out_dir, 'best_model.pth')
    test_preds_npz = os.path.join(out_dir, 'test_preds.npz')

    # Resume: skip if both files exist
    if os.path.exists(metrics_csv) and os.path.exists(test_json):
        print(f"    [skip] {model_name}/subject_{subject_id:02d} -- already complete")
        with open(test_json) as f:
            return json.load(f)

    model = model.to(DEVICE)
    train_dl, val_dl, test_dl = get_dataloaders(subject_data)

    optimizer  = AdamW(model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY)
    scheduler  = CosineAnnealingLR(optimizer, T_max=epochs)
    criterion  = nn.CrossEntropyLoss()

    best_val_loss = float('inf')
    patience_ctr  = 0
    stopped_epoch = epochs

    # Write CSV header
    with open(metrics_csv, 'w') as f:
        f.write('epoch,train_loss,val_loss,val_acc_valence,val_acc_arousal\n')

    for epoch in range(1, epochs + 1):
        # ── Train ──
        model.train()
        train_losses = []
        for X_b, y_b in train_dl:
            X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
            optimizer.zero_grad()
            v_logits, a_logits = model(X_b)
            loss = (config.VALENCE_WEIGHT * criterion(v_logits, y_b[:, 0]) +
                    config.AROUSAL_WEIGHT * criterion(a_logits, y_b[:, 1]))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())

        scheduler.step()  # step regardless of early stopping

        # ── Validate ──
        model.eval()
        val_losses, n_correct_v, n_correct_a, n_total = [], 0, 0, 0
        with torch.no_grad():
            for X_b, y_b in val_dl:
                X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
                v_logits, a_logits = model(X_b)
                val_losses.append(
                    (config.VALENCE_WEIGHT * criterion(v_logits, y_b[:, 0]) +
                     config.AROUSAL_WEIGHT * criterion(a_logits, y_b[:, 1])).item()
                )
                n_correct_v += (v_logits.argmax(1) == y_b[:, 0]).sum().item()
                n_correct_a += (a_logits.argmax(1) == y_b[:, 1]).sum().item()
                n_total     += y_b.size(0)

        train_loss = float(np.mean(train_losses))
        val_loss   = float(np.mean(val_losses))
        val_acc_v  = n_correct_v / n_total
        val_acc_a  = n_correct_a / n_total

        # ── Log ──
        with open(metrics_csv, 'a') as f:
            f.write(f'{epoch},{train_loss:.6f},{val_loss:.6f},'
                    f'{val_acc_v:.6f},{val_acc_a:.6f}\n')

        # ── Early stopping & checkpoint ──
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_ctr  = 0
            torch.save(model.state_dict(), best_pth)
        else:
            patience_ctr += 1

        if epoch % 10 == 0 or epoch == 1:
            print(f"      ep {epoch:3d}/{epochs}  "
                  f"train={train_loss:.4f}  val={val_loss:.4f}  "
                  f"val_v={val_acc_v:.4f}  val_a={val_acc_a:.4f}")

        if patience_ctr >= config.EARLY_STOPPING_PATIENCE:
            stopped_epoch = epoch
            print(f"      Early stop at epoch {epoch}")
            break

    # ── Test evaluation ──
    model.load_state_dict(torch.load(best_pth, map_location=DEVICE))
    model.eval()

    v_true_all, a_true_all = [], []
    v_pred_all, a_pred_all = [], []
    v_prob_all, a_prob_all = [], []

    with torch.no_grad():
        for X_b, y_b in test_dl:
            X_b = X_b.to(DEVICE)
            v_logits, a_logits = model(X_b)

            v_true_all.extend(y_b[:, 0].numpy())
            a_true_all.extend(y_b[:, 1].numpy())
            v_pred_all.extend(v_logits.argmax(1).cpu().numpy())
            a_pred_all.extend(a_logits.argmax(1).cpu().numpy())
            v_prob_all.extend(torch.softmax(v_logits, 1)[:, 1].cpu().numpy())
            a_prob_all.extend(torch.softmax(a_logits, 1)[:, 1].cpu().numpy())

    v_true = np.array(v_true_all)
    a_true = np.array(a_true_all)
    v_pred = np.array(v_pred_all)
    a_pred = np.array(a_pred_all)
    v_prob = np.array(v_prob_all)
    a_prob = np.array(a_prob_all)

    def safe_auc(y_true, y_prob):
        return float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.5

    test_metrics = {
        'acc_v':  float((v_pred == v_true).mean()),
        'acc_a':  float((a_pred == a_true).mean()),
        'f1_v':   float(f1_score(v_true, v_pred, average='macro', zero_division=0)),
        'f1_a':   float(f1_score(a_true, a_pred, average='macro', zero_division=0)),
        'auc_v':  safe_auc(v_true, v_prob),
        'auc_a':  safe_auc(a_true, a_prob),
        'stopped_epoch': stopped_epoch,
    }

    with open(test_json, 'w') as f:
        json.dump(test_metrics, f, indent=2)

    # Save raw predictions for confusion matrices in evaluate.py
    np.savez(test_preds_npz,
             y_true_v=v_true, y_true_a=a_true,
             y_pred_v=v_pred, y_pred_a=a_pred)

    print(f"      TEST  acc_v={test_metrics['acc_v']:.4f}  acc_a={test_metrics['acc_a']:.4f}  "
          f"f1_v={test_metrics['f1_v']:.4f}  f1_a={test_metrics['f1_a']:.4f}")
    return test_metrics


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='Train EEG DL models on DEAP')
    parser.add_argument('--models', nargs='+',
                        default=['eegnet', 'eegnet_transformer', 'convnext_eeg'],
                        help='Which models to train')
    parser.add_argument('--subjects', nargs='+', type=int, default=None,
                        help='Subject IDs (default: 1-32)')
    parser.add_argument('--debug', action='store_true',
                        help='Subjects 1-3 only, max 20 epochs')
    parser.add_argument('--no-eval', action='store_true',
                        help='Skip evaluate.py after training')
    args = parser.parse_args()

    print(f"\nDevice: {DEVICE}")

    subjects = args.subjects or list(range(1, config.N_SUBJECTS + 1))
    epochs   = config.EPOCHS

    if args.debug:
        subjects = subjects[:3]
        epochs   = 20
        print("DEBUG MODE: subjects 1-3, max 20 epochs\n")

    # Save split file once
    save_splits()

    # Build all requested models (triggers param-count print)
    builders = _get_builders(args.models)

    # ── Parameter count summary table ──
    print("\n" + "=" * 57)
    print(f"{'Model':<28} {'Total params':>14} {'Trainable':>13}")
    print("=" * 57)
    for name, build_fn in builders.items():
        m         = build_fn()
        total     = sum(p.numel() for p in m.parameters())
        trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
        print(f"  {name:<26} {total:>14,} {trainable:>13,}")
        del m
    print("=" * 57 + "\n")

    # ── Training loop ──
    for model_name, build_fn in builders.items():
        print(f"\n{'='*60}")
        print(f"  Model: {model_name}")
        print(f"{'='*60}")

        for sid in subjects:
            print(f"\n  Subject {sid:02d}:")

            # Check before loading data
            out_dir  = os.path.join(config.RESULTS_DIR, model_name, f'subject_{sid:02d}')
            done = (os.path.exists(os.path.join(out_dir, 'metrics.csv')) and
                    os.path.exists(os.path.join(out_dir, 'test_metrics.json')))
            if done:
                print(f"    [skip] already complete")
                continue

            try:
                subject_data = load_subject(sid)
            except FileNotFoundError:
                print(f"    s{sid:02d}.dat not found -- skipping")
                continue

            model = build_fn()
            train_model(model, subject_data, model_name, sid, epochs=epochs)

    # ── Auto-evaluate ──
    if not args.no_eval:
        print("\n\n" + "=" * 60)
        print("  Running evaluate.py ...")
        print("=" * 60 + "\n")
        import evaluate
        evaluate.run_evaluation()


if __name__ == '__main__':
    main()
