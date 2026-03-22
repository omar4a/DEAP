"""10-fold CV training of EEGNet-DE on DEAP differential entropy features.

Usage:
    python train.py                  # all 32 subjects
    python train.py --debug          # subjects 1-3 only
    python train.py --subject 5      # single subject
    python train.py --no-eval        # skip evaluate.py at end
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
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from de_extractor import extract_de_features
from model import build_model

# ── Reproducibility ───────────────────────────────────────────────────────────
torch.manual_seed(config.SEED)
np.random.seed(config.SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'results', 'eegnet_de')


# ── Reshaping ─────────────────────────────────────────────────────────────────

def reshape_for_eegnet(de_features: np.ndarray) -> np.ndarray:
    """Reshape DE features for EEGNet input.

    Args:
        de_features: (n_trials, 60, 32, 5) — [trial, time, channel, band]
    Returns:
        (n_trials, 1, 32, 300) float32
        Bands concatenated along time axis: [delta×60, theta×60, ..., gamma×60]
    """
    n_trials, n_time, n_ch, n_bands = de_features.shape
    # (n_trials, 60, 32, 5) → (n_trials, 32, 5, 60)
    x = de_features.transpose(0, 2, 3, 1)
    # → (n_trials, 32, 300)
    x = x.reshape(n_trials, n_ch, n_bands * n_time)
    # → (n_trials, 1, 32, 300)
    return x[:, np.newaxis, :, :].astype(np.float32)


# ── DataLoader factory ────────────────────────────────────────────────────────

def make_loader(X: np.ndarray, y: np.ndarray, shuffle: bool) -> DataLoader:
    ds = TensorDataset(torch.from_numpy(X).float(),
                       torch.from_numpy(y).long())
    return DataLoader(ds, batch_size=config.BATCH_SIZE,
                      shuffle=shuffle, num_workers=0, pin_memory=False)


# ── Single fold training ──────────────────────────────────────────────────────

def train_fold(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
               epochs: int, metrics_csv: str) -> str:
    """Train one fold. Returns path to best model weights file."""
    best_pth = metrics_csv.replace('metrics.csv', 'best.pth')
    model    = model.to(DEVICE)

    optimizer = AdamW(model.parameters(), lr=config.LR,
                      weight_decay=config.WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float('inf')
    patience_ctr  = 0

    with open(metrics_csv, 'w') as f:
        f.write('epoch,train_loss,val_loss,val_acc_v,val_acc_a\n')

    for epoch in range(1, epochs + 1):
        # ── Train ──
        model.train()
        train_losses = []
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
            optimizer.zero_grad()
            v_logits, a_logits = model(X_b)
            loss = (config.VALENCE_WEIGHT * criterion(v_logits, y_b[:, 0]) +
                    config.AROUSAL_WEIGHT * criterion(a_logits, y_b[:, 1]))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())

        scheduler.step()

        # ── Validate ──
        model.eval()
        val_losses, n_v, n_a, n_tot = [], 0, 0, 0
        with torch.no_grad():
            for X_b, y_b in val_loader:
                X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
                v_logits, a_logits = model(X_b)
                val_losses.append(
                    (config.VALENCE_WEIGHT * criterion(v_logits, y_b[:, 0]) +
                     config.AROUSAL_WEIGHT * criterion(a_logits, y_b[:, 1])).item()
                )
                n_v   += (v_logits.argmax(1) == y_b[:, 0]).sum().item()
                n_a   += (a_logits.argmax(1) == y_b[:, 1]).sum().item()
                n_tot += y_b.size(0)

        tl = float(np.mean(train_losses))
        vl = float(np.mean(val_losses))
        va_v = n_v / n_tot if n_tot else 0.0
        va_a = n_a / n_tot if n_tot else 0.0

        with open(metrics_csv, 'a') as f:
            f.write(f'{epoch},{tl:.6f},{vl:.6f},{va_v:.6f},{va_a:.6f}\n')

        if vl < best_val_loss:
            best_val_loss = vl
            patience_ctr  = 0
            torch.save(model.state_dict(), best_pth)
        else:
            patience_ctr += 1

        if patience_ctr >= config.EARLY_STOPPING_PATIENCE:
            break

    return best_pth


# ── Test evaluation ───────────────────────────────────────────────────────────

def eval_fold(model: nn.Module, best_pth: str,
              test_loader: DataLoader, preds_npz: str) -> dict:
    model.load_state_dict(torch.load(best_pth, map_location=DEVICE))
    model.eval()

    v_true, a_true, v_pred, a_pred, v_prob, a_prob = [], [], [], [], [], []
    with torch.no_grad():
        for X_b, y_b in test_loader:
            X_b = X_b.to(DEVICE)
            v_l, a_l = model(X_b)
            v_true.extend(y_b[:, 0].numpy())
            a_true.extend(y_b[:, 1].numpy())
            v_pred.extend(v_l.argmax(1).cpu().numpy())
            a_pred.extend(a_l.argmax(1).cpu().numpy())
            v_prob.extend(torch.softmax(v_l, 1)[:, 1].cpu().numpy())
            a_prob.extend(torch.softmax(a_l, 1)[:, 1].cpu().numpy())

    v_true, a_true = np.array(v_true), np.array(a_true)
    v_pred, a_pred = np.array(v_pred), np.array(a_pred)
    v_prob, a_prob = np.array(v_prob), np.array(a_prob)

    def safe_auc(yt, yp):
        return float(roc_auc_score(yt, yp)) if len(np.unique(yt)) > 1 else 0.5

    np.savez(preds_npz, y_true_v=v_true, y_true_a=a_true,
             y_pred_v=v_pred, y_pred_a=a_pred)

    return {
        'acc_v': float((v_pred == v_true).mean()),
        'acc_a': float((a_pred == a_true).mean()),
        'f1_v':  float(f1_score(v_true, v_pred, average='macro', zero_division=0)),
        'f1_a':  float(f1_score(a_true, a_pred, average='macro', zero_division=0)),
        'auc_v': safe_auc(v_true, v_prob),
        'auc_a': safe_auc(a_true, a_prob),
    }


# ── Subject-level 10-fold CV ──────────────────────────────────────────────────

def run_subject(subject_id: int, epochs: int = None) -> None:
    if epochs is None:
        epochs = config.EPOCHS

    subj_dir    = os.path.join(RESULTS_DIR, f'subject_{subject_id:02d}')
    summary_csv = os.path.join(subj_dir, 'fold_summary.csv')

    if os.path.exists(summary_csv):
        print(f"  [skip] subject_{subject_id:02d} -- fold_summary.csv exists")
        return

    os.makedirs(subj_dir, exist_ok=True)

    # Load DE features
    data       = extract_de_features(subject_id)
    de_all     = data['de_features']   # (40, 60, 32, 5)
    labels_all = data['labels']        # (40, 2)

    # Reshape all trials at once
    X_all = reshape_for_eegnet(de_all)    # (40, 1, 32, 300)

    trial_indices = np.arange(config.N_TRIALS)
    kf = KFold(n_splits=config.CV_FOLDS, shuffle=True,
               random_state=config.SEED)

    fold_rows = []

    for fold_idx, (train_val_idx, test_idx) in enumerate(kf.split(trial_indices)):
        # Split train_val into actual train (first 32) and val (last 4)
        val_idx   = train_val_idx[-4:]
        train_idx = train_val_idx[:-4]

        X_train, y_train = X_all[train_idx],   labels_all[train_idx]
        X_val,   y_val   = X_all[val_idx],     labels_all[val_idx]
        X_test,  y_test  = X_all[test_idx],    labels_all[test_idx]

        train_dl = make_loader(X_train, y_train, shuffle=True)
        val_dl   = make_loader(X_val,   y_val,   shuffle=False)
        test_dl  = make_loader(X_test,  y_test,  shuffle=False)

        metrics_csv = os.path.join(subj_dir, f'fold_{fold_idx:02d}_metrics.csv')
        preds_npz   = os.path.join(subj_dir, f'fold_{fold_idx:02d}_preds.npz')

        model    = build_model()
        best_pth = train_fold(model, train_dl, val_dl, epochs, metrics_csv)
        metrics  = eval_fold(model, best_pth, test_dl, preds_npz)

        fold_rows.append({'fold': fold_idx, **metrics})

        print(f"    fold {fold_idx:02d}  "
              f"acc_v={metrics['acc_v']:.4f}  acc_a={metrics['acc_a']:.4f}  "
              f"f1_v={metrics['f1_v']:.4f}  f1_a={metrics['f1_a']:.4f}")

    # Save fold summary
    import csv
    fieldnames = ['fold', 'acc_v', 'acc_a', 'f1_v', 'f1_a', 'auc_v', 'auc_a']
    with open(summary_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(fold_rows)

    mean_v = np.mean([r['acc_v'] for r in fold_rows])
    mean_a = np.mean([r['acc_a'] for r in fold_rows])
    print(f"  Subject {subject_id:02d} DONE -- "
          f"mean acc_v={mean_v:.4f}  mean acc_a={mean_a:.4f}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='EEGNet-DE 10-fold CV on DEAP')
    parser.add_argument('--debug',   action='store_true',
                        help='Subjects 1-3 only, max 50 epochs')
    parser.add_argument('--subject', type=int, default=None,
                        help='Run a single subject ID')
    parser.add_argument('--no-eval', action='store_true',
                        help='Skip evaluate.py at the end')
    args = parser.parse_args()

    print(f"\nDevice: {DEVICE}")

    # Print startup info
    m = build_model()
    print(f"\nInput shape per sample: (1, 1, {config.N_CHANNELS}, {config.DE_SEQ_LEN})")
    print(f"Trials per subject: {config.N_TRIALS}")
    print(f"Per-fold sizes: train~{config.N_TRIALS - config.N_TRIALS//config.CV_FOLDS - 4}  "
          f"val=4  test~{config.N_TRIALS//config.CV_FOLDS}")
    del m
    print()

    if args.subject is not None:
        subjects = [args.subject]
        epochs   = config.EPOCHS
    elif args.debug:
        subjects = [1, 2, 3]
        epochs   = 50
        print("DEBUG MODE: subjects 1-3, max 50 epochs per fold\n")
    else:
        subjects = list(range(1, config.N_SUBJECTS + 1))
        epochs   = config.EPOCHS

    for sid in subjects:
        print(f"\nSubject {sid:02d}:")
        try:
            run_subject(sid, epochs=epochs)
        except FileNotFoundError as e:
            print(f"  s{sid:02d}.dat not found -- skipping  ({e})")

    if not args.no_eval:
        print("\n\n" + "=" * 60)
        print("  Running evaluate.py ...")
        print("=" * 60 + "\n")
        import evaluate
        evaluate.run_evaluation()


if __name__ == '__main__':
    main()
