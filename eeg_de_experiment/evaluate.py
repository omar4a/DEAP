"""Evaluation and visualisation for the EEGNet-DE experiment.

Outputs saved to results/eegnet_de/final/:
  summary_table.csv, grand_summary.txt,
  per_subject_barplot.png, valence_vs_arousal_scatter.png,
  fold_variance_plot.png, confusion_matrices.png,
  learning_curves_sample.png
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, 'results', 'eegnet_de')
OUT_DIR     = os.path.join(RESULTS_DIR, 'final')

PREV_VAL_ACC = 0.6387   # from raw-EEG fixed-split experiment
PREV_ARO_ACC = 0.4894


# ── Data loading ──────────────────────────────────────────────────────────────

def load_subject_folds(subject_id: int) -> pd.DataFrame | None:
    """Load fold_summary.csv for one subject. Returns None if missing."""
    path = os.path.join(RESULTS_DIR, f'subject_{subject_id:02d}', 'fold_summary.csv')
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    df['subject_id'] = subject_id
    return df


def load_all_folds() -> pd.DataFrame:
    frames = []
    for sid in range(1, config.N_SUBJECTS + 1):
        df = load_subject_folds(sid)
        if df is not None:
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_all_predictions() -> tuple:
    """Aggregate test predictions across all subjects and folds."""
    v_true_all, a_true_all = [], []
    v_pred_all, a_pred_all = [], []

    for sid in range(1, config.N_SUBJECTS + 1):
        for k in range(config.CV_FOLDS):
            npz = os.path.join(RESULTS_DIR, f'subject_{sid:02d}',
                               f'fold_{k:02d}_preds.npz')
            if not os.path.exists(npz):
                continue
            d = np.load(npz)
            v_true_all.append(d['y_true_v'])
            a_true_all.append(d['y_true_a'])
            v_pred_all.append(d['y_pred_v'])
            a_pred_all.append(d['y_pred_a'])

    concat = lambda lst: np.concatenate(lst) if lst else np.array([])
    return (concat(v_true_all), concat(a_true_all),
            concat(v_pred_all), concat(a_pred_all))


# ── Main evaluation ───────────────────────────────────────────────────────────

def run_evaluation():
    os.makedirs(OUT_DIR, exist_ok=True)

    all_folds = load_all_folds()
    if all_folds.empty:
        print("No results found — run train.py first.")
        return

    subjects = sorted(all_folds['subject_id'].unique())
    n_done   = len(subjects)

    # Per-subject means across folds
    per_subj = (all_folds.groupby('subject_id')
                          .agg(val_acc_mean=('acc_v', 'mean'),
                               val_acc_std= ('acc_v', 'std'),
                               ar_acc_mean= ('acc_a', 'mean'),
                               ar_acc_std=  ('acc_a', 'std'),
                               val_f1_mean= ('f1_v',  'mean'),
                               val_f1_std=  ('f1_v',  'std'),
                               ar_f1_mean=  ('f1_a',  'mean'),
                               ar_f1_std=   ('f1_a',  'std'),
                               val_auc_mean=('auc_v', 'mean'),
                               val_auc_std= ('auc_v', 'std'),
                               ar_auc_mean= ('auc_a', 'mean'),
                               ar_auc_std=  ('auc_a', 'std'))
                          .reset_index())

    # ── 1. summary_table.csv ──────────────────────────────────────────────────
    grand_row = pd.DataFrame([{
        'subject_id': 'GRAND',
        'val_acc_mean': per_subj['val_acc_mean'].mean(),
        'val_acc_std':  per_subj['val_acc_mean'].std(),
        'ar_acc_mean':  per_subj['ar_acc_mean'].mean(),
        'ar_acc_std':   per_subj['ar_acc_mean'].std(),
        'val_f1_mean':  per_subj['val_f1_mean'].mean(),
        'val_f1_std':   per_subj['val_f1_mean'].std(),
        'ar_f1_mean':   per_subj['ar_f1_mean'].mean(),
        'ar_f1_std':    per_subj['ar_f1_mean'].std(),
        'val_auc_mean': per_subj['val_auc_mean'].mean(),
        'val_auc_std':  per_subj['val_auc_mean'].std(),
        'ar_auc_mean':  per_subj['ar_auc_mean'].mean(),
        'ar_auc_std':   per_subj['ar_auc_mean'].std(),
    }])
    summary = pd.concat([per_subj, grand_row], ignore_index=True)
    summary.to_csv(os.path.join(OUT_DIR, 'summary_table.csv'), index=False)
    print("Saved: summary_table.csv")

    # ── 2. grand_summary.txt ─────────────────────────────────────────────────
    gv_mean  = per_subj['val_acc_mean'].mean()
    gv_std   = per_subj['val_acc_mean'].std()
    ga_mean  = per_subj['ar_acc_mean'].mean()
    ga_std   = per_subj['ar_acc_mean'].std()
    n_v70    = (per_subj['val_acc_mean'] > 0.70).sum()
    n_v80    = (per_subj['val_acc_mean'] > 0.80).sum()
    n_a70    = (per_subj['ar_acc_mean']  > 0.70).sum()

    lines = [
        "EEGNet-DE Experiment — Grand Summary\n",
        "=" * 50 + "\n",
        f"Subjects completed: {n_done} / {config.N_SUBJECTS}\n",
        f"Folds per subject:  {config.CV_FOLDS}\n\n",
        f"Valence accuracy:  {gv_mean:.4f} +/- {gv_std:.4f}  ({gv_mean*100:.1f}%)\n",
        f"Arousal accuracy:  {ga_mean:.4f} +/- {ga_std:.4f}  ({ga_mean*100:.1f}%)\n\n",
        f"Subjects with valence acc > 70%: {n_v70} / {n_done}\n",
        f"Subjects with valence acc > 80%: {n_v80} / {n_done}\n",
        f"Subjects with arousal acc > 70%: {n_a70} / {n_done}\n\n",
        "Comparison:\n",
        f"  Previous experiment (raw EEG, fixed split): "
        f"Valence {PREV_VAL_ACC*100:.1f}%, Arousal {PREV_ARO_ACC*100:.1f}%\n",
        f"  This experiment    (DE, 10-fold CV):        "
        f"Valence {gv_mean*100:.1f}%, Arousal {ga_mean*100:.1f}%\n",
        f"  Delta valence: {(gv_mean - PREV_VAL_ACC)*100:+.1f} pp\n",
        f"  Delta arousal: {(ga_mean - PREV_ARO_ACC)*100:+.1f} pp\n",
    ]
    with open(os.path.join(OUT_DIR, 'grand_summary.txt'), 'w') as f:
        f.writelines(lines)
    print("Saved: grand_summary.txt")

    # ── 3. per_subject_barplot.png ────────────────────────────────────────────
    df_sorted = per_subj.sort_values('val_acc_mean', ascending=True)
    colors = ['#2ca02c' if v > 0.70 else '#ff7f0e' if v > 0.60 else '#d62728'
              for v in df_sorted['val_acc_mean']]

    fig, ax = plt.subplots(figsize=(8, 10))
    bars = ax.barh(df_sorted['subject_id'].astype(str),
                   df_sorted['val_acc_mean'],
                   xerr=df_sorted['val_acc_std'],
                   color=colors, alpha=0.85, capsize=3,
                   edgecolor='black', linewidth=0.5)
    ax.axvline(0.50, color='black',  linestyle='--', linewidth=1.2, label='Chance (50%)')
    ax.axvline(0.70, color='blue',   linestyle='--', linewidth=1.0, label='70%')
    ax.axvline(0.80, color='purple', linestyle='--', linewidth=1.0, label='80%')
    ax.set_xlabel('Mean Valence Accuracy (10-fold CV)')
    ax.set_title('Per-Subject Valence Accuracy (sorted descending)')
    ax.legend(fontsize=8)
    ax.set_xlim(0, 1)
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'per_subject_barplot.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: per_subject_barplot.png")

    # ── 4. valence_vs_arousal_scatter.png ────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(per_subj['val_acc_mean'], per_subj['ar_acc_mean'],
               s=60, color='steelblue', alpha=0.8, zorder=3)
    for _, row in per_subj.iterrows():
        ax.annotate(str(int(row['subject_id'])),
                    (row['val_acc_mean'], row['ar_acc_mean']),
                    textcoords='offset points', xytext=(5, 3), fontsize=7)
    ax.axhline(0.5, color='gray', linestyle='--', linewidth=1)
    ax.axvline(0.5, color='gray', linestyle='--', linewidth=1)
    ax.set_xlabel('Valence Accuracy')
    ax.set_ylabel('Arousal Accuracy')
    ax.set_title('Per-Subject: Valence vs Arousal Accuracy')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'valence_vs_arousal_scatter.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: valence_vs_arousal_scatter.png")

    # ── 5. fold_variance_plot.png ─────────────────────────────────────────────
    # Box per subject, sorted by median valence accuracy
    fold_pivot = (all_folds.pivot_table(index='fold', columns='subject_id',
                                        values='acc_v')
                            .reindex(columns=subjects))
    medians    = fold_pivot.median()
    order      = medians.sort_values().index.tolist()
    data_boxes = [fold_pivot[s].dropna().values for s in order]

    fig, ax = plt.subplots(figsize=(14, 5))
    bp = ax.boxplot(data_boxes, patch_artist=True, notch=False,
                    medianprops={'color': 'red', 'linewidth': 1.5})
    for patch in bp['boxes']:
        patch.set_facecolor('lightsteelblue')
        patch.set_alpha(0.7)
    ax.axhline(0.5, color='black', linestyle='--', linewidth=1)
    ax.set_xticks(range(1, len(order) + 1))
    ax.set_xticklabels([str(s) for s in order], rotation=45, fontsize=8)
    ax.set_xlabel('Subject ID (sorted by median valence accuracy)')
    ax.set_ylabel('Test Valence Accuracy')
    ax.set_title('Fold Variance: Test Valence Accuracy across 10 Folds per Subject')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'fold_variance_plot.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: fold_variance_plot.png")

    # ── 6. confusion_matrices.png ─────────────────────────────────────────────
    v_true, a_true, v_pred, a_pred = load_all_predictions()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, y_t, y_p, title in [
        (axes[0], v_true, v_pred, 'Valence'),
        (axes[1], a_true, a_pred, 'Arousal'),
    ]:
        if y_t.size == 0:
            ax.set_visible(False)
            continue
        cm      = confusion_matrix(y_t, y_p)
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        sns.heatmap(cm_norm, annot=True, fmt='.2f', ax=ax, cmap='Blues',
                    xticklabels=['Low', 'High'], yticklabels=['Low', 'High'],
                    vmin=0, vmax=1, cbar=False)
        ax.set_title(f'{title} (all subjects + folds)')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')

    plt.suptitle('Aggregated Confusion Matrices (row-normalised)', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'confusion_matrices.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: confusion_matrices.png")

    # ── 7. learning_curves_sample.png (subject 01, all 10 folds) ─────────────
    subj01_dir = os.path.join(RESULTS_DIR, 'subject_01')
    if os.path.isdir(subj01_dir):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        cmap = plt.cm.tab10

        for k in range(config.CV_FOLDS):
            csv_path = os.path.join(subj01_dir, f'fold_{k:02d}_metrics.csv')
            if not os.path.exists(csv_path):
                continue
            df = pd.read_csv(csv_path)
            col = cmap(k / config.CV_FOLDS)
            axes[0].plot(df['train_loss'], color=col, alpha=0.7, linewidth=1.2,
                         label=f'Fold {k}')
            axes[1].plot(df['val_loss'],   color=col, alpha=0.7, linewidth=1.2)

        axes[0].set_title('Train Loss — Subject 01 (all 10 folds)')
        axes[1].set_title('Val Loss   — Subject 01 (all 10 folds)')
        for ax in axes:
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.grid(alpha=0.3)
        axes[0].legend(fontsize=7, ncol=2)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, 'learning_curves_sample.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()
        print("Saved: learning_curves_sample.png")
    else:
        print("Skipped learning_curves_sample.png (subject_01 not found)")

    # ── Console summary ───────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"{'Subject':>8}  {'Val Acc':>8}  {'Aro Acc':>8}  {'Val F1':>8}  {'Aro F1':>8}")
    print('='*60)
    for _, row in per_subj.iterrows():
        print(f"  s{int(row['subject_id']):02d}     "
              f"{row['val_acc_mean']:>7.4f}   {row['ar_acc_mean']:>7.4f}   "
              f"{row['val_f1_mean']:>7.4f}   {row['ar_f1_mean']:>7.4f}")
    print('='*60)
    print(f"  GRAND  {gv_mean:>7.4f}   {ga_mean:>7.4f}")
    print(f"  Prev   {PREV_VAL_ACC:>7.4f}   {PREV_ARO_ACC:>7.4f}  (raw EEG, fixed split)")
    print(f"  Delta  {(gv_mean-PREV_VAL_ACC)*100:>+6.1f}pp  {(ga_mean-PREV_ARO_ACC)*100:>+6.1f}pp")
    print(f"\nAll outputs written to {OUT_DIR}")


if __name__ == '__main__':
    run_evaluation()
