"""Evaluate trained models and generate comparison outputs.

Outputs saved to results/final_comparison/:
  summary_table.csv, parameter_count_table.csv,
  learning_curves.png, confusion_matrices.png,
  comparison_barplot.png, per_subject_accuracy.png,
  statistical_tests.txt

Run standalone:
    python evaluate.py
Or called automatically at the end of train.py.
"""
import os
import sys
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import confusion_matrix

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

MODEL_NAMES  = ['eegnet', 'eegnet_transformer', 'convnext_eeg']
MODEL_LABELS = {
    'eegnet':              'EEGNet',
    'eegnet_transformer':  'EEGNet+Transformer',
    'convnext_eeg':        'ConvNeXt-EEG',
}
COLORS = ['#4C72B0', '#DD8452', '#55A868']


# ── Data loading ─────────────────────────────────────────────────────────────

def load_all_results() -> dict:
    """Load metrics.csv and test_metrics.json for every model × subject."""
    results = {m: {} for m in MODEL_NAMES}
    for model_name in MODEL_NAMES:
        for sid in range(1, config.N_SUBJECTS + 1):
            subj_dir   = os.path.join(config.RESULTS_DIR, model_name, f'subject_{sid:02d}')
            csv_path   = os.path.join(subj_dir, 'metrics.csv')
            json_path  = os.path.join(subj_dir, 'test_metrics.json')
            if not (os.path.exists(csv_path) and os.path.exists(json_path)):
                continue
            results[model_name][sid] = {
                'metrics': pd.read_csv(csv_path),
                'test':    json.load(open(json_path)),
            }
    return results


def load_test_predictions() -> dict:
    """Load test-set predictions for confusion matrix aggregation."""
    preds = {m: {'v_true': [], 'v_pred': [], 'a_true': [], 'a_pred': []}
             for m in MODEL_NAMES}
    for model_name in MODEL_NAMES:
        for sid in range(1, config.N_SUBJECTS + 1):
            npz_path = os.path.join(config.RESULTS_DIR, model_name,
                                    f'subject_{sid:02d}', 'test_preds.npz')
            if not os.path.exists(npz_path):
                continue
            d = np.load(npz_path)
            preds[model_name]['v_true'].append(d['y_true_v'])
            preds[model_name]['v_pred'].append(d['y_pred_v'])
            preds[model_name]['a_true'].append(d['y_true_a'])
            preds[model_name]['a_pred'].append(d['y_pred_a'])

        for key in ['v_true', 'v_pred', 'a_true', 'a_pred']:
            lst = preds[model_name][key]
            preds[model_name][key] = np.concatenate(lst) if lst else np.array([])
    return preds


def get_param_counts() -> dict:
    """Build each model fresh and count parameters."""
    counts = {}
    try:
        from models.eegnet import EEGNet
        from models.eegnet_transformer import EEGNetTransformer
        from models.convnext_eeg import ConvNeXtEEG
        for name, cls in [('eegnet', EEGNet),
                           ('eegnet_transformer', EEGNetTransformer),
                           ('convnext_eeg', ConvNeXtEEG)]:
            m = cls()
            total     = sum(p.numel() for p in m.parameters())
            trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
            counts[name] = {'total': total, 'trainable': trainable}
    except Exception as e:
        print(f"Warning: could not build models for param count: {e}")
    return counts


# ── Helpers ───────────────────────────────────────────────────────────────────

def _pad_mean(arrays: list, max_len: int) -> np.ndarray:
    """Pad each 1-D array to max_len (repeat last value) then take mean."""
    padded = [np.pad(a, (0, max_len - len(a)), mode='edge') for a in arrays]
    return np.mean(padded, axis=0)


def _subjects_with_data(results: dict, model_name: str) -> list:
    return sorted(results[model_name].keys())


# ── Main evaluation routine ───────────────────────────────────────────────────

def run_evaluation():
    out_dir = os.path.join(config.RESULTS_DIR, 'final_comparison')
    os.makedirs(out_dir, exist_ok=True)

    results = load_all_results()
    preds   = load_test_predictions()
    params  = get_param_counts()

    # ── 1. Summary table ──────────────────────────────────────────────────────
    rows = []
    for model_name in MODEL_NAMES:
        subs = _subjects_with_data(results, model_name)
        if not subs:
            continue

        def col(key):
            return [results[model_name][s]['test'][key] for s in subs]

        def vcol(metric_col):   # best val accuracy across epochs
            return [results[model_name][s]['metrics'][metric_col].max() for s in subs]

        row = {'model_name': model_name,
               'params': params.get(model_name, {}).get('total', 'N/A')}
        for tag, vals in [
            ('val_acc_v',  vcol('val_acc_valence')),
            ('val_acc_a',  vcol('val_acc_arousal')),
            ('test_acc_v', col('acc_v')),
            ('test_acc_a', col('acc_a')),
            ('test_f1_v',  col('f1_v')),
            ('test_f1_a',  col('f1_a')),
            ('test_auc_v', col('auc_v')),
            ('test_auc_a', col('auc_a')),
        ]:
            row[f'{tag}_mean'] = float(np.mean(vals))
            row[f'{tag}_std']  = float(np.std(vals))
        rows.append(row)

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(os.path.join(out_dir, 'summary_table.csv'), index=False)
    print("Saved: summary_table.csv")

    # ── 2. Parameter count table ──────────────────────────────────────────────
    param_rows = [
        {'model_name': m,
         'total_params':     params.get(m, {}).get('total',     'N/A'),
         'trainable_params': params.get(m, {}).get('trainable', 'N/A')}
        for m in MODEL_NAMES
    ]
    pd.DataFrame(param_rows).to_csv(
        os.path.join(out_dir, 'parameter_count_table.csv'), index=False)
    print("Saved: parameter_count_table.csv")

    # ── 3. Learning curves ───────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, model_name, color in zip(axes, MODEL_NAMES, COLORS):
        subs = _subjects_with_data(results, model_name)
        if not subs:
            ax.set_title(MODEL_LABELS[model_name])
            continue
        all_train, all_val = [], []
        for sid in subs:
            df = results[model_name][sid]['metrics']
            tr = df['train_loss'].values
            vl = df['val_loss'].values
            ax.plot(tr, color='steelblue', alpha=0.15, linewidth=0.7)
            ax.plot(vl, color='tomato',    alpha=0.15, linewidth=0.7)
            all_train.append(tr)
            all_val.append(vl)
        max_len = max(len(a) for a in all_train)
        ax.plot(_pad_mean(all_train, max_len), color='steelblue',
                linewidth=2.5, label='Train (mean)')
        ax.plot(_pad_mean(all_val,   max_len), color='tomato',
                linewidth=2.5, label='Val (mean)')
        ax.set_title(MODEL_LABELS[model_name], fontsize=13)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    plt.suptitle('Learning Curves (per-subject thin, mean bold)', fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'learning_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: learning_curves.png")

    # ── 4. Confusion matrices ─────────────────────────────────────────────────
    # Layout: rows = [Valence, Arousal], cols = [EEGNet, EEGNet+T, ConvNeXt]
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    dims = [('Valence', 'v_true', 'v_pred'), ('Arousal', 'a_true', 'a_pred')]
    for col_idx, model_name in enumerate(MODEL_NAMES):
        for row_idx, (dim_label, true_key, pred_key) in enumerate(dims):
            ax = axes[row_idx][col_idx]
            y_t = preds[model_name][true_key]
            y_p = preds[model_name][pred_key]
            if y_t.size == 0:
                ax.set_visible(False)
                continue
            cm      = confusion_matrix(y_t, y_p)
            cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
            sns.heatmap(cm_norm, annot=True, fmt='.2f', ax=ax, cmap='Blues',
                        xticklabels=['Low', 'High'],
                        yticklabels=['Low', 'High'],
                        vmin=0, vmax=1, cbar=False)
            ax.set_title(f'{MODEL_LABELS[model_name]}\n{dim_label}', fontsize=10)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('True')

    plt.suptitle('Confusion Matrices (normalised by row — recall per class)',
                 fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'confusion_matrices.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: confusion_matrices.png")

    # ── 5. Comparison barplot ─────────────────────────────────────────────────
    # Grouped bars: X-axis = [Valence, Arousal], 3 bars per group
    fig, ax = plt.subplots(figsize=(10, 5))
    x       = np.arange(2)          # Valence, Arousal
    n       = len(rows)
    width   = 0.22
    offsets = np.linspace(-(n - 1) / 2 * width, (n - 1) / 2 * width, n)

    for i, (row, color) in enumerate(zip(rows, COLORS)):
        means = [row['test_acc_v_mean'], row['test_acc_a_mean']]
        stds  = [row['test_acc_v_std'],  row['test_acc_a_std']]
        bars  = ax.bar(x + offsets[i], means, width, yerr=stds, capsize=4,
                       color=color, alpha=0.85, label=MODEL_LABELS[row['model_name']],
                       edgecolor='black', linewidth=0.7)
        for bar, m, s in zip(bars, means, stds):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    m + s + 0.01, f'{m:.3f}',
                    ha='center', va='bottom', fontsize=8)

    ax.axhline(0.5, color='black', linestyle='--', linewidth=1.2, label='Chance (50%)')
    ax.set_xticks(x)
    ax.set_xticklabels(['Valence', 'Arousal'], fontsize=12)
    ax.set_ylabel('Mean Test Accuracy ± 1 std')
    ax.set_title('Model Comparison — Test Accuracy')
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'comparison_barplot.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: comparison_barplot.png")

    # ── 6. Per-subject accuracy ───────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    for ax, dim_label, key in zip(axes, ['Valence', 'Arousal'], ['acc_v', 'acc_a']):
        for model_name, color in zip(MODEL_NAMES, COLORS):
            subs = _subjects_with_data(results, model_name)
            if not subs:
                continue
            accs = [results[model_name][s]['test'][key] for s in subs]
            ax.plot(subs, accs, marker='o', markersize=4, linewidth=1.5,
                    color=color, label=MODEL_LABELS[model_name], alpha=0.85)
        ax.axhline(0.5, color='gray', linestyle='--', linewidth=1, alpha=0.6)
        ax.set_xlabel('Subject ID', fontsize=11)
        ax.set_ylabel('Test Accuracy', fontsize=11)
        ax.set_title(f'{dim_label} — Per-Subject Test Accuracy', fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'per_subject_accuracy.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: per_subject_accuracy.png")

    # ── 7. Statistical tests ──────────────────────────────────────────────────
    comparisons = [
        ('eegnet', 'eegnet_transformer', 'EEGNet vs EEGNet+Transformer'),
        ('eegnet', 'convnext_eeg',       'EEGNet vs ConvNeXt-EEG'),
        ('eegnet_transformer', 'convnext_eeg', 'EEGNet+Transformer vs ConvNeXt-EEG'),
    ]

    lines = [
        'Statistical Tests — Paired t-test (across subjects)\n',
        '=' * 60 + '\n\n',
    ]

    for m1, m2, label in comparisons:
        common = sorted(set(_subjects_with_data(results, m1)) &
                        set(_subjects_with_data(results, m2)))
        if len(common) < 2:
            lines.append(f'{label}: insufficient data (n={len(common)})\n\n')
            continue

        for dim_key, dim_label in [('acc_v', 'Valence'), ('acc_a', 'Arousal')]:
            a1 = [results[m1][s]['test'][dim_key] for s in common]
            a2 = [results[m2][s]['test'][dim_key] for s in common]
            t, p = stats.ttest_rel(a1, a2)
            sig  = ' *' if p < 0.05 else ''
            lines.append(f'{label} ({dim_label}):\n')
            lines.append(f'  n={len(common)}, t={t:.4f}, p={p:.4f}{sig}\n\n')

    stat_path = os.path.join(out_dir, 'statistical_tests.txt')
    with open(stat_path, 'w') as f:
        f.writelines(lines)
    print("Saved: statistical_tests.txt")

    print(f'\nAll outputs written to {out_dir}')
    _print_summary(rows)


def _print_summary(rows: list):
    if not rows:
        print("No results to summarise.")
        return
    print('\n' + '=' * 80)
    print(f"{'Model':<25} {'Val acc_v':>10} {'Val acc_a':>10} "
          f"{'Test acc_v':>11} {'Test acc_a':>11} {'Test F1_v':>10} {'Test F1_a':>10}")
    print('=' * 80)
    for r in rows:
        print(f"{r['model_name']:<25} "
              f"{r['val_acc_v_mean']:>8.4f}   {r['val_acc_a_mean']:>8.4f}   "
              f"{r['test_acc_v_mean']:>9.4f}   {r['test_acc_a_mean']:>9.4f}   "
              f"{r['test_f1_v_mean']:>8.4f}   {r['test_f1_a_mean']:>8.4f}")
    print('=' * 80)


if __name__ == '__main__':
    run_evaluation()
