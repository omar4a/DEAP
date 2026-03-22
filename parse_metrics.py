import os, json, glob
import numpy as np

base_dl = 'c:/Omar/Education/GP/DEAP/eeg_dl_experiment'
res = {}

for root, dirs, files in os.walk(base_dl):
    if 'fold_summary.json' in files:
        path = os.path.join(root, 'fold_summary.json')
        parts = path.split(os.sep)
        try:
            idx = parts.index('eeg_dl_experiment') + 1
            model = parts[idx]
        except: continue
        tgt = ''
        if 'valence' in parts: tgt = 'valence'
        if 'arousal' in parts: tgt = 'arousal'
        
        if model not in res:
            res[model] = {'valence': {'acc': [], 'f1': []}, 'arousal': {'acc': [], 'f1': []}}
            
        if tgt:
            try:
                with open(path) as f:
                    d = json.load(f)
                    res[model][tgt]['acc'].append(d.get('video_accuracy', 0))
                    res[model][tgt]['f1'].append(d.get('video_f1', 0))
            except: pass

print("--- DL EXPERIMENTS ---")
for m, v in res.items():
    print(f'Model: {m}')
    val_acc = np.mean(v['valence']['acc']) if v['valence']['acc'] else 0
    val_f1 = np.mean(v['valence']['f1']) if v['valence']['f1'] else 0
    aro_acc = np.mean(v['arousal']['acc']) if v['arousal']['acc'] else 0
    aro_f1 = np.mean(v['arousal']['f1']) if v['arousal']['f1'] else 0
    print(f'  Valence Acc: {val_acc:.4f} | F1: {val_f1:.4f} | Folds: {len(v["valence"]["acc"])}')
    print(f'  Arousal Acc: {aro_acc:.4f} | F1: {aro_f1:.4f} | Folds: {len(v["arousal"]["acc"])}')

import pandas as pd
print("\n--- EEGNET BASELINE CSVs ---")
csvs = glob.glob('c:/Omar/Education/GP/DEAP/eeg_eegnet_grouped_experiment/results*/overall_summary.csv')
for c in csvs:
    print(f"\n[{os.path.basename(os.path.dirname(c))}]")
    try:
        df = pd.read_csv(c)
        print(df[['target', 'video_accuracy_mean', 'video_f1_mean']].to_string(index=False))
    except Exception as e:
        print(e)
