import glob, os, pandas as pd

base = 'c:/Omar/Education/GP/DEAP/eeg_tsception_baseline/results_tsception_baseline'
csvs = glob.glob(base + '/**/*.csv', recursive=True)

for c in sorted(csvs):
    if 'overall' in c or 'test_video_predictions' in c: 
        continue
    try:
        df = pd.read_csv(c)
        c_norm = os.path.normpath(c)
        parts = c_norm.split(os.sep)
        subj_idx = parts.index('results_tsception_baseline') + 1
        subject = parts[subj_idx]
        fold = parts[subj_idx + 3]
        
        ep = df['epoch'].max()
        val_acc = df['val_acc'].iloc[-1]
        val_f1 = df['val_f1'].iloc[-1]
        
        print(f"{subject} {fold} | Epoch {ep} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")
    except Exception as e:
        print(f"Error parsing {c}: {e}")
