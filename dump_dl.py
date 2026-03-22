import os, glob, json, numpy as np

for exp_dir in ['c:/Omar/Education/GP/DEAP/eeg_dl_experiment/results', 'c:/Omar/Education/GP/DEAP/eeg_de_experiment/results']:
    models = [d for d in glob.glob(exp_dir + '/*') if os.path.isdir(d)]
    for m in models:
        m_name = os.path.basename(m)
        for target in ['valence', 'arousal']:
            jsons = glob.glob(m + f'/**/{target}/**/fold_summary.json', recursive=True)
            if not jsons: continue
            
            accs, f1s = [], []
            for j in jsons:
                try:
                    with open(j) as f:
                        data = json.load(f)
                        if 'video_accuracy' in data: accs.append(data['video_accuracy'])
                        if 'video_f1' in data: f1s.append(data['video_f1'])
                except:
                    pass
            if accs:
                print(f"{m_name} ({target}): Acc {np.mean(accs):.4f} - F1 {np.mean(f1s):.4f} (Folds: {len(accs)})")
