import pandas as pd
import numpy as np
import sys
from pathlib import Path

# 1. Load DL Subject Summary
dl_summary_path = Path("c:/Omar/Education/GP/DEAP/eeg_eegnet_grouped_experiment/results_weighted_baseline_macrof1_fullrun/subject_summary.csv")
dl_df = pd.read_csv(dl_summary_path)

def analyze_dl_bottom_quarter(target):
    target_df = dl_df[dl_df['target'] == target].copy()
    num_subjects = len(target_df)
    quarter = num_subjects // 4
    
    # Sort by accuracy
    sorted_df = target_df.sort_values(by='video_accuracy_mean', ascending=True)
    bottom_quarter = sorted_df.head(quarter)
    
    bottom_subject_ids = bottom_quarter['subject_id'].tolist()
    
    # Calculate means
    original_mean = target_df['video_accuracy_mean'].mean()
    
    top_three_quarters = sorted_df.iloc[quarter:]
    new_mean = top_three_quarters['video_accuracy_mean'].mean()
    
    print(f"\n=== DL (EEGNet) - {target.capitalize()} ===")
    print(f"Original Mean Accuracy (32 subjects): {original_mean:.2%}")
    print(f"Bottom 25% Subjects (8 subjects): {bottom_subject_ids}")
    print(f"Bottom 25% Mean Accuracy: {bottom_quarter['video_accuracy_mean'].mean():.2%}")
    print(f"New Mean Accuracy (Top 24 subjects): {new_mean:.2%}")
    print(f"Absolute Jump: {new_mean - original_mean:.2%}")
    
    return bottom_subject_ids

dl_val_bottom = analyze_dl_bottom_quarter('valence')
dl_aro_bottom = analyze_dl_bottom_quarter('arousal')
