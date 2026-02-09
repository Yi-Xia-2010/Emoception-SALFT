import os
import json
import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedGroupKFold

# ================= Configuration =================

PATH_TO_OURS = "Results/new/ours" 
PATH_TO_FULL = "Results/new/full"
PATH_TO_EXPLORATION = "Results/new/exploration/new/full"
DATASET_DIR = "Dataset"  

FOLD_NAME = "fold_1"        
LOG_FILE = "validation_logs.json"
GFLOPS_FULL = 6402.90
GFLOPS_OURS = 4449.00
BATCH_SIZE = 4


def get_exploration_epochs(game_name):
    exp_dir = os.path.join(PATH_TO_EXPLORATION, game_name, FOLD_NAME)
    
    if not os.path.exists(exp_dir):
        default_epochs = 3 if game_name.lower() in ['gallery', 'platform'] else 2
        print(f"      [Exploration] Path not exist, use: {default_epochs}")
        return default_epochs

    subdirs = [d for d in os.listdir(exp_dir) if os.path.isdir(os.path.join(exp_dir, d))]
    
    epochs = []
    for d in subdirs:
        if d.endswith("epoch"):
            try:
                num = int(d.replace("epoch", ""))
                epochs.append(num)
            except ValueError:
                pass
    
    epochs.sort()
    
    max_continuous = 0
    for i, ep in enumerate(epochs):
        expected = i + 1
        if ep == expected:
            max_continuous = ep
        else:
            break
            
    if max_continuous == 0:
        default_epochs = 3 if game_name.lower() in ['gallery', 'platform'] else 2
        print(f"      [Exploration] Couldn't found continuous Epoch，use default: {default_epochs}")
        return default_epochs

    print(f"      [Exploration] ✅ explore epoch: {max_continuous}")
    return max_continuous


def get_player_id_simple(file_path, game_name):
    file_path = os.path.normpath(file_path)
    file_paths = file_path.split(os.sep)
    try:
        video_name = file_paths[-2]
        parts = video_name.split(f'_{game_name}_')
        if len(parts) >= 1: return parts[0]
        return "unknown"
    except: return "unknown"

def get_actual_train_steps(game_name):
    possible_paths = [
        os.path.join(DATASET_DIR, f"new_{game_name}.csv"),
        os.path.join("..", "Dataset", f"new_{game_name}.csv"),
        f"new_{game_name}.csv"
    ]
    csv_path = next((p for p in possible_paths if os.path.exists(p)), None)
            
    if not csv_path:
        print(f"    Couldn't found CSV (new_{game_name}.csv)")
        return None

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"   Failed to read CSV: {e}")
        return None

    if 'start_time' not in df.columns:
        def extract_frame_num(path):
            try: return int(os.path.splitext(os.path.basename(path))[0])
            except: return 0
        df['temp_frame_num'] = df['start_frame'].apply(extract_frame_num)
        df['start_time'] = df['temp_frame_num'] / 30.0

    df['player_id'] = df['start_frame'].apply(lambda x: get_player_id_simple(x, game_name))
    df = df.sort_values(by=['player_id', 'start_time']).reset_index(drop=True)
    df['arousal_change'] = df.groupby('player_id')['arousal_change'].shift(-1)
    df = df.dropna(subset=['arousal_change']).reset_index(drop=True)
    
    BLOCK_DURATION = 6
    df['block_id'] = df.apply(lambda r: f"{r['player_id']}_{int(r['start_time'] // BLOCK_DURATION)}", axis=1)
    groups = df['block_id'].values
    
    class_labels = ['down', 'same', 'up']
    label2id = {l: i for i, l in enumerate(class_labels)}
    df['arousal_change'] = df['arousal_change'].astype(str)
    y_all = df['arousal_change'].map(lambda x: label2id.get(x, 1)).values

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    splits = list(sgkf.split(df, y_all, groups=groups))
    
    if len(splits) == 0: return None

    train_val_idx, test_idx = splits[0] 
    train_val_df = df.iloc[train_val_idx]
    train_val_y = y_all[train_val_idx]
    train_val_groups = groups[train_val_idx]

    inner_sgkf = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=42)
    inner_train_idx, inner_val_idx = next(inner_sgkf.split(train_val_df, train_val_y, groups=train_val_groups))
    
    num_train_samples = len(inner_train_idx)
    steps_per_epoch = int(np.ceil(num_train_samples / BATCH_SIZE))
    
    print(f"   ✅ [Dataset] {game_name}: {num_train_samples} samples -> {steps_per_epoch} steps/epoch")
    return steps_per_epoch


def calculate_cumulative_cost(f1_history, steps_per_epoch, is_ours, exploration_epochs):
    cumulative_cost = []
    current_total_gflops = 0.0
    
    cost_full_epoch = steps_per_epoch * GFLOPS_FULL
    cost_ours_epoch = steps_per_epoch * GFLOPS_OURS
    
    for epoch_idx in range(len(f1_history)):
        if is_ours:
            if epoch_idx < exploration_epochs:
                step_cost = cost_full_epoch 
            else:
                step_cost = cost_ours_epoch 
        else:
            step_cost = cost_full_epoch      
            
        current_total_gflops += step_cost
        cumulative_cost.append(current_total_gflops)
        
    return cumulative_cost

def load_f1_history(log_path):
    if not os.path.exists(log_path): 
        print(f"      [Missing] {log_path}")
        return []
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            return [entry.get('f1_weighted', 0) for entry in json.load(f)]
    except: return []


def main():
    print(f"--- Efficiency Analysis (Academic Unit Scaling) ---")
    
    if not os.path.exists(PATH_TO_OURS):
        print(f"Error :Path not exist: {PATH_TO_OURS}"); return

    games = [d for d in os.listdir(PATH_TO_OURS) if os.path.isdir(os.path.join(PATH_TO_OURS, d))]
    print(f"found game: {games}")

    for game in games:
        print(f"\n" + "-"*50 + f"\nProcessing: {game}\n" + "-"*50)
        
        steps_per_epoch = get_actual_train_steps(game)
        if steps_per_epoch is None: continue

        exp_epochs = get_exploration_epochs(game)

        path_ours = os.path.join(PATH_TO_OURS, game, FOLD_NAME, LOG_FILE)
        path_full = os.path.join(PATH_TO_FULL, game, FOLD_NAME, LOG_FILE)
        
        f1_ours = load_f1_history(path_ours)
        f1_full = load_f1_history(path_full)
        
        if not f1_ours or not f1_full:
            print("   Skip, missing logs")
            continue

        
        cost_ours_raw = calculate_cumulative_cost(f1_ours, steps_per_epoch, is_ours=True, exploration_epochs=exp_epochs)
        cost_full_raw = calculate_cumulative_cost(f1_full, steps_per_epoch, is_ours=False, exploration_epochs=999)

        
        max_val = max(cost_full_raw) if cost_full_raw else 0
        
        # 1 PFLOP = 1,000,000 GFLOPs (10^15 FLOPs)
        # 1 TFLOP = 1,000 GFLOPs (10^12 FLOPs)
        
        if max_val >= 1e6:
            scale_factor = 1e6
            unit_label = "PFLOPs" # Peta FLOPs
        elif max_val >= 1e3:
            scale_factor = 1e3
            unit_label = "TFLOPs" # Tera FLOPs
        else:
            scale_factor = 1.0
            unit_label = "GFLOPs" # Giga FLOPs 

       
        cost_full_plot = [c / scale_factor for c in cost_full_raw]
        cost_ours_plot = [c / scale_factor for c in cost_ours_raw]
        
        print(f"   📏 Auto-Scaling: Max={max_val:.2f} GFLOPs -> Scaled by /{scale_factor} -> Unit: {unit_label}")
        

        
        plt.figure(figsize=(8, 5))
       
        plt.rcParams.update({'font.size': 12, 'font.family': 'sans-serif'})
        
        # Plot Full
        plt.plot(cost_full_plot, f1_full, marker='o', linestyle='-', color='#0072B2', 
                 label=f'Full Finetuning', linewidth=2)
        
        # Plot Ours
        plt.plot(cost_ours_plot, f1_ours, marker='s', linestyle='--', color='#D55E00', 
                 label=f'SALFT (Ours)', linewidth=2)

       
        plt.xlabel(f"Cumulative Computational Cost ({unit_label})")
        plt.ylabel("Weighted F1 Score")
        plt.title(f"Efficiency Analysis: {game}")
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend(loc='lower right')
        plt.tight_layout()
        
        out_name = f"Results/new/efficiency_{game}.png"
        plt.savefig(out_name, dpi=300)
        print(f"   Saved: {out_name}")
        plt.close()

if __name__ == "__main__":
    main()