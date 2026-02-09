from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold
import numpy as np
import os
import random
import pandas as pd
import torch
from torch.utils.data import Dataset
import json
import joblib
import scipy.stats as stats
from tqdm import tqdm

# Set random seeds
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Configuration
GAMES_TO_RUN = ['apex', 'endless', 'fps', 'gallery','gun', 'platfrom', 'solid', 'tiny', 'topdown']
CLEAN_DATA_CSV_PATH = "../Dataset/clean_data.csv"
DATASET_BASE_PATH = "../Dataset/"
RESULTS_ROOT = "Results/new/rf"

# Helpers
def get_file_name_and_parent_folder(file_path, game_name):
    file_path = os.path.normpath(file_path)
    file_paths = file_path.split(os.sep)
    file_name, _ = os.path.splitext(os.path.basename(file_path))
    parent_folder = os.path.dirname(file_path)
    video_name = file_paths[-2]
    try:
        player_id, session_id = video_name.split(f'_{game_name}_')
    except ValueError:
        player_id, session_id = "unknown", "unknown"
    return file_name, parent_folder, video_name, player_id, session_id

def get_session_id(file_path):
    file_path = os.path.normpath(file_path)
    file_paths = file_path.split(os.sep)
    return file_paths[-2]

def get_player_id_simple(file_path, game_name):
    file_path = os.path.normpath(file_path)
    file_paths = file_path.split(os.sep)
    video_name = file_paths[-2]
    try:
        parts = video_name.split(f'_{game_name}_')
        if len(parts) == 2:
            return parts[0]
        return "unknown"
    except ValueError:
        return "unknown"

class MyCSVDataset(Dataset):
    def __init__(self, data_df, csv_file_2, game_name, label2id):
        self.game_name = game_name
        self.data = data_df.reset_index(drop=True)
        self.label2id = label2id
        self.unknown_labels_seen = set()

        print(f"Loading and indexing clean data from {csv_file_2}...")
        self.gf = pd.read_csv(csv_file_2)

        # Drop logic
        control_drop = [col for col in self.gf.columns if "control" in col and
                        col not in ["[control]player_id", "[control]session_id"]]
        self.gf = self.gf.drop(columns=control_drop, errors='ignore')
        self.gf = self.gf.drop(columns=['[output]arousal'], errors='ignore')
        string_drop = [col for col in self.gf.columns if '[string]' in col]
        if len(string_drop) > 0:
            self.gf = self.gf.drop(columns=string_drop)

        # Indexing
        self.data_lookup = {}
        grouped = self.gf.groupby(['[control]player_id', '[control]session_id'])
        for keys, group in tqdm(grouped, desc="Indexing Data"):
            feature_data = group.drop(columns=['[control]player_id', '[control]session_id'], errors='ignore')
            self.data_lookup[keys] = feature_data
        
        if len(self.data_lookup) > 0:
            sample_val = next(iter(self.data_lookup.values()))
            self.feature_dim = sample_val.shape[1]
        else:
            self.feature_dim = self.gf.shape[1] - 2 

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        clip_path = sample['start_frame']
        
        _, _, _, player_id, session_id = get_file_name_and_parent_folder(clip_path, self.game_name)

        start = int(sample['start_time']) * 4
        end = start + 24

        lookup_key = (player_id, session_id)
        
        if lookup_key in self.data_lookup:
            game_vector = self.data_lookup[lookup_key]
            pd.set_option('future.no_silent_downcasting', True)
            game_vector_slice = game_vector.iloc[start:end]
            game_vector_slice = game_vector_slice.fillna(0).infer_objects(copy=False)
            game_array = np.array(game_vector_slice.values)
        else:
            game_array = np.empty((0, self.feature_dim))

        if len(game_array) == 0:
             game_array = np.zeros((24, self.feature_dim))
        elif game_array.shape[0] < 24:
            pad_width = ((0, 24 - game_array.shape[0]), (0, 0))
            game_array = np.pad(game_array, pad_width, mode='constant', constant_values=0)

        game_array_flat = game_array.flatten()
        
        label_str = sample.get('arousal_change', 'same')
        
        if label_str in self.label2id:
            label_id = self.label2id[label_str]
        else:
            if label_str not in self.unknown_labels_seen:
                print(f"\n[WARNING] Found unknown label: '{label_str}' at index {idx}. Defaulting to class 'same' (ID: 1).")
                self.unknown_labels_seen.add(label_str)
            label_id = self.label2id.get('same', 1)

        return game_array_flat, label_id

def prepare_full_dataset(dataset):
    print("Extracting features from dataset for RF...")
    X_list = []
    y_list = []
    for i in tqdm(range(len(dataset)), desc="Building Arrays"):
        x, y = dataset[i]
        X_list.append(x)
        y_list.append(y)
    X = np.array(X_list)
    y = np.array(y_list)
    return X, y

def process_game(game_name):
    print(f"\nProcessing game: {game_name}")

    csv_file = os.path.join(DATASET_BASE_PATH, f'new_{game_name}.csv')
    if not os.path.exists(csv_file):
        print(f"Error: Data file '{csv_file}' not found. Skipping.")
        return None

    all_data_df = pd.read_csv(csv_file)
    all_data_df['player_id'] = all_data_df['start_frame'].apply(lambda x: get_player_id_simple(x, game_name))
    all_data_df['session_id'] = all_data_df['start_frame'].apply(get_session_id)
    
    all_data_df = all_data_df.sort_values(by=['session_id', 'start_time']).reset_index(drop=True)
    all_data_df['start_frame'] = all_data_df.groupby('session_id')['start_frame'].shift(1)
    all_data_df['start_time'] = all_data_df.groupby('session_id')['start_time'].shift(1)
    all_data_df = all_data_df.dropna(subset=['start_frame', 'start_time']).reset_index(drop=True)
    
    BLOCK_DURATION = 6 
    if 'start_time' not in all_data_df.columns:
        def extract_frame_num(path):
            try: return int(os.path.splitext(os.path.basename(path))[0])
            except: return 0
        all_data_df['temp_frame_num'] = all_data_df['start_frame'].apply(extract_frame_num)
        all_data_df['start_time'] = all_data_df['temp_frame_num'] / 30.0

    all_data_df['block_id'] = all_data_df.apply(
        lambda row: f"{row['player_id']}_{int(row['start_time'] // BLOCK_DURATION)}", 
        axis=1
    )
    groups = all_data_df['block_id'].values

    print(f"Total samples: {len(all_data_df)}")
    print(f"Total blocks (groups): {len(np.unique(groups))}")

    game_output_dir = os.path.join(RESULTS_ROOT, game_name)
    os.makedirs(game_output_dir, exist_ok=True)

    class_labels = ['down', 'same', 'up']
    label2id = {label: i for i, label in enumerate(class_labels)}

    full_ds = MyCSVDataset(all_data_df, CLEAN_DATA_CSV_PATH, game_name, label2id)
    X_all, y_all = prepare_full_dataset(full_ds)

    k_folds = 5
    sgkf = StratifiedGroupKFold(n_splits=k_folds, shuffle=True, random_state=42)

    fold_accs = []
    fold_wf1s = []
    fold_mf1s = []
    
    # Global Confusion Matrix
    total_conf_matrix = np.zeros((3, 3), dtype=int)
    class_metrics_accumulator = {c: {'precision': [], 'recall': [], 'f1-score': []} for c in class_labels}

    print(f"Starting {k_folds}-Fold Micro-Block Stratified CV...")

    for fold, (train_idx, test_idx) in enumerate(sgkf.split(X_all, y_all, groups=groups)):
        print(f"\n--- Fold {fold+1}/{k_folds} ---")
        
        X_train, y_train = X_all[train_idx], y_all[train_idx]
        X_test, y_test = X_all[test_idx], y_all[test_idx]
        
        if len(X_train) == 0 or len(X_test) == 0:
            print("Warning: Empty partition. Skipping.")
            continue

        clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        clf.fit(X_train, y_train)

        test_pred = clf.predict(X_test)
        
        acc = accuracy_score(y_test, test_pred)
        w_f1 = f1_score(y_test, test_pred, average='weighted', zero_division=0)
        m_f1 = f1_score(y_test, test_pred, average='macro', zero_division=0)

        cm = confusion_matrix(y_test, test_pred, labels=[0, 1, 2])
        total_conf_matrix += cm

        fold_accs.append(acc)
        fold_wf1s.append(w_f1)
        fold_mf1s.append(m_f1)

        print(f"  Result: Acc={acc:.4f}, W-F1={w_f1:.4f}, M-F1={m_f1:.4f}")

        report = classification_report(y_test, test_pred, target_names=class_labels, output_dict=True, zero_division=0)
        
        class_metrics_fold = {}
        for cls in class_labels:
            metrics = report[cls]
            class_metrics_fold[cls] = metrics
            class_metrics_accumulator[cls]['precision'].append(metrics['precision'])
            class_metrics_accumulator[cls]['recall'].append(metrics['recall'])
            class_metrics_accumulator[cls]['f1-score'].append(metrics['f1-score'])

        fold_res = {
            "game": game_name,
            "fold": fold + 1,
            "accuracy": acc,
            "weighted_f1": w_f1,
            "macro_f1": m_f1,
            "confusion_matrix": cm.tolist(),
            "class_metrics": class_metrics_fold,
            "train_samples": len(X_train),
            "test_samples": len(X_test)
        }
        
        fold_dir = os.path.join(game_output_dir, f"fold_{fold+1}")
        os.makedirs(fold_dir, exist_ok=True)
        
        with open(os.path.join(fold_dir, "results.json"), 'w') as f:
            json.dump(fold_res, f, indent=4)
            
        joblib.dump(clf, os.path.join(fold_dir, "model.joblib"))

    if not fold_accs: return None

    def compute_ci95(data):
        a = 1.0 * np.array(data)
        n = len(a)
        if n < 2: return np.mean(a), 0.0
        m, se = np.mean(a), stats.sem(a)
        h = se * stats.t.ppf((1 + 0.95) / 2., n-1)
        return m, h

    mean_acc, ci_acc = compute_ci95(fold_accs)
    mean_wf1, ci_wf1 = compute_ci95(fold_wf1s)
    mean_mf1, ci_mf1 = compute_ci95(fold_mf1s)

    avg_class_metrics = {}
    for cls in class_labels:
        avg_class_metrics[cls] = {
            "precision_mean": np.mean(class_metrics_accumulator[cls]['precision']),
            "precision_std": np.std(class_metrics_accumulator[cls]['precision']),
            "recall_mean": np.mean(class_metrics_accumulator[cls]['recall']),
            "recall_std": np.std(class_metrics_accumulator[cls]['recall']),
            "f1_mean": np.mean(class_metrics_accumulator[cls]['f1-score']),
            "f1_std": np.std(class_metrics_accumulator[cls]['f1-score']),
        }

    # --- [New Feature] Percentage Confusion Matrix ---
    # Normalize by row (True Class) -> Percentage of correct/incorrect predictions per class
    with np.errstate(divide='ignore', invalid='ignore'):
        total_conf_matrix_percent = total_conf_matrix.astype('float') / total_conf_matrix.sum(axis=1)[:, np.newaxis] * 100
    # Replace NaN with 0 (occurs if a class has 0 samples in ground truth)
    total_conf_matrix_percent = np.nan_to_num(total_conf_matrix_percent)
    
    print(f"\n=== Final Results for '{game_name}' ===")
    print(f"Accuracy:    {mean_acc:.4f} ± {ci_acc:.4f}")
    print(f"Weighted F1: {mean_wf1:.4f} ± {ci_wf1:.4f}")
    print(f"Macro F1:    {mean_mf1:.4f} ± {ci_mf1:.4f}")
    print("Aggregate Confusion Matrix (Count):")
    print(total_conf_matrix)
    print("Aggregate Confusion Matrix (Percent):")
    print(np.round(total_conf_matrix_percent, 2))

    result_dict = {
        "game": game_name,
        "method": "5-Fold Micro-Block (6s)",
        "accuracy":      {"mean": mean_acc, "ci95": ci_acc, "std": np.std(fold_accs)},
        "weighted_f1":   {"mean": mean_wf1, "ci95": ci_wf1, "std": np.std(fold_wf1s)},
        "macro_f1":      {"mean": mean_mf1, "ci95": ci_mf1, "std": np.std(fold_mf1s)},
        "aggregate_confusion_matrix": total_conf_matrix.tolist(),
        "aggregated_confusion_matrix_percent": np.round(total_conf_matrix_percent, 2).tolist(), # Added this
        "avg_class_metrics": avg_class_metrics,
        "fold_details": {
            "accuracies": fold_accs,
            "weighted_f1s": fold_wf1s,
            "macro_f1s": fold_mf1s
        }
    }

    with open(os.path.join(game_output_dir, "summary.json"), 'w') as f:
        json.dump(result_dict, f, indent=4)

    return result_dict

def main():
    if not os.path.exists(RESULTS_ROOT):
        os.makedirs(RESULTS_ROOT)

    all_results = []
    for game in GAMES_TO_RUN:
        res = process_game(game)
        if res:
            all_results.append(res)

    if all_results:
        flat_results = []
        for res in all_results:
            row = {
                "game": res['game'],
                "acc_mean": res['accuracy']['mean'],
                "acc_ci": res['accuracy']['ci95'],
                "wf1_mean": res['weighted_f1']['mean'],
                "wf1_ci": res['weighted_f1']['ci95'],
                "mf1_mean": res['macro_f1']['mean'],
                "mf1_ci": res['macro_f1']['ci95']
            }
            
            # 1. Flatten Count CM
            labels = ['down', 'same', 'up']
            cm_count = np.array(res['aggregate_confusion_matrix'])
            for i, true_label in enumerate(labels):
                for j, pred_label in enumerate(labels):
                    row[f"cm_count_true_{true_label}_pred_{pred_label}"] = cm_count[i, j]

            # 2. Flatten Percent CM (New)
            cm_pct = np.array(res['aggregated_confusion_matrix_percent'])
            for i, true_label in enumerate(labels):
                for j, pred_label in enumerate(labels):
                    row[f"cm_pct_true_{true_label}_pred_{pred_label}"] = cm_pct[i, j]

            for cls, metrics in res['avg_class_metrics'].items():
                row[f"{cls}_f1"] = metrics['f1_mean']
                row[f"{cls}_prec"] = metrics['precision_mean']
                row[f"{cls}_rec"] = metrics['recall_mean']
            flat_results.append(row)

        df = pd.DataFrame(flat_results)
        print("\nFinal Results Summary:")
        print(df)
        df.to_csv(os.path.join(RESULTS_ROOT, "all_games_rf_results.csv"), index=False)

if __name__ == '__main__':
    main()