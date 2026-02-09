import os
import json
import time
import random
import argparse
import gc
from collections import defaultdict, Counter
from tqdm import tqdm

import numpy as np
import pandas as pd
import scipy.stats as stats
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.manifold import TSNE
from transformers import VivitImageProcessor, VivitForVideoClassification, AdamW
from torchvision import transforms
import matplotlib.pyplot as plt
import matplotlib
from torch.cuda.amp import autocast, GradScaler

# Set Matplotlib backend to Agg
matplotlib.use('Agg')

# [Optimization] Enable TF32 acceleration
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True # Faster for fixed input size

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = False # False is faster for ViViT
    torch.backends.cudnn.benchmark = True

# === Helper: Parse Player ID ===
def get_player_id_simple(file_path, game_name):
    file_path = os.path.normpath(file_path)
    file_paths = file_path.split(os.sep)
    try:
        video_name = file_paths[-2]
        parts = video_name.split(f'_{game_name}_')
        if len(parts) >= 1: return parts[0]
        return "unknown"
    except: return "unknown"

# === Helper: Statistics ===
def compute_statistics(data):
    n = len(data)
    if n < 2: return np.mean(data), 0.0, 0.0
    mean_val = np.mean(data)
    std_val = np.std(data, ddof=1)
    se = std_val / np.sqrt(n)
    h = se * stats.t.ppf((1 + 0.95) / 2., n-1)
    return mean_val, std_val, h

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super(NpEncoder, self).default(obj)

def save_results_to_json(filepath, data):
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False, cls=NpEncoder)

# === Test A: User Entropy Analysis (Fixed args) ===
def analyze_user_label_entropy(df, output_dir, label2id=None):
    print("\n[Analysis] Starting User Label Entropy Analysis...")
    user_distributions = df.groupby('player_id')['arousal_change'].value_counts(normalize=True).unstack(fill_value=0)
    results, user_entropies = [], []
    for player_id, row in user_distributions.iterrows():
        ent = stats.entropy(row.values, base=3) 
        user_entropies.append(ent)
        results.append({"player_id": player_id, "entropy": ent})
    
    mean_ent = np.mean(user_entropies)
    pd.DataFrame(results).to_csv(os.path.join(output_dir, "user_entropy_details.csv"), index=False)
    
    plt.figure(figsize=(10, 6))
    plt.hist(user_entropies, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
    plt.axvline(mean_ent, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_ent:.2f}')
    plt.title('Distribution of Intra-User Label Entropy')
    plt.savefig(os.path.join(output_dir, "user_label_entropy_distribution.png"))
    plt.close()
    
    summary = {"mean_entropy": mean_ent, "min_entropy": np.min(user_entropies), "max_entropy": np.max(user_entropies)}
    save_results_to_json(os.path.join(output_dir, "user_entropy_summary.json"), summary)

# === Test B: t-SNE Analysis ===
def perform_tsne_analysis(model, dataloader, df, device, output_dir, fold_idx):
    print(f"\n[t-SNE] Extracting features for Fold {fold_idx} Test Set...")
    model.eval()
    features_list, labels_list = [], []
    all_players = df['player_id'].values
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Embeddings"):
            pixel_values = batch["pixel_values"].to(device, non_blocking=True)
            labels = batch["label"].to(device).squeeze(1)
            
            with autocast(dtype=torch.bfloat16): 
                outputs = model(pixel_values=pixel_values, output_hidden_states=True)
                cls_emb = outputs.hidden_states[-1][:, 0, :] 
            features_list.append(cls_emb.cpu().float().numpy())
            labels_list.append(labels.cpu().numpy())

    X = np.concatenate(features_list, axis=0)
    y_mood = np.concatenate(labels_list, axis=0)
    
    if len(X) != len(all_players):
        min_len = min(len(X), len(all_players))
        X, y_mood, all_players = X[:min_len], y_mood[:min_len], all_players[:min_len]

    tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')
    X_embedded = tsne.fit_transform(X)

    tsne_df = pd.DataFrame({'tsne_x': X_embedded[:, 0], 'tsne_y': X_embedded[:, 1], 'label_mood': y_mood, 'player_id': all_players})
    tsne_df.to_csv(os.path.join(output_dir, f"tsne_coordinates_fold_{fold_idx}.csv"), index=False)

    fig, axes = plt.subplots(1, 2, figsize=(20, 9))
    scatter1 = axes[0].scatter(X_embedded[:, 0], X_embedded[:, 1], c=y_mood, cmap='viridis', alpha=0.6, s=15)
    axes[0].set_title(f"Feature Space by Mood")
    axes[0].legend(*scatter1.legend_elements(), title="Mood")
    
    unique_players = np.unique(all_players)
    pid_map = {pid: i for i, pid in enumerate(unique_players)}
    y_pid = np.array([pid_map[p] for p in all_players])
    axes[1].scatter(X_embedded[:, 0], X_embedded[:, 1], c=y_pid, cmap='tab20', alpha=0.6, s=15)
    axes[1].set_title(f"Feature Space by Player ID")
    plt.tight_layout(); plt.savefig(os.path.join(output_dir, f"tsne_fold_{fold_idx}.png"), dpi=300); plt.close()

# === Pre-packing Logic (Original logic for pre-processed frames) ===
def preprocess_dataset(df, root_dir, cache_dir, game_name):
    os.makedirs(cache_dir, exist_ok=True)
    print(f"\n[Pre-packing] Starting to pack images into {cache_dir}...")
    unique_samples = df.drop_duplicates(subset=['start_frame'])
    for idx, row in tqdm(unique_samples.iterrows(), total=len(unique_samples), desc="Packing"):
        clip_path = row['start_frame']
        file_name_str = os.path.splitext(os.path.basename(clip_path))[0]
        parent_folder = os.path.dirname(clip_path)
        safe_name = f"{parent_folder}_{file_name_str}.pt".replace(os.sep, "_")
        save_path = os.path.join(cache_dir, safe_name)
        if os.path.exists(save_path): continue
        frames = []
        try: file_name_int = int(file_name_str)
        except ValueError: file_name_int = 0
        for i in range(32):
            frame_path = os.path.join(root_dir, parent_folder, f"{file_name_int + i:04d}.png")
            try:
                with Image.open(frame_path) as img:
                    img = img.convert('RGB').resize((256, 256))
                    tensor_img = transforms.PILToTensor()(img) 
                    frames.append(tensor_img)
            except: frames.append(torch.zeros((3, 256, 256), dtype=torch.uint8))
        torch.save(torch.stack(frames), save_path)
    print("[Pre-packing] Done!\n")

# === Dataset Class ===
class MyCSVDataset(Dataset):
    def __init__(self, data_df, game_name, image_processor, label2id, root_dir="../Dataset/", cache_dir=None):
        self.data = data_df.reset_index(drop=True)
        self.game_name = game_name
        self.image_processor = image_processor
        self.label2id = label2id
        self.root_dir = root_dir
        self.cache_dir = cache_dir
        self.to_pil = transforms.ToPILImage()

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        clip_path = sample['start_frame']
        file_name_str = os.path.splitext(os.path.basename(clip_path))[0]
        parent_folder = os.path.dirname(clip_path)
        frames, loaded = [], False

        if self.cache_dir:
            safe_name = f"{parent_folder}_{file_name_str}.pt".replace(os.sep, "_")
            cache_path = os.path.join(self.cache_dir, safe_name)
            if os.path.exists(cache_path):
                try:
                    video_tensor = torch.load(cache_path)
                    frames = [self.to_pil(video_tensor[i]) for i in range(video_tensor.shape[0])]
                    loaded = True
                except: pass
        
        if not loaded:
            try: file_name_int = int(file_name_str)
            except ValueError: file_name_int = 0
            for i in range(32):
                try: frames.append(Image.open(os.path.join(self.root_dir, parent_folder, f"{file_name_int + i:04d}.png")).convert('RGB'))
                except: frames.append(Image.new('RGB', (224, 224), (0, 0, 0)))

        inputs = self.image_processor(list(frames), return_tensors="pt")
        inputs['pixel_values'] = inputs['pixel_values'].squeeze(0)
        
        raw_label = str(sample.get('arousal_change', 'same'))
        try:
            label_id = self.label2id[raw_label]
        except KeyError:            
            raise KeyError(f"❌ Data Integrity Error: Found unknown label '{raw_label}' at index {idx}. Allowed labels: {list(self.label2id.keys())}")
            
        inputs['label'] = torch.LongTensor([label_id])
        return inputs

# === Evaluation ===
def evaluate(model, dataloader, device, case_name, epoch, label2id, id2label):
    model.eval()
    all_labels, all_preds, total_loss = [], [], 0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for batch in dataloader:
            pixel_values = batch["pixel_values"].to(device, non_blocking=True)
            labels = batch["label"].to(device).squeeze(1)
            
            with autocast(dtype=torch.bfloat16): 
                outputs = model(pixel_values=pixel_values)
                loss = criterion(outputs.logits, labels)
            total_loss += loss.item()
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(outputs.logits.argmax(-1).cpu().numpy())
    
    avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0.0
    accuracy = accuracy_score(all_labels, all_preds)
    f1_w = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    print(f"[{case_name}] Ep{epoch} | Acc: {accuracy:.4f} | W-F1: {f1_w:.4f} | Loss: {avg_loss:.4f}")
    
    possible_labels = sorted(list(label2id.values()))
    cm = confusion_matrix(all_labels, all_preds, labels=possible_labels)
    with np.errstate(divide='ignore', invalid='ignore'):
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_percent = np.nan_to_num(cm_percent) * 100
    
    class_metrics = {}
    labels_present = sorted(list(set(all_labels) | set(all_preds)))
    p_class = precision_score(all_labels, all_preds, average=None, labels=labels_present, zero_division=0)
    r_class = recall_score(all_labels, all_preds, average=None, labels=labels_present, zero_division=0)
    f_class = f1_score(all_labels, all_preds, average=None, labels=labels_present, zero_division=0)
    
    for i, label_id in enumerate(labels_present):
        if int(label_id) in id2label:
            c_name = id2label[int(label_id)]
            class_metrics[c_name] = {"precision": p_class[i], "recall": r_class[i], "f1_score": f_class[i]}

    return {
        "epoch": epoch, "loss": avg_loss, 
        "accuracy": accuracy, "f1_weighted": f1_w, 
        "f1_macro": f1_score(all_labels, all_preds, average='macro', zero_division=0), 
        "confusion_matrix": cm.tolist(), 
        "confusion_matrix_percent": np.round(cm_percent, 2).tolist(),
        "class_metrics": class_metrics
    }

def plot_curves(train_losses, val_losses, train_accs, val_accs, output_path):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_losses, 'b-', label="Train Loss")
    if val_losses: plt.plot(epochs, val_losses, 'r-', label="Val Loss")
    plt.title("Loss Curve"); plt.legend(); plt.grid(True)
    plt.savefig(os.path.join(output_path, 'loss_curve.png')); plt.close()
    
    if train_accs:
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, train_accs, 'b--', label="Train Acc")
        if val_accs: plt.plot(epochs, val_accs, 'r--', label="Val Acc")
        plt.title("Accuracy Curve"); plt.legend(); plt.grid(True)
        plt.savefig(os.path.join(output_path, 'acc_curve.png')); plt.close()

# === Main Function ===
def main(args):
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    scaler = GradScaler(enabled=False) 
    ACCUM_STEPS = 1 # [Config] No Accumulation
    
    CACHE_DIR = f"../Dataset/cache_{args.game_name}"
    class_labels = ['down', 'same', 'up']
    label2id = {l: i for i, l in enumerate(class_labels)}
    id2label = {i: l for l, i in label2id.items()}
    
    base_output_dir = os.path.join(args.output_dir, args.game_name)
    os.makedirs(base_output_dir, exist_ok=True)
    
    model_ckpt = "google/vivit-b-16x2-kinetics400"
    image_processor = VivitImageProcessor.from_pretrained(model_ckpt)

    # 1. Load Data
    csv_file = f'../Dataset/new_{args.game_name}.csv'
    if not os.path.exists(csv_file):
        print(f" Error: CSV file not found: {csv_file}")
        return
        
    print(f"Loaded Data: {csv_file}")
    all_data_df = pd.read_csv(csv_file)
    
    # [Lag Logic]
    print("\n[Lag Logic] Applying 1-second label delay...")
    if 'start_time' not in all_data_df.columns:
        def extract_frame_num(path):
            try: return int(os.path.splitext(os.path.basename(path))[0])
            except: return 0
        all_data_df['temp_frame_num'] = all_data_df['start_frame'].apply(extract_frame_num)
        all_data_df['start_time'] = all_data_df['temp_frame_num'] / 30.0

    all_data_df['player_id'] = all_data_df['start_frame'].apply(lambda x: get_player_id_simple(x, args.game_name))
    all_data_df = all_data_df.sort_values(by=['player_id', 'start_time']).reset_index(drop=True)
    all_data_df['arousal_change'] = all_data_df.groupby('player_id')['arousal_change'].shift(-1)
    all_data_df = all_data_df.dropna(subset=['arousal_change']).reset_index(drop=True)
    
    # [Safety] Ensure labels are strings
    all_data_df['arousal_change'] = all_data_df['arousal_change'].astype(str)
    print(f"[Lag Logic] Dropped tail samples.")

    if args.do_preprocess: 
        preprocess_dataset(all_data_df, "../Dataset/", CACHE_DIR, args.game_name)
    
    analyze_user_label_entropy(all_data_df, base_output_dir, label2id)
    
    # [KEY LOGIC] Micro-Block Split (Mixed-Subject, but Leak-Resistant)
    BLOCK_DURATION = 6
    all_data_df['block_id'] = all_data_df.apply(lambda r: f"{r['player_id']}_{int(r['start_time'] // BLOCK_DURATION)}", axis=1)
    groups = all_data_df['block_id'].values
    y_all = all_data_df['arousal_change'].map(label2id).values

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    all_test_results = []
    
    finetune_schedule = None
    if args.finetune_schedule_file:
        with open(args.finetune_schedule_file, 'r') as f: finetune_schedule = json.load(f)

    for fold, (train_val_idx, test_idx) in enumerate(sgkf.split(all_data_df, y_all, groups=groups)):
        fold_num = fold + 1
        print(f"\n" + "="*40 + f"\nFOLD {fold_num}/5 (Lag: 1s, Block: 6s, AccSteps: {ACCUM_STEPS})\n" + "="*40)
        
        fold_dir = os.path.join(base_output_dir, f"fold_{fold_num}")
        model_dir = os.path.join(fold_dir, "checkpoints")
        os.makedirs(model_dir, exist_ok=True)
        
        test_df = all_data_df.iloc[test_idx].copy()
        train_val_df = all_data_df.iloc[train_val_idx]
        train_val_y = y_all[train_val_idx]
        train_val_groups = groups[train_val_idx]
        
        inner_sgkf = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=42)
        train_idx, val_idx = next(inner_sgkf.split(train_val_df, train_val_y, groups=train_val_groups))
        
        train_loader = DataLoader(MyCSVDataset(train_val_df.iloc[train_idx], args.game_name, image_processor, label2id, cache_dir=CACHE_DIR), batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
        val_loader = DataLoader(MyCSVDataset(train_val_df.iloc[val_idx], args.game_name, image_processor, label2id, cache_dir=CACHE_DIR), batch_size=4, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)
        test_loader = DataLoader(MyCSVDataset(test_df, args.game_name, image_processor, label2id, cache_dir=CACHE_DIR), batch_size=4, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)

        model = VivitForVideoClassification.from_pretrained(model_ckpt, label2id=label2id, id2label=id2label, ignore_mismatched_sizes=True).to(device)

        unfreeze = finetune_schedule.get(f"fold_{fold_num}", []) if finetune_schedule else args.trainable_layers
        if unfreeze:
            for p in model.parameters(): p.requires_grad = False
            for n, p in model.named_parameters():
                if any(k in n for k in (unfreeze + ["classifier"])): p.requires_grad = True
            print(f"  [Freezing] Unfrozen: {unfreeze}")
        else:
            print("  [Freezing] Full Finetuning")

        optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)
        best_f1, best_path = -1.0, ""
        best_epoch = 0 
        validation_logs, train_losses, val_losses, train_accs, val_accs = [], [], [], [], []

        for epoch in range(args.num_epochs):
            model.train()
            run_loss = 0
            train_correct = 0
            train_total = 0
            optimizer.zero_grad() 
            
            for i, batch in enumerate(train_loader):
                pixel_values = batch["pixel_values"].to(device, non_blocking=True)
                labels = batch["label"].to(device).squeeze(1)
                
                with autocast(dtype=torch.bfloat16):
                    outputs = model(pixel_values=pixel_values, labels=labels)
                    loss = outputs.loss / ACCUM_STEPS
                    
                    logits = outputs.logits
                    preds = logits.argmax(dim=-1)
                    train_correct += (preds == labels).sum().item()
                    train_total += labels.size(0)
                
                # Scaler acts as a no-op when enabled=False, keeping logic consistent
                scaler.scale(loss).backward()
                
                if (i + 1) % ACCUM_STEPS == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                
                run_loss += loss.item() * ACCUM_STEPS

            avg_train_loss = run_loss / len(train_loader)
            avg_train_acc = train_correct / train_total if train_total > 0 else 0.0
            print(f"Epoch {epoch+1}: Loss {avg_train_loss:.4f} | Train Acc {avg_train_acc:.4f}")

            val_res = evaluate(model, val_loader, device, "Val", epoch+1, label2id, id2label)
            val_res['train_loss'] = avg_train_loss
            validation_logs.append(val_res)
            
            train_losses.append(avg_train_loss); train_accs.append(avg_train_acc) 
            val_losses.append(val_res['loss']); val_accs.append(val_res['accuracy'])
            
            model.save_pretrained(os.path.join(model_dir, f"checkpoint_epoch_{epoch+1}"))

            if val_res['f1_weighted'] > best_f1:
                best_f1 = val_res['f1_weighted']
                best_path = os.path.join(fold_dir, "checkpoint_best")
                best_epoch = epoch + 1 
                model.save_pretrained(best_path)
                image_processor.save_pretrained(best_path)
            
            save_results_to_json(os.path.join(fold_dir, "validation_logs.json"), validation_logs)
        
        plot_curves(train_losses, val_losses, train_accs, val_accs, fold_dir)

        # Final Test
        model = VivitForVideoClassification.from_pretrained(best_path).to(device)
        test_res = evaluate(model, test_loader, device, "Test", best_epoch, label2id, id2label)
        save_results_to_json(os.path.join(fold_dir, "test_results.json"), test_res)
        all_test_results.append(test_res)
        
        if fold_num == 1: 
            perform_tsne_analysis(model, test_loader, test_df, device, base_output_dir, fold_num)
        
        del model; torch.cuda.empty_cache(); gc.collect()

    # === Final Aggregated Summary ===
    if all_test_results:
        accs = [res['accuracy'] for res in all_test_results]
        f1s_w = [res['f1_weighted'] for res in all_test_results]
        mean_acc, std_acc, ci_acc = compute_statistics(accs)
        mean_f1, std_f1, ci_f1 = compute_statistics(f1s_w)
        
        num_classes = len(label2id)
        agg_cm = np.zeros((num_classes, num_classes))
        for r in all_test_results: agg_cm += np.array(r['confusion_matrix'])
        
        with np.errstate(divide='ignore', invalid='ignore'):
            agg_cm_pct = agg_cm / agg_cm.sum(axis=1, keepdims=True)
        agg_cm_pct = np.nan_to_num(agg_cm_pct) * 100

        avg_class_metrics = defaultdict(lambda: defaultdict(list))
        for r in all_test_results:
            for cls_name, metrics in r['class_metrics'].items():
                for metric_name, val in metrics.items():
                    avg_class_metrics[cls_name][metric_name].append(val)
        
        final_class_summary = {}
        for cls_name, metrics in avg_class_metrics.items():
            final_class_summary[cls_name] = {k: np.mean(v) for k, v in metrics.items()}

        print(f"\nFINAL CV RESULT -> Acc: {mean_acc:.4f} ± {ci_acc:.4f} | W-F1: {mean_f1:.4f} ± {ci_f1:.4f}")
        print(f"Class Summary: {final_class_summary}")
        
        agg_res = {
            "mean_accuracy": mean_acc, "ci95_accuracy": ci_acc,
            "mean_f1_weighted": mean_f1, "ci95_f1_weighted": ci_f1,
            "aggregated_confusion_matrix": agg_cm.tolist(),
            "aggregated_confusion_matrix_percent": np.round(agg_cm_pct, 2).tolist(),
            "avg_class_metrics": final_class_summary,
            "fold_results": all_test_results
        }
        save_results_to_json(os.path.join(base_output_dir, "aggregated_cv_results.json"), agg_res)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--game_name', type=str, default="solid")
    parser.add_argument('--output_dir', type=str, default="Results/new/full")
    parser.add_argument('--num_epochs', type=int, default=14)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=5e-5) 
    parser.add_argument('--do_preprocess', action='store_true')
    parser.add_argument('--trainable_layers', nargs='+', default=None)
    parser.add_argument('--finetune_schedule_file', type=str, default=None)
    args = parser.parse_args()
    main(args)