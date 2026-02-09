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

# === [NEW] PEFT Imports ===
from peft import get_peft_model, LoraConfig, PeftModel

# Set Matplotlib backend
matplotlib.use('Agg')

# [Optimization] Enable TF32
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True 

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = False 
    torch.backends.cudnn.benchmark = True

def get_player_id_simple(file_path, game_name):
    file_path = os.path.normpath(file_path)
    file_paths = file_path.split(os.sep)
    try:
        video_name = file_paths[-2]
        parts = video_name.split(f'_{game_name}_')
        if len(parts) >= 1: return parts[0]
        return "unknown"
    except: return "unknown"

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

# === Analysis Functions (Entropy & t-SNE) ===
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

def perform_tsne_analysis(model, dataloader, df, device, output_dir, fold_idx):
    print(f"\n[t-SNE] Extracting features for Fold {fold_idx} Test Set...")
    model.eval()
    features_list, labels_list, player_ids_list = [], [], []
    
    with torch.no_grad():
        batch_idx = 0
        for batch in tqdm(dataloader, desc="Embeddings"):
            pixel_values = batch["pixel_values"].to(device, non_blocking=True)
            labels = batch["label"].to(device).squeeze(1)
            batch_size = pixel_values.size(0)
            
            with autocast(dtype=torch.bfloat16):
                outputs = model(pixel_values=pixel_values, output_hidden_states=True)
                # Handle potential PEFT wrapping
                if hasattr(outputs, "hidden_states"):
                    hidden_states = outputs.hidden_states
                elif hasattr(outputs, "base_model_output") and hasattr(outputs.base_model_output, "hidden_states"):
                    hidden_states = outputs.base_model_output.hidden_states
                else:
                    hidden_states = outputs.hidden_states
                
                cls_emb = hidden_states[-1][:, 0, :] 
            
            features_list.append(cls_emb.cpu().float().numpy())
            labels_list.append(labels.cpu().numpy())
            
            start_idx = batch_idx * dataloader.batch_size
            end_idx = start_idx + batch_size
            batch_player_ids = df.iloc[start_idx:end_idx]['player_id'].values
            player_ids_list.append(batch_player_ids)
            
            batch_idx += 1

    X = np.concatenate(features_list, axis=0)
    y_mood = np.concatenate(labels_list, axis=0)
    all_players = np.concatenate(player_ids_list, axis=0)
    
    if len(X) != len(all_players):
        print(f" Warning: Feature length ({len(X)}) != Player ID length ({len(all_players)})")
        print(f"   This may indicate a data loading issue. Truncating to minimum length.")
        min_len = min(len(X), len(all_players), len(y_mood))
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

def preprocess_dataset(df, root_dir, cache_dir, game_name):
    os.makedirs(cache_dir, exist_ok=True)
    print(f"\n[Pre-packing] Starting to pack images into {cache_dir}...")
    unique_samples = df.drop_duplicates(subset=['start_frame'])
    
    failed_samples = []
    for idx, row in tqdm(unique_samples.iterrows(), total=len(unique_samples), desc="Packing"):
        clip_path = row['start_frame']
        file_name_str = os.path.splitext(os.path.basename(clip_path))[0]
        parent_folder = os.path.dirname(clip_path)
        safe_name = f"{parent_folder}_{file_name_str}.pt".replace(os.sep, "_")
        save_path = os.path.join(cache_dir, safe_name)
        if os.path.exists(save_path): continue
        
        frames = []
        try: 
            file_name_int = int(file_name_str)
        except ValueError: 
            file_name_int = 0
            
        try:
            for i in range(32):
                frame_path = os.path.join(root_dir, parent_folder, f"{file_name_int + i:04d}.png")
                try:
                    with Image.open(frame_path) as img:
                        img = img.convert('RGB').resize((256, 256))
                        tensor_img = transforms.PILToTensor()(img) 
                        frames.append(tensor_img)
                except Exception as e:
                    frames.append(torch.zeros((3, 256, 256), dtype=torch.uint8))
            
            torch.save(torch.stack(frames), save_path)
        except Exception as e:
            print(f"Failed to process {clip_path}: {e}")
            failed_samples.append(clip_path)
    
    if failed_samples:
        print(f"Failed to process {len(failed_samples)} samples")
        with open(os.path.join(cache_dir, "failed_samples.txt"), 'w') as f:
            f.write('\n'.join(failed_samples))
    
    print("[Pre-packing] Done!\n")

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
        label_id = self.label2id.get(str(sample.get('arousal_change', 'same')), 1) 
        inputs['label'] = torch.LongTensor([label_id])
        return inputs

def evaluate(model, dataloader, device, case_name, epoch, label2id, id2label, criterion=None):
    model.eval()
    all_labels, all_preds, total_loss = [], [], 0
    if criterion is None:
        criterion = nn.CrossEntropyLoss()
        
    with torch.no_grad():
        for batch in dataloader:
            pixel_values = batch["pixel_values"].to(device, non_blocking=True)
            labels = batch["label"].to(device).squeeze(1)
            logits = model(pixel_values=pixel_values).logits
            loss = criterion(logits, labels)
            
            total_loss += loss.item()
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(logits.argmax(-1).cpu().numpy())
    
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

def main(args):
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    use_amp = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8  # Ampere+
    scaler = GradScaler(enabled=False)  
    
    ACCUM_STEPS = 1 
    CACHE_DIR = f"../Dataset/cache_{args.game_name}"
    
    # Labels
    class_labels = ['down', 'same', 'up']
    label2id = {l: i for i, l in enumerate(class_labels)}
    id2label = {i: l for l, i in label2id.items()}
    
    base_output_dir = os.path.join(args.output_dir, args.game_name)
    os.makedirs(base_output_dir, exist_ok=True)
    
    model_ckpt = "google/vivit-b-16x2-kinetics400"
    
    try:
        image_processor = VivitImageProcessor.from_pretrained(model_ckpt)
    except Exception as e:
        print(f" Error loading image processor: {e}")
        return

    # Load Data
    csv_file = f'../Dataset/new_{args.game_name}.csv'
    if not os.path.exists(csv_file):
        print(f" Error: CSV file not found: {csv_file}"); return
    
    print(f"Loaded Data: {csv_file}")
    try:
        all_data_df = pd.read_csv(csv_file)
    except Exception as e:
        print(f" Error loading CSV: {e}")
        return
    
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
    all_data_df['arousal_change'] = all_data_df['arousal_change'].astype(str)

    if args.do_preprocess: 
        preprocess_dataset(all_data_df, "../Dataset/", CACHE_DIR, args.game_name)
    
    analyze_user_label_entropy(all_data_df, base_output_dir, label2id)
    
    # Splitting
    BLOCK_DURATION = 6
    all_data_df['block_id'] = all_data_df.apply(lambda r: f"{r['player_id']}_{int(r['start_time'] // BLOCK_DURATION)}", axis=1)
    groups = all_data_df['block_id'].values
    y_all = all_data_df['arousal_change'].map(label2id).values
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    all_test_results = []
    
    # === [BASELINE CONF] LoRA Configuration ===
    print(f"[LoRA Setup] Rank: {args.lora_r}, Alpha: {args.lora_alpha}, LR: {args.learning_rate}")
    peft_config = LoraConfig(
        inference_mode=False,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["query", "value", "key", "dense"], # [BEST PRACTICE] All attention linear layers
        modules_to_save=["classifier"] # Fully train the new head
    )

    for fold, (train_val_idx, test_idx) in enumerate(sgkf.split(all_data_df, y_all, groups=groups)):
        fold_num = fold + 1
        print(f"\n" + "="*40 + f"\nFOLD {fold_num}/5 (Baseline: ViViT LoRA)\n" + "="*40)
        
        fold_dir = os.path.join(base_output_dir, f"fold_{fold_num}")
        model_dir = os.path.join(fold_dir, "checkpoints")
        os.makedirs(model_dir, exist_ok=True)
        
        try:
            test_df = all_data_df.iloc[test_idx].copy()
            train_val_df = all_data_df.iloc[train_val_idx]
            train_val_y = y_all[train_val_idx]
            train_val_groups = groups[train_val_idx]
            
            inner_sgkf = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=42)
            train_idx, val_idx = next(inner_sgkf.split(train_val_df, train_val_y, groups=train_val_groups))
            
            train_loader = DataLoader(MyCSVDataset(train_val_df.iloc[train_idx], args.game_name, image_processor, label2id, cache_dir=CACHE_DIR), batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
            val_loader = DataLoader(MyCSVDataset(train_val_df.iloc[val_idx], args.game_name, image_processor, label2id, cache_dir=CACHE_DIR), batch_size=4, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)
            test_loader = DataLoader(MyCSVDataset(test_df, args.game_name, image_processor, label2id, cache_dir=CACHE_DIR), batch_size=4, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)

            # 1. Load Base Model
            base_model = VivitForVideoClassification.from_pretrained(model_ckpt, label2id=label2id, id2label=id2label, ignore_mismatched_sizes=True)
            
            # 2. Wrap with LoRA
            model = get_peft_model(base_model, peft_config)
            model.to(device)
            model.print_trainable_parameters()

            optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)
            criterion = nn.CrossEntropyLoss()
            
            best_f1, best_path, best_epoch = -1.0, "", 0
            validation_logs, train_losses, val_losses, train_accs, val_accs = [], [], [], [], []

            for epoch in range(args.num_epochs):
                model.train()
                run_loss, train_correct, train_total = 0, 0, 0
                optimizer.zero_grad() 
                
                for i, batch in enumerate(train_loader):
                    pixel_values = batch["pixel_values"].to(device, non_blocking=True)
                    labels = batch["label"].to(device).squeeze(1)
                    
                    # [BF16] Force bfloat16 for training on A100
                    with autocast(dtype=torch.bfloat16):
                        outputs = model(pixel_values=pixel_values, labels=labels)
                        loss = outputs.loss / ACCUM_STEPS
                        train_correct += (outputs.logits.argmax(dim=-1) == labels).sum().item()
                        train_total += labels.size(0)
                    
                    scaler.scale(loss).backward()
                    if (i + 1) % ACCUM_STEPS == 0:
                        scaler.step(optimizer); scaler.update(); optimizer.zero_grad()
                    run_loss += loss.item() * ACCUM_STEPS
                
                if len(train_loader) % ACCUM_STEPS != 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

                avg_train_loss = run_loss / len(train_loader)
                avg_train_acc = train_correct / train_total if train_total > 0 else 0.0
                print(f"Epoch {epoch+1}: Loss {avg_train_loss:.4f} | Acc {avg_train_acc:.4f}")

                val_res = evaluate(model, val_loader, device, "Val", epoch+1, label2id, id2label, criterion)
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

            # === Test Phase: Load Base + Adapter ===
            print(f"Loading best LoRA model from {best_path}...")
            model.cpu()
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            base_model_test = VivitForVideoClassification.from_pretrained(model_ckpt, label2id=label2id, id2label=id2label, ignore_mismatched_sizes=True)
            model_test = PeftModel.from_pretrained(base_model_test, best_path)
            model_test.to(device)

            test_res = evaluate(model_test, test_loader, device, "Test", best_epoch, label2id, id2label)
            save_results_to_json(os.path.join(fold_dir, "test_results.json"), test_res)
            all_test_results.append(test_res)
            
            if fold_num == 1: 
                perform_tsne_analysis(model_test, test_loader, test_df, device, base_output_dir, fold_num)
            
            model_test.cpu()
            del model_test
            del base_model_test
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            gc.collect()
        
        except Exception as e:
            print(f" Error in Fold {fold_num}: {e}")
            import traceback
            traceback.print_exc()
            print(f"Skipping Fold {fold_num} and continuing...")
            continue

    if all_test_results:
        accs = [res['accuracy'] for res in all_test_results]
        f1s_w = [res['f1_weighted'] for res in all_test_results]
        mean_acc, _, ci_acc = compute_statistics(accs)
        mean_f1, _, ci_f1 = compute_statistics(f1s_w)
        
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

        print(f"\n{'='*50}")
        print(f"FINAL CV RESULT -> Acc: {mean_acc:.4f} (±{ci_acc:.4f}) | W-F1: {mean_f1:.4f} (±{ci_f1:.4f})")
        print(f"{'='*50}")
        
        agg_res = {
            "mean_accuracy": mean_acc, "ci95_accuracy": ci_acc,
            "mean_f1_weighted": mean_f1, "ci95_f1_weighted": ci_f1,
            "aggregated_confusion_matrix": agg_cm.tolist(),
            "aggregated_confusion_matrix_percent": np.round(agg_cm_pct, 2).tolist(),
            "avg_class_metrics": final_class_summary,
            "fold_results": all_test_results
        }
        save_results_to_json(os.path.join(base_output_dir, "aggregated_cv_results.json"), agg_res)
    else:
        print(" No successful folds completed!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--game_name', type=str, default="solid")
    parser.add_argument('--output_dir', type=str, default="Results/new/lora")
    parser.add_argument('--num_epochs', type=int, default=14)
    parser.add_argument('--batch_size', type=int, default=4)
    
    # [Baseline Config] Higher LR for LoRA is standard
    parser.add_argument('--learning_rate', type=float, default=5e-4) 
    parser.add_argument('--do_preprocess', action='store_true')
    
    # [Baseline Config] Golden Set for LoRA
    parser.add_argument('--lora_r', type=int, default=16, help="Rank (16 is stable for Vision)")
    parser.add_argument('--lora_alpha', type=int, default=32, help="Alpha (2x Rank)")
    parser.add_argument('--lora_dropout', type=float, default=0)
    
    args = parser.parse_args()
    main(args)