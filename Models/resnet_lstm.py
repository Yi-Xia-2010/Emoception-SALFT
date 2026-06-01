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
from torch.optim import AdamW
from PIL import Image
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
import torchvision.models as models
from torchvision import transforms
import torchvision.transforms.functional as F
import torch.nn.functional as F_torch
import matplotlib.pyplot as plt
import matplotlib
from torch.cuda.amp import autocast, GradScaler

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
        if len(parts) >= 1:
            return parts[0]
        return "unknown"
    except:
        return "unknown"

def compute_statistics(data):
    n = len(data)
    if n < 2:
        return np.mean(data), 0.0, 0.0
    mean_val = np.mean(data)
    std_val = np.std(data, ddof=1)
    se = std_val / np.sqrt(n)
    h = se * stats.t.ppf((1 + 0.95) / 2., n-1)
    return mean_val, std_val, h

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

def save_results_to_json(filepath, data):
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False, cls=NpEncoder)

def preprocess_dataset(df, root_dir, cache_dir, game_name):
    os.makedirs(cache_dir, exist_ok=True)
    print(f"\n[Pre-packing] Starting to pack images into {cache_dir}...")
    unique_samples = df.drop_duplicates(subset=['start_frame'])
    
    error_count = 0
    for idx, row in tqdm(unique_samples.iterrows(), total=len(unique_samples), desc="Packing"):
        clip_path = row['start_frame']
        file_name_str = os.path.splitext(os.path.basename(clip_path))[0]
        parent_folder = os.path.dirname(clip_path)
        
        safe_name = f"{parent_folder}_{file_name_str}.pt".replace(os.sep, "_")
        save_path = os.path.join(cache_dir, safe_name)
        
        if os.path.exists(save_path):
            continue

        frames = []
        try:
            file_name_int = int(file_name_str)
        except ValueError:
            file_name_int = 0

        for i in range(32):
            frame_path = os.path.join(root_dir, parent_folder, f"{file_name_int + i:04d}.png")
            try:
                with Image.open(frame_path) as img:
                    img = img.convert('RGB').resize((256, 256))
                    tensor_img = transforms.ToTensor()(img)
                    frames.append(tensor_img)
            except Exception as e:
                if error_count == 0:
                    print(f" Error reading image: {frame_path} -> {e}")
                error_count += 1
                frames.append(torch.zeros((3, 256, 256), dtype=torch.float32))
        
        torch.save(torch.stack(frames), save_path)
        
    if error_count > 0:
        print(f"\n WARNING: {error_count} frames failed to load (replaced with black images).")
        print("Please check your '../Dataset/' path and image filenames.\n")
    else:
        print("[Pre-packing] Done successfully!\n")

class ResNetLSTM(nn.Module):
    def __init__(self, num_classes, backbone_name='resnet50', lstm_hidden_size=256, lstm_layers=1):
        super(ResNetLSTM, self).__init__()
        
        if backbone_name == 'resnet18':
            weights = models.ResNet18_Weights.DEFAULT
            resnet = models.resnet18(weights=weights)
            input_features = 512
        elif backbone_name == 'resnet34':
            weights = models.ResNet34_Weights.DEFAULT
            resnet = models.resnet34(weights=weights)
            input_features = 512
        elif backbone_name == 'resnet50':
            weights = models.ResNet50_Weights.DEFAULT
            resnet = models.resnet50(weights=weights)
            input_features = 2048
        elif backbone_name == 'resnet101':
            weights = models.ResNet101_Weights.DEFAULT
            resnet = models.resnet101(weights=weights)
            input_features = 2048
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")
        
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        
        self.lstm = nn.LSTM(
            input_size=input_features, 
            hidden_size=lstm_hidden_size, 
            num_layers=lstm_layers, 
            batch_first=True
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        batch_size, num_frames, C, H, W = x.shape
        c_in = x.view(batch_size * num_frames, C, H, W)
        features = self.backbone(c_in) 
        features = features.view(batch_size, num_frames, -1) 
        self.lstm.flatten_parameters() 
        _, (hn, _) = self.lstm(features)
        return self.classifier(hn[-1])

class ResNetDataset(Dataset):
    def __init__(self, data_df, game_name, label2id, root_dir="../Dataset/", cache_dir=None, is_train=False, enable_aug=False):
        self.data = data_df.reset_index(drop=True)
        self.game_name = game_name
        self.label2id = label2id
        self.root_dir = root_dir
        self.cache_dir = cache_dir
        self.is_train = is_train
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.to_pil = transforms.ToPILImage()
        self.basic_transform = transforms.Resize((224, 224))

    def __len__(self):
        return len(self.data)

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
            try: 
                file_name_int = int(file_name_str)
            except ValueError: 
                file_name_int = 0
            
            for i in range(32):
                frame_path = os.path.join(self.root_dir, parent_folder, f"{file_name_int + i:04d}.png")
                try:
                    frames.append(Image.open(frame_path).convert('RGB').resize((256, 256)))
                except:
                    frames.append(Image.new('RGB', (256, 256), (0, 0, 0)))

        frames = [self.basic_transform(img) for img in frames]
        frame_tensors = []
        for img in frames:
            t = transforms.ToTensor()(img)
            t = self.normalize(t)
            frame_tensors.append(t)
        
        pixel_values = torch.stack(frame_tensors)
        label_id = self.label2id.get(str(sample.get('arousal_change', 'same')), 1)
        return {"pixel_values": pixel_values, "label": torch.LongTensor([label_id])}

def evaluate(model, dataloader, device, case_name, epoch, label2id, id2label, criterion=None, use_amp=False):
    model.eval()
    all_labels = []
    all_preds = []
    total_loss = 0.0

    with torch.no_grad():
        for batch in dataloader:
            pixel_values = batch["pixel_values"].to(device, non_blocking=True)
            labels = batch["label"].to(device).squeeze(1)
            
            # 评估时统一使用FP32
            logits = model(pixel_values)
            if criterion:
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
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    class_metrics = {}
    labels_present = sorted(list(set(all_labels) | set(all_preds)))
    p_class = precision_score(all_labels, all_preds, average=None, labels=labels_present, zero_division=0)
    r_class = recall_score(all_labels, all_preds, average=None, labels=labels_present, zero_division=0)
    f_class = f1_score(all_labels, all_preds, average=None, labels=labels_present, zero_division=0)

    for i, label_id in enumerate(labels_present):
        if int(label_id) in id2label:
            c_name = id2label[int(label_id)]
            class_metrics[c_name] = {
                "precision": p_class[i],
                "recall": r_class[i],
                "f1_score": f_class[i]
            }

    return {
        "epoch": epoch,
        "loss": avg_loss,
        "accuracy": accuracy,
        "f1_weighted": f1_w,
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_percent": np.round(cm_percent, 2).tolist(),
        "class_metrics": class_metrics
    }

def plot_curves(train_losses, val_losses, train_accs, val_accs, output_path):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_losses, 'b-', label="Train Loss")
    if val_losses:
        plt.plot(epochs, val_losses, 'r-', label="Val Loss")
    plt.title("Loss Curve"); plt.legend(); plt.grid(True)
    plt.savefig(os.path.join(output_path, 'loss_curve.png')); plt.close()
    
    if train_accs:
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, train_accs, 'b--', label="Train Acc")
        if val_accs:
            plt.plot(epochs, val_accs, 'r--', label="Val Acc")
        plt.title("Accuracy Curve"); plt.legend(); plt.grid(True)
        plt.savefig(os.path.join(output_path, 'acc_curve.png')); plt.close()

def main(args):
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    ACCUM_STEPS = 1 
    CACHE_DIR = f"../Dataset/cache_{args.game_name}"
    
    base_output_dir = os.path.join(args.output_dir, args.game_name)
    os.makedirs(base_output_dir, exist_ok=True)

    if args.load_from:
        source_dir = os.path.join(args.load_from, args.game_name)
        print(f" Loading skipped folds from: {source_dir}")
    else:
        source_dir = base_output_dir 

    print(f" Saving new results to: {base_output_dir}")

    # 1. Load Data
    csv_file = f'../Dataset/new_{args.game_name}.csv'
    if not os.path.exists(csv_file):
        print(f" Error: CSV file not found: {csv_file}")
        return
    all_data_df = pd.read_csv(csv_file)
    
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

    BLOCK_DURATION = 6
    all_data_df['block_id'] = all_data_df.apply(lambda r: f"{r['player_id']}_{int(r['start_time'] // BLOCK_DURATION)}", axis=1)
    groups = all_data_df['block_id'].values
    
    class_labels = ['down', 'same', 'up']
    label2id = {l: i for i, l in enumerate(class_labels)}
    id2label = {i: l for l, i in label2id.items()}
    y_all = all_data_df['arousal_change'].map(label2id).values

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    all_test_results = []
    
    finetune_schedule = None
    if args.finetune_schedule_file:
        with open(args.finetune_schedule_file, 'r') as f:
            finetune_schedule = json.load(f)

    for fold, (train_val_idx, test_idx) in enumerate(sgkf.split(all_data_df, y_all, groups=groups)):
        fold_num = fold + 1

        is_target_fold = (args.target_folds is not None) and (fold_num in args.target_folds)
        
        if (args.target_folds is not None) and (not is_target_fold):
            fold_res_path = os.path.join(source_dir, f"fold_{fold_num}", "test_results.json")
            
            if os.path.exists(fold_res_path):
                print(f"Skipping Fold {fold_num} (Loading existing result from source)...")
                with open(fold_res_path, 'r') as f:
                    all_test_results.append(json.load(f))
            else:
                print(f"Skipping Fold {fold_num} (WARNING: No existing result found in source: {source_dir}).")
            continue

        print(f"\n" + "="*40 + f"\nFOLD {fold_num}/5 (ResNet+LSTM)\n" + "="*40)
        
        use_amp = not is_target_fold  # 非目标fold使用BF16，目标fold使用FP32
        
        if is_target_fold:
            print(f" APPLYING FIXES FOR FOLD {fold_num}: AMP Disabled (using FP32)")
        else:
            print(f" Using BF16 AMP for FOLD {fold_num}")

        fold_dir = os.path.join(base_output_dir, f"fold_{fold_num}")
        model_dir = os.path.join(fold_dir, "checkpoints")
        os.makedirs(model_dir, exist_ok=True)
        
        test_df = all_data_df.iloc[test_idx].copy()
        train_val_df = all_data_df.iloc[train_val_idx]
        train_val_y = y_all[train_val_idx]
        train_val_groups = groups[train_val_idx]
        
        inner_sgkf = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=42)
        train_idx, val_idx = next(inner_sgkf.split(train_val_df, train_val_y, groups=train_val_groups))
        
        train_dataset = ResNetDataset(train_val_df.iloc[train_idx], args.game_name, label2id, cache_dir=CACHE_DIR, is_train=True, enable_aug=False)
        val_dataset   = ResNetDataset(train_val_df.iloc[val_idx], args.game_name, label2id, cache_dir=CACHE_DIR, is_train=False, enable_aug=False)
        test_dataset  = ResNetDataset(test_df, args.game_name, label2id, cache_dir=CACHE_DIR, is_train=False, enable_aug=False)

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
        val_loader   = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)
        test_loader  = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)

        model = ResNetLSTM(num_classes=len(class_labels), backbone_name=args.backbone).to(device)

        # Freezing Logic
        layer_alias_mapping = {
            'stem': ['backbone.0', 'backbone.1'], 
            'layer1': ['backbone.4'], 
            'layer2': ['backbone.5'], 
            'layer3': ['backbone.6'], 
            'layer4': ['backbone.7'], 
            'lstm': ['lstm'], 
            'classifier': ['classifier']
        }
        layers_to_unfreeze = []
        should_freeze = False
        
        if finetune_schedule:
            layers_to_unfreeze = finetune_schedule.get(f"fold_{fold_num}", [])
            should_freeze = True
            print(f"  [Freezing] Schedule: {layers_to_unfreeze}")
        elif args.trainable_layers:
            layers_to_unfreeze = args.trainable_layers
            should_freeze = True
            print(f"  [Freezing] Args: {layers_to_unfreeze}")
        
        if should_freeze:
            if not any(k in layers_to_unfreeze for k in ['lstm', 'classifier']):
                layers_to_unfreeze.extend(['lstm', 'classifier'])
            
            actual_keywords = []
            for item in layers_to_unfreeze:
                actual_keywords.extend(layer_alias_mapping.get(item, [item]))
            
            for param in model.parameters():
                param.requires_grad = False
            
            c = 0
            for name, param in model.named_parameters():
                for kw in actual_keywords:
                    if kw in name:
                        param.requires_grad = True
                        c += 1
                        break
            print(f"  [Freezing] Unfrozen Params: {c}")
        else:
            print("  [Freezing] Full Finetuning")

        optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)
        
        # [FIXED] BF16不需要GradScaler
        scaler = None
        if use_amp:
            print("  [AMP] Using BF16 without GradScaler")
        
        criterion = nn.CrossEntropyLoss()
        
        best_f1 = -1.0
        best_path = ""
        best_epoch = 0
        validation_logs, train_losses, val_losses, train_accs, val_accs = [], [], [], [], []

        for epoch in range(args.num_epochs):
            model.train()
            # [REMOVED] BN冻结逻辑已移除，保持原有行为
            
            run_loss = 0
            train_correct = 0
            train_total = 0
            
            # [FIXED] 移除训练循环外的zero_grad
            for i, batch in enumerate(train_loader):
                pixel_values = batch["pixel_values"].to(device, non_blocking=True)
                labels = batch["label"].to(device).squeeze(1)
                
                if use_amp:
                    # [FIXED] BF16训练：直接用autocast，不用scaler
                    with autocast(dtype=torch.bfloat16):
                        logits = model(pixel_values)
                        loss = criterion(logits, labels) / ACCUM_STEPS
                    
                    loss.backward()
                    
                    if (i + 1) % ACCUM_STEPS == 0:
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)
                else:
                    # FP32训练
                    logits = model(pixel_values)
                    loss = criterion(logits, labels) / ACCUM_STEPS
                    loss.backward()
                    
                    if (i + 1) % ACCUM_STEPS == 0:
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)
                
                run_loss += loss.item() * ACCUM_STEPS
                train_correct += (logits.argmax(-1) == labels).sum().item()
                train_total += labels.size(0)

            avg_train_loss = run_loss / len(train_loader)
            avg_train_acc = train_correct / train_total if train_total > 0 else 0
            print(f"Epoch {epoch+1}: Loss {avg_train_loss:.4f} | Acc {avg_train_acc:.4f}")

            # [FIXED] 验证时使用FP32
            val_res = evaluate(model, val_loader, device, "Val", epoch+1, label2id, id2label, criterion, use_amp=False)
            val_res['train_loss'] = avg_train_loss
            validation_logs.append(val_res)
            
            train_losses.append(avg_train_loss)
            train_accs.append(avg_train_acc)
            val_losses.append(val_res['loss'])
            val_accs.append(val_res['accuracy'])
            
            ckpt_path = os.path.join(model_dir, f"checkpoint_epoch_{epoch+1}.pt")
            torch.save(model.state_dict(), ckpt_path)

            if val_res['f1_weighted'] > best_f1:
                best_f1 = val_res['f1_weighted']
                best_path = os.path.join(fold_dir, "checkpoint_best.pt")
                best_epoch = epoch + 1
                torch.save(model.state_dict(), best_path)
            
            save_results_to_json(os.path.join(fold_dir, "validation_logs.json"), validation_logs)
        
        plot_curves(train_losses, val_losses, train_accs, val_accs, fold_dir)

        if best_path:
            model.load_state_dict(torch.load(best_path))
            # [FIXED] 测试时使用FP32
            test_res = evaluate(model, test_loader, device, "Test", best_epoch, label2id, id2label, criterion, use_amp=False)
            test_res["best_epoch"] = best_epoch
            save_results_to_json(os.path.join(fold_dir, "test_results.json"), test_res)
            all_test_results.append(test_res)
        
        del model; torch.cuda.empty_cache(); gc.collect()

    # Final Aggregation
    if all_test_results:
        accs = [r['accuracy'] for r in all_test_results]
        f1s = [r['f1_weighted'] for r in all_test_results]
        mean_acc, _, ci_acc = compute_statistics(accs)
        mean_f1, _, ci_f1 = compute_statistics(f1s)
        
        agg_res = {
            "mean_accuracy": mean_acc, 
            "ci95_accuracy": ci_acc, 
            "mean_f1_weighted": mean_f1, 
            "ci95_f1_weighted": ci_f1, 
            "fold_results": all_test_results
        }
        save_results_to_json(os.path.join(base_output_dir, "aggregated_cv_results.json"), agg_res)
        print(f"\nFINAL CV: Acc {mean_acc:.4f} | F1 {mean_f1:.4f} | (Aggregated {len(all_test_results)} folds)")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--game_name', type=str, default="solid")
    parser.add_argument('--output_dir', type=str, default="Results/new/resnet_full")
    parser.add_argument('--backbone', type=str, default='resnet50')
    parser.add_argument('--num_epochs', type=int, default=14)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--do_preprocess', action='store_true')
    parser.add_argument('--finetune_schedule_file', type=str, default=None)
    parser.add_argument('--trainable_layers', nargs='+', default=None)
    parser.add_argument('--target_folds', nargs='+', type=int, default=None, 
                        help='Specific folds to re-run with fixes (e.g., 3 5).')
    parser.add_argument('--load_from', type=str, default=None, 
                        help='Directory to load existing results from if fold is skipped.')
    
    args = parser.parse_args()
    main(args)
