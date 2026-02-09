#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import random
import argparse
import sys
from typing import List, Tuple, Dict, Optional

import matplotlib
matplotlib.use('Agg') # 强制非交互后端，适配服务器环境

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from PIL import Image

from transformers import VivitImageProcessor, VivitForVideoClassification
from tqdm import tqdm
from sklearn.model_selection import StratifiedGroupKFold

# ================= 🛠️ 基础工具函数 =================

def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_player_id_simple(file_path: str, game_name: str) -> str:
    file_path = os.path.normpath(file_path)
    file_paths = file_path.split(os.sep)
    try:
        video_name = file_paths[-2]
        parts = video_name.split(f'_{game_name}_')
        if len(parts) >= 1: return parts[0]
        return "unknown"
    except: return "unknown"

# ================= 📂 数据集类 =================

class MyCSVDataset(Dataset):
    def __init__(self, data_df: pd.DataFrame, base_path: str, game_name: str, label2id: Dict[str, int]):
        self.data = data_df.reset_index(drop=True)
        self.base_path = base_path
        self.game_name = game_name
        self.label2id = label2id
        self.image_processor = VivitImageProcessor.from_pretrained("google/vivit-b-16x2-kinetics400")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Optional[Dict[str, torch.Tensor]]:
        sample = self.data.iloc[idx]
        clip_path = sample['start_frame']
        file_name_str = os.path.splitext(os.path.basename(clip_path))[0]
        parent_folder = os.path.dirname(clip_path)
        
        try: start_frame_num = int(file_name_str)
        except ValueError: start_frame_num = 0

        frames = []
        for i in range(32):
            current_frame_name = f"{start_frame_num + i:04d}.png"
            frame_path = os.path.join(self.base_path, parent_folder, current_frame_name)
            try:
                frame = Image.open(frame_path).convert('RGB')
                frames.append(frame)
            except FileNotFoundError:
                frames.append(Image.new('RGB', (224, 224), (0, 0, 0)))

        inputs = self.image_processor(list(frames), return_tensors="pt")
        pixel_values = inputs['pixel_values'].squeeze(0)
        label_str = str(sample['arousal_change'])
        label = self.label2id.get(label_str, 1) 
        
        return {
            'pixel_values': pixel_values,
            'label': torch.tensor(label, dtype=torch.long)
        }

def prepare_input(sample_data: Dict, device: torch.device) -> Dict[str, torch.Tensor]:
    pixel_values = sample_data['pixel_values'].unsqueeze(0)
    return {"pixel_values": pixel_values.to(device)}

# ================= 🧠 模型与解释逻辑 (LRP) =================

def load_model(model_path: str, model_ckpt_hub: str, id2label: Dict, label2id: Dict, device: torch.device) -> VivitForVideoClassification:
    print(f"Loading model from: {model_path}")
    model = VivitForVideoClassification.from_pretrained(
        model_ckpt_hub, label2id=label2id, id2label=id2label, ignore_mismatched_sizes=True
    )
    if model_path.endswith('.pt') or model_path.endswith('.pth'):
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
    elif os.path.isdir(model_path):
        model = VivitForVideoClassification.from_pretrained(model_path, label2id=label2id, id2label=id2label)
    
    model.to(device)
    model.eval()
    return model

def calculate_new_relevance(attentions, grads, device):
    num_tokens = attentions[0].shape[-1]
    relevance = torch.eye(num_tokens, device=device).unsqueeze(0)

    for attn, grad in zip(reversed(attentions), grads):
        attn_heads_fused = attn.mean(dim=1)
        grad_heads_fused = grad.mean(dim=1)
        R_layer = torch.clamp(attn_heads_fused * grad_heads_fused, min=0)
        R_layer_with_residual = R_layer + torch.eye(num_tokens, device=device).unsqueeze(0)
        R_layer_normalized = R_layer_with_residual / (R_layer_with_residual.sum(dim=-1, keepdim=True) + 1e-6)
        relevance = torch.matmul(relevance, R_layer_normalized)
    
    return relevance[:, 0, 1:]

def generate_relevance(model, inputs, target_id):
    model.zero_grad()
    attns, grads, hooks = [], [], []

    def hook_fn(module, input, output):
        attns.append(output[1].detach())
        output[1].register_hook(lambda g: grads.append(g))

    for layer in model.vivit.encoder.layer:
        hooks.append(layer.attention.attention.register_forward_hook(hook_fn))

    with torch.set_grad_enabled(True):
        logits = model(**inputs, output_attentions=True).logits
        logits[0, target_id].backward()

    for h in hooks: h.remove()
    return calculate_new_relevance(attns, grads, model.device)

# ================= 🎨 可视化绘图逻辑 =================

def process_frames_and_maps(cls_relevance, original_frames_tensor):
    """预处理：生成所有32帧的 Heatmap 和 Overlay"""
    num_frames, _, height, width = original_frames_tensor.shape
    num_temporal = num_frames // 2
    grid_size = int(np.sqrt(cls_relevance.shape[-1] // num_temporal))
    temporal_maps_16 = cls_relevance.reshape(1, num_temporal, grid_size, grid_size)[0]
    
    temporal_maps_32 = []
    for t in range(num_temporal):
        temporal_maps_32.append(temporal_maps_16[t])
        temporal_maps_32.append(temporal_maps_16[t])

    processed_data = []
    for i in range(num_frames):
        frame_tensor = original_frames_tensor[i].cpu()
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        frame_tensor = torch.clamp(frame_tensor * std + mean, 0, 1)
        orig_img = (frame_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

        heatmap = temporal_maps_32[i].detach().cpu().numpy()
        heatmap_upsampled = cv2.resize(heatmap, (width, height), interpolation=cv2.INTER_LINEAR)
        heatmap_norm = (heatmap_upsampled - np.min(heatmap_upsampled)) / (np.max(heatmap_upsampled) - np.min(heatmap_upsampled) + 1e-8)
        heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_norm), cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

        overlay = cv2.addWeighted(orig_img, 0.6, heatmap_rgb, 0.4, 0)
        
        processed_data.append({'orig': orig_img, 'heat': heatmap_rgb, 'over': overlay})
    return processed_data

def save_full_visualization(processed_data, save_path, class_name, game_name, sample_index):
    """完整版 (32帧, 3视图)"""
    rows, cols = 6, 16
    fig, axs = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.6))
    
    for i in range(32):
        if i < 16: col = i; base_row = 0
        else: col = i - 16; base_row = 3
            
        data = processed_data[i]
        axs[base_row, col].imshow(data['orig']); axs[base_row, col].axis('off')
        axs[base_row + 1, col].imshow(data['heat']); axs[base_row + 1, col].axis('off')
        axs[base_row + 2, col].imshow(data['over']); axs[base_row + 2, col].axis('off')

    fig.text(0.01, 0.92, 'Original (0-15)', va='center', rotation='vertical', fontsize=12)
    fig.text(0.01, 0.75, 'Heatmap (0-15)', va='center', rotation='vertical', fontsize=12)
    fig.text(0.01, 0.58, 'Overlay (0-15)', va='center', rotation='vertical', fontsize=12)
    fig.text(0.01, 0.42, 'Original (16-31)', va='center', rotation='vertical', fontsize=12)
    fig.text(0.01, 0.25, 'Heatmap (16-31)', va='center', rotation='vertical', fontsize=12)
    fig.text(0.01, 0.08, 'Overlay (16-31)', va='center', rotation='vertical', fontsize=12)

    fig.suptitle(f"FULL: {class_name} | {game_name} | Idx: {sample_index}", fontsize=16, y=0.99)
    plt.tight_layout(rect=[0.02, 0, 1, 0.98])
    plt.subplots_adjust(wspace=0.02, hspace=0.02)
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  [Saved] Full Vis: {save_path}")

def save_simplified_visualization(processed_data, save_path, class_name, game_name, sample_index):
    """精简版 (16帧, 2视图, 无Heatmap)"""
    sampled_indices = list(range(0, 32, 2))
    rows, cols = 4, 8
    fig, axs = plt.subplots(rows, cols, figsize=(cols * 2.0, rows * 2.0))
    
    for idx, frame_idx in enumerate(sampled_indices):
        if idx < 8: base_row = 0; col = idx
        else: base_row = 2; col = idx - 8
            
        data = processed_data[frame_idx]
        axs[base_row, col].imshow(data['orig']); axs[base_row, col].axis('off')
        axs[base_row + 1, col].imshow(data['over']); axs[base_row + 1, col].axis('off')

    fig.text(0.01, 0.88, 'Original (T:0-15)', va='center', rotation='vertical', fontsize=14)
    fig.text(0.01, 0.63, 'Overlay (T:0-15)', va='center', rotation='vertical', fontsize=14)
    fig.text(0.01, 0.38, 'Original (T:16-31)', va='center', rotation='vertical', fontsize=14)
    fig.text(0.01, 0.13, 'Overlay (T:16-31)', va='center', rotation='vertical', fontsize=14)

    fig.suptitle(f" {class_name} | {game_name} | Idx: {sample_index}", fontsize=16, y=0.98)
    plt.tight_layout(rect=[0.03, 0, 1, 0.96])
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  [Saved] Simple Vis: {save_path}")

# ================= 🚀 主程序 =================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--game_name', type=str, default='solid')
    parser.add_argument('--model_path', type=str, default='Results/new5/full/solid/fold_1/checkpoint_best')
    parser.add_argument('--dataset_dir', type=str, default='../Dataset/')
    parser.add_argument('--output_dir', type=str, default='interpretation_results/v3_dual_mode/full')
    parser.add_argument('--seed', type=int, default=42)
    # [新增参数] 指定样本 Index
    parser.add_argument('--sample_idx', type=int, default=-1, help='Specific sample index to visualize. Default -1 means auto-search.')
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        print(f"❌ Error: Model path not found: {args.model_path}"); return

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 准备数据
    class_labels = ['down', 'same', 'up']
    label2id = {l: i for i, l in enumerate(class_labels)}
    id2label = {i: l for l, i in label2id.items()}

    csv_file = os.path.join(args.dataset_dir, f'new_{args.game_name}.csv')
    if not os.path.exists(csv_file): print(f"❌ Error: CSV not found {csv_file}"); return
    
    all_data_df = pd.read_csv(csv_file)

    # 2. 复刻预处理 (Lag & Split)
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

    BLOCK_DURATION = 6
    all_data_df['block_id'] = all_data_df.apply(lambda r: f"{r['player_id']}_{int(r['start_time'] // BLOCK_DURATION)}", axis=1)
    groups = all_data_df['block_id'].values
    y_all = all_data_df['arousal_change'].map(label2id).values

    print("Splitting data (Fold 1)...")
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    splits = list(sgkf.split(all_data_df, y_all, groups=groups))
    test_df = all_data_df.iloc[splits[0][1]].copy() # Fold 1 Test Set
    
    dataset = MyCSVDataset(test_df, args.dataset_dir, args.game_name, label2id)
    print(f"Test Set Size: {len(dataset)}")

    # 3. 加载模型
    model_ckpt_hub = "google/vivit-b-16x2-kinetics400"
    model = load_model(args.model_path, model_ckpt_hub, id2label, label2id, device)

    game_out_dir = os.path.join(args.output_dir, args.game_name)
    os.makedirs(game_out_dir, exist_ok=True)

    # ================= 4. 处理逻辑 (分支) =================
    
    # 模式 A: 指定样本 Index
    if args.sample_idx >= 0:
        idx = args.sample_idx
        if idx >= len(dataset):
            print(f"❌ Error: Index {idx} out of range (0 ~ {len(dataset)-1})"); return
            
        print(f"\n--- Processing Single Sample Mode (Index: {idx}) ---")
        try: 
            sample = dataset[idx]
        except Exception as e:
            print(f"❌ Error loading sample {idx}: {e}"); return

        gt_id = sample['label'].item()
        gt_name = id2label[gt_id]
        
        inputs = prepare_input(sample, device)
        with torch.no_grad():
            logits = model(**inputs).logits
            pred_id = logits.argmax(-1).item()
            pred_name = id2label[pred_id]
            
        # 打印预测结果
        status = "✅ Correct" if pred_id == gt_id else "❌ Wrong"
        print(f"Sample {idx}: GT='{gt_name}', Pred='{pred_name}' -> {status}")
        
        # 无论对错，都针对【真实标签】生成解释 (看看模型有没有关注到真实类别该关注的地方)
        print(f"Generating interpretation for class: '{gt_name}'...")
        relevance = generate_relevance(model, inputs, gt_id)
        processed_data = process_frames_and_maps(relevance, inputs['pixel_values'][0])
        
        # 保存文件名带上 "_specific" 标记
        full_name = os.path.join(game_out_dir, f"full_{gt_name}_{idx}_specific.png")
        simple_name = os.path.join(game_out_dir, f"simple_{gt_name}_{idx}_specific.png")
        
        save_full_visualization(processed_data, full_name, gt_name, args.game_name, idx)
        save_simplified_visualization(processed_data, simple_name, gt_name, args.game_name, idx)

    # 模式 B: 自动搜索 (默认)
    else:
        print("\n--- Auto-Search Mode (Find 1 correct sample per class) ---")
        found_classes = set()
        print("Starting search (from idx 10)...")
        for idx in tqdm(range(10, len(dataset))):
            if len(found_classes) == len(class_labels): break
            
            try: sample = dataset[idx]
            except: continue
            
            gt_id = sample['label'].item()
            gt_name = id2label[gt_id]
            if gt_name in found_classes: continue

            inputs = prepare_input(sample, device)
            with torch.no_grad():
                pred_id = model(**inputs).logits.argmax(-1).item()

            if pred_id == gt_id:
                print(f"\n✅ Found correct '{gt_name}' at idx {idx}")
                found_classes.add(gt_name)
                
                relevance = generate_relevance(model, inputs, gt_id)
                processed_data = process_frames_and_maps(relevance, inputs['pixel_values'][0])
                
                full_name = os.path.join(game_out_dir, f"full_{gt_name}_{idx}.png")
                simple_name = os.path.join(game_out_dir, f"simple_{gt_name}_{idx}.png")
                
                save_full_visualization(processed_data, full_name, gt_name, args.game_name, idx)
                save_simplified_visualization(processed_data, simple_name, gt_name, args.game_name, idx)

    print("\nDone! Check output directory.")

if __name__ == '__main__':
    main()