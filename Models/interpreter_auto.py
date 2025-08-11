#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import random
import argparse
import shutil
import sys
from typing import List, Tuple, Dict, Optional, Union

import matplotlib
matplotlib.use('Agg')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from PIL import Image
import cv2

from transformers import VivitImageProcessor, VivitForVideoClassification
from tqdm import tqdm

def set_seed(seed: int):
    """
    Sets the seed for all random number generators to ensure reproducibility.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)
    print(f"Seed set to {seed} for reproducibility.")

def get_file_name_and_parent_folder(file_path: str, game_name: str) -> Tuple[str, str, str, str]:
    file_path = os.path.normpath(file_path)
    file_paths = file_path.split(os.sep)
    file_name, _ = os.path.splitext(os.path.basename(file_path))
    parent_folder = os.path.dirname(file_path)
    video_name = file_paths[-2]
    player_id, session_id = video_name.split(f'_{game_name}_')
    return file_name, parent_folder, player_id, session_id


class MyCSVDataset(Dataset):
    """
    Custom PyTorch Dataset to load video frames based on a CSV file.
    """
    def __init__(self, csv_file: str, csv_file_2: str, base_path: str = "../Dataset/", game_name: str = 'solid'):
        self.data = pd.read_csv(csv_file)
        self.gf = pd.read_csv(csv_file_2, low_memory=False)
        self.base_path = base_path
        self.game_name = game_name
        self.image_processor = VivitImageProcessor.from_pretrained("google/vivit-b-16x2-kinetics400")
        
        self.class_labels =['down','same','up']
        self.label2id = {label: i for i, label in enumerate(self.class_labels)}

        columns_to_drop = [col for col in self.gf.columns if "control" in col and col not in ["[control]player_id", "[control]session_id"]]
        self.gf = self.gf.drop(columns=columns_to_drop)
        self.gf = self.gf.drop(columns=['[output]arousal'])

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Optional[Dict[str, torch.Tensor]]:
        sample = self.data.iloc[idx]
        clip_path = sample['start_frame']
        file_name_str, parent_folder, _, _ = get_file_name_and_parent_folder(clip_path, self.game_name)
        start_frame_num = int(file_name_str)

        frames = []
        for i in range(32):
            current_frame_name = f"{start_frame_num + i:04d}.png"
            frame_path = os.path.join(self.base_path, parent_folder, current_frame_name)
            
            try:
                frame = Image.open(frame_path).convert('RGB')
                frames.append(frame)
            except FileNotFoundError:
                # Return None if any frame is missing, will be skipped in the main loop
                return None

        inputs = self.image_processor(list(frames), return_tensors="pt")
        pixel_values = inputs['pixel_values'].squeeze(0)

        label = self.label2id[sample['arousal_change']]
        
        return {
            'pixel_values': pixel_values,
            'label': torch.tensor(label, dtype=torch.long)
        }

def prepare_input_from_sample(sample_data: Dict, device: torch.device) -> Dict[str, torch.Tensor]:
    """
    Prepares a preprocessed sample dictionary for the model.
    """
    pixel_values = sample_data['pixel_values'].unsqueeze(0)
    return {"pixel_values": pixel_values.to(device)}



# Model and Interpretation Logic
def load_model(model_path: Optional[str], model_ckpt_hub: str, id2label: Dict, label2id: Dict, device: torch.device) -> VivitForVideoClassification:
    """
    Loads the ViViT model.
    """
    model = None
    if model_path and os.path.exists(model_path):
        if os.path.isdir(model_path):
            print(f"Loading full model from local directory: {model_path}")
            model = VivitForVideoClassification.from_pretrained(
                model_path,
                label2id=label2id,
                id2label=id2label,
                ignore_mismatched_sizes=True
            )
        elif model_path.endswith(('.pt', '.pth')):
            print(f"Loading base model structure from Hugging Face: {model_ckpt_hub}")
            model = VivitForVideoClassification.from_pretrained(
                model_ckpt_hub,
                label2id=label2id,
                id2label=id2label,
                ignore_mismatched_sizes=True
            )
            print(f"Applying weights from local state dictionary: {model_path}")
            model.load_state_dict(torch.load(model_path, map_location=device))
        else:
            print(f"Warning: Provided model_path '{model_path}' is not a valid directory or .pt/.pth file.")

    if model is None:
        if model_path:
             print(f"Falling back to loading from Hugging Face hub: {model_ckpt_hub}")
        else:
             print(f"Loading model from Hugging Face hub: {model_ckpt_hub}")
        model = VivitForVideoClassification.from_pretrained(
            model_ckpt_hub,
            label2id=label2id,
            id2label=id2label,
            ignore_mismatched_sizes=True
        )
    
    model.to(device)
    model.config.output_attentions = True
    model.eval()
    return model

def calculate_new_relevance(attentions: List[torch.Tensor], grads: List[torch.Tensor], device: torch.device) -> torch.Tensor:
    num_tokens = attentions[0].shape[-1]
    relevance = torch.eye(num_tokens, device=device).unsqueeze(0)
    """
    Calculate relevance matrix, note the relevance here is not the same as defined in Chefer's paper and is replaced by raw attetion
    """

    for attn, grad in zip(reversed(attentions), grads):
        if attn.shape != grad.shape:
            print(f"Skipping layer due to shape mismatch: attn {attn.shape}, grad {grad.shape}")
            continue

        attn_heads_fused = attn.mean(dim=1)
        grad_heads_fused = grad.mean(dim=1)

        R_layer = attn_heads_fused * grad_heads_fused
        R_layer = torch.clamp(R_layer, min=0)

        R_layer_with_residual = R_layer + torch.eye(num_tokens, device=device).unsqueeze(0)
        R_layer_normalized = R_layer_with_residual / (R_layer_with_residual.sum(dim=-1, keepdim=True) + 1e-6)

        relevance = torch.matmul(relevance, R_layer_normalized)
    
    cls_relevance = relevance[:, 0, 1:]
    return cls_relevance

def generate_new_relevance_for_target(model, inputs, target_id, retain_graph=False):

    model.zero_grad()
    all_attentions, all_attn_grads = [], []
    hooks = []

    def save_attention_hook(module, input, output):
        attn_weights = output[1]
        all_attentions.append(attn_weights.detach())
        attn_weights.register_hook(lambda grad: all_attn_grads.append(grad))

    for layer in model.vivit.encoder.layer:
        hooks.append(layer.attention.attention.register_forward_hook(save_attention_hook))

    outputs = model(**inputs)
    logits = outputs.logits

    target_logit = logits[0, target_id]
    target_logit.backward(retain_graph=retain_graph)

    for h in hooks:
        h.remove()
        
    if not all_attentions or not all_attn_grads:
        raise RuntimeError("Attention or gradients were not captured.")

    relevance = calculate_new_relevance(all_attentions, all_attn_grads, model.device)
    return relevance


# Visualization
def generate_and_save_visualization(
    cls_relevance: torch.Tensor,
    original_frames_tensor: torch.Tensor,
    output_dir: str,
    game_name: str,
    sample_index: int,
    class_name: str
):
    # Create the main output directory for the game
    game_output_dir = os.path.join(output_dir, game_name)
    os.makedirs(game_output_dir, exist_ok=True)
    
    num_frames, channels, height, width = original_frames_tensor.shape
    num_temporal_tokens = num_frames // 2
    num_spatial_tokens = cls_relevance.shape[-1] // num_temporal_tokens
    grid_size = int(np.sqrt(num_spatial_tokens))
    
    cls_relevance_reshaped = cls_relevance.reshape(1, num_temporal_tokens, grid_size, grid_size)

    temporal_maps = cls_relevance_reshaped[0]
    frame_heatmaps = []
    for t in range(num_temporal_tokens):
        heatmap = temporal_maps[t].detach().cpu().numpy()
        heatmap_upsampled = cv2.resize(heatmap, (width, height), interpolation=cv2.INTER_LINEAR)
        frame_heatmaps.extend([heatmap_upsampled, heatmap_upsampled.copy()])

    all_original_frames, all_heatmaps, all_overlays = [], [], []
    for i in range(num_frames):
        # Prepare original frame
        frame_tensor = original_frames_tensor[i].cpu()
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        frame_tensor = torch.clamp(frame_tensor * std + mean, 0, 1)
        original_frame = (frame_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        all_original_frames.append(original_frame)

        # Prepare raw heatmap
        heatmap = frame_heatmaps[i]
        heatmap_norm = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap) + 1e-8)
        heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_norm), cv2.COLORMAP_JET)
        heatmap_color_rgb = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
        all_heatmaps.append(heatmap_color_rgb)

        # Prepare overlay
        overlay = cv2.addWeighted(original_frame, 0.6, heatmap_color_rgb, 0.4, 0)
        all_overlays.append(overlay)

    # Create a 6x16 grid for plotting
    rows, cols = 6, 16
    fig, axs = plt.subplots(rows, cols, figsize=(cols * 1.7, rows * 1.75))
    
    for i in range(num_frames):
        row_offset = (i // 16) * 3
        col = i % 16
        
        axs[row_offset, col].imshow(all_original_frames[i])
        axs[row_offset, col].axis('off')
        
        axs[row_offset + 1, col].imshow(all_heatmaps[i])
        axs[row_offset + 1, col].axis('off')

        axs[row_offset + 2, col].imshow(all_overlays[i])
        axs[row_offset + 2, col].axis('off')

    # Add descriptive labels for each row
    fig.text(0.01, 0.917, 'Originals\n(1-16)', ha='left', va='center', rotation='vertical', fontsize=18)
    fig.text(0.01, 0.750, 'Heatmaps\n(1-16)', ha='left', va='center', rotation='vertical', fontsize=18)
    fig.text(0.01, 0.583, 'Overlays\n(1-16)', ha='left', va='center', rotation='vertical', fontsize=18)
    fig.text(0.01, 0.417, 'Originals\n(17-32)', ha='left', va='center', rotation='vertical', fontsize=18)
    fig.text(0.01, 0.250, 'Heatmaps\n(17-32)', ha='left', va='center', rotation='vertical', fontsize=18)
    fig.text(0.01, 0.083, 'Overlays\n(17-32)', ha='left', va='center', rotation='vertical', fontsize=18)

    # Add a main title to the figure for clarity
    fig.suptitle(f"Interpretation for Class: '{class_name}' (Game: {game_name}, Sample: {sample_index})", fontsize=24, y=1.0)

    plt.tight_layout(rect=[0.03, 0, 1, 0.98])
    plt.subplots_adjust(wspace=0.01, hspace=0.01)
    
    # New descriptive filename
    save_path = os.path.join(
        game_output_dir,
        f"class_{class_name}_sample_{sample_index}.png"
    )
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Saved visualization for class '{class_name}' to: {save_path}")
    return save_path

# Main Execution
def main():
    parser = argparse.ArgumentParser(description="ViViT interpretation script to find one correct sample per class.")
    parser.add_argument('--game_name', type=str, default='solid', help='Name of the game for dataset loading.')
    parser.add_argument('--model_path', type=str, default='finetune_layer0_results/solid/checkpoints/checkpoint_epoch_10.pt', help='Path to a local model (.pt, .pth, or directory).')
    parser.add_argument('--base_path', type=str, default='../Dataset/', help='Base directory path for the dataset.')
    parser.add_argument('--output_dir', type=str, default='interpretation_samples/all_games_with_heatmap', help='Directory to save the output visualizations.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')
    args = parser.parse_args()

    # MODIFIED: Check if the model checkpoint file exists before proceeding.
    if not os.path.exists(args.model_path):
        print(f"Error: The specified model checkpoint file was not found: '{args.model_path}'")
        print("Please check the path and try again.")
        sys.exit(1) # Exit the script if the file is not found.

    set_seed(args.seed)

    # --- Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model_ckpt_hub = "google/vivit-b-16x2-kinetics400"
    class_labels = ['down', 'same', 'up']
    label2id = {label: i for i, label in enumerate(class_labels)}
    id2label = {i: label for label, i in label2id.items()}
    print(f"Target classes: {class_labels}")

    # --- Load Model and Data ---
    model = load_model(args.model_path, model_ckpt_hub, id2label, label2id, device)
    
    dataset = MyCSVDataset(
        csv_file=os.path.join(args.base_path, f'new_{args.game_name}.csv'),
        csv_file_2=os.path.join(args.base_path, "clean_data.csv"),
        base_path=args.base_path,
        game_name=args.game_name
    )
    
    # --- Find and Process One Correct Sample Per Class ---
    found_classes = set()
    num_classes = len(class_labels)

    print("\nStarting search for correctly predicted samples...")
    # MODIFIED: Search range now starts from index 10 to the end of the dataset.
    for sample_index in tqdm(range(10, len(dataset)), desc=f"Searching samples from index 10 for '{args.game_name}'"):
        if len(found_classes) == num_classes:
            print("\nAll classes have been found. Stopping search.")
            break

        sample_data = dataset[sample_index]
        if sample_data is None:
            continue # Skip if frames were missing for this sample
            
        ground_truth_id = sample_data['label'].item()
        ground_truth_name = id2label[ground_truth_id]

        # If we already found a sample for this class, skip to the next.
        if ground_truth_name in found_classes:
            continue

        # Prepare input for the model
        inputs = prepare_input_from_sample(sample_data, device)

        # Get model prediction
        with torch.no_grad():
            outputs = model(**inputs)
            predicted_class_id = outputs.logits.argmax(dim=-1).item()

        # Check if prediction is correct
        if predicted_class_id == ground_truth_id:
            print(f"\nFound correct sample for class '{ground_truth_name}' at index {sample_index}.")
            
            found_classes.add(ground_truth_name)
            original_frames_tensor = inputs['pixel_values'][0]

            # Generate explanation
            print(f"--- Generating interpretation for class '{ground_truth_name}' ---")
            relevance = generate_new_relevance_for_target(model, inputs, ground_truth_id, retain_graph=False)
            
            # Generate and save the visualization
            generate_and_save_visualization(
                relevance,
                original_frames_tensor,
                args.output_dir,
                args.game_name,
                sample_index,
                ground_truth_name
            )

    # --- Final Report ---
    print("\n--- Search Complete ---")
    if len(found_classes) == num_classes:
        print("Successfully found and generated explanations for all classes:")
        print(sorted(list(found_classes)))
    else:
        print("Warning: Could not find a correctly predicted sample for all classes after index 10.")
        print(f"Found: {sorted(list(found_classes))}")
        missing_classes = set(class_labels) - found_classes
        print(f"Missing: {sorted(list(missing_classes))}")

    print("\nScript finished successfully.")

if __name__ == '__main__':
    main()