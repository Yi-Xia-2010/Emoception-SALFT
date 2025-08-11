#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script performs Layer-wise Relevance Propagation (LRP) on a fine-tuned
Video Vision Transformer (ViViT) model to interpret its predictions.
"""
import os
import random
import argparse
from typing import List, Tuple, Dict, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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
    print(f"Seed set to {seed} for reproducibility.")

def get_file_name_and_parent_folder(file_path: str, game_name: str) -> Tuple[str, str, str, str]:
    """
    Extracts metadata from a given file path.
    """
    file_path = os.path.normpath(file_path)
    file_paths = file_path.split(os.sep)
    file_name, _ = os.path.splitext(os.path.basename(file_path))
    parent_folder = os.path.dirname(file_path)
    video_name = file_paths[-2]
    player_id, session_id = video_name.split(f'_{game_name}_')
    return file_name, parent_folder, player_id, session_id


# Data Loading
class MyCSVDataset(Dataset):
    """
    Custom PyTorch Dataset to load video frames based on a CSV file.
    """
    def __init__(self, csv_file: str, csv_file_2: str, base_path: str = "../Dataset/", game_name: str = 'solid'):
        self.data = pd.read_csv(csv_file)
        self.gf = pd.read_csv(csv_file_2)
        self.base_path = base_path
        self.game_name = game_name
        self.image_processor = VivitImageProcessor.from_pretrained("google/vivit-b-16x2-kinetics400")
        
        # Define class labels
        self.class_labels =['down','same','up']
        self.label2id = {label: i for i, label in enumerate(self.class_labels)}

        # Clean up game feature dataframe
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
            # Use os.path.join for robust path construction
            frame_path = os.path.join(self.base_path, parent_folder, current_frame_name)
            
            try:
                frame = Image.open(frame_path).convert('RGB')
                frames.append(frame)
            except FileNotFoundError:
                print(f"Warning: Frame not found at {frame_path}. Skipping sample {idx}.")
                return None

        # Process video frames with the image processor
        inputs = self.image_processor(list(frames), return_tensors="pt")
        pixel_values = inputs['pixel_values'].squeeze(0)  # Remove batch dim added by processor

        # Process label
        label = self.label2id[sample['arousal_change']]
        
        return {
            'pixel_values': pixel_values,
            'label': torch.tensor(label, dtype=torch.long)
        }

# Gets a preprocessed sample from the dataset and prepares it for the model.
def prepare_input_from_dataset(dataset: MyCSVDataset, index: int, device: torch.device) -> Dict[str, torch.Tensor]:
    sample_data = dataset[index]
    if sample_data is None:
        raise RuntimeError(f"Could not load sample at index {index}. Check file paths.")
    
    # Add the batch dimension required by the model and move to the correct device
    pixel_values = sample_data['pixel_values'].unsqueeze(0)
    return {"pixel_values": pixel_values.to(device)}


# Model and Interpretation Logic
def load_model(model_ckpt_hub: str, local_model_path: Optional[str], id2label: Dict, label2id: Dict, device: torch.device) -> VivitForVideoClassification:
    # Check if a valid local model path is provided
    use_local_checkpoint = local_model_path and os.path.exists(local_model_path) and local_model_path.endswith(('.pt', '.pth'))

    print(f"Loading base model structure from Hugging Face: {model_ckpt_hub}")
    model = VivitForVideoClassification.from_pretrained(
        model_ckpt_hub,
        label2id=label2id,
        id2label=id2label,
        ignore_mismatched_sizes=True # Necessary for custom classification heads
    )

    if use_local_checkpoint:
        print(f"Loading weights from local checkpoint: {local_model_path}")
        model.load_state_dict(torch.load(local_model_path, map_location=device))
    else:
        print(f"Local model not found or path is incorrect. Using pre-trained weights from Hugging Face.")
    
    model.to(device)
    model.config.output_attentions = True # Ensure attention outputs are enabled
    model.eval() # Set model to evaluation mode
    return model

# Calculates end-to-end relevance using the LRP method based on attention and gradients.
def calculate_lrp_relevance(attentions: List[torch.Tensor], grads: List[torch.Tensor], device: torch.device) -> torch.Tensor:

    num_tokens = attentions[0].shape[-1]
    relevance = torch.eye(num_tokens, device=device).unsqueeze(0)

    # Iterate through layers from last to first
    for attn, grad in zip(reversed(attentions), reversed(grads)):
        if attn.shape != grad.shape:
            print(f"Skipping layer due to shape mismatch: attn {attn.shape}, grad {grad.shape}")
            continue

        # Fuse attention heads by averaging
        attn_heads_fused = attn.mean(dim=1)
        grad_heads_fused = grad.mean(dim=1)

        # LRP rule: relevance is proportional to (gradient * attention)
        R_layer = attn_heads_fused * grad_heads_fused
        R_layer = torch.clamp(R_layer, min=0) # Keep only positive contributions

        # Add identity for residual connection and normalize
        R_layer_with_residual = R_layer + torch.eye(num_tokens, device=device).unsqueeze(0)
        R_layer_normalized = R_layer_with_residual / (R_layer_with_residual.sum(dim=-1, keepdim=True) + 1e-6)

        # Propagate relevance
        relevance = torch.matmul(relevance, R_layer_normalized)
    
    # Extract relevance of all tokens w.r.t the [CLS] token, and discard self-relevance
    cls_relevance = relevance[:, 0, 1:]
    return cls_relevance


# Visualization
def generate_and_save_visualization(
    cls_relevance: torch.Tensor,
    original_frames_tensor: torch.Tensor,
    output_dir: str,
    game_name: str,
    sample_index: int,
    predicted_class_name: str
):
    # Create the output directory
    viz_output_dir = os.path.join(output_dir, f"{game_name}_lrp_sample{sample_index}_cls_{predicted_class_name}")
    os.makedirs(viz_output_dir, exist_ok=True)
    
    # --- Reshape Spatiotemporal Relevance ---
    num_frames, channels, height, width = original_frames_tensor.shape
    # For vivit-b-16x2, temporal tubelet size is 2, patch size is 16
    num_temporal_tokens = num_frames // 2
    num_spatial_tokens = cls_relevance.shape[-1] // num_temporal_tokens
    grid_size = int(np.sqrt(num_spatial_tokens))

    print(f"Reshaping relevance: Temporal tokens: {num_temporal_tokens}, Spatial tokens/frame: {num_spatial_tokens}, Grid size: {grid_size}x{grid_size}")
    
    cls_relevance_reshaped = cls_relevance.reshape(1, num_temporal_tokens, grid_size, grid_size)

    # --- Generate Per-Frame Heatmaps ---
    temporal_maps = cls_relevance_reshaped[0]
    frame_heatmaps = []
    for t in range(num_temporal_tokens):
        heatmap = temporal_maps[t].detach().cpu().numpy()
        heatmap_upsampled = cv2.resize(heatmap, (width, height), interpolation=cv2.INTER_LINEAR)
        # Each temporal token's heatmap applies to 2 consecutive frames
        frame_heatmaps.extend([heatmap_upsampled, heatmap_upsampled.copy()])

    # --- Prepare All Visualization Images in Memory ---
    all_original_frames, all_heatmaps_norm, all_overlays = [], [], []

    for i in range(num_frames):
        # Denormalize original frame for viewing
        frame_tensor = original_frames_tensor[i].cpu()
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        frame_tensor = torch.clamp(frame_tensor * std + mean, 0, 1)
        original_frame = (frame_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        all_original_frames.append(original_frame)

        # Normalize heatmap and apply colormap
        heatmap = frame_heatmaps[i]
        heatmap_norm = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap) + 1e-8)
        all_heatmaps_norm.append(heatmap_norm)
        
        heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_norm), cv2.COLORMAP_JET)
        heatmap_color_rgb = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

        # Create overlay
        overlay = cv2.addWeighted(original_frame, 0.6, heatmap_color_rgb, 0.4, 0)
        all_overlays.append(overlay)

    # --- Create and Save the Consolidated Plot ---
    rows, cols = 6, 16
    fig, axs = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2.2))
    
    # Plot all the images
    for i in range(num_frames):
        row_offset = (i // 16) * 3
        col = i % 16
        
        axs[row_offset, col].imshow(all_original_frames[i])
        axs[row_offset, col].axis('off')
        
        axs[row_offset + 1, col].imshow(all_heatmaps_norm[i], cmap='jet')
        axs[row_offset + 1, col].axis('off')
        
        axs[row_offset + 2, col].imshow(all_overlays[i])
        axs[row_offset + 2, col].axis('off')

    # Add a main title to the entire figure.
    fig.suptitle(f"LRP Visualization for Sample {sample_index} (Predicted: {predicted_class_name})", fontsize=24)

    # Adjust subplot parameters for a tight layout.
    # `rect` leaves space: [left, bottom, right, top]. We leave space on left for row labels and top for suptitle.
    fig.tight_layout(rect=[0.04, 0, 1, 0.95])
    
    # Fine-tune the spacing between subplots, which is the correct way to use wspace/hspace
    fig.subplots_adjust(wspace=0.05, hspace=0.05)

    # Add row labels with dynamically calculated positions for robustness.
    # This is done *after* layout adjustments so that axes positions are finalized.
    row_labels = ['Originals', 'Heatmaps', 'Overlays']
    for i in range(rows):
        # Get the position of the first subplot in the current row.
        pos = axs[i, 0].get_position()
        
        # Determine the label text and frame range.
        frame_range = "(Frames 1-16)" if i < 3 else "(Frames 17-32)"
        label_text = f"{row_labels[i % 3]}\n{frame_range}"
        
        # Place text in figure coordinates. The y-coordinate is the vertical center
        # of the subplot row, making the placement robust to figure size changes.
        fig.text(0.015, pos.y0 + pos.height / 2, label_text,
                 ha='left', va='center', rotation='vertical', fontsize=18)
    
    save_path = os.path.join(viz_output_dir, f"{game_name}_lrp_sample{sample_index}_cls_{predicted_class_name}_w_heatmap.png")
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    
    print(f"\nConsolidated visualization has been saved to: {save_path}")


# Main Execution
def main():
    parser = argparse.ArgumentParser(description="ViViT LRP Interpretation Script")
    parser.add_argument('--game_name', type=str, default='solid', help='Name of the game for dataset loading.')
    parser.add_argument('--sample_index', type=int, default=30, help='Index of the sample to process from the dataset.')
    parser.add_argument('--model_path', type=str, default='finetune_layer0_results/solid/checkpoints/checkpoint_epoch_10.pt', help='Path to the local fine-tuned model checkpoint.')
    parser.add_argument('--base_path', type=str, default='../Dataset/', help='Base directory path for the dataset.')
    parser.add_argument('--output_dir', type=str, default='new_interpretation', help='Directory to save the output visualizations.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')
    args = parser.parse_args()

    set_seed(args.seed)

    # --- Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model_ckpt_hub = "google/vivit-b-16x2-kinetics400"
    class_labels =['down','same','up']
    label2id = {label: i for i, label in enumerate(class_labels)}
    id2label = {i: label for label, i in label2id.items()}
    print(f"Class mapping: {id2label}")

    # --- Load Model and Data ---
    model = load_model(model_ckpt_hub, args.model_path, id2label, label2id, device)
    
    dataset = MyCSVDataset(
        csv_file=os.path.join(args.base_path, f'new_{args.game_name}.csv'),
        csv_file_2=os.path.join(args.base_path, "clean_data.csv"),
        base_path=args.base_path,
        game_name=args.game_name
    )
    
    inputs = prepare_input_from_dataset(dataset, args.sample_index, device)
    original_frames_tensor = inputs['pixel_values'][0]

    # --- Register Hooks ---
    all_attentions, all_attn_grads = [], []
    hooks = []

    def save_attention_hook(module, input, output):
        attn_weights = output[1]
        all_attentions.append(attn_weights.detach())
        attn_weights.register_hook(lambda grad: all_attn_grads.append(grad))

    for layer in model.vivit.encoder.layer:
        hooks.append(layer.attention.attention.register_forward_hook(save_attention_hook))

    # --- Forward and Backward Pass ---
    print("\nRunning forward and backward pass...")
    model.zero_grad()
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = logits.argmax(dim=-1).item()
    
    target_logit = logits[0, predicted_class]
    target_logit.backward()
    print(f"Predicted class: '{id2label[predicted_class]}' (ID: {predicted_class})")
    print(f"Captured {len(all_attentions)} attention tensors and {len(all_attn_grads)} gradients.")

    # --- Calculate LRP Relevance ---
    if not all_attentions or not all_attn_grads:
        raise RuntimeError("Attention or gradients were not captured. Check hooks.")
        
    cls_relevance = calculate_lrp_relevance(all_attentions, all_attn_grads, device)

    # --- Generate Visualization ---
    generate_and_save_visualization(
        cls_relevance,
        original_frames_tensor,
        args.output_dir,
        args.game_name,
        args.sample_index,
        id2label[predicted_class]
    )

    # --- Clean Up ---
    for h in hooks:
        h.remove()
    print("Hooks removed. Script finished.")

if __name__ == '__main__':
    main()