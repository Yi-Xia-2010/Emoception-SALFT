#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script performs qualitative evaluation of
various interpretability methods for a ViViT model.
"""

import os
import random
import argparse
import json
from typing import List, Tuple, Dict, Optional, Union, Type
from abc import ABC, abstractmethod

# Set the backend for Matplotlib before importing pyplot
import matplotlib
matplotlib.use('Agg')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset, random_split

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
import cv2

from transformers import VivitImageProcessor, VivitForVideoClassification
from tqdm import tqdm

def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
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
    def __init__(self, csv_file: str, base_path: str = "../Dataset/", game_name: str = 'solid'):
        self.data = pd.read_csv(csv_file)
        self.base_path = base_path
        self.game_name = game_name
        self.image_processor = VivitImageProcessor.from_pretrained("google/vivit-b-16x2-kinetics400")
        self.class_labels =['down','same','up']
        self.label2id = {label: i for i, label in enumerate(self.class_labels)}

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Optional[Dict[str, Union[torch.Tensor, str]]]:
        # If this dataset is a Subset, the index needs to be mapped to the original dataset's index
        if isinstance(idx, tuple):
            idx = idx[0]
        
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
                print(f"Warning: Frame not found at {frame_path}. Skipping sample {idx}.")
                return None

        inputs = self.image_processor(list(frames), return_tensors="pt")
        pixel_values = inputs['pixel_values'].squeeze(0)
        label = self.label2id[sample['arousal_change']]
        
        return {
            'pixel_values': pixel_values,
            'label': torch.tensor(label, dtype=torch.long),
            'clip_path': clip_path
        }

def load_model(model_path: Optional[str], model_ckpt_hub: str, id2label: Dict, label2id: Dict, device: torch.device) -> VivitForVideoClassification:
    model = None
    if model_path and os.path.exists(model_path):
        if os.path.isdir(model_path):
            model = VivitForVideoClassification.from_pretrained(model_path, label2id=label2id, id2label=id2label, ignore_mismatched_sizes=True)
        elif model_path.endswith(('.pt', '.pth')):
            model = VivitForVideoClassification.from_pretrained(model_ckpt_hub, label2id=label2id, id2label=id2label, ignore_mismatched_sizes=True)
            model.load_state_dict(torch.load(model_path, map_location=device))
    
    if model is None:
        model = VivitForVideoClassification.from_pretrained(model_ckpt_hub, label2id=label2id, id2label=id2label, ignore_mismatched_sizes=True)
    
    model.to(device)
    model.config.output_attentions = True
    model.eval()
    return model


class Interpreter(ABC):
    """Abstract base class for all interpretability methods."""
    def __init__(self, model: VivitForVideoClassification, device: torch.device):
        self.model = model
        self.device = device

    @abstractmethod
    def interpret(self, inputs: torch.Tensor, target_class: int) -> torch.Tensor:
        """Generates a relevance map for the given input and target class."""
        pass

class OurMethodInterpreter(Interpreter):
    """Implementation of our method"""
    def interpret(self, inputs: torch.Tensor, target_class: int) -> torch.Tensor:
        all_attentions, all_attn_grads = [], []
        hooks = []
        def save_attention_hook(module, input, output):
            attn_weights = output[1]
            all_attentions.append(attn_weights.detach())
            attn_weights.register_hook(lambda grad: all_attn_grads.insert(0, grad))
        for layer in self.model.vivit.encoder.layer:
            hooks.append(layer.attention.attention.register_forward_hook(save_attention_hook))
        self.model.zero_grad()
        outputs = self.model(pixel_values=inputs)
        logits = outputs.logits
        target_logit = logits[0, target_class]
        target_logit.backward(retain_graph=True)
        for h in hooks: h.remove()
        num_tokens = all_attentions[0].shape[-1]
        relevance = torch.eye(num_tokens, device=self.device).unsqueeze(0)
        for attn, grad in zip(all_attentions, all_attn_grads):
            attn_heads_fused = attn.mean(dim=1)
            grad_heads_fused = grad.mean(dim=1)
            R_layer = attn_heads_fused * grad_heads_fused
            R_layer = torch.clamp(R_layer, min=0) 
            R_layer_with_residual = R_layer + torch.eye(num_tokens, device=self.device).unsqueeze(0)
            R_layer_normalized = R_layer_with_residual / (R_layer_with_residual.sum(dim=-1, keepdim=True) + 1e-6)
            relevance = torch.matmul(relevance, R_layer_normalized)
        return relevance[0, 0, 1:]

class RolloutInterpreter(Interpreter):
    """Implementation of vanilla Rollout"""
    def interpret(self, inputs: torch.Tensor, target_class: int) -> torch.Tensor:
        all_attentions = []
        hooks = []
        def save_attention_hook(module, input, output):
            all_attentions.append(output[1].detach())
        for layer in self.model.vivit.encoder.layer:
            hooks.append(layer.attention.attention.register_forward_hook(save_attention_hook))
        self.model(pixel_values=inputs)
        for h in hooks: h.remove()
        num_tokens = all_attentions[0].shape[-1]
        relevance = torch.eye(num_tokens, device=self.device).unsqueeze(0)
        for attn in all_attentions:
            attn_heads_fused = attn.mean(dim=1)
            R_layer_with_residual = attn_heads_fused + torch.eye(num_tokens, device=self.device).unsqueeze(0)
            relevance = torch.matmul(relevance, R_layer_with_residual)
        return relevance[0, 0, 1:]

class GradCAMInterpreter(Interpreter):
    """Implementation of Grad-CAM for Vision Transformers."""
    def interpret(self, inputs: torch.Tensor, target_class: int) -> torch.Tensor:
        final_layer_features, final_layer_grads = None, None
        hooks = []
        def save_feature_hook(module, input, output): nonlocal final_layer_features; final_layer_features = output[0]
        def save_grad_hook(module, grad_input, grad_output): nonlocal final_layer_grads; final_layer_grads = grad_output[0]
        final_layer = self.model.vivit.encoder.layer[-1]
        hooks.append(final_layer.register_forward_hook(save_feature_hook))
        hooks.append(final_layer.register_full_backward_hook(save_grad_hook))
        self.model.zero_grad()
        outputs = self.model(pixel_values=inputs)
        logits = outputs.logits
        target_logit = logits[0, target_class]
        target_logit.backward(retain_graph=True)
        for h in hooks: h.remove()
        weights = torch.mean(final_layer_grads, dim=1).squeeze(0)
        features = final_layer_features.squeeze(0)
        relevance = F.relu(torch.einsum('d,td->t', weights, features))
        return relevance[1:]

class GradSAMInterpreter(Interpreter):
    """Implementation of Grad-SAM"""
    def interpret(self, inputs: torch.Tensor, target_class: int) -> torch.Tensor:
        all_attentions, all_attn_grads = [], []
        hooks = []
        def save_attention_hook(module, input, output):
            attn_weights = output[1]
            all_attentions.append(attn_weights.detach())
            attn_weights.register_hook(lambda grad: all_attn_grads.insert(0, grad))
        for layer in self.model.vivit.encoder.layer:
            hooks.append(layer.attention.attention.register_forward_hook(save_attention_hook))
        self.model.zero_grad()
        outputs = self.model(pixel_values=inputs)
        logits = outputs.logits
        target_logit = logits[0, target_class]
        target_logit.backward(retain_graph=True)
        for h in hooks: h.remove()
        L = len(all_attentions)
        M = all_attentions[0].shape[1]
        N = all_attentions[0].shape[-1]
        all_H_lm_per_layer_head = []
        for l in range(L):
            attn = all_attentions[l]
            grad = all_attn_grads[l]
            relu_grad = F.relu(grad)
            H_lm = attn * relu_grad
            all_H_lm_per_layer_head.append(H_lm)
        stacked_H_lm = torch.cat(all_H_lm_per_layer_head, dim=0).squeeze(1)
        sum_j_H_lm = stacked_H_lm.sum(dim=-1)
        avg_m_sum_j_H_lm = sum_j_H_lm.mean(dim=1)
        final_relevance = avg_m_sum_j_H_lm.mean(dim=0)
        return final_relevance[1:]

def run_perturbation_test(model: VivitForVideoClassification, interpreter: Interpreter, inputs: torch.Tensor, target_class: int, mode: str = 'positive', steps: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    relevance_map = interpreter.interpret(inputs, target_class).detach()
    sorted_indices = torch.argsort(relevance_map, descending=(mode == 'positive'))
    with torch.no_grad():
        initial_embeddings = model.vivit.embeddings(inputs)
    num_tubelets = initial_embeddings.shape[1] - 1
    confidence_scores = []
    perturbation_levels = np.linspace(0.1, 0.9, steps)
    for frac in perturbation_levels:
        num_to_perturb = int(frac * num_tubelets)
        perturbed_embeddings = initial_embeddings.clone()
        if num_to_perturb > 0:
            indices_to_perturb = sorted_indices[:num_to_perturb] + 1 
            perturbed_embeddings[0, indices_to_perturb, :] = 0.0
        with torch.no_grad():
            encoder_outputs = model.vivit.encoder(perturbed_embeddings)
            logits = model.classifier(encoder_outputs[0][:, 0, :])
            probabilities = F.softmax(logits, dim=-1)
            confidence_scores.append(probabilities[0, target_class].item())
    return perturbation_levels, np.array(confidence_scores)

def calculate_auc(scores: np.ndarray) -> float:
    return np.mean(scores)

def generate_qualitative_comparison(
    model: VivitForVideoClassification,
    interpreters: Dict[str, Type[Interpreter]],
    inputs: torch.Tensor,
    original_frames_tensor: torch.Tensor,
    target_class: int,
    id2label: Dict,
    output_dir: str,
    sample_index: int,
    num_frames_to_show: int = 8
):
    """Generates a side-by-side comparison plot of heatmaps from different interpreters."""
    print(f"\n--- Generating Qualitative Comparison for sample {sample_index} ---")
    device = inputs.device
    num_interpreters = len(interpreters)
    
    total_frames = original_frames_tensor.shape[0]
    indices_to_show = np.linspace(0, total_frames - 1, num_frames_to_show, dtype=int)

    fig, axs = plt.subplots(num_interpreters, num_frames_to_show, figsize=(num_frames_to_show * 2, num_interpreters * 2 + 0.5))
    # fig.suptitle(f'Qualitative Comparison for Sample {sample_index} (Predicted: {id2label[target_class]})', fontsize=16)

    if num_interpreters == 1:
        axs = np.expand_dims(axs, axis=0)

    for i, (name, interpreter_class) in enumerate(interpreters.items()):
        interpreter = interpreter_class(model, device)
        relevance_map = interpreter.interpret(inputs, target_class).detach()

        num_temporal_tubelets = total_frames // 2
        num_spatial_tokens_per_tubelet = relevance_map.shape[0] // num_temporal_tubelets
        grid_size = int(np.sqrt(num_spatial_tokens_per_tubelet))
        
        relevance_reshaped = relevance_map.reshape(num_temporal_tubelets, grid_size, grid_size)

        frame_heatmaps = []
        for t_idx in range(num_temporal_tubelets):
            heatmap = relevance_reshaped[t_idx].cpu().numpy()
            heatmap_upsampled = cv2.resize(heatmap, (224, 224), interpolation=cv2.INTER_LINEAR)
            frame_heatmaps.append(heatmap_upsampled)
            frame_heatmaps.append(heatmap_upsampled.copy())

        axs[i, 0].set_ylabel(name, rotation=90, size='large', labelpad=20)

        for j, frame_idx in enumerate(indices_to_show):
            frame_tensor = original_frames_tensor[frame_idx].cpu()
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            frame_tensor = torch.clamp(frame_tensor * std + mean, 0, 1)
            original_frame = (frame_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

            heatmap = frame_heatmaps[frame_idx]
            heatmap_norm = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap) + 1e-8)
            heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_norm), cv2.COLORMAP_JET)
            heatmap_color_rgb = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
            
            overlay = cv2.addWeighted(original_frame, 0.6, heatmap_color_rgb, 0.4, 0)

            ax_to_plot = axs[i, j]
            ax_to_plot.imshow(overlay)
            ax_to_plot.set_xticks([])
            ax_to_plot.set_yticks([])
            if i == 0:
                ax_to_plot.set_title(f'Frame {frame_idx}')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_path = os.path.join(output_dir, f'qualitative_comparison_sample_{sample_index}.png')
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"Qualitative comparison plot saved to: {save_path}")

def main():
    parser = argparse.ArgumentParser(description="ViViT Interpretability Evaluation Script")
    parser.add_argument('--game_name', type=str, default='solid', help='Name of the game for dataset loading.')
    parser.add_argument('--model_path', type=str, default='finetune_layer0_results/solid/checkpoints/checkpoint_epoch_10.pt', help='Path to a local model state dict.')
    parser.add_argument('--base_path', type=str, default='../Dataset/', help='Base directory path for the dataset.')
    parser.add_argument('--output_dir', type=str, help='Directory to save the output plots and results.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')
    # MODIFICATION: Updated help text for the qualitative sample index argument.
    parser.add_argument('--qualitative_sample_idx', type=int, default=-1, help='Index of a sample for qualitative plot. Set to -1 to auto-find a sample where true and predicted are "up".')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'], help="The dataset split to evaluate on (train, val, or test)")
    args = parser.parse_args()
    if not args.output_dir:
        args.output_dir = f'interpretation_evaluation_results/{args.game_name}/with_target_vs_predicted'   

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # --- Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_ckpt_hub = "google/vivit-b-16x2-kinetics400"
    class_labels =['down','same','up']
    label2id = {label: i for i, label in enumerate(class_labels)}
    id2label = {i: label for label, i in label2id.items()}

    # --- Load Model and Data ---
    model = load_model(args.model_path, model_ckpt_hub, id2label, label2id, device)
    
    # Load the full dataset and then split it using PyTorch's random_split.
    full_csv_path = os.path.join(args.base_path, f'new_{args.game_name}.csv')
    print(f"Info: Loading full dataset from '{full_csv_path}'...")
    
    full_dataset = MyCSVDataset(
        csv_file=full_csv_path,
        base_path=args.base_path,
        game_name=args.game_name
    )

    # Define split proportions
    dataset_size = len(full_dataset)
    train_size = int(0.7 * dataset_size)
    val_size = int(0.15 * dataset_size)
    test_size = dataset_size - train_size - val_size

    # Create a generator for reproducible splits
    generator = torch.Generator().manual_seed(args.seed)
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size], generator=generator
    )

    split_datasets = {
        'train': train_dataset,
        'val': val_dataset,
        'test': test_dataset
    }

    print(f"Dataset split (seed={args.seed}):")
    print(f"  - Train set: {len(train_dataset)} samples")
    print(f"  - Validation set: {len(val_dataset)} samples")
    print(f"  - Test set: {len(test_dataset)} samples")

    # Select the dataset for evaluation based on the --split argument
    evaluation_dataset = split_datasets[args.split]
    print(f"\nRunning evaluation on the '{args.split}' split ({len(evaluation_dataset)} samples).")

    # --- Initialize Interpreters ---
    interpreters_to_test = {
        "Rollout": RolloutInterpreter,
        "Grad-CAM": GradCAMInterpreter,
        "Grad-SAM": GradSAMInterpreter,
        "Our Method": OurMethodInterpreter
    }
    
    # --- MODIFICATION: Generate Qualitative Comparison Plot if Requested ---
    if args.qualitative_sample_idx is not None:
        # If index is -1, automatically search for a suitable sample
        if args.qualitative_sample_idx == -1:
            print("\n--- Searching for a suitable sample for Qualitative Comparison ---")
            print("Condition: Start from index 20, True Label == 'up', Predicted Label == 'up'")
            found_sample_for_qualitative = False
            # Iterate from index 20 to the end of the selected dataset split
            for i in range(20, len(evaluation_dataset)):
                try:
                    qual_sample_data = evaluation_dataset[i]
                    if not qual_sample_data:
                        continue

                    true_target_class = qual_sample_data['label'].item()
                    
                    # Optimization: Check if the true label is 'up' before running the model
                    if id2label[true_target_class] != 'up':
                        continue

                    qual_inputs = qual_sample_data['pixel_values'].unsqueeze(0).to(device)
                    with torch.no_grad():
                        qual_predicted_class = model(pixel_values=qual_inputs).logits.argmax(dim=-1).item()
                    
                    # Check if both true and predicted labels are 'up'
                    if id2label[true_target_class] == 'up' and id2label[qual_predicted_class] == 'up':
                        print(f"Found suitable sample at index {i}. Generating plot...")
                        generate_qualitative_comparison(
                            model, interpreters_to_test, qual_inputs, qual_sample_data['pixel_values'],
                            qual_predicted_class, id2label, args.output_dir, i
                        )
                        found_sample_for_qualitative = True
                        break # Exit the loop once a sample is found and processed
                except Exception as e:
                    print(f"\n[Error] An error occurred while processing sample {i} for qualitative search: {e}")
                    continue # Continue to the next sample
            
            if not found_sample_for_qualitative:
                print("Search complete. No suitable sample was found that meets the criteria.")

        # If a specific, non-negative index is given, use the original logic
        else:
            try:
                print(f"\n--- Generating Qualitative Comparison for specified sample index {args.qualitative_sample_idx} ---")
                qual_sample_data = evaluation_dataset[args.qualitative_sample_idx]
                if qual_sample_data:
                    qual_inputs = qual_sample_data['pixel_values'].unsqueeze(0).to(device)
                    with torch.no_grad():
                        qual_predicted_class = model(pixel_values=qual_inputs).logits.argmax(dim=-1).item()
                    
                    generate_qualitative_comparison(
                        model, interpreters_to_test, qual_inputs, qual_sample_data['pixel_values'],
                        qual_predicted_class, id2label, args.output_dir, args.qualitative_sample_idx
                    )
            except IndexError:
                print(f"\n[Error] Qualitative sample index {args.qualitative_sample_idx} is out of range. The '{args.split}' split only has {len(evaluation_dataset)} samples.")
            except Exception as e:
                print(f"\n[Error] Could not generate qualitative plot for sample {args.qualitative_sample_idx}: {e}")

    # --- Run Quantitative Perturbation Tests ---
    num_total_samples = len(evaluation_dataset)
    print(f"\n--- Running perturbation tests on {num_total_samples} samples ---")
    
    results = {
        name: {
            'target_class_test': {'positive': [], 'negative': []},
            'predicted_class_test': {'positive': [], 'negative': []}
        } for name in interpreters_to_test
    }
    
    # The loop now iterates over the selected dataset split
    for i in tqdm(range(num_total_samples), desc=f"Processing '{args.split}' split for perturbation test"):
        try:
            sample_data = evaluation_dataset[i]
            if sample_data is None:
                print(f"Skipping sample {i} as it returned None.")
                continue

            inputs = sample_data['pixel_values'].unsqueeze(0).to(device)
            true_target_class = sample_data['label'].item() # Ground truth label

            with torch.no_grad():
                predicted_class = model(pixel_values=inputs).logits.argmax(dim=-1).item()

            for name, interpreter_class in interpreters_to_test.items():
                interpreter = interpreter_class(model, device)

                # Test against TRUE TARGET class
                _, pos_scores_true = run_perturbation_test(model, interpreter, inputs, true_target_class, mode='positive')
                results[name]['target_class_test']['positive'].append(pos_scores_true)
                _, neg_scores_true = run_perturbation_test(model, interpreter, inputs, true_target_class, mode='negative')
                results[name]['target_class_test']['negative'].append(neg_scores_true)

                # Test against PREDICTED class
                _, pos_scores_pred = run_perturbation_test(model, interpreter, inputs, predicted_class, mode='positive')
                results[name]['predicted_class_test']['positive'].append(pos_scores_pred)
                _, neg_scores_pred = run_perturbation_test(model, interpreter, inputs, predicted_class, mode='negative')
                results[name]['predicted_class_test']['negative'].append(neg_scores_pred)

        except Exception as e:
            print(f"\n[Error] Could not process sample {i} for perturbation test: {e}")

    # --- Aggregate, Plot, and Summarize Perturbation Results ---
    print("\n--- Perturbation Test Results ---")
    
    summary_data = {
        "dataset_split_info": {
            "train_samples": len(train_dataset),
            "validation_samples": len(val_dataset),
            "test_samples": len(test_dataset),
            "total_samples": dataset_size,
            "evaluated_split": args.split,
            "evaluated_samples": len(evaluation_dataset)
        },
        "results": {}
    }

    test_types = [
        ('target_class_test', 'Target Class'),
        ('predicted_class_test', 'Predicted Class')
    ]

    for test_key, test_name in test_types:
        summary_data["results"][test_key] = {}
        print(f"\n--- Results for Perturbation against {test_name} ---")
        
        # Positive Perturbation Plot
        plt.figure(figsize=(10, 7))
        summary_data["results"][test_key]['positive'] = {}
        for name in interpreters_to_test.keys():
            if results[name][test_key]['positive']:
                avg_scores = np.mean(np.array(results[name][test_key]['positive']), axis=0)
                auc = calculate_auc(avg_scores)
                levels = np.linspace(0.1, 0.9, len(avg_scores))
                plt.plot(levels, avg_scores, marker='o', linestyle='-', label=f'{name} (AUC: {auc:.3f})')
                print(f"[Positive - {test_name}] {name}: AUC = {auc:.4f}")
                summary_data["results"][test_key]['positive'][name] = {'auc': auc, 'curve': avg_scores.tolist()}
        
        # plt.title(f'Positive Perturbation (vs. {test_name}) on {args.split} split')
        plt.xlabel('Fraction of Tubelets Perturbed')
        plt.ylabel(f'Model Confidence for {test_name}')
        plt.grid(True, linestyle='--'); plt.legend(); plt.savefig(os.path.join(args.output_dir, f'positive_perturbation_{args.split}_{test_key}.png'), dpi=300); plt.close()

        # Negative Perturbation Plot
        plt.figure(figsize=(10, 7))
        summary_data["results"][test_key]['negative'] = {}
        for name in interpreters_to_test.keys():
            if results[name][test_key]['negative']:
                avg_scores = np.mean(np.array(results[name][test_key]['negative']), axis=0)
                auc = calculate_auc(avg_scores)
                levels = np.linspace(0.1, 0.9, len(avg_scores))
                plt.plot(levels, avg_scores, marker='o', linestyle='-', label=f'{name} (AUC: {auc:.3f})')
                print(f"[Negative - {test_name}] {name}: AUC = {auc:.4f}")
                summary_data["results"][test_key]['negative'][name] = {'auc': auc, 'curve': avg_scores.tolist()}

        # plt.title(f'Negative Perturbation (vs. {test_name}) on {args.split} split')
        plt.xlabel('Fraction of Tubelets Perturbed')
        plt.ylabel(f'Model Confidence for {test_name}')
        plt.grid(True, linestyle='--'); plt.legend(); plt.savefig(os.path.join(args.output_dir, f'negative_perturbation_{args.split}_{test_key}.png'), dpi=300); plt.close()

    # --- Save Results to JSON ---
    json_path = os.path.join(args.output_dir, f'evaluation_summary_{args.split}.json')
    with open(json_path, 'w') as f:
        json.dump(summary_data, f, indent=4)
    print(f"\nSummary of results saved to {json_path}")
    
    print(f"\nAll evaluation plots and summary saved to {args.output_dir}")

if __name__ == '__main__':
    main()
