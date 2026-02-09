#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Complete Interpretability Evaluation Script - Memory Optimized for A100
Evaluates: 4 interpreters × 2 target types × 2 perturbation modes = 16 combinations
Optimizations: Batch cleanup, gradient management, periodic GC
"""

import os
import gc
import random
import argparse
import json
from typing import Dict, Tuple

import matplotlib
matplotlib.use('Agg')

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import StratifiedGroupKFold

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

from transformers import VivitImageProcessor, VivitForVideoClassification
from tqdm import tqdm


def set_seed(seed: int):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_player_id_simple(file_path: str, game_name: str) -> str:
    """Extract player ID from file path"""
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


class MemoryManager:
    """Memory manager - periodically cleans GPU cache"""
    
    def __init__(self, cleanup_interval=50):
        self.cleanup_interval = cleanup_interval
        self.call_count = 0
    
    def maybe_cleanup(self):
        """Conditional cleanup"""
        self.call_count += 1
        if self.call_count % self.cleanup_interval == 0:
            self.force_cleanup()
    
    def force_cleanup(self):
        """Force GPU cache cleanup"""
        gc.collect()
        torch.cuda.empty_cache()
    
    def get_memory_info(self):
        """Get GPU memory usage information"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            reserved = torch.cuda.memory_reserved() / 1024**3    # GB
            return f"Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB"
        return "CPU mode"


class MyCSVDataset:
    """Dataset with caching support"""
    
    def __init__(self, data_df: pd.DataFrame, base_path: str, game_name: str, 
                 label2id: Dict[str, int], cache_dir: str = None):
        self.data = data_df.reset_index(drop=True)
        self.base_path = base_path
        self.game_name = game_name
        self.label2id = label2id
        self.cache_dir = cache_dir
        self.image_processor = VivitImageProcessor.from_pretrained("google/vivit-b-16x2-kinetics400")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        sample = self.data.iloc[idx]
        clip_path = sample['start_frame']
        file_name_str = os.path.splitext(os.path.basename(clip_path))[0]
        parent_folder = os.path.dirname(clip_path)
        
        frames = []
        loaded_from_cache = False

        # Try loading from cache
        if self.cache_dir:
            safe_name = f"{parent_folder}_{file_name_str}.pt".replace(os.sep, "_")
            cache_path = os.path.join(self.cache_dir, safe_name)
            if os.path.exists(cache_path):
                try:
                    video_tensor = torch.load(cache_path, weights_only=True)
                    frames = [video_tensor[i] for i in range(video_tensor.shape[0])]
                    loaded_from_cache = True
                except:
                    pass
        
        # Load from PNG files if cache failed
        if not loaded_from_cache:
            try: 
                start_frame_num = int(file_name_str)
            except ValueError: 
                start_frame_num = 0
            
            for i in range(32):
                current_frame_name = f"{start_frame_num + i:04d}.png"
                frame_path = os.path.join(self.base_path, parent_folder, current_frame_name)
                try: 
                    frames.append(Image.open(frame_path).convert('RGB'))
                except: 
                    # Use black frame as fallback
                    frames.append(Image.new('RGB', (224, 224), (0, 0, 0)))

        # Process frames
        inputs = self.image_processor(list(frames), return_tensors="pt")
        pixel_values = inputs['pixel_values'].squeeze(0)
        
        # Get label
        label_str = str(sample['arousal_change'])
        label = self.label2id.get(label_str, 1)

        return {
            'pixel_values': pixel_values, 
            'label': torch.tensor(label, dtype=torch.long), 
            'clip_path': clip_path
        }


def load_model(model_path: str, device: torch.device):
    """Load model with eager attention for hook compatibility"""
    model_ckpt_hub = "google/vivit-b-16x2-kinetics400"
    class_labels = ['down', 'same', 'up']
    label2id = {l: i for i, l in enumerate(class_labels)}
    id2label = {i: l for i, l in enumerate(class_labels)}
    
    # Force eager implementation for A100 compatibility
    model = VivitForVideoClassification.from_pretrained(
        model_ckpt_hub, 
        label2id=label2id, 
        id2label=id2label, 
        ignore_mismatched_sizes=True,
        attn_implementation="eager"
    )
    
    # Load weights if path exists
    if model_path and os.path.exists(model_path):
        if model_path.endswith(('.pt', '.pth')):
            print(f"✓ Loading state dict from {model_path}")
            model.load_state_dict(torch.load(model_path, map_location=device))
        elif os.path.isdir(model_path):
            print(f"✓ Loading from directory {model_path}")
            model = VivitForVideoClassification.from_pretrained(
                model_path, 
                label2id=label2id, 
                id2label=id2label, 
                attn_implementation="eager"
            )
    
    model.to(device)
    model.config.output_attentions = True
    model.eval()
    
    # Enable gradients for interpretation
    for param in model.parameters(): 
        param.requires_grad = True
    
    return model, id2label, label2id


class Interpreter:
    """Base interpreter class"""
    def __init__(self, model, device):
        self.model = model
        self.device = device
    
    def interpret(self, inputs: torch.Tensor, target_class: int) -> torch.Tensor:
        """Generate relevance map for given input and target class"""
        raise NotImplementedError
    
    def cleanup(self):
        """Clean up temporary variables"""
        pass

class OurMethodInterpreter(Interpreter):
    """
    Our proposed method: Attention Rollout with Gradient Weighting
    Combines attention flow with gradient information for better localization
    """
    
    def interpret(self, inputs: torch.Tensor, target_class: int) -> torch.Tensor:
        all_attentions, all_attn_grads = [], []
        hooks = []
        
        def save_attention_hook(module, input, output):
            attn_weights = output[1]
            all_attentions.append(attn_weights.detach())
            # Use append - gradients arrive in backward order
            attn_weights.register_hook(lambda grad: all_attn_grads.append(grad))
        
        # Register hooks for all transformer layers
        for layer in self.model.vivit.encoder.layer:
            hooks.append(layer.attention.attention.register_forward_hook(save_attention_hook))
        
        # Forward + Backward pass
        self.model.zero_grad()
        outputs = self.model(pixel_values=inputs)
        target_logit = outputs.logits[0, target_class]
        target_logit.backward(retain_graph=True)
        
        # Clean up hooks
        for h in hooks: 
            h.remove()
        
        # Reverse attentions to match gradient order
        # all_attentions (before): [layer_0, layer_1, ..., layer_L] (forward order)
        # all_attn_grads (after backward): [grad_L, grad_{L-1}, ..., grad_0] (backward order)
        all_attentions.reverse()
        # Now both are in backward order: [last, ..., first]
        
        # Propagate relevance from last layer to first
        num_tokens = all_attentions[0].shape[-1]
        relevance = torch.eye(num_tokens, device=self.device).unsqueeze(0)
        
        for attn, grad in zip(all_attentions, all_attn_grads):
            # Average over attention heads
            attn_heads_fused = attn.mean(dim=1)
            grad_heads_fused = grad.mean(dim=1)
            
            # Gradient-weighted attention (keep only positive contributions)
            R_layer = torch.clamp(attn_heads_fused * grad_heads_fused, min=0)
            
            # Add residual connection (identity matrix)
            R_layer_with_residual = R_layer + torch.eye(num_tokens, device=self.device).unsqueeze(0)
            
            # Normalize to make it a proper transition matrix
            R_layer_normalized = R_layer_with_residual / (R_layer_with_residual.sum(dim=-1, keepdim=True) + 1e-6)
            
            # Accumulate relevance through matrix multiplication
            relevance = torch.matmul(relevance, R_layer_normalized)
        
        # Return relevance for spatial tokens (exclude CLS token)
        result = relevance[0, 0, 1:].detach()
        
        # Explicit cleanup to save memory
        del all_attentions, all_attn_grads, relevance
        
        return result

class RolloutInterpreter(Interpreter):
    """
    Vanilla Attention Rollout (Abnar & Zuidema, 2020)
    Propagates attention weights through layers without gradient information
    """
    
    def interpret(self, inputs: torch.Tensor, target_class: int) -> torch.Tensor:
        all_attentions = []
        hooks = []
        
        def save_attention_hook(module, input, output):
            all_attentions.append(output[1].detach())
        
        for layer in self.model.vivit.encoder.layer:
            hooks.append(layer.attention.attention.register_forward_hook(save_attention_hook))
        
        # Forward pass only (no gradients needed)
        with torch.no_grad():
            self.model(pixel_values=inputs)
        
        for h in hooks: 
            h.remove()
        
        # Rollout attention through layers
        num_tokens = all_attentions[0].shape[-1]
        relevance = torch.eye(num_tokens, device=self.device).unsqueeze(0)
        
        for attn in all_attentions:
            attn_heads_fused = attn.mean(dim=1)
            R_layer_with_residual = attn_heads_fused + torch.eye(num_tokens, device=self.device).unsqueeze(0)
            relevance = torch.matmul(relevance, R_layer_with_residual)
        
        result = relevance[0, 0, 1:].detach()
        
        del all_attentions, relevance
        
        return result

class GradCAMInterpreter(Interpreter):
    """
    Grad-CAM adapted for Vision Transformers
    Uses gradients of output w.r.t. final layer features
    """
    
    def interpret(self, inputs: torch.Tensor, target_class: int) -> torch.Tensor:
        # Use dict to avoid closure issues
        cache = {'feat': None, 'grad': None}
        
        def save_feature_hook(module, input, output): 
            cache['feat'] = output[0].detach()
        
        def save_grad_hook(module, grad_input, grad_output): 
            cache['grad'] = grad_output[0].detach()
        
        # Hook the final transformer layer
        final_layer = self.model.vivit.encoder.layer[-1]
        h1 = final_layer.register_forward_hook(save_feature_hook)
        h2 = final_layer.register_full_backward_hook(save_grad_hook)
        
        # Forward + Backward
        self.model.zero_grad()
        outputs = self.model(pixel_values=inputs)
        outputs.logits[0, target_class].backward(retain_graph=True)
        
        h1.remove()
        h2.remove()
        
        # Compute Grad-CAM: weighted sum of features
        weights = torch.mean(cache['grad'], dim=1).squeeze(0)
        features = cache['feat'].squeeze(0)
        relevance = F.relu(torch.einsum('d,td->t', weights, features))
        
        result = relevance[1:].detach()
        
        del cache, weights, features, relevance
        
        return result

class GradSAMInterpreter(Interpreter):
    """
    Grad-SAM (Gradient-weighted Self-Attention Map)
    FIXED: Proper gradient collection with explicit reverse
    """
    
    def interpret(self, inputs: torch.Tensor, target_class: int) -> torch.Tensor:
        all_attentions, all_attn_grads = [], []
        hooks = []
        
        def save_attention_hook(module, input, output):
            attn_weights = output[1]
            all_attentions.append(attn_weights.detach())
            # FIXED: Use append instead of insert(0)
            attn_weights.register_hook(lambda grad: all_attn_grads.append(grad))
        
        for layer in self.model.vivit.encoder.layer:
            hooks.append(layer.attention.attention.register_forward_hook(save_attention_hook))
        
        self.model.zero_grad()
        outputs = self.model(pixel_values=inputs)
        outputs.logits[0, target_class].backward(retain_graph=True)
        
        for h in hooks: 
            h.remove()
        
        # FIXED: Reverse gradients to match attention order
        # all_attentions: [layer_0, ..., layer_L] (forward order)
        # all_attn_grads after backward: [grad_L, ..., grad_0] (backward order)
        all_attn_grads.reverse()
        # Now both are in forward order: [layer_0, ..., layer_L]
        
        # Compute H_lm (attention × ReLU(gradient)) for each layer and head
        all_H = []
        for l in range(len(all_attentions)):
            H_lm = all_attentions[l] * F.relu(all_attn_grads[l])
            all_H.append(H_lm)
        
        # Stack across layers and aggregate
        stacked_H = torch.cat(all_H, dim=0).squeeze(1)  # [L*M, N, N]
        sum_j_H_lm = stacked_H.sum(dim=-1)  # Sum over columns: [L*M, N]
        avg_m_sum_j_H_lm = sum_j_H_lm.mean(dim=1)  # Average over heads: [L, N] -> [N]
        final_relevance = avg_m_sum_j_H_lm.mean(dim=0)  # Average over layers: [N]
        
        result = final_relevance[1:].detach()
        
        del all_attentions, all_attn_grads, all_H, stacked_H
        
        return result


def run_perturbation_test(model, interpreter, inputs: torch.Tensor, 
                         target_class: int, mode: str, steps: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Memory-optimized perturbation test
    Key optimizations:
    1. Use torch.no_grad() to reduce memory
    2. Delete intermediate variables promptly
    3. Cleanup after each perturbation level
    
    Args:
        model: The ViViT model
        interpreter: Interpreter instance
        inputs: Input tensor [1, C, T, H, W]
        target_class: Target class index
        mode: 'positive' (mask most relevant) or 'negative' (mask least relevant)
        steps: Number of perturbation levels
    
    Returns:
        perturbation_levels: Array of perturbation fractions
        confidence_scores: Model confidence at each level
    """
    # Get relevance map
    relevance_map = interpreter.interpret(inputs, target_class).detach()
    
    # Sort tubelets by relevance
    sorted_indices = torch.argsort(relevance_map, descending=(mode == 'positive'))
    
    # Get initial embeddings
    with torch.no_grad():
        initial_embeddings = model.vivit.embeddings(inputs)
    
    num_tubelets = initial_embeddings.shape[1] - 1  # Exclude CLS token
    confidence_scores = []
    perturbation_levels = np.linspace(0.1, 0.9, steps)
    
    for frac in perturbation_levels:
        num_to_perturb = int(frac * num_tubelets)
        
        # Use no_grad to reduce memory
        with torch.no_grad():
            perturbed_embeddings = initial_embeddings.clone()
            
            if num_to_perturb > 0:
                # Get indices to perturb (add 1 to skip CLS token)
                indices_to_perturb = sorted_indices[:num_to_perturb] + 1
                # Zero out selected embeddings
                perturbed_embeddings[0, indices_to_perturb, :] = 0.0
            
            # Forward pass with perturbed embeddings
            encoder_outputs = model.vivit.encoder(perturbed_embeddings)
            logits = model.classifier(encoder_outputs[0][:, 0, :])
            probabilities = F.softmax(logits, dim=-1)
            confidence = probabilities[0, target_class].item()
            confidence_scores.append(confidence)
            
            # Delete temporary variables
            del perturbed_embeddings, encoder_outputs, logits, probabilities
    
    # Final cleanup
    del relevance_map, sorted_indices, initial_embeddings
    
    return perturbation_levels, np.array(confidence_scores)


def main():
    parser = argparse.ArgumentParser(description="Complete Interpretability Evaluation (Memory Optimized)")
    parser.add_argument('--game_name', type=str, default='solid', help='Game name')
    parser.add_argument('--model_path', type=str, default='Results/new5/ours/solid/fold_1/checkpoint_best',
                       help='Path to model checkpoint')
    parser.add_argument('--base_path', type=str, default='../Dataset/', help='Dataset base path')
    parser.add_argument('--output_dir', type=str, default='perturbation_results',
                       help='Output directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--fold', type=int, default=0, help='Fold index (0-4)')
    parser.add_argument('--steps', type=int, default=10, help='Number of perturbation steps')
    parser.add_argument('--max_samples', type=int, default=-1, 
                       help='Maximum samples to test (-1 for all)')
    parser.add_argument('--cleanup_interval', type=int, default=50, 
                       help='GPU cleanup interval (samples)')
    parser.add_argument('--save_interval', type=int, default=100,
                       help='Save checkpoint interval (samples)')
    args = parser.parse_args()

    # Initialize
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Memory manager
    mem_manager = MemoryManager(cleanup_interval=args.cleanup_interval)
    
    print("=" * 80)
    print("Complete Interpretability Evaluation (Memory Optimized for A100)")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Game: {args.game_name}, Fold: {args.fold}")
    print(f"Cleanup interval: {args.cleanup_interval} samples")
    print(f"Initial GPU memory: {mem_manager.get_memory_info()}")
    print("=" * 80)

    # Load model
    print("\n[1/4] Loading model...")
    model, id2label, label2id = load_model(args.model_path, device)
    print(f"After model loading: {mem_manager.get_memory_info()}")

    # Load data
    print("\n[2/4] Loading data...")
    csv_path = os.path.join(args.base_path, f'new_{args.game_name}.csv')
    all_df = pd.read_csv(csv_path)
    
    # Extract player IDs
    all_df['player_id'] = all_df['start_frame'].apply(
        lambda x: get_player_id_simple(x, args.game_name)
    )
    
    # Shift arousal_change within player groups for sequence prediction
    all_df['arousal_change'] = all_df.groupby('player_id')['arousal_change'].shift(-1)
    all_df = all_df.dropna(subset=['arousal_change']).reset_index(drop=True)
    all_df['block_id'] = all_df['player_id']
    
    # Create stratified group splits
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    splits = list(sgkf.split(all_df, all_df['arousal_change'].map(label2id), groups=all_df['block_id']))
    test_idx = splits[args.fold][1]
    test_df = all_df.iloc[test_idx].reset_index(drop=True)
    
    cache_dir = os.path.join(args.base_path, f"cache_{args.game_name}")
    dataset = MyCSVDataset(test_df, args.base_path, args.game_name, label2id, cache_dir=cache_dir)
    
    num_samples = len(dataset) if args.max_samples == -1 else min(args.max_samples, len(dataset))
    
    print(f" Test set size: {len(test_df)}")
    print(f" Will test {num_samples} samples")
    print(f"  Label distribution: {test_df['arousal_change'].value_counts().to_dict()}")

    # Initialize interpreters
    print("\n[3/4] Initializing interpreters...")
    interpreters = {
        "Rollout": RolloutInterpreter,
        "Grad-CAM": GradCAMInterpreter,
        "Grad-SAM": GradSAMInterpreter,
        "Our Method": OurMethodInterpreter
    }
    print(f" {len(interpreters)} interpreters ready")

    # Initialize result storage
    results = {
        name: {
            'target': {'positive': [], 'negative': []},
            'predicted': {'positive': [], 'negative': []}
        } for name in interpreters
    }
    
    checkpoint_path = os.path.join(args.output_dir, 'checkpoint.json')

    # Run evaluation
    print(f"\n[4/4] Running complete evaluation...")
    print(f"Total tests per sample: {len(interpreters)} × 2 types × 2 modes = {len(interpreters)*4}")
    print(f"Expected total tests: {num_samples * len(interpreters) * 4}")
    print("=" * 80)
    
    error_count = 0
    
    for i in tqdm(range(num_samples), desc="Processing samples"):
        try:
            sample = dataset[i]
            if sample is None:
                error_count += 1
                continue
            
            inputs = sample['pixel_values'].unsqueeze(0).to(device)
            true_class = sample['label'].item()
            
            # Get prediction
            with torch.no_grad():
                pred_class = model(pixel_values=inputs).logits.argmax(-1).item()
            
            # Run tests for each interpreter
            for name, interp_cls in interpreters.items():
                interp = interp_cls(model, device)
                
                # Target class tests
                _, pos_t = run_perturbation_test(model, interp, inputs, true_class, 'positive', args.steps)
                results[name]['target']['positive'].append(pos_t)
                
                _, neg_t = run_perturbation_test(model, interp, inputs, true_class, 'negative', args.steps)
                results[name]['target']['negative'].append(neg_t)
                
                # Predicted class tests
                _, pos_p = run_perturbation_test(model, interp, inputs, pred_class, 'positive', args.steps)
                results[name]['predicted']['positive'].append(pos_p)
                
                _, neg_p = run_perturbation_test(model, interp, inputs, pred_class, 'negative', args.steps)
                results[name]['predicted']['negative'].append(neg_p)
                
                # Clean up interpreter
                del interp
            
            # Clean up inputs
            del inputs
            
            # Periodic GPU memory cleanup
            mem_manager.maybe_cleanup()
            
            # Periodic checkpoint saving
            if (i + 1) % args.save_interval == 0:
                print(f"\n[Checkpoint] Saving at sample {i+1}")
                print(f"GPU memory: {mem_manager.get_memory_info()}")
                
                checkpoint_data = {
                    'meta': {
                        'samples_processed': i + 1,
                        'total_samples': num_samples,
                        'errors': error_count
                    },
                    'results': results
                }
                
                with open(checkpoint_path, 'w') as f:
                    json.dump(checkpoint_data, f, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
                
                print(f"✓ Checkpoint saved to {checkpoint_path}")
            
        except Exception as e:
            error_count += 1
            print(f"\n Error at sample {i}: {type(e).__name__}: {str(e)}")
            if error_count > 10:
                print("\n Too many errors, stopping...")
                break
            
            # Force cleanup after error
            mem_manager.force_cleanup()

    # Final cleanup
    mem_manager.force_cleanup()

    # Calculate and save results
    print(f"\n{'='*80}")
    print("Results Summary")
    print(f"{'='*80}")
    
    successful = num_samples - error_count
    print(f"Successful samples: {successful}/{num_samples}")
    print(f"Failed samples: {error_count}")
    print(f"Final GPU memory: {mem_manager.get_memory_info()}")
    
    summary = {
        'meta': {
            'game': args.game_name,
            'fold': args.fold,
            'total_samples': num_samples,
            'successful_samples': successful,
            'errors': error_count,
            'seed': args.seed,
            'steps': args.steps
        },
        'results': {}
    }
    
    # Calculate AUC for all combinations
    for t_type in ['target', 'predicted']:
        summary['results'][t_type] = {}
        print(f"\n--- {t_type.upper()} CLASS ---")
        
        for mode in ['positive', 'negative']:
            summary['results'][t_type][mode] = {}
            
            for name in interpreters:
                if len(results[name][t_type][mode]) > 0:
                    avg = np.mean(np.array(results[name][t_type][mode]), axis=0)
                    auc = float(np.mean(avg))
                    summary['results'][t_type][mode][name] = {
                        'auc': auc,
                        'curve': avg.tolist()
                    }
                    print(f"  [{mode.upper():8s}] {name:15s}: AUC = {auc:.4f}")

    # Save final results
    final_json = os.path.join(args.output_dir, f'summary_fold{args.fold}.json')
    with open(final_json, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✓ Final results saved to {final_json}")
    
    # Plot summary comparison
    plot_summary_comparison(summary, args.output_dir, args.fold)
    
    print(f"\n{'='*80}")
    print("Evaluation Complete!")
    print(f"{'='*80}")


def plot_summary_comparison(summary, output_dir, fold):
    """Plot comparison of all results"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Complete Evaluation - Fold {fold}', fontsize=16, fontweight='bold')
    
    test_types = [('target', 'Target Class'), ('predicted', 'Predicted Class')]
    modes = [('positive', 'Positive Perturbation'), ('negative', 'Negative Perturbation')]
    
    for i, (t_type, t_name) in enumerate(test_types):
        for j, (mode, m_name) in enumerate(modes):
            ax = axes[i, j]
            
            data = summary['results'][t_type][mode]
            
            for name, result in data.items():
                levels = np.linspace(0.1, 0.9, len(result['curve']))
                ax.plot(levels, result['curve'], marker='o', label=f"{name} (AUC: {result['auc']:.3f})")
            
            ax.set_xlabel('Fraction Perturbed')
            ax.set_ylabel('Confidence')
            ax.set_title(f'{t_name} - {m_name}')
            ax.grid(True, alpha=0.3)
            ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'summary_fold{fold}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Summary plot saved")

if __name__ == '__main__':
    main()