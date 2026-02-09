import os
import json
import time
import random
import argparse
import re
import collections
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from torch import nn
from PIL import Image
import torchvision.models as models
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

import matplotlib
matplotlib.use('Agg')


class ResNetLSTMModel(nn.Module):
    def __init__(self, num_classes, backbone_name='resnet50', lstm_hidden_size=256, lstm_layers=1):
        super(ResNetLSTMModel, self).__init__()
        
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


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_resnet_layer_name(name):
    parts = name.split('.')
    if parts[0] == 'backbone':
        idx = parts[1]
        if idx in ['0', '1']: return 'stem'
        if idx == '4': return 'layer1' 
        if idx == '5': return 'layer2' 
        if idx == '6': return 'layer3' 
        if idx == '7': return 'layer4' 
    return None 

def natural_sort_key(s):
    order_map = {
        'stem': 0, 
        'layer1': 2, 
        'layer2': 3, 
        'layer3': 4, 
        'layer4': 5
    }
    return order_map.get(s, 99)



def compare_models(original_model, fine_tuned_model, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    ACADEMIC_MAP = {
        'stem':   'Stem (conv1 + bn1)',
        'layer1': 'Stage 1 (conv2_x)',
        'layer2': 'Stage 2 (conv3_x)',
        'layer3': 'Stage 3 (conv4_x)',
        'layer4': 'Stage 4 (conv5_x)'
    }

    layer_params_original = defaultdict(list)
    layer_params_finetuned = defaultdict(list)
    
    target_prefix = "backbone"

    params1 = dict(original_model.named_parameters())
    params2 = dict(fine_tuned_model.named_parameters())

    for name, param1 in params1.items():
        if name not in params2: continue
        if not name.startswith(target_prefix): continue

        readable_name = get_resnet_layer_name(name)
        if readable_name:
            layer_params_original[readable_name].append(param1.data.view(-1))
            layer_params_finetuned[readable_name].append(params2[name].data.view(-1))

    comparison_results = {}
    
    sorted_layers = sorted(layer_params_original.keys(), key=natural_sort_key)

    for layer in sorted_layers:
        if not layer_params_original[layer]: continue
        
        p1 = torch.cat(layer_params_original[layer])
        p2 = torch.cat(layer_params_finetuned[layer])
        
        diff = p1 - p2
        
        # Calculate Metrics
        param_count = p1.numel()
        param_l1 = diff.abs().sum().item()
        param_l2 = torch.norm(diff).item()
        
        # Avg L2 = L2 / sqrt(N)
        param_avg_l2 = param_l2 / np.sqrt(param_count) if param_count > 0 else 0.0
        
        param_cos = nn.functional.cosine_similarity(p1, p2, dim=0).item()
        
        comparison_results[layer] = {
            'param_l1_diff': param_l1, 
            'param_l2_diff': param_l2,
            'param_avg_l2_diff': param_avg_l2,
            'param_cosine_similarity': param_cos
        }

    json_path = os.path.join(output_dir, "comparison_results_raw.json")
    with open(json_path, 'w') as f:
        json.dump(comparison_results, f, indent=2)
    
    layers = list(comparison_results.keys())
    
    display_labels = [ACADEMIC_MAP.get(l, l) for l in layers]
    x_indices = range(len(layers))
    
    def plot_metric(metric_name, data_dict, ylabel, file_name, ylim=None):
        values = [data_dict[l][metric_name] for l in layers]
        
        plt.figure(figsize=(11, 7))
        
        plt.plot(x_indices, values, marker='o', linestyle='-', linewidth=2, markersize=8)
        
        plt.xticks(ticks=x_indices, labels=display_labels, rotation=45, ha='right', fontsize=11)
        
        plt.title(f"{ylabel} per ResNet Block", fontsize=16, fontweight='bold')
        plt.ylabel(ylabel, fontsize=12)
        plt.xlabel("Network Stage (Paper Terminology)", fontsize=12)
        plt.grid(True, which="both", ls="--", alpha=0.7)
        if ylim: plt.ylim(*ylim)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{file_name}.png"), dpi=150)
        plt.close()

    plot_metric("param_l2_diff", comparison_results, "L2 Difference (Euclidean)", "param_l2_diff")
    plot_metric("param_avg_l2_diff", comparison_results, "Avg L2 (Normalized by Size)", "param_avg_l2_diff") 
    plot_metric("param_cosine_similarity", comparison_results, "Cosine Similarity", "param_cosine_similarity", ylim=(0.95, 1.001))


def main_generate_comparison(finetuned_model_path, output_dir, num_classes, backbone_name):

    if os.path.exists(os.path.join(output_dir, "comparison_results_raw.json")):
        return True, os.path.join(output_dir, "comparison_results_raw.json")

    print(f"\n  --- Generating comparison for: {os.path.basename(finetuned_model_path)} ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        original_model = ResNetLSTMModel(
            num_classes=num_classes, 
            backbone_name=backbone_name
        )
        original_model.to(device)
        print(f"    > Baseline loaded with ImageNet pretrained weights")
    except Exception as e:
        print(f"    > Error loading baseline: {e}")
        return False, None

    # 2. Load Fine-tuned Model
    if not os.path.exists(finetuned_model_path): 
        print(f"    > Fine-tuned model not found: {finetuned_model_path}")
        return False, None

    try:
        fine_tuned_model = ResNetLSTMModel(
            num_classes=num_classes, 
            backbone_name=backbone_name
        )
        state_dict = torch.load(finetuned_model_path, map_location=device)
        fine_tuned_model.load_state_dict(state_dict)
        fine_tuned_model.to(device)
        print(f"    > Fine-tuned model loaded successfully")
    except Exception as e:
        print(f"    > Error loading fine-tuned weights: {e}")
        return False, None

    # 3. Compare Models
    compare_models(original_model, fine_tuned_model, output_dir)
    
    del original_model
    del fine_tuned_model
    torch.cuda.empty_cache()
    
    return True, os.path.join(output_dir, "comparison_results_raw.json")


def load_l2_differences(filepath, metric_key="param_l2_diff"):
    if not os.path.exists(filepath): return None, None
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        l2_diffs = {k: v[metric_key] for k, v in data.items()}
        if not l2_diffs: return None, None
        max_l2_layer = max(l2_diffs, key=l2_diffs.get)
        return l2_diffs, max_l2_layer
    except: return None, None


def compare_and_save(file1, file2, spearman_output_dir, metric_key="param_l2_diff"):
    metric_analysis_dir = os.path.join(spearman_output_dir, f"analysis_{metric_key}")
    os.makedirs(metric_analysis_dir, exist_ok=True)

    l2_1, _ = load_l2_differences(file1, metric_key)
    l2_2, _ = load_l2_differences(file2, metric_key)

    if l2_1 is None or l2_2 is None: return False

    shared_keys = sorted(list(set(l2_1.keys()) & set(l2_2.keys())), key=natural_sort_key)
    if not shared_keys: return False

    values1 = [l2_1[k] for k in shared_keys]
    values2 = [l2_2[k] for k in shared_keys]

    if len(values1) < 2: return False

    rho, p = spearmanr(values1, values2)
    
    # Save Report
    match1 = re.search(r"(?P<epoch>\d+)epoch", file1)
    match2 = re.search(r"(?P<epoch>\d+)epoch", file2)
    e1_tag = f"e{match1.group('epoch')}" if match1 else "e_prev"
    e2_tag = f"e{match2.group('epoch')}" if match2 else "e_curr"
    
    print(f"    > [{metric_key}] {e1_tag} vs {e2_tag}: Rho={rho:.4f}")

    report = {
        "epoch_pair": f"{e1_tag}_vs_{e2_tag}",
        "spearman_rho": round(rho, 6),
        "p_value": round(p, 6),
        "metric": metric_key
    }
    
    with open(os.path.join(metric_analysis_dir, f"corr_{e1_tag}_vs_{e2_tag}.json"), 'w') as f:
        json.dump(report, f, indent=2)

    if rho > 0.8 and p < 0.05:
        _, max_layer = load_l2_differences(file2, metric_key)
        if max_layer:
            print(f"      -> Stability Reached. Selected: {max_layer}")
            return max_layer
    return None



def main(args):
    set_seed(42) 
    
    num_classes = 2 if args.binary_mode else 3
    
    norm_input_path = os.path.normpath(args.base_model_dir)
    path_parts = norm_input_path.split(os.sep)
    if len(path_parts) >= 3:
        sub_structure = os.path.join(*path_parts[-1:])
        game_name_log = path_parts[-1]
    else:
        sub_structure = os.path.basename(norm_input_path)
        game_name_log = sub_structure

    game_exploration_dir = os.path.join(args.output_base_dir, sub_structure)
    os.makedirs(game_exploration_dir, exist_ok=True)
    

    path_l2_dir = os.path.join(game_exploration_dir, "metric_l2")
    path_avg_dir = os.path.join(game_exploration_dir, "metric_avg_l2")
    os.makedirs(path_l2_dir, exist_ok=True)
    os.makedirs(path_avg_dir, exist_ok=True)

    print(f"--- Starting Automated Analysis ({args.backbone}) ---")
    print(f"Game:                 {game_name_log}")
    print(f"Output Base:          {game_exploration_dir}")

    schedule_l2 = {} 
    schedule_avg = {}

    fold_dirs = sorted(
        [d for d in os.listdir(args.base_model_dir) if d.startswith('fold_') and os.path.isdir(os.path.join(args.base_model_dir, d))],
        key=lambda x: int(re.search(r'(\d+)', x).group(1)) 
    )
    
    if not fold_dirs:
        print(f"Error: No 'fold_...' directories found in {args.base_model_dir}")
        return

    for fold_dir_name in fold_dirs:
        print(f"\n--- Processing {fold_dir_name} ---")
        
        fold_exploration_dir = os.path.join(game_exploration_dir, fold_dir_name)
        os.makedirs(fold_exploration_dir, exist_ok=True)
        
        checkpoints_dir = os.path.join(args.base_model_dir, fold_dir_name, "checkpoints")
        if not os.path.isdir(checkpoints_dir): continue

        epoch_models = [] 
        try:
            items = os.listdir(checkpoints_dir)
            for item in items:
                if item.startswith('checkpoint_epoch_') and item.endswith('.pt'):
                    match = re.search(r'(\d+)', item)
                    if match:
                        epoch_num = int(match.group(1))
                        epoch_models.append((epoch_num, os.path.join(checkpoints_dir, item)))
            epoch_models.sort(key=lambda x: x[0]) 
        except: continue

        if not epoch_models: continue
        print(f"  Found {len(epoch_models)} epochs.")

        default_l2, default_avg = None, None
        last_epoch_num, last_model_path = epoch_models[-1]
        last_epoch_out = os.path.join(fold_exploration_dir, f"{last_epoch_num}epoch")
        
        success_last, last_json_path = main_generate_comparison(
            last_model_path, last_epoch_out, num_classes, args.backbone
        )
        
        if success_last:
            _, default_l2 = load_l2_differences(last_json_path, "param_l2_diff")
            _, default_avg = load_l2_differences(last_json_path, "param_avg_l2_diff")
            print(f"    [Fallback] Last Epoch ({last_epoch_num}): L2='{default_l2}', Avg='{default_avg}'")
        
        generated_jsons = {last_epoch_num: last_json_path if success_last else None}
        
        # Init Choices
        chosen_l2 = [default_l2] if default_l2 else []
        chosen_avg = [default_avg] if default_avg else []
        
        stop_l2 = False
        stop_avg = False

        # --- Phase 2: Analysis ---
        for i in range(len(epoch_models) - 1):
            if stop_l2 and stop_avg: break

            curr_epoch, curr_path = epoch_models[i]
            next_epoch, next_path = epoch_models[i+1]

            if next_epoch != curr_epoch + 1: continue

            if curr_epoch not in generated_jsons:
                out_dir = os.path.join(fold_exploration_dir, f"{curr_epoch}epoch")
                ok, path = main_generate_comparison(curr_path, out_dir, num_classes, args.backbone)
                generated_jsons[curr_epoch] = path if ok else None
            
            if next_epoch not in generated_jsons:
                out_dir = os.path.join(fold_exploration_dir, f"{next_epoch}epoch")
                ok, path = main_generate_comparison(next_path, out_dir, num_classes, args.backbone)
                generated_jsons[next_epoch] = path if ok else None

            path1 = generated_jsons[curr_epoch]
            path2 = generated_jsons[next_epoch]

            if path1 and path2:
                # 1. Metric L2
                if not stop_l2:
                    layer = compare_and_save(path1, path2, fold_exploration_dir, "param_l2_diff")
                    if layer:
                        chosen_l2 = [layer]
                        stop_l2 = True
                
                # 2. Metric Avg L2
                if not stop_avg:
                    layer = compare_and_save(path1, path2, fold_exploration_dir, "param_avg_l2_diff")
                    if layer:
                        chosen_avg = [layer]
                        stop_avg = True
        
        schedule_l2[fold_dir_name] = chosen_l2
        schedule_avg[fold_dir_name] = chosen_avg

    # Save Results
    with open(os.path.join(path_l2_dir, 'finetuning_schedule.json'), 'w') as f:
        json.dump(schedule_l2, f, indent=2)
    with open(os.path.join(path_avg_dir, 'finetuning_schedule.json'), 'w') as f:
        json.dump(schedule_avg, f, indent=2)
    
    print(f"\n--- Analysis Complete ---")
    print(f"L2 Schedule:  {path_l2_dir}")
    print(f"Avg Schedule: {path_avg_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ResNet Exploration")
    
    parser.add_argument("--base_model_dir", type=str, required=True, 
                        help="Input path (e.g. Results/10_percent_test/resnet_lstm/solid)")
    parser.add_argument("--output_base_dir", type=str, default="Results/new/exploration_resnet", 
                        help="Root for output.")
    
    parser.add_argument("--backbone", type=str, default="resnet50", choices=['resnet18', 'resnet34', 'resnet50', 'resnet101'],
                        help="Choose backbone architecture (must match what was trained).")
    
    parser.add_argument('--binary_mode', action='store_true', help="Set True if checkpoints are 2-class")
    
    args = parser.parse_args()
    main(args)