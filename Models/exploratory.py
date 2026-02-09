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
from transformers import VivitImageProcessor, VivitForVideoClassification
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

import matplotlib
matplotlib.use('Agg')


class_labels = ['down', 'same', 'up']
label2id_global = {label: i for i, label in enumerate(class_labels)}
id2label_global = {i: label for label, i in label2id_global.items()}


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def compare_models(original_model, fine_tuned_model, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    def extract_layer_name(name):
        parts = name.split(".")
        if "encoder" in parts and "layer" in parts:
            i = parts.index("layer")
            return ".".join(parts[:i+2])
        return "embeddings"

    layer_params_original = defaultdict(list)
    layer_params_finetuned = defaultdict(list)

    for (name1, param1), (name2, param2) in zip(original_model.named_parameters(), fine_tuned_model.named_parameters()):
        if name1 != name2 or 'classifier' in name1:
            continue
        layer = extract_layer_name(name1)
        layer_params_original[layer].append(param1.data.view(-1))
        layer_params_finetuned[layer].append(param2.data.view(-1))

    comparison_results = {}
    for layer in layer_params_original:
        p1 = torch.cat(layer_params_original[layer])
        p2 = torch.cat(layer_params_finetuned[layer])
        
        diff = p1 - p2
        
        # Extended metrics
        param_count = p1.numel()
        param_l1 = diff.abs().sum().item()
        param_l2 = torch.norm(diff).item()
        param_avg_l2 = param_l2 / np.sqrt(param_count) if param_count > 0 else 0.0
        param_cos = nn.functional.cosine_similarity(p1, p2, dim=0).item()
        
        comparison_results[layer] = {
            'param_l1_diff': param_l1, 
            'param_l2_diff': param_l2, 
            'param_avg_l2_diff': param_avg_l2,
            'param_cosine_similarity': param_cos
        }
    
    comparison_results_simplified = {
        name.replace("vivit.", ""): stats for name, stats in comparison_results.items()
    }
    
    json_path = os.path.join(output_dir, "comparison_results_raw.json")
    with open(json_path, 'w') as f:
        json.dump(comparison_results, f, indent=2)

    layers = list(comparison_results_simplified.keys())
    
    def plot_metric(metric_name, values, ylabel, file_name, log=False, ylim=None):
        plt.figure(figsize=(12, 6))
        if log: values = np.log10(np.array(values) + 1e-9)
        plt.plot(layers, values, marker='o', linestyle='-')
        plt.xticks(rotation=45, ha='right')
        plt.title(metric_name, fontsize=16)
        plt.ylabel(ylabel, fontsize=12)
        plt.xlabel("Layer", fontsize=12)
        plt.grid(True, which="both", ls="--")
        if ylim: plt.ylim(*ylim)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{file_name}.png"))
        plt.close()

    plot_metric("L2 Parameter Difference", [comparison_results_simplified[l]['param_l2_diff'] for l in layers], "L2 Difference", "param_l2_diff")
    plot_metric("Average L2 Difference (Per Param)", [comparison_results_simplified[l]['param_avg_l2_diff'] for l in layers], "Avg L2 Difference", "param_avg_l2_diff")
    plot_metric("Cosine Similarity", [comparison_results_simplified[l]['param_cosine_similarity'] for l in layers], "Cosine Similarity", "param_cosine_similarity", ylim=(0.95, 1.001))

def main_generate_comparison(finetuned_model_path, model_ckpt, output_dir, label2id, id2label):
    expected_json = os.path.join(output_dir, "comparison_results_raw.json")
    if os.path.exists(expected_json):
        return True, expected_json

    print(f"\n  --- Generating comparison for: {finetuned_model_path} ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        original_model = VivitForVideoClassification.from_pretrained(
            model_ckpt, label2id=label2id, id2label=id2label, ignore_mismatched_sizes=True
        )
        original_model.to(device)
    except:
        print("  > Error loading base model.")
        return False, None

    if not os.path.exists(finetuned_model_path):
        return False, None

    try:
        if os.path.isdir(finetuned_model_path):
            fine_tuned_model = VivitForVideoClassification.from_pretrained(finetuned_model_path)
        else:
            fine_tuned_model = VivitForVideoClassification.from_pretrained(
                model_ckpt, label2id=label2id, id2label=id2label, ignore_mismatched_sizes=True
            )
            state_dict = torch.load(finetuned_model_path, map_location=device)
            fine_tuned_model.load_state_dict(state_dict)
        fine_tuned_model.to(device)
    except Exception as e:
        print(f"  > Error loading fine-tuned model: {e}")
        return False, None

    compare_models(original_model, fine_tuned_model, output_dir)
    del original_model
    del fine_tuned_model
    torch.cuda.empty_cache()
    
    return True, os.path.join(output_dir, "comparison_results_raw.json")


def natural_sort_key(s):
    match = re.match(r'^(.*?)(\d+)$', s)
    if match: return (match.group(1), int(match.group(2)))
    return (s, -1)

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

    print(f"\n  > Comparing '{os.path.basename(os.path.dirname(file1))}' AND '{os.path.basename(os.path.dirname(file2))}' [{metric_key}]")

    l2_1, _ = load_l2_differences(file1, metric_key)
    l2_2, _ = load_l2_differences(file2, metric_key)

    if l2_1 is None or l2_2 is None: return None

    shared_keys = sorted(list(set(l2_1.keys()) & set(l2_2.keys())), key=natural_sort_key)
    values1 = [l2_1[k] for k in shared_keys]
    values2 = [l2_2[k] for k in shared_keys]

    # Spearman
    rho, p = spearmanr(values1, values2)
    print(f"    Spearman rho={rho:.4f}, p={p:.4f}")

    #  Save JSON 
    match1 = re.search(r"(?P<epoch>\d+)epoch", file1)
    match2 = re.search(r"(?P<epoch>\d+)epoch", file2)
    e1_tag = f"e{match1.group('epoch')}" if match1 else "e_prev"
    e2_tag = f"e{match2.group('epoch')}" if match2 else "e_curr"

    filename = f"{e1_tag}_vs_{e2_tag}_{metric_key}_corr.json"
    with open(os.path.join(metric_analysis_dir, filename), 'w') as f:
        json.dump({"rho": rho, "p": p}, f)
    
    if rho > 0.8 and p < 0.05:
        print(f"    Correlation HIGH (rho > 0.8). Logic triggered.")
        _, max_layer_2 = load_l2_differences(file2, metric_key)  
        print(f"    Selected Layer (from {e2_tag}): {max_layer_2}")
        return max_layer_2
    else:
        return None


def main(args):
    set_seed(42)
    
    if args.binary_mode:
        class_labels = ['down', 'up']
    else:
        class_labels = ['down', 'same', 'up']
    label2id = {label: i for i, label in enumerate(class_labels)}
    id2label = {i: label for label, i in label2id.items()}

    # ---------------------------------------------------------
    norm_input_path = os.path.normpath(args.base_model_dir)
    path_parts = norm_input_path.split(os.sep)
    
    # Path Analysis
    if len(path_parts) >= 3:
        sub_structure = os.path.join(*path_parts[-1:])
        game_name_log = path_parts[-1]
    else:
        sub_structure = os.path.basename(norm_input_path)
        game_name_log = sub_structure

    game_exploration_dir = os.path.join(args.output_base_dir, sub_structure)
    os.makedirs(game_exploration_dir, exist_ok=True)
    # ---------------------------------------------------------
    
    print(f"--- Starting Analysis for {game_name_log} ---")
    print(f"--- Mode: {'Binary' if args.binary_mode else 'Multi-class'} ---")
    print(f"--- Output Dir: {game_exploration_dir} ---")

    schedule_l2 = {}     
    schedule_avg = {} 

    fold_dirs = sorted(
        [d for d in os.listdir(args.base_model_dir) if d.startswith('fold_') and os.path.isdir(os.path.join(args.base_model_dir, d))],
        key=lambda x: int(re.search(r'(\d+)', x).group(1))
    )
    
    for fold_dir_name in fold_dirs:
        print(f"\n--- Processing {fold_dir_name} ---")
        fold_exploration_dir = os.path.join(game_exploration_dir, fold_dir_name)
        os.makedirs(fold_exploration_dir, exist_ok=True)
        
        checkpoints_dir = os.path.join(args.base_model_dir, fold_dir_name, "checkpoints")
        if not os.path.isdir(checkpoints_dir): continue

        epoch_models = []
        for item in os.listdir(checkpoints_dir):
            if item.startswith('checkpoint_epoch_'):
                match = re.search(r'(\d+)', item)
                if match: epoch_models.append((int(match.group(1)), os.path.join(checkpoints_dir, item)))
        epoch_models.sort(key=lambda x: x[0])
        
        if not epoch_models: continue

        # Fallback Setup (Last Epoch)
        default_l2, default_avg = None, None
        last_ep, last_path = epoch_models[-1]
        last_out = os.path.join(fold_exploration_dir, f"{last_ep}epoch")
        
        ok, last_json = main_generate_comparison(last_path, args.model_ckpt, last_out, label2id, id2label)
        if ok:
            _, default_l2 = load_l2_differences(last_json, "param_l2_diff")
            _, default_avg = load_l2_differences(last_json, "param_avg_l2_diff")
        
        generated_jsons = {last_ep: last_json if ok else None}
        
        chosen_l2 = [default_l2] if default_l2 else []
        chosen_avg = [default_avg] if default_avg else []
        
        stop_l2, stop_avg = False, False

        # Analysis Loop
        for i in range(len(epoch_models) - 1):
            if stop_l2 and stop_avg: break

            curr, curr_path = epoch_models[i]
            next_e, next_path = epoch_models[i+1]
            if next_e != curr + 1: continue

            if curr not in generated_jsons:
                out = os.path.join(fold_exploration_dir, f"{curr}epoch")
                ok, p = main_generate_comparison(curr_path, args.model_ckpt, out, label2id, id2label)
                generated_jsons[curr] = p if ok else None
            
            if next_e not in generated_jsons:
                out = os.path.join(fold_exploration_dir, f"{next_e}epoch")
                ok, p = main_generate_comparison(next_path, args.model_ckpt, out, label2id, id2label)
                generated_jsons[next_e] = p if ok else None

            p1, p2 = generated_jsons[curr], generated_jsons[next_e]

            if p1 and p2:
                # Metric A: L2
                if not stop_l2:
                    layer = compare_and_save(p1, p2, fold_exploration_dir, "param_l2_diff")
                    if layer:
                        print(f"    -> [L2] Stability Reached. Selected: {layer}")
                        chosen_l2 = [layer]
                        stop_l2 = True

                # Metric B: Avg L2
                if not stop_avg:
                    layer = compare_and_save(p1, p2, fold_exploration_dir, "param_avg_l2_diff")
                    if layer:
                        print(f"    -> [Avg] Stability Reached. Selected: {layer}")
                        chosen_avg = [layer]
                        stop_avg = True
        
        schedule_l2[fold_dir_name] = chosen_l2
        schedule_avg[fold_dir_name] = chosen_avg

    # Save Schedules
    path_l2 = os.path.join(game_exploration_dir, "metric_l2")
    path_avg = os.path.join(game_exploration_dir, "metric_avg_l2")
    os.makedirs(path_l2, exist_ok=True)
    os.makedirs(path_avg, exist_ok=True)
    
    with open(os.path.join(path_l2, "finetuning_schedule.json"), 'w') as f:
        json.dump(schedule_l2, f, indent=2)
    with open(os.path.join(path_avg, "finetuning_schedule.json"), 'w') as f:
        json.dump(schedule_avg, f, indent=2)

    print(f"\nDone. Results in:\n  {path_l2}\n  {path_avg}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_dir", type=str, required=True, help="Input directory (e.g., .../full_finetune/solid)")
    parser.add_argument("--output_base_dir", type=str, default="Results/new/exploration")
    parser.add_argument("--model_ckpt", type=str, default="google/vivit-b-16x2-kinetics400")
    parser.add_argument('--binary_mode', action='store_true')
    args = parser.parse_args()
    main(args)