import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import pandas as pd
import json
import argparse
import re
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from transformers import VivitImageProcessor, VivitForVideoClassification
from collections import defaultdict

# Define class labels
class_labels = ['down', 'same', 'up']
label2id = {label: i for i, label in enumerate(class_labels)}
id2label = {i: label for label, i in label2id.items()}

def get_file_name_and_parent_folder(file_path, game_name):
    """
    Extract file name, parent folder, player ID, and session ID from a file path.
    """
    file_path = os.path.normpath(file_path)
    file_paths = file_path.split(os.sep)
    file_name, _ = os.path.splitext(os.path.basename(file_path))
    parent_folder = os.path.dirname(file_path)
    video_name = file_paths[-2]
    player_id, session_id = video_name.split(f'_{game_name}_')
    return file_name, parent_folder, player_id, session_id

class MyCSVDataset(Dataset):
    """
    Custom Dataset class for loading data from CSV files.
    """
    def __init__(self, csv_file, csv_file_2, game_name, image_processor):
        self.data = pd.read_csv(csv_file)
        self.gf = pd.read_csv(csv_file_2)
        self.game_name = game_name
        self.image_processor = image_processor

        # Clean up data
        columns_to_drop = [col for col in self.gf.columns if "control" in col and
                           col != "[control]player_id" and
                           col != "[control]session_id"]
        self.gf = self.gf.drop(columns=columns_to_drop)
        self.gf = self.gf.drop(columns=['[output]arousal'])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        clip_path = sample['start_frame']
        file_name, parent_folder, player_id, session_id = get_file_name_and_parent_folder(clip_path, self.game_name)
        file_name = int(file_name)

        start = int(sample['start_time']) * 4
        end = start + 24

        # Convert game vector to tensor
        game_vector = self.gf[(self.gf['[control]player_id'] == player_id) & (self.gf['[control]session_id'] == session_id)]
        game_vector = game_vector.drop(columns=['[control]player_id', '[control]session_id'])
        game_vector = game_vector.iloc[start:end]

        for col in game_vector.columns:
            if '[string]' in col:
                game_vector[col] = game_vector[col].apply(lambda x: 1 if isinstance(x, str) and x != '' else 0)
        
        game_vector = game_vector.fillna(0).infer_objects(copy=False)
        game_array = np.array(game_vector.values)
        game_tensor = torch.from_numpy(game_array).float()

        # Load and process frames
        frames = []
        for i in range(32):
            frame_num_str = str(file_name + i).zfill(4)
            frame_path = os.path.join("../Dataset/", parent_folder, f"{frame_num_str}.png")
            frame_path = os.path.normpath(frame_path)
            
            try:
                frame = Image.open(frame_path).convert('RGB')
                frames.append(frame)
            except FileNotFoundError:
                print(f"Warning: Frame not found at {frame_path}. Skipping.")
                continue
        
        if not frames:
            return None

        inputs = self.image_processor(list(frames), return_tensors="pt")
        pixel_values = inputs['pixel_values'].squeeze(0)
        inputs['pixel_values'] = pixel_values

        # Process label
        label = sample['arousal_change']
        label_id = label2id[label]
        label_tensor = torch.LongTensor([label_id])
        
        inputs['label'] = label_tensor
        inputs['game_tensor'] = game_tensor

        return inputs

def compare_models(original_model, fine_tuned_model, output_dir):
    """
    Compare the weights of the original and fine-tuned models.
    """
    # Create a directory to save the results
    os.makedirs(output_dir, exist_ok=True)

    # Function to extract layer name
    def extract_layer_name(name):
        parts = name.split(".")
        if "encoder" in parts and "layer" in parts:
            i = parts.index("layer")
            return ".".join(parts[:i+2])
        return "embeddings"

    # Aggregate parameters for each layer
    layer_params_original = defaultdict(list)
    layer_params_finetuned = defaultdict(list)

    for (name1, param1), (name2, param2) in zip(original_model.named_parameters(), fine_tuned_model.named_parameters()):
        if name1 != name2 or 'classifier' in name1:
            continue

        layer = extract_layer_name(name1)
        layer_params_original[layer].append(param1.data.view(-1))
        layer_params_finetuned[layer].append(param2.data.view(-1))

    # Concatenate and compute differences
    comparison_results = {}
    for layer in layer_params_original:
        p1 = torch.cat(layer_params_original[layer])
        p2 = torch.cat(layer_params_finetuned[layer])
        diff = p1 - p2

        param_l1 = diff.abs().sum().item()
        param_l2 = torch.norm(diff).item()
        param_cos = nn.functional.cosine_similarity(p1, p2, dim=0).item()

        result = {
            'param_l1_diff': param_l1,
            'param_l2_diff': param_l2,
            'param_cosine_similarity': param_cos
        }
        comparison_results[layer] = result
    
    # Simplify key names
    comparison_results_simplified = {
        name.replace("vivit.", ""): stats for name, stats in comparison_results.items()
    }
    
    # Print results
    for name, stats in comparison_results_simplified.items():
        print(f"\nParameter: {name}")
        for key, value in stats.items():
            print(f"  {key}: {value:.6f}")

    # Save raw comparison results to a JSON file
    json_path = os.path.join(output_dir, "comparison_results_raw.json")
    with open(json_path, 'w') as f:
        json.dump(comparison_results, f, indent=2)
    print(f"Saved raw comparison results to: {json_path}")

    # Visualization
    layers = list(comparison_results_simplified.keys())

    def plot_metric(metric_name, values, ylabel, file_name, log=False, ylim=None):
        plt.figure(figsize=(12, 6))
        if log:
            values = np.log10(np.array(values) + 1e-9)
        plt.plot(layers, values, marker='o', linestyle='-')
        plt.xticks(rotation=45, ha='right')
        plt.title(metric_name, fontsize=16)
        plt.ylabel(ylabel, fontsize=12)
        plt.xlabel("Layer", fontsize=12)
        plt.grid(True, which="both", ls="--")
        if ylim:
            plt.ylim(*ylim)
        plt.tight_layout()
        save_path = os.path.join(output_dir, f"{file_name}.png")
        plt.savefig(save_path)
        plt.close()
        print(f"Saved whole-layer comparison plot to {save_path}")

    plot_metric("L2 Parameter Difference", [comparison_results_simplified[l]['param_l2_diff'] for l in layers], "L2 Difference", "param_l2_diff")
    plot_metric("Cosine Similarity of Parameters", [comparison_results_simplified[l]['param_cosine_similarity'] for l in layers], "Cosine Similarity", "param_cosine_similarity", ylim=(0.95, 1.001))


def main():
    parser = argparse.ArgumentParser(description="Compare weights of a pre-trained and a fine-tuned ViViT model.")
    parser.add_argument("--game_name", type=str, required=True, help="Name of the game (e.g., 'endles') for dataset loading.")
    parser.add_argument("--finetuned_model_path", type=str, required=True, help="Path to the fine-tuned model. Can be a directory (Hugging Face format) or a .pt file (state_dict).")
    parser.add_argument("--model_ckpt", type=str, default="google/vivit-b-16x2-kinetics400", help="Base model checkpoint from Hugging Face.")
    parser.add_argument("--data_dir", type=str, default="../Dataset", help="Directory containing the dataset files.")
    parser.add_argument("--epoch", type=int, default=1, help="Epoch number to use for the output directory name. Overrides path parsing.")
    
    args = parser.parse_args()

    # Check if data files exist
    csv_file = os.path.join(args.data_dir, f'new_{args.game_name}.csv')
    csv_file_2 = os.path.join(args.data_dir, 'clean_data.csv')
    if not os.path.exists(csv_file) or not os.path.exists(csv_file_2):
        print(f"Error: Dataset CSV files not found in {args.data_dir}")
        return

    # --- Determine the output directory path ---
    epoch_num = args.epoch
    game_name_for_path = args.game_name

    # If epoch is not provided via argument, try to extract it from the model path
    if epoch_num is None:
        path_pattern = re.compile(r"Results[\\/]full_finetuning_results[\\/](?P<game>.+?)[\\/]checkpoints[\\/]checkpoint_epoch_(?P<epoch>\d+)")
        match = path_pattern.search(args.finetuned_model_path)
        if match:
            game_name_for_path = match.group('game') # Use more specific game name from path
            epoch_num = match.group('epoch')
            print(f"Epoch number extracted from path: {epoch_num}")

    if epoch_num is not None:
        # An epoch number was either provided directly or extracted from the path
        output_dir = os.path.join("ExploratoryStage", game_name_for_path, f"{epoch_num}epoch")
        print(f"Custom output path rule applied. Setting output directory to: {output_dir}")
    else:
        # Fallback if no epoch is specified or found in the path
        output_folder_id = os.path.normpath(args.finetuned_model_path).replace(os.sep, '_')
        output_folder_id = os.path.splitext(output_folder_id)[0].lstrip('._')
        output_dir = f"comparison_weight_grad/{output_folder_id}"
        print(f"No epoch specified or found. Using default output path rule. Setting output directory to: {output_dir}")


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the original model
    print("Loading original model...")
    original_model = VivitForVideoClassification.from_pretrained(
        args.model_ckpt,
        label2id=label2id,
        id2label=id2label,
        ignore_mismatched_sizes=True,
    )
    original_model.to(device)

    # Load the fine-tuned model
    print(f"Loading fine-tuned model from {args.finetuned_model_path}...")
    if not os.path.exists(args.finetuned_model_path):
        print(f"Error: Fine-tuned model path does not exist: {args.finetuned_model_path}")
        return

    try:
        if os.path.isdir(args.finetuned_model_path):
            # Load in Hugging Face format
            fine_tuned_model = VivitForVideoClassification.from_pretrained(args.finetuned_model_path)
        elif args.finetuned_model_path.endswith('.pt'):
            # Load .pt file (state_dict)
            fine_tuned_model = VivitForVideoClassification.from_pretrained(
                args.model_ckpt,
                label2id=label2id,
                id2label=id2label,
                ignore_mismatched_sizes=True,
            )
            state_dict = torch.load(args.finetuned_model_path, map_location=device)
            fine_tuned_model.load_state_dict(state_dict)
        else:
            print(f"Error: Unsupported model format at {args.finetuned_model_path}. Please provide a directory or a .pt file.")
            return
            
        fine_tuned_model.to(device)
        print("Models loaded successfully.")
    except Exception as e:
        print(f"Error loading fine-tuned model: {e}")
        return

    # Compare models
    print("Comparing model weights...")
    compare_models(original_model, fine_tuned_model, output_dir)
    print("Comparison finished.")

if __name__ == "__main__":
    main()