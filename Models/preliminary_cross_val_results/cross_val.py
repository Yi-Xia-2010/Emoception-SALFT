import os
import json
import time
import random
import argparse
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from transformers import VivitImageProcessor, VivitForVideoClassification, AdamW

# Set a seed for reproducibility
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class MyCSVDataset(Dataset):
    def __init__(self, csv_file, csv_file_2, game_name, image_processor, label2id, root_dir="../Dataset/"):
        self.data = pd.read_csv(csv_file)
        self.gf = pd.read_csv(csv_file_2) 
        self.game_name = game_name
        self.image_processor = image_processor
        self.label2id = label2id
        self.root_dir = root_dir
        
        columns_to_drop = [col for col in self.gf.columns if "control" in col and 
                                 col != "[control]player_id" and 
                                 col != "[control]session_id"]
        self.gf = self.gf.drop(columns=columns_to_drop)
        self.gf = self.gf.drop(columns=['[output]arousal'])

    def __len__(self):
        return len(self.data)

    def _get_file_info(self, file_path):
        file_path = os.path.normpath(file_path)
        file_paths = file_path.split(os.sep)
        video_name = file_paths[-2] 
        player_id, session_id = video_name.split(f'_{self.game_name}_')
        return player_id, session_id

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        clip_path = sample['start_frame']
        
        # Extract metadata from file path
        # file_name_str is the frame number (e.g., "0001")
        file_name_str = os.path.splitext(os.path.basename(clip_path))[0]
        parent_folder = os.path.dirname(clip_path)
        # player_id, session_id = self._get_file_info(clip_path) # Not used in current model input

        # Load and process video frames
        frames = []
        file_name_int = int(file_name_str)
        for i in range(32): # Load 32 consecutive frames
            frame_path = os.path.join(self.root_dir, parent_folder, f"{file_name_int + i:04d}.png")
            try:
                frame = Image.open(frame_path).convert('RGB')
                frames.append(frame)
            except FileNotFoundError:
                print(f"Warning: Frame not found at {frame_path}. Skipping clip.")
                # If a frame is missing, break and mark this sample as invalid
                frames = [] # Clear frames to indicate invalid sample
                break 

        if not frames or len(frames) < 32: # Ensure we have enough frames
             # This sample is invalid, return None and handle in collate_fn
             return None

        # Process video frames using VivitImageProcessor
        inputs = self.image_processor(list(frames), return_tensors="pt")
        inputs['pixel_values'] = inputs['pixel_values'].squeeze(0) # Remove batch dimension added by processor

        # Process label
        label_str = sample['arousal_change']
        label_id = self.label2id[label_str]
        inputs['label'] = torch.LongTensor([label_id]) # Ensure label is a LongTensor for classification loss
        
        return inputs

# Custom collate_fn to handle None values from dataset (e.g., if frames are missing)
def custom_collate_fn(batch):
    batch = [item for item in batch if item is not None] # Filter out None samples
    if not batch:
        return None # Return None if batch is empty after filtering

    # Stack pixel values and labels
    pixel_values = torch.stack([item['pixel_values'] for item in batch])
    labels = torch.stack([item['label'] for item in batch])
    
    return {
        'pixel_values': pixel_values,
        'label': labels
    }


def evaluate(model, dataloader, device, case_name, epoch, label2id, id2label):
    """Evaluates the model and returns performance metrics."""
    model.eval()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for batch in dataloader:
            if batch is None: # Skip empty batches from custom_collate_fn
                continue
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["label"].to(device)
            
            outputs = model(pixel_values=pixel_values)
            logits = outputs.logits
            predicted_ids = logits.argmax(-1)
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted_ids.cpu().numpy())

    # Ensure all_labels and all_preds are 1D arrays
    all_labels = np.array(all_labels).flatten()
    all_preds = np.array(all_preds).flatten()
    
    # Handle cases where all_labels or all_preds might be empty (e.g., if dataloader was empty)
    if len(all_labels) == 0:
        print(f"[Epoch {epoch} - {case_name}] No samples to evaluate. Returning zero metrics.")
        return {
            "epoch": epoch,
            "case_name": case_name,
            "accuracy": 0.0,
            "f1_weighted": 0.0,
            "precision_weighted": 0.0,
            "recall_weighted": 0.0,
            "f1_micro": 0.0,
            "f1_macro": 0.0,
            "class_metrics": {}
        }

    # Calculate metrics
    accuracy = (all_preds == all_labels).mean()
    f1_weighted = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    print(f"[Epoch {epoch} - {case_name}] Accuracy: {accuracy:.4f}, Weighted F1: {f1_weighted:.4f}")
    
    results = {
        "epoch": epoch,
        "case_name": case_name,
        "accuracy": accuracy,
        "f1_weighted": f1_weighted,
        "precision_weighted": precision_score(all_labels, all_preds, average='weighted', zero_division=0),
        "recall_weighted": recall_score(all_labels, all_preds, average='weighted', zero_division=0),
        "f1_micro": f1_score(all_labels, all_preds, average='micro', zero_division=0),
        "f1_macro": f1_score(all_labels, all_preds, average='macro', zero_division=0),
        "class_metrics": {}
    }
    
    # Per-class metrics
    labels_present = sorted(list(set(all_labels)))
    prec_per_class = precision_score(all_labels, all_preds, average=None, labels=labels_present, zero_division=0)
    rec_per_class = recall_score(all_labels, all_preds, average=None, labels=labels_present, zero_division=0)
    f1_per_class = f1_score(all_labels, all_preds, average=None, labels=labels_present, zero_division=0)

    for i, label_id in enumerate(labels_present):
        class_name = id2label[label_id]
        results["class_metrics"][class_name] = {
            "precision": prec_per_class[i],
            "recall": rec_per_class[i],
            "f1_score": f1_per_class[i]
        }
        
    return results

def save_results_to_json(filepath, data):
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def main(args):
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    class_labels = ['down', 'same', 'up']
    label2id = {label: i for i, label in enumerate(class_labels)}
    id2label = {i: label for label, i in label2id.items()}
    
    # Define output directories for this specific experiment type
    output_base_dir = args.output_dir
    os.makedirs(output_base_dir, exist_ok=True)
    
    log_dir = os.path.join(output_base_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    model_ckpt = "google/vivit-b-16x2-kinetics400"
    image_processor = VivitImageProcessor.from_pretrained(model_ckpt)
    
    csv_file = f'../Dataset/new_{args.game_name}.csv'
    csv_file_2 = "../Dataset/clean_data.csv"
    dataset = MyCSVDataset(csv_file, csv_file_2, args.game_name, image_processor, label2id)

    # Split dataset
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(42)
    )

    # Use custom_collate_fn to handle potential None values from dataset
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=custom_collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, collate_fn=custom_collate_fn) # Use batch_size for val/test too
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=custom_collate_fn)

    # Determine trainable sub-modules (excluding the classifier, which is always trained)
    # If specific layers are provided, use them. Otherwise, determine all possible layers.
    if args.finetune_layers:
        trainable_sub_modules_list = [layer.strip() for layer in args.finetune_layers.split(',')]
        # Ensure 'classifier_only' is always an option if not explicitly excluded
        if "classifier_only" not in trainable_sub_modules_list:
            trainable_sub_modules_list.append("classifier_only")
    else:

        temp_model_for_structure = VivitForVideoClassification.from_pretrained(
            model_ckpt, 
            num_labels=len(class_labels),
            ignore_mismatched_sizes=True 
        )
        trainable_sub_modules_list = []

        # Add embeddings in trainable list
        if hasattr(temp_model_for_structure.vivit, 'embeddings'):
            trainable_sub_modules_list.append("vivit.embeddings")

        # Add encoder layers in trainable list
        if hasattr(temp_model_for_structure.vivit.encoder, 'layer'):
            for i in range(len(temp_model_for_structure.vivit.encoder.layer)):
                trainable_sub_modules_list.append(f"vivit.encoder.layer.{i}")
        
        
        # Add a special case for only fine-tuning the classifier
        trainable_sub_modules_list.append("classifier_only")

        del temp_model_for_structure # Clean up temporary model

    all_experiment_results = defaultdict(dict) # To store results for all experiments

    # --- Main Loop for different fine-tuning strategies ---
    for layer_name_to_finetune in trainable_sub_modules_list:
        print(f"\n--- Starting experiment: Fine-tuning {layer_name_to_finetune} + Classifier ---")
        
        # Re-initialize model for each experiment to ensure a clean slate
        model = VivitForVideoClassification.from_pretrained(
            model_ckpt,
            label2id=label2id,
            id2label=id2label,
            ignore_mismatched_sizes=True # Important for adapting pre-trained head to new num_labels
        ).to(device)

        # Freeze all parameters initially
        for param in model.parameters():
            param.requires_grad = False

        # Unfreeze the classifier head
        for param in model.classifier.parameters():
            param.requires_grad = True

        # Unfreeze the specific target layer
        if layer_name_to_finetune == "vivit.embeddings":
            for param in model.vivit.embeddings.parameters():
                param.requires_grad = True
        elif layer_name_to_finetune.startswith("vivit.encoder.layer."):
            try:
                layer_idx = int(layer_name_to_finetune.split('.')[-1])
                for param in model.vivit.encoder.layer[layer_idx].parameters():
                    param.requires_grad = True
            except (ValueError, IndexError):
                print(f"Error: Invalid encoder layer name format: {layer_name_to_finetune}. Skipping.")
                continue # Skip this layer if format is wrong
        
        elif layer_name_to_finetune == "classifier_only":
            pass # Classifier is already unfrozen
        else:
            print(f"Warning: Layer '{layer_name_to_finetune}' not recognized for unfreezing. Check model structure or spelling. Skipping this layer.")
            continue # Skip unrecognized layers
        
        # Verify which parameters are trainable
        trainable_params = [name for name, param in model.named_parameters() if param.requires_grad]
        if not trainable_params:
            print(f"No trainable parameters found for {layer_name_to_finetune}. Skipping this experiment.")
            continue # Skip if no parameters are trainable (e.g., if a specified layer doesn't exist)

        print(f"Trainable parameters for this run ({len(trainable_params)}): {trainable_params}")
        trainable_params_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total number of trainable parameters: {trainable_params_count}")

        # Initialize optimizer with only the trainable parameters
        optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)

        # Training Loop for the current layer's experiment
        for epoch in range(args.num_epochs):
            print(f"\n--- Epoch {epoch+1}/{args.num_epochs} for {layer_name_to_finetune} ---")
            model.train()
            total_loss = 0
            start_time = time.time()
            
            for i, batch in enumerate(train_dataloader):
                if batch is None: # Skip empty batches
                    continue
                pixel_values = batch["pixel_values"].to(device)
                labels = batch["label"].to(device).squeeze(1) # Remove singleton dimension for labels

                optimizer.zero_grad()
                
                outputs = model(pixel_values=pixel_values, labels=labels)
                loss = outputs.loss
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                if (i + 1) % 100 == 0:
                    print(f"Epoch {epoch+1}, Batch {i+1}/{len(train_dataloader)}, Loss: {loss.item():.4f}")

            avg_train_loss = total_loss / len(train_dataloader) if len(train_dataloader) > 0 else 0.0
            epoch_time = time.time() - start_time
            print(f"Epoch {epoch+1} finished. Avg Train Loss: {avg_train_loss:.4f}, Time: {epoch_time:.2f}s")
            
            # --- Evaluate and Log at specific epochs (e.g., 3 or 5 and 8) ---
            if (epoch + 1) == 3:
            # if (epoch + 1) == 5 or (epoch + 1) == 8:
                print(f"\n--- Evaluating {layer_name_to_finetune} at Epoch {epoch+1} ---")
                
                # Test evaluation
                test_case_name = f"{layer_name_to_finetune}_epoch_{epoch+1}_test"
                test_results = evaluate(model, test_dataloader, device, test_case_name, epoch + 1, label2id, id2label)
                
                # Store results in the main results dictionary
                if args.log_test_only:
                    all_experiment_results[layer_name_to_finetune][f"epoch_{epoch+1}"] = {
                        "test": test_results
                    }
                else:
                    # Run validation evaluation only if it needs to be logged
                    val_case_name = f"{layer_name_to_finetune}_epoch_{epoch+1}_validation"
                    val_results = evaluate(model, val_dataloader, device, val_case_name, epoch + 1, label2id, id2label)
                    all_experiment_results[layer_name_to_finetune][f"epoch_{epoch+1}"] = {
                        "validation": val_results,
                        "test": test_results
                    }
                
                # Save intermediate results for safety and progress tracking
                save_results_to_json(os.path.join(log_dir, f"all_results_{args.game_name}.json"), all_experiment_results)

    # Final save of all experiment results
    final_results_path = os.path.join(log_dir, f"all_results_{args.game_name}.json")
    save_results_to_json(final_results_path, all_experiment_results)
    print(f"\nAll experiments complete. All results saved to {final_results_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train and evaluate a ViViT model with single-layer fine-tuning.")
    parser.add_argument('--game_name', type=str, default="solid", help="Name of the game dataset.")
    parser.add_argument('--output_dir', type=str, default="./cross_val_finetuning_results", help="Directory to save logs for single-layer fine-tuning experiments.")
    parser.add_argument('--num_epochs', type=int, default=8, help="Number of training epochs for each experiment (will evaluate at epoch 5 and 8).")
    parser.add_argument('--batch_size', type=int, default=4, help="Batch size for training and evaluation.")
    parser.add_argument('--learning_rate', type=float, default=5e-5, help="Learning rate for the optimizer.")
    parser.add_argument('--finetune_layers', type=str, default=None, 
                        help="Comma-separated list of layer names to fine-tune (e.g., 'vivit.embeddings,vivit.encoder.layer.0'). If None, all layers will be iterated.")
    # MODIFICATION: Added a new argument to control logging behavior.
    parser.add_argument('--log_test_only', action='store_true', 
                        help="If set, only logs test results to the JSON file, skipping validation results in the output file.")
        
    args = parser.parse_args()
    
    main(args)