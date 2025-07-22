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
import matplotlib.pyplot as plt

# Set a seed for reproducibility
def set_seed(seed):
    """Sets the seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# --- Dataset Definition ---
class MyCSVDataset(Dataset):
    """Custom Dataset for loading video frames and game data."""
    def __init__(self, csv_file, csv_file_2, game_name, image_processor, label2id, root_dir="../Dataset/"):
        self.data = pd.read_csv(csv_file)
        self.gf = pd.read_csv(csv_file_2)
        self.game_name = game_name
        self.image_processor = image_processor
        self.label2id = label2id
        self.root_dir = root_dir
        
        # Pre-process game feature data
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
        file_name_str, parent_folder = os.path.splitext(os.path.basename(clip_path))
        parent_folder = os.path.dirname(clip_path)
        player_id, session_id = self._get_file_info(clip_path)
        
        # Load and process video frames
        frames = []
        file_name_int = int(file_name_str)
        for i in range(32):
            frame_path = os.path.join(self.root_dir, parent_folder, f"{file_name_int + i:04d}.png")
            frame = Image.open(frame_path).convert('RGB')
            frames.append(frame)

        inputs = self.image_processor(list(frames), return_tensors="pt")
        inputs['pixel_values'] = inputs['pixel_values'].squeeze(0)

        # Process label
        label_str = sample['arousal_change']
        label_id = self.label2id[label_str]
        inputs['label'] = torch.LongTensor([label_id])
        
        return inputs

# --- Evaluation and Logging ---
def evaluate(model, dataloader, device, case_name, epoch, label2id, id2label):
    """Evaluates the model and returns performance metrics."""
    model.eval()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for batch in dataloader:
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
    """Saves evaluation results to a JSON file."""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

# --- MODIFICATION: New Plotting Helper Function ---
def plot_loss_curve(loss_list, output_path):
    """
    Plots the training loss curve based on the user's specified style 
    and saves it to a file.
    """
    plt.figure(figsize=(10, 5))

    # The x-axis represents epochs, starting from 1 for clarity.
    epochs = range(1, len(loss_list) + 1)
    
    # Plotting training loss curves
    plt.plot(epochs, loss_list, color='#e4007f', label="train/loss curve")

    # Plotting axes and legends
    plt.ylabel("loss", fontsize='large')
    plt.xlabel("epoch", fontsize='large')
    plt.legend(loc='upper right', fontsize='x-large')
    plt.xticks(epochs)

    # Construct the full save path and save the figure
    save_path = os.path.join(output_path, 'pytorch_vivit_loss_curve.png')
    plt.savefig(save_path)
    plt.close() # Close the figure to free up memory
    print(f"Training loss curve saved to {save_path}")

# --- Main Training and Evaluation Script ---
def main(args):
    """Main function to run the training and evaluation pipeline."""
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Configuration ---
    class_labels = ['down', 'same', 'up']
    label2id = {label: i for i, label in enumerate(class_labels)}
    id2label = {i: label for label, i in label2id.items()}
    
    # Create game-specific output directories
    game_output_dir = os.path.join(args.output_dir, args.game_name)
    model_dir = os.path.join(game_output_dir, "checkpoints")
    os.makedirs(model_dir, exist_ok=True)
    print(f"Results will be saved in: {game_output_dir}")
    
    # --- Data Loading ---
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

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size)
    val_dataloader = DataLoader(val_dataset, batch_size=1)
    test_dataloader = DataLoader(test_dataset, batch_size=1)

    # --- Model and Optimizer ---
    model = VivitForVideoClassification.from_pretrained(
        model_ckpt,
        label2id=label2id,
        id2label=id2label,
        ignore_mismatched_sizes=True
    ).to(device)
    
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)

    # --- Training Loop ---
    best_val_f1 = -1.0
    best_checkpoint_path = ""
    best_epoch = -1  
    validation_logs = []
    train_losses = []

    for epoch in range(args.num_epochs):
        print(f"\n--- Starting Epoch {epoch+1}/{args.num_epochs} ---")
        model.train()
        total_loss = 0
        start_time = time.time()
        
        for i, batch in enumerate(train_dataloader):
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["label"].to(device).squeeze(1)

            optimizer.zero_grad()
            
            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            if (i + 1) % 100 == 0:
                print(f"Epoch {epoch+1}, Batch {i+1}/{len(train_dataloader)}, Loss: {loss.item():.4f}")

        avg_train_loss = total_loss / len(train_dataloader)
        train_losses.append(avg_train_loss)
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1} finished. Avg Train Loss: {avg_train_loss:.4f}, Time: {epoch_time:.2f}s")
        
        # --- Validation and Checkpoint Saving ---
        val_case_name = f"full_finetuning_{args.game_name}_{epoch + 1}_validation"
        val_results = evaluate(model, val_dataloader, device, val_case_name, epoch + 1, label2id, id2label)
        validation_logs.append(val_results)
        
        # --- MODIFICATION: Choose save format ---
        # Decide the path based on the save format.
        # Hugging Face format saves to a directory, state_dict saves to a file.
        if args.save_format == 'huggingface':
            checkpoint_path = os.path.join(model_dir, f"checkpoint_epoch_{epoch+1}")
            model.save_pretrained(checkpoint_path)
            image_processor.save_pretrained(checkpoint_path) # Also save the processor
            print(f"Hugging Face model checkpoint saved to {checkpoint_path}")
        else: # 'state_dict' format
            checkpoint_path = os.path.join(model_dir, f"checkpoint_epoch_{epoch+1}.pt")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")

        if val_results["f1_weighted"] > best_val_f1:
            best_val_f1 = val_results["f1_weighted"]
            best_checkpoint_path = checkpoint_path
            best_epoch = epoch + 1
            print(f"*** New best validation F1 score: {best_val_f1:.4f} at epoch {epoch+1} ***")

    # Save validation logs
    validation_logs_path = os.path.join(game_output_dir, "validation_logs.json")
    save_results_to_json(validation_logs_path, validation_logs)
    print(f"\nTraining complete. All validation logs saved to {validation_logs_path}")

    # --- MODIFICATION: Call the new plot helper ---
    plot_loss_curve(train_losses, game_output_dir)

    # --- Final Evaluation on Test Set ---
    print("\n--- Starting Final Evaluation on Test Set ---")
    if not best_checkpoint_path:
        print("No best checkpoint found. Exiting.")
        return

    # --- MODIFICATION: Load model based on the save format ---
    print(f"Loading best model from: {best_checkpoint_path}")
    if args.save_format == 'huggingface':
        # For Hugging Face format, load the entire model from the directory
        model = VivitForVideoClassification.from_pretrained(best_checkpoint_path).to(device)
    else: # 'state_dict' format
        # For state_dict, load the weights into the existing model structure
        model.load_state_dict(torch.load(best_checkpoint_path))
    
    test_case_name = f"full_finetuning_{args.game_name}_{best_epoch}_test"
    test_results = evaluate(model, test_dataloader, device, test_case_name, best_epoch, label2id, id2label)
    
    # Save test results
    test_results_path = os.path.join(game_output_dir, "test_results.json")
    save_results_to_json(test_results_path, test_results)
    print(f"Final test results saved to {test_results_path}")
    print(f"Test Accuracy: {test_results['accuracy']:.4f}, Test Weighted F1: {test_results['f1_weighted']:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train and evaluate a ViViT model.")
    parser.add_argument('--game_name', type=str, default="solid", help="Name of the game dataset. Used to create the output sub-directory.")
    parser.add_argument('--output_dir', type=str, default="Results/full_finetuning_results", help="Base directory to save results.")
    parser.add_argument('--num_epochs', type=int, default=14, help="Number of training epochs.")
    parser.add_argument('--batch_size', type=int, default=4, help="Batch size for training.")
    parser.add_argument('--learning_rate', type=float, default=5e-5, help="Learning rate for the optimizer.")
    # --- MODIFICATION: New argument to select save format ---
    parser.add_argument(
        '--save_format', 
        type=str, 
        default="state_dict", 
        choices=['state_dict', 'huggingface'], 
        help="Format to save the model checkpoints ('state_dict' or 'huggingface')."
    )
    
    args = parser.parse_args()
    
    main(args)