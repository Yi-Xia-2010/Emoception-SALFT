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
    # Optional: For stronger determinism, uncomment the following lines
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)


class MyCSVDataset(Dataset):
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

def evaluate(model, dataloader, device, case_name, epoch, label2id, id2label):
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
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def main(args):
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    class_labels = ['down', 'same', 'up']
    label2id = {label: i for i, label in enumerate(class_labels)}
    id2label = {i: label for label, i in label2id.items()}
    
    base_dir_name = os.path.basename(args.output_dir)
    case_prefix = base_dir_name.removesuffix('_results')
    print(f"Automatically extracted case prefix: '{case_prefix}'")
    
    output_dir = os.path.join(args.output_dir, args.game_name)
    model_dir = os.path.join(output_dir, "checkpoints")
 
    os.makedirs(model_dir, exist_ok=True)

    print(f"All outputs will be saved to: {output_dir}")
    
    model_ckpt_hub = "google/vivit-b-16x2-kinetics400"

    image_processor = VivitImageProcessor.from_pretrained(model_ckpt_hub)
    
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

    # --- Model Loading ---
    if args.local_model_path and args.local_model_path.endswith(('.pt', '.pth')):

        print(f"Loading base model structure from Hugging Face: {model_ckpt_hub}")

        model = VivitForVideoClassification.from_pretrained(
            model_ckpt_hub,
            label2id=label2id,
            id2label=id2label,
            ignore_mismatched_sizes=True
        )
        print(f"Loading weights from local checkpoint file: {args.local_model_path}")

        model.load_state_dict(torch.load(args.local_model_path, map_location=device))
        model.to(device)

    else:
        model_name_or_path = args.local_model_path if args.local_model_path else model_ckpt_hub
        if args.local_model_path:
            print(f"Loading full model from local directory: {model_name_or_path}")
        else:
            print(f"Loading pre-trained model from Hugging Face: {model_name_or_path}")
        
        model = VivitForVideoClassification.from_pretrained(
            model_name_or_path,
            label2id=label2id,
            id2label=id2label,
            ignore_mismatched_sizes=True
        ).to(device)
    
    # --- Layer Freezing ---
    print("Freezing all model parameters by default.")
    for param in model.parameters():
        param.requires_grad = False
        
    if args.finetune_layers:
        print(f"Unfreezing layers for fine-tuning: {args.finetune_layers}")
        for name, param in model.named_parameters():
            for layer_name in args.finetune_layers:
                if layer_name in name:
                    param.requires_grad = True
                    print(f"  - Unfroze {name}")

    # --- Optimizer ---
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = AdamW(trainable_params, lr=args.learning_rate)
    
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nNumber of trainable parameters: {num_trainable_params}")
    if num_trainable_params == 0:
        print("\nWARNING: There are no trainable parameters. Please check your --finetune_layers argument.")
        print("Defaulting to unfreezing the classifier layer to prevent errors.")
        for name, param in model.named_parameters():
            if "classifier" in name:
                param.requires_grad = True
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = AdamW(trainable_params, lr=args.learning_rate)
        num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Number of trainable parameters is now: {num_trainable_params}")

    # --- Training Loop ---
    best_val_f1 = -1.0
    best_checkpoint_path = ""
    best_epoch = -1
    validation_logs = []

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
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1} finished. Avg Train Loss: {avg_train_loss:.4f}, Time: {epoch_time:.2f}s")
        
        # --- Validation and Checkpoint Saving ---

        val_case_name = f"{case_prefix}_{args.game_name}_{epoch + 1}_validation"
        val_results = evaluate(model, val_dataloader, device, val_case_name, epoch + 1, label2id, id2label)
        validation_logs.append(val_results)
        
        if args.save_format == 'huggingface':
            checkpoint_path = os.path.join(model_dir, f"checkpoint_epoch_{epoch+1}")
            model.save_pretrained(checkpoint_path)
            image_processor.save_pretrained(checkpoint_path)
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

    # Save all validation logs
    save_results_to_json(os.path.join(output_dir, "validation_logs.json"), validation_logs)
    print("\nTraining complete. All validation logs saved.")

    # --- Final Evaluation on Test Set ---
    print("\n--- Starting Final Evaluation on Test Set ---")
    if not best_checkpoint_path:
        print("No best checkpoint found. Exiting.")
        return

    # --- Load model based on the save format ---
    print(f"Loading best model from: {best_checkpoint_path}")
    if args.save_format == 'huggingface':
        model = VivitForVideoClassification.from_pretrained(best_checkpoint_path).to(device)
    else: # 'state_dict' format
        # Re-initialize model structure before loading state_dict
        model = VivitForVideoClassification.from_pretrained(model_ckpt_hub, label2id=label2id, id2label=id2label, ignore_mismatched_sizes=True)
        model.load_state_dict(torch.load(best_checkpoint_path))
        model.to(device)
    
    test_case_name = f"{case_prefix}_{args.game_name}_{best_epoch}_test"
    test_results = evaluate(model, test_dataloader, device, test_case_name, best_epoch, label2id, id2label)
    
    test_results_path = os.path.join(log_dir, "test_results.json")
    save_results_to_json(test_results_path, test_results)
    print(f"Final test results saved to {test_results_path}")
    print(f"Test Accuracy: {test_results['accuracy']:.4f}, Test Weighted F1: {test_results['f1_weighted']:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train and evaluate a ViViT model.")
    parser.add_argument('--game_name', type=str, default="solid", help="Name of the game dataset.")
    parser.add_argument('--output_dir', type=str, default="Results/continue_finetune_results", help="Directory to save checkpoints and logs.")
    parser.add_argument('--num_epochs', type=int, default=13, help="Number of training epochs.")
    parser.add_argument('--batch_size', type=int, default=4, help="Batch size for training.")
    parser.add_argument('--learning_rate', type=float, default=5e-5, help="Learning rate for the optimizer.")
    parser.add_argument('--local_model_path', type=str, default="full_finetuning_results/solid/checkpoints/checkpoint_epoch_1.pt", help="Path to a local pre-trained model directory for fine-tuning.")
    parser.add_argument('--finetune_layers', nargs='+', default=["vivit.encoder.layer.0", "classifier"], help="List of layer names to fine-tune (unfreeze). Example: 'classifier' 'vivit.encoder.layer.11'")
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