from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import os
import random
from PIL import Image
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import VivitImageProcessor
import json
import math

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)



# Add all the game names you need to run to this list
GAMES_TO_RUN = ['apex', 'endless', 'fps', 'gallery', 'gun', 'platform', 'solid', 'tiny', 'topdown' ] 

# Paths to dataset files, please modify according to your setup
CLEAN_DATA_CSV_PATH = "../Dataset/clean_data.csv"
DATASET_BASE_PATH = "../Dataset/"
# ==============================================================================

# Initialize Vivit Image Processor
model_ckpt = "google/vivit-b-16x2-kinetics400"
image_processor = VivitImageProcessor.from_pretrained(model_ckpt)

# Define class labels
class_labels = ['down', 'same', 'up']
label2id = {label: i for i, label in enumerate(class_labels)}
id2label = {i: label for label, i in label2id.items()}
print(f"Unique classes: {list(label2id.keys())}.")


# These functions and classes are used for parsing file paths and loading data.
# The `get_file_name_and_parent_folder` function now accepts `game_name` as a parameter to correctly handle paths for different games.
# The `MyCSVDataset` class is also modified to accept `game_name`.

def get_file_name_and_parent_folder(file_path, game_name):
    """
    Extracts file name, parent folder, player ID, and session ID from a file path.
    """
    file_path = os.path.normpath(file_path)
    file_paths = file_path.split(os.sep)
    file_name, _ = os.path.splitext(os.path.basename(file_path))
    parent_folder = os.path.dirname(file_path)
    video_name = file_paths[-2]
    
    # Split the string based on game_name
    try:
        player_id, session_id = video_name.split(f'_{game_name}_')
    except ValueError:
        print(f"Warning: Could not split '{video_name}' using '{game_name}'. Check filename format.")
        player_id, session_id = "unknown", "unknown"
        
    return file_name, parent_folder, player_id, session_id


class MyCSVDataset(Dataset):
    """
    Custom dataset class for loading and preprocessing data for each game.
    """
    def __init__(self, csv_file, csv_file_2, game_name):
        self.game_name = game_name
        self.data = pd.read_csv(csv_file)
        self.gf = pd.read_csv(csv_file_2)
        
        # Remove unnecessary columns
        columns_to_drop = [col for col in self.gf.columns if "control" in col and 
                           col != "[control]player_id" and 
                           col != "[control]session_id"]
        self.gf = self.gf.drop(columns=columns_to_drop)
        self.gf = self.gf.drop(columns=['[output]arousal'])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        gf = self.gf
        clip_path = sample['start_frame']
        
        # Pass game_name
        file_name, parent_folder, player_id, session_id = get_file_name_and_parent_folder(clip_path, self.game_name)
        file_name = int(file_name)

        start = int(sample['start_time']) * 4
        end = start + 24

        game_vector = gf[(gf['[control]player_id'] == player_id) & (gf['[control]session_id'] == session_id)]
        game_vector = game_vector.drop(columns=['[control]player_id', '[control]session_id'])
        
        pd.set_option('future.no_silent_downcasting', True)
        
        game_vector = game_vector.iloc[start:end]
        for col in game_vector.columns:
            if '[string]' in col:
                game_vector[col] = game_vector[col].apply(lambda x: 1 if isinstance(x, str) and x != '' else 0)
        
        game_vector = game_vector.fillna(0).infer_objects(copy=False)
        game_array = np.array(game_vector.values)
        
        # Ensure game_array has 24 rows, pad with zeros if not
        if game_array.shape[0] < 24:
            pad_width = ((0, 24 - game_array.shape[0]), (0, 0))
            game_array = np.pad(game_array, pad_width, mode='constant', constant_values=0)

        game_tensor = torch.from_numpy(game_array).float()

        frames = []
        for i in range(32):
            frame_number = file_name + i
            frame_path = os.path.join(DATASET_BASE_PATH, parent_folder, f"{frame_number:04d}.png")
            frame_path = os.path.normpath(frame_path)
            
            try:
                frame = Image.open(frame_path).convert('RGB')
                frames.append(frame)
            except FileNotFoundError:
                # If a frame is not found, use a black image as a placeholder
                print(f"Warning: Frame file not found '{frame_path}', using a black image instead.")
                frames.append(Image.new('RGB', (224, 224), 'black'))

        inputs = image_processor(list(frames), return_tensors="pt")
        pixel_values = inputs['pixel_values'].squeeze(0)
        inputs['pixel_values'] = pixel_values

        label = label2id[sample['arousal_change']]
        label_tensor = torch.LongTensor([label])
        
        inputs['label'] = label_tensor
        inputs['game_tensor'] = game_tensor

        return inputs


# The `process_game` function encapsulates the complete workflow for a single game: loading data, training the model, testing, and returning results.

def process_game(game_name):
    """
    Executes the full pipeline for a single game: data loading, training, testing, and evaluation.
    """
    print(f"\n{'='*25}")
    print(f"Processing game: {game_name}")
    print(f"{'='*25}")

    # Check if data files exist
    csv_file = os.path.join(DATASET_BASE_PATH, f'new_{game_name}.csv')
    if not os.path.exists(csv_file):
        print(f"Error: Data file '{csv_file}' not found. Skipping this game.")
        return None

    # 1. Load Data
    dataset = MyCSVDataset(csv_file, CLEAN_DATA_CSV_PATH, game_name)
    
    if len(dataset) == 0:
        print(f"Warning: Dataset for game '{game_name}' is empty. Skipping.")
        return None

    # 2. Split Train/Test Data
    data_num = len(dataset)
    train_num = math.ceil(data_num * 0.7)
    test_num = data_num - train_num
    
    if test_num < 1:
        print(f"Warning: Not enough data for game '{game_name}' to create a test set. Skipping.")
        return None
        
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_num, test_num])
    print(f"Dataset size: {data_num}, Train: {len(train_dataset)}, Test: {len(test_dataset)}")

    # 3. Prepare Training Data
    game_tensors = []
    labels = []
    for data in train_dataset:
        game_tensors.append(data['game_tensor'].numpy())
        labels.append(data['label'].numpy())

    game_tensors_np = np.array(game_tensors)
    labels_np = np.array(labels).ravel() # Use .ravel() to convert to a 1D array
    
    # Reshape
    game_tensors_np = game_tensors_np.reshape(game_tensors_np.shape[0], -1)

    # 4. Train Model
    print("Training RandomForest model...")
    clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    clf.fit(game_tensors_np, labels_np)
    print("Training complete.")

    # 5. Prepare Test Data
    test_game_tensors = []
    test_labels = []
    for data in test_dataset:
        test_game_tensors.append(data['game_tensor'].numpy())
        test_labels.append(data['label'].numpy())

    test_game_tensors_np = np.array(test_game_tensors)
    test_labels_np = np.array(test_labels).ravel() # Use .ravel()
    
    # Reshape
    test_game_tensors_np = test_game_tensors_np.reshape(test_game_tensors_np.shape[0], -1)

    # 6. Predict and Evaluate
    print("Making predictions...")
    test_pred = clf.predict(test_game_tensors_np)

    acc = accuracy_score(test_labels_np, test_pred)
    w_f1 = f1_score(test_labels_np, test_pred, average='weighted')
    # ADDED: Calculate macro F1 score
    macro_f1 = f1_score(test_labels_np, test_pred, average='macro')


    print(f"Results for game '{game_name}':")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  Weighted F1 Score: {w_f1:.4f}")
    # ADDED: Print macro F1 score
    print(f"  Macro F1 Score: {macro_f1:.4f}")

    # ADDED: Return macro_f1_score in the results dictionary
    return {"game": game_name, "accuracy": acc, "weighted_f1_score": w_f1, "macro_f1_score": macro_f1}


### 4. Main Execution Flow
# This is the main entry point of the script. It iterates through the `GAMES_TO_RUN` list, calls the `process_game` function,
# and then saves all the results to JSON and CSV files.

if __name__ == '__main__':
    all_results = []
    
    # Iterate over all specified games
    for game in GAMES_TO_RUN:
        result = process_game(game)
        if result:
            all_results.append(result)

    print(f"\n{'='*30}")
    print("All games processed.")
    print(f"{'='*30}")

    if all_results:
        # Convert results to DataFrame
        results_df = pd.DataFrame(all_results)
        print("Final Results Summary:")
        print(results_df)

        # Define output filenames
        json_output_path = "Results/random_forest_results.json"
        csv_output_path = "Results/random_forest_results.csv"

        # Save to JSON file
        with open(json_output_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=4)
        print(f"\nResults saved to: {json_output_path}")

        # Save to CSV file
        results_df.to_csv(csv_output_path, index=False, encoding='utf-8-sig')
        print(f"Results saved to: {csv_output_path}")
    else:
        print("No results were generated. Please check your configuration and data files.")