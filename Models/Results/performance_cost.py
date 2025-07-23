# -*- coding: utf-8 -*-
import argparse
import json
import logging
import math
from pathlib import Path
from typing import Dict, Any, Optional

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# --- Global Constants ---
# Define the per-step computational cost in GFLOPs for each type
COST_FULL_FINETUNING = 6402.90
COST_FINETUNE_LAYER0 = 4449.00

# --- Logging Configuration ---
# Configure the logger to replace print for more flexible output control
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


# --- Mock Objects ---
# To allow this script to run independently, create mock objects for dependencies
# not required for calculating the dataset length.
class MockImageProcessor:
    """A mock image processor for instantiating the Dataset without installing transformers."""
    def __call__(self, images, return_tensors):
        return None

# An example label2id dictionary. The actual content does not affect length calculation.
mock_label2id: Dict[str, int] = {"decrease": 0, "increase": 1, "maintain": 2}


class MyCSVDataset(Dataset):
    def __init__(self, csv_file: Path, csv_file_2: Path, game_name: str,
                 image_processor: Any, label2id: Dict[str, int], root_dir: Path):
        self.game_name = game_name
        self.image_processor = image_processor
        self.label2id = label2id
        self.root_dir = root_dir
        if not csv_file.is_file():
            logging.warning(f"Primary CSV file not found: {csv_file}. Dataset size will be 0.")
            self.data = pd.DataFrame()
        else:
            self.data = pd.read_csv(csv_file)

        if csv_file_2.is_file():
            try:
                self.gf = pd.read_csv(csv_file_2)
                columns_to_drop = [col for col in self.gf.columns if "control" in col and
                                   col != "[control]player_id" and
                                   col != "[control]session_id"]
                self.gf = self.gf.drop(columns=columns_to_drop)
                self.gf = self.gf.drop(columns=['[output]arousal'])
            except KeyError as e:
                logging.warning(f"Expected column not found in {csv_file_2}: {e}")
                self.gf = pd.DataFrame()
        else:
            logging.warning(f"Second CSV file not found: {csv_file_2}. Related functionality will be affected.")
            self.gf = pd.DataFrame()

    def __len__(self) -> int:
        return len(self.data)


# --- Core Functions ---
def read_json_files_from_directory(json_dir: Path) -> Optional[Dict[str, Any]]:
    """
    Safely reads all .json files from a specified directory.
    """
    if not json_dir.is_dir():
        logging.error(f"The specified JSON directory does not exist: {json_dir}")
        return None

    all_data = {}
    logging.info(f"Starting to read JSON files from '{json_dir}' directory...")
    for file_path in json_dir.glob('*_summary.json'):
        # Avoid reading previously updated files
        if '_updated' in file_path.stem:
            continue
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                all_data[file_path.name] = json.load(f)
                logging.info(f"  -> Successfully read: {file_path.name}")
        except json.JSONDecodeError:
            logging.warning(f"  -> Skipping invalid JSON file: {file_path.name}")
        except Exception as e:
            logging.error(f"  -> An unknown error occurred while reading file {file_path.name}: {e}")
    return all_data


def calculate_and_add_info(json_data: Dict[str, Any], config: argparse.Namespace) -> None:
    """
    Instantiates the dataset to calculate its size, computes the training steps,
    cost per epoch, and total training cost, then updates the JSON data.
    """
    logging.info(f"Starting to calculate Steps and Computational Cost for each game (Batch Size: {config.batch_size})...")
    mock_processor = MockImageProcessor()
    
    dataset_root_dir = Path(config.dataset_dir)
    game_data_csv_path = dataset_root_dir / "clean_data.csv"

    for filename, data in json_data.items():
        # Extract game name from "game_name_summary.json"
        game_name = Path(filename).stem.replace('_summary', '')
        csv_file_path = dataset_root_dir / f'new_{game_name}.csv'
        
        try:
            dataset = MyCSVDataset(
                csv_file=csv_file_path,
                csv_file_2=game_data_csv_path,
                game_name=game_name,
                image_processor=mock_processor,
                label2id=mock_label2id,
                root_dir=dataset_root_dir
            )
            
            dataset_size = len(dataset)
            if dataset_size == 0:
                data['steps_per_epoch'] = 0
                continue

            train_size = int(0.7 * dataset_size)
            steps = math.ceil(train_size / config.batch_size)
            
            data['steps_per_epoch'] = steps
            logging.info(f"  -> {game_name}: (Total size: {dataset_size}, Train size: {train_size}) -> {steps} steps/epoch")
            
            # --- Calculate and add cost information ---
            
            cost_full_per_epoch = steps * COST_FULL_FINETUNING
            cost_layer0_per_epoch = steps * COST_FINETUNE_LAYER0

            # --- Full Finetuning Cost Calculation ---
            # MODIFIED: Check for both 'full_finetuning' and 'full_finetuning_results' keys
            full_key = 'full_finetuning' if 'full_finetuning' in data else 'full_finetuning_results'
            
            if full_key in data and data.get(full_key):
                # Directly add cumulative cost to the dictionaries in the original list
                for epoch_data in data[full_key]:
                    current_epoch = epoch_data.get('epoch', 0)
                    cumulative_cost = cost_full_per_epoch * current_epoch
                    epoch_data['cumulative_cost_gflops'] = round(cumulative_cost, 2)
                
                # Add total cost info at the top level
                num_epochs = len(data[full_key])
                total_cost = cost_full_per_epoch * num_epochs
                data['full_finetuning_cost_per_epoch_gflops'] = round(cost_full_per_epoch, 2)
                data['full_finetuning_total_cost_gflops'] = round(total_cost, 2)
                logging.info(f"       -> Full Finetuning: {cost_full_per_epoch:.2f} GFLOPs/epoch | Total ({num_epochs} epochs): {total_cost:.2f} GFLOPs")

            # --- Finetune Layer0 Cost Calculation ---
            # MODIFIED: Check for both 'finetune_layer0' and 'finetune_layer0_results' keys
            layer0_key = 'finetune_layer0' if 'finetune_layer0' in data else 'finetune_layer0_results'

            if layer0_key in data and data.get(layer0_key):
                # only gallery and platform fine-tuned 3 epoch in first stage, other games fine-tuned 2epoch in first stage.
                if game_name.lower() in ['gallery', 'platform']:
                    initial_cost = COST_FULL_FINETUNING * 3
                    logging.info(f"       -> Finetune Layer0: Applying 3-step initial cost for '{game_name}'.")
                else:
                    initial_cost = COST_FULL_FINETUNING * 2
                    logging.info(f"       -> Finetune Layer0: Applying 2-step initial cost for '{game_name}'.")
                
                # Directly add cumulative cost to the dictionaries in the original list
                for epoch_data in data[layer0_key]:
                    current_epoch = epoch_data.get('epoch', 0)
                    cumulative_cost = initial_cost + (cost_layer0_per_epoch * current_epoch)
                    epoch_data['cumulative_cost_gflops'] = round(cumulative_cost, 2)

                # Add total cost info at the top level
                num_epochs = len(data[layer0_key])
                total_cost_with_initial = initial_cost + (cost_layer0_per_epoch * num_epochs)
                data['finetune_layer0_cost_per_epoch_gflops'] = round(cost_layer0_per_epoch, 2)
                data['finetune_layer0_total_cost_gflops'] = round(total_cost_with_initial, 2)
                logging.info(f"       -> Finetune Layer0: {cost_layer0_per_epoch:.2f} GFLOPs/epoch | Total ({num_epochs} epochs + initial cost): {total_cost_with_initial:.2f} GFLOPs")

        except Exception as e:
            logging.error(f"  -> An error occurred while processing {game_name}: {e}")
            data['steps_per_epoch'] = "Calculation Error"

def save_json_files(json_data: Dict[str, Any], save_dir: Path) -> None:
    """
    Saves the updated JSON data to the specified directory with new filenames.
    """
    logging.info(f"Starting to save updated data to '{save_dir}' directory...")
    
    for original_filename, data in json_data.items():
        p = Path(original_filename)
        # Create new filename from "game_name_summary.json" to "game_name_summary_updated.json"
        new_filename = f"{p.stem}_updated{p.suffix}"
        output_path = save_dir / new_filename
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
            logging.info(f"  -> Successfully saved: {output_path}")
        except Exception as e:
            logging.error(f"  -> An error occurred while saving file {output_path}: {e}")

def plot_results(json_data: Dict[str, Any], save_dir: Path) -> None:
    """
    For each game, plots f1_weighted vs. cumulative_cost_gflops with academic journal styling.
    """
    logging.info(f"Starting to generate result plots and saving to '{save_dir}' directory...")

    # Set global parameters for academic styling
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 12,
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 16
    })

    for filename, data in json_data.items():
        game_name = Path(filename).stem.replace('_summary', '')
        fig, ax = plt.subplots(figsize=(8, 5)) # Adjust size for journal layout
        
        plot_has_data = False
        
        # Define colors and markers
        colors = ['#0072B2', '#D55E00'] # Blue, Orange (colorblind-friendly)
        markers = ['o', 's']

        # --- Data Extraction and Scaling ---
        all_costs = []
        full_costs, full_f1s = [], []
        layer0_costs, layer0_f1s = [], []

        # MODIFIED: Check for both 'full_finetuning' and 'full_finetuning_results' keys
        full_key = 'full_finetuning' if 'full_finetuning' in data else 'full_finetuning_results'
        if full_key in data and data.get(full_key):
            full_costs = [d.get('cumulative_cost_gflops', 0) for d in data[full_key]]
            full_f1s = [d.get('f1_weighted', 0) for d in data[full_key]]
            if full_costs: all_costs.extend(full_costs)
        
        # MODIFIED: Check for both 'finetune_layer0' and 'finetune_layer0_results' keys
        layer0_key = 'finetune_layer0' if 'finetune_layer0' in data else 'finetune_layer0_results'
        if layer0_key in data and data.get(layer0_key):
            layer0_costs = [d.get('cumulative_cost_gflops', 0) for d in data[layer0_key]]
            layer0_f1s = [d.get('f1_weighted', 0) for d in data[layer0_key]]
            if layer0_costs: all_costs.extend(layer0_costs)

        if all_costs:
            max_cost = max(all_costs) if all_costs else 0
            if max_cost > 1000: # Only apply scientific notation for large numbers
                exponent = int(math.floor(math.log10(max_cost)))
                scale_factor = 10**exponent
                xlabel = f'Cumulative Cost ($\\times10^{{{exponent}}}$ GFLOPs)'
            else:
                scale_factor = 1
                xlabel = 'Cumulative Cost (GFLOPs)'

            # --- Plotting ---
            if full_costs and full_f1s:
                scaled_full_costs = [c / scale_factor for c in full_costs]
                ax.plot(scaled_full_costs, full_f1s, marker=markers[0], linestyle='-', color=colors[0], label='Full Finetuning', linewidth=2)
                plot_has_data = True

            if layer0_costs and layer0_f1s:
                scaled_layer0_costs = [c / scale_factor for c in layer0_costs]
                ax.plot(scaled_layer0_costs, layer0_f1s, marker=markers[1], linestyle='--', color=colors[1], label='SALFT (ours)', linewidth=2)
                plot_has_data = True
        
        if plot_has_data:
            # ax.set_title(f'Weighted F1-Score vs. Cumulative Cost for {game_name}')
            ax.set_xlabel(xlabel)
            ax.set_ylabel('Weighted F1-Score')
            ax.legend(frameon=False) # Frameless legend
            
            # Set grid and spines
            ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            # Save the plot
            plot_filename = f"{Path(filename).stem}_f1_vs_cost.png"
            plot_path = save_dir / plot_filename
            try:
                fig.savefig(plot_path, dpi=300, bbox_inches='tight') # Save as a high-resolution image
                logging.info(f"  -> Successfully saved plot: {plot_path}")
            except Exception as e:
                logging.error(f"  -> An error occurred while saving plot {plot_path}: {e}")
            plt.close(fig) # Close the current figure to prepare for the next one
        else:
            logging.warning(f"  -> Skipping plot generation for {filename} as no plottable data was found.")


def main():
    parser = argparse.ArgumentParser(
        description="Read game experiment JSON data, calculate additional info, and save back to the original directory with new filenames.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--json_dir", 
        type=str, 
        default="validation_trace_results", 
        help="Directory containing the original JSON files, which will also be the save directory."
    )
    parser.add_argument(
        "--dataset_dir", 
        type=str, 
        default="../../Dataset/", 
        help="Root directory of the dataset, which should contain 'new_GAME.csv' and 'clean_data.csv'."
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=4, 
        help="The batch size used during training."
    )
    parser.add_argument(
        "--plot",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Generates result plots by default. Use --no-plot to disable."
    )
    args = parser.parse_args()

    json_dir_path = Path(args.json_dir)

    # Read JSON files
    json_data = read_json_files_from_directory(json_dir_path)

    if not json_data:
        logging.warning("No JSON data was read. Exiting program.")
        return

    # Calculate and add information
    calculate_and_add_info(json_data, args)
    
    # Save the updated data to the original directory with new filenames
    save_json_files(json_data, json_dir_path)

    # Generate plots if requested
    if args.plot:
        plot_results(json_data, json_dir_path)
    
    logging.info("--- All processing and saving operations have been completed ---")


if __name__ == "__main__":
    main()