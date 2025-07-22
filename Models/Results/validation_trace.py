import json
import argparse
from pathlib import Path
from collections import defaultdict

def aggregate_data(base_dirs):
    """
    Searches for all result files in the specified base directories
    and aggregates the data by game and training type.

    Args:
        base_dirs (list[Path]): A list of root directory paths containing game folders.

    Returns:
        defaultdict | None: A dictionary containing all aggregated data, or None if no files were found.
                            Structure: {game: {training_type: [metrics]}}
    """
    # Using defaultdict simplifies the code, avoiding checks for key existence.
    # Structure: { "apex": { "full_finetuning": [...] } }
    aggregated_data = defaultdict(lambda: defaultdict(list))
    
    # Mapping from directory name to training type
    TRAINING_TYPE_MAP = {
        "full_finetuning_results": "full_finetuning",
        "finetune_layer0_results": "finetune_layer0",
        "continue_finetune_layer0_results": "continue_finetuning"
    }

    files_found = 0
    # Iterate over all specified base directories
    for base_path in base_dirs:
        if not base_path.is_dir():
            print(f"Warning: '{base_path}' is not a valid directory, skipping.")
            continue
            
        # Adjusted glob pattern to search within the 'logs' subdirectory.
        target_files = list(base_path.glob('*/validation_logs.json'))
        
        if not target_files:
            continue

        files_found += len(target_files)
        print(f"Found {len(target_files)} files to aggregate in '{base_path}'.")
        
        # Determine the training_type from the base directory name
        base_dir_name = base_path.name
        training_type = TRAINING_TYPE_MAP.get(base_dir_name, base_dir_name)

        for file_path in target_files:
            # Go up one more parent level to get the game_name
            game_name = file_path.parent.parent.name
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Could not read or parse file '{file_path}'. Error: {e}")
                continue

            for entry in data:
                # Extract the required information from each entry
                epoch = entry.get('epoch')
                f1_weighted = entry.get('f1_weighted')
                
                # Ensure all necessary data exists and is of the correct type
                if not all([isinstance(epoch, int), isinstance(f1_weighted, (float, int))]):
                    print(f"Warning: Skipping an incomplete entry in file '{file_path}': {entry}")
                    continue
                
                # Add the extracted data to the aggregation dictionary
                aggregated_data[game_name][training_type].append({
                    "epoch": epoch,
                    "f1_weighted": f1_weighted
                })

    return aggregated_data if files_found > 0 else None

def write_output_files(output_dir, aggregated_data):
    """
    Writes the aggregated data into separate JSON files for each game.

    Args:
        output_dir (str): The target directory to save the summary files.
        aggregated_data (defaultdict): The dictionary containing all aggregated data.
    """
    output_path = Path(output_dir)
    # Ensure the output directory exists
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("\nGenerating summary files...")
    for game_name, game_data in aggregated_data.items():
        # Before writing, sort each list by epoch
        for training_type in game_data:
            game_data[training_type].sort(key=lambda x: x['epoch'])
            
        # Define the output filename
        file_path = output_path / f"{game_name}_summary.json"
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(game_data, f, indent=4, ensure_ascii=False)
            print(f"    Success -> Wrote aggregated results for '{game_name}' to '{file_path}'")
        except IOError as e:
            print(f"    Failure -> Could not write file '{file_path}'. Error: {e}")


def main():
    """
    Main function to parse command-line arguments and start the aggregation process.
    """
    parser = argparse.ArgumentParser(
        description="Aggregate 'validation_logs.json' files. The script searches for training type subdirectories within a given root directory."
    )
    
    # MODIFIED: Argument changed to accept a single root directory.
    parser.add_argument(
        "--root-dir", 
        type=str,
        default='.',
        help="The root directory containing all training type subdirectories (e.g., 'finetune_all', 'finetune_layer0')."
    )
    
    parser.add_argument(
        "--output-dir", 
        type=str,
        default='validation_trace_results',
        help="The target directory to save the aggregated JSON files (default: validation_trace_results)."
    )
    
    args = parser.parse_args()
    
    # MODIFIED: New logic to find subdirectories automatically.
    root_path = Path(args.root_dir)
    if not root_path.is_dir():
        print(f"Error: The specified root directory '{root_path}' does not exist or is not a directory.")
        return

    # Get all subdirectories in the root path, which will be the base directories for aggregation.
    base_dirs = [d for d in root_path.iterdir() if d.is_dir()]
    if not base_dirs:
        print(f"No subdirectories found in '{root_path}', nothing to process.")
        return
    
    # Pass the list of found subdirectories to the aggregation function.
    aggregated_data = aggregate_data(base_dirs)
    
    if not aggregated_data:
        print("No 'validation_logs.json' files found to aggregate in any 'logs' subdirectories.")
        return
        
    write_output_files(args.output_dir, aggregated_data)
    
    print("\nData aggregation for all games complete.")


if __name__ == '__main__':
    main()