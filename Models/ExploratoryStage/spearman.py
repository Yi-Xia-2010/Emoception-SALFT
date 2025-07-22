import json
import os
import re
import numpy as np
from scipy.stats import spearmanr
import collections

def natural_sort_key(s):
    """
    Creates a key for natural sorting.
    e.g. "vivit.encoder.layer.10" -> ("vivit.encoder.layer.", 10)
    This ensures that numerical parts of strings are sorted numerically.
    """
    match = re.match(r'^(.*?)(\d+)$', s)
    if match:
        prefix, number = match.groups()
        return (prefix, int(number))
    # For keys without numbers like "embeddings", sort them by the string itself.
    return (s, -1)

def load_l2_differences(filepath):
    """Loads the param_l2_diff values from a JSON file."""
    if not os.path.exists(filepath):
        print(f"Error: File not found at {filepath}")
        return None
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return {k: v["param_l2_diff"] for k, v in data.items()}
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Warning: Could not process file {filepath}. Reason: {e}")
        return None

def compare_and_save(file1, file2, results_dir):
    """
    Compares two JSON files by calculating the Spearman correlation,
    then saves the result to a new JSON file inside the specified directory.
    """
    print(f"Comparing '{file1}' AND '{file2}'")

    l2_1 = load_l2_differences(file1)
    l2_2 = load_l2_differences(file2)

    if l2_1 is None or l2_2 is None:
        print("Comparison aborted because one or more files failed to load.\n")
        return

    # Find the intersection of keys between the two files
    shared_keys_set = set(l2_1.keys()) & set(l2_2.keys())
    
    if not shared_keys_set:
        print("Error: No shared layers found between the two files.\n")
        return

    # Sort the shared keys using the natural sort function
    shared_keys = sorted(list(shared_keys_set), key=natural_sort_key)

    values1 = [l2_1[k] for k in shared_keys]
    values2 = [l2_2[k] for k in shared_keys]

    rho, p = spearmanr(values1, values2)

    result = {
        "file1": file1,
        "file2": file2,
        "shared_layers_count": len(shared_keys),
        "shared_layers": shared_keys,
        "param_l2_diff_file1": {k: l2_1[k] for k in shared_keys},
        "param_l2_diff_file2": {k: l2_2[k] for k in shared_keys},
        "spearman": {
            "correlation": round(rho, 6),
            "p_value": round(p, 6)
        }
    }

    # Determine the output file path
    abs_path = os.path.abspath(file1)
    parts = abs_path.split(os.sep)
    game_name = parts[-3] if len(parts) >= 3 else "unknown_game"
    
    epoch1_match = re.search(r'(\d+epoch)', file1)
    epoch2_match = re.search(r'(\d+epoch)', file2)
    
    tag1 = epoch1_match.group(1) if epoch1_match else "file1"
    tag2 = epoch2_match.group(1) if epoch2_match else "file2"

    filename = f"{game_name}_{tag1}_vs_{tag2}_corr.json"
    output_path = os.path.join(results_dir, filename)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"Comparison complete. Results saved to: {output_path}\n")

def main():
    """
    Automatically discovers and processes raw comparison files in subdirectories.
    """
    results_dir = "spearman_results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Regex to find the target files, e.g., '.../1epoch/comparison_results_raw.json'
    # It handles both / and \ as path separators.
    file_pattern = re.compile(r"(\d+)epoch[\\/](comparison_results_raw\.json)$")
    
    # Group found files by game
    games = collections.defaultdict(list)

    print("Starting file discovery...")
    for root, _, files in os.walk('.'):
        # Avoid searching in the results directory itself
        if os.path.abspath(root).startswith(os.path.abspath(results_dir)):
            continue
            
        for name in files:
            if name == 'comparison_results_raw.json':
                full_path = os.path.join(root, name)
                match = file_pattern.search(full_path)
                if match:
                    epoch_number = int(match.group(1))
                    # Assumes game name is the directory containing the '...epoch' folders
                    game_name = os.path.basename(os.path.dirname(root))
                    games[game_name].append((epoch_number, full_path))

    if not games:
        print("No files matching the pattern '.../{n}epoch/comparison_results_raw.json' were found.")
        return

    print(f"Found {len(games)} game(s) to process.")
    # Process each game's files
    for game_name, files_found in games.items():
        print(f"\nProcessing game: {game_name}")
        
        # Sort the list of (epoch_number, path) tuples numerically by epoch number.
        files_found.sort(key=lambda item: item[0])

        # Identify all consecutive pairs from the sorted list.
        pairs_to_process = []
        for i in range(len(files_found) - 1):
            current_epoch, current_path = files_found[i]
            next_epoch, next_path = files_found[i+1]
            
            # Check if the next epoch is exactly one greater than the current one.
            if next_epoch == current_epoch + 1:
                pairs_to_process.append((current_path, next_path))
        
        if not pairs_to_process:
            print(f"No consecutive epoch pairs found for game '{game_name}'.")
            continue # Move to the next game

        # Then, process the identified pairs
        print(f"Found {len(pairs_to_process)} pairs to compare for game '{game_name}'.")
        for file1, file2 in pairs_to_process:
            compare_and_save(file1, file2, results_dir)

if __name__ == "__main__":
    main()