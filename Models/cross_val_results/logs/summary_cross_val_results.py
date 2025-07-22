import json
import pandas as pd
import re
import os
import argparse

def process_json_files(directory_path, output_csv_prefix):
    """
    Reads all JSON files in a specified directory, extracts key metrics
    (Accuracy, F1 Weighted, F1 Macro), and consolidates the data into
    separate CSV files for each game.

    Args:
        directory_path (str): The path to the directory containing the JSON files.
        output_csv_prefix (str): The prefix for the output CSV file names.
    """
    all_extracted_data = []

    json_file_pattern = re.compile(r'^all_results_(.*)\.json$')

    if not os.path.isdir(directory_path):
        print(f"Error: Directory not found at {directory_path}")
        return

    # Define the custom order for layers, which matches the keys
    custom_layer_order = ['embeddings'] + [f'layer.{i}' for i in range(12)] + ['classifier']

    for filename in os.listdir(directory_path):
        if json_file_pattern.match(filename):
            file_path = os.path.join(directory_path, filename)
            print(f"Processing file: {file_path}")

            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError) as e:
                print(f"  Error reading file {file_path}: {e}")
                continue

            # Extract game name from a filename 
            game_name_match = re.search(r'^all_results_(.*?)\.json', filename)
            game_name = game_name_match.group(1) if game_name_match else "unknown_game"

            # Iterate through the new nested JSON structure
            for model_name, epoch_data in data.items():
                # Extract the layer name from the model key (e.g., "vivit.encoder.layer.0")
                layer_name = model_name.replace("vivit.encoder.", "").replace("vivit.", "")
                
                if not epoch_data:
                    continue
                # Assuming we are interested in the first epoch found for each model
                first_epoch_key = list(epoch_data.keys())[0]
                metrics = epoch_data[first_epoch_key].get('test', {})

                if not metrics:
                    continue

                # Get metrics using the new snake_case keys
                accuracy = metrics.get('accuracy')
                f1_weighted = metrics.get('f1_weighted')
                f1_macro = metrics.get('f1_macro') # Added f1_macro extraction

                all_extracted_data.append({
                    'Game': game_name,
                    'Layer': layer_name,
                    'Accuracy': accuracy,
                    'F1 Weighted': f1_weighted,
                    'F1 Macro': f1_macro  # Added f1_macro to the record
                })

    if not all_extracted_data:
        print("No matching JSON files found or no data extracted.")
        return

    # Convert all extracted data into a single DataFrame
    df = pd.DataFrame(all_extracted_data)
    # Convert 'Layer' column to a Categorical type with the custom order
    df['Layer'] = pd.Categorical(df['Layer'], categories=custom_layer_order, ordered=True)

    # Process and save data for each unique game
    for game in df['Game'].unique():
        game_df = df[df['Game'] == game].copy()
        game_df = game_df.sort_values(by='Layer') # Sort by the custom-ordered 'Layer'

        # Save the summarized data to a CSV file
        output_csv_name = f"{output_csv_prefix}_{game}.csv"
        game_df.to_csv(output_csv_name, index=False)
        print(f"\nData for '{game}' successfully extracted and saved to {output_csv_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process JSON files to generate CSV summaries.")

    parser.add_argument(
        "-d", "--directory",
        type=str,
        default=".",
        help="The path to the directory containing the JSON files (default: './output_data')."
    )
    parser.add_argument(
        "-c", "--csv_prefix",
        type=str,
        default="cross_val_summary",
        help="The prefix for the output CSV file names. (e.g., 'summary/data_summary')."
    )

    args = parser.parse_args()

    # Ensure the parent directory for the CSV output exists
    if os.path.dirname(args.csv_prefix):
        os.makedirs(os.path.dirname(args.csv_prefix), exist_ok=True)

    process_json_files(args.directory, args.csv_prefix)