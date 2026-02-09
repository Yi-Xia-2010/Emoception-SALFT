import pandas as pd
import os
import argparse
import re

def consolidate_summary_csvs(directory_path, weighted_output_path, macro_output_path):
    """
    Reads all game-specific summary CSVs from a directory, pivots the data
    to consolidate 'F1 Weighted' and 'F1 Macro' metrics into two separate
    CSV files, with layers as rows and games as columns.

    Args:
        directory_path (str): The path to the directory containing the summary CSV files.
        weighted_output_path (str): The output file path for the consolidated F1 Weighted data.
        macro_output_path (str): The output file path for the consolidated F1 Macro data.
    """
    all_game_data = []
    # This pattern looks for files starting with 'data_summary_' and ending with '.csv'
    csv_file_pattern = re.compile(r'cross_val_summary_.*\.csv$')

    if not os.path.isdir(directory_path):
        print(f"Error: Directory not found at {directory_path}")
        return

    print(f"Searching for summary CSVs in: {directory_path}")
    for filename in os.listdir(directory_path):
        if csv_file_pattern.match(filename):
            file_path = os.path.join(directory_path, filename)
            print(f"Reading file: {file_path}")
            try:
                # Read each game's summary CSV into a DataFrame
                game_df = pd.read_csv(file_path)
                all_game_data.append(game_df)
            except Exception as e:
                print(f"  Error reading or processing file {file_path}: {e}")
                continue

    if not all_game_data:
        print("No summary CSV files found to process.")
        return

    # Concatenate all individual game DataFrames into one large DataFrame
    consolidated_df = pd.concat(all_game_data, ignore_index=True)

    # Define the custom order for layers to ensure correct row sorting in the output
    custom_layer_order = ['embeddings'] + [f'layer.{i}' for i in range(12)] + ['classifier']
    consolidated_df['Layer'] = pd.Categorical(consolidated_df['Layer'], categories=custom_layer_order, ordered=True)
    
    # --- Pivot the data for each metric ---

    # Pivot for 'F1 Weighted'
    try:
        f1_weighted_pivot = consolidated_df.pivot(index='Layer', columns='Game', values='F1 Weighted')
        # Save the pivoted DataFrame to a CSV file
        f1_weighted_pivot.to_csv(weighted_output_path)
        print(f"\nConsolidated 'F1 Weighted' data saved to: {weighted_output_path}")
    except Exception as e:
        print(f"\nError creating or saving the F1 Weighted pivot table: {e}")

    # Pivot for 'F1 Macro'
    try:
        f1_macro_pivot = consolidated_df.pivot(index='Layer', columns='Game', values='F1 Macro')
        # Save the pivoted DataFrame to a CSV file
        f1_macro_pivot.to_csv(macro_output_path)
        print(f"Consolidated 'F1 Macro' data saved to: {macro_output_path}")
    except Exception as e:
        print(f"Error creating or saving the F1 Macro pivot table: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Consolidate game summary CSVs into two final metric files.")

    parser.add_argument(
        "-d", "--directory",
        type=str,
        default=".",
        help="The path to the directory containing the game summary CSV files (default: './summary')."
    )
    parser.add_argument(
        "-w", "--weighted_output",
        type=str,
        default="summary/consolidated_f1_weighted.csv",
        help="The output file path for the consolidated F1 Weighted data (default: './summary/consolidated_f1_weighted.csv')."
    )
    parser.add_argument(
        "-m", "--macro_output",
        type=str,
        default="summary/consolidated_f1_macro.csv",
        help="The output file path for the consolidated F1 Macro data (default: './summary/consolidated_f1_macro.csv')."
    )

    args = parser.parse_args()

    # Ensure the output directory exists
    output_dir = os.path.dirname(args.weighted_output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    consolidate_summary_csvs(args.directory, args.weighted_output, args.macro_output)