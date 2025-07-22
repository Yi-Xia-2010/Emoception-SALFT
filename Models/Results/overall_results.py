import os
import pandas as pd
import argparse
import glob

def consolidate_results(input_dir, output_file):
    """
    Reads all CSV files in a directory, consolidates them into a summary CSV,
    including accuracy, f1_score, and f1_macro.

    Args:
        input_dir (str): The directory path containing multiple *_results.csv files.
        output_file (str): The full path for the final summary CSV file.
    """
    # Use glob to find all files matching the *_results.csv pattern
    search_pattern = os.path.join(input_dir, '*_results.csv')
    csv_files = glob.glob(search_pattern)

    if not csv_files:
        print(f"Error: No '*_results.csv' files found in the directory '{input_dir}'.")
        return

    print(f"Found {len(csv_files)} CSV files to process...")

    all_methods_data = []

    # Iterate over each found CSV file
    for file_path in csv_files:
        # Extract the method name from the filename
        method_name = os.path.basename(file_path).replace('_results.csv', '')
        
        # Create a dictionary for the current method to store its data
        current_method_data = {'Method': method_name}
        
        try:
            # Read the CSV file
            df = pd.read_csv(file_path)

            # --- Find the correct column names (case-insensitive) ---
            # Create a mapping from lowercase/stripped name to original name
            col_map = {c.lower().strip(): c for c in df.columns}

            game_col = col_map.get('game')
            accuracy_col = col_map.get('accuracy')
            # Check for both weighted f1 variations
            f1_weighted_col = col_map.get('weighted avg f1')
            # --- Added f1_macro column search ---
            f1_macro_col = col_map.get('f1_macro')

            # Check if all required columns were found
            if not all([game_col, accuracy_col, f1_weighted_col, f1_macro_col]):
                print(f"Warning: Skipping file '{file_path}' due to missing required columns. "
                      "Expected 'Game', 'Accuracy', 'Weighted avg f1', and 'f1_macro'.")
                continue # Skip to the next file

            # Iterate over each row of the DataFrame (i.e., each game)
            for index, row in df.iterrows():
                game_name = row[game_col]
                accuracy = row[accuracy_col]
                f1_weighted = row[f1_weighted_col]
                # --- Added f1_macro data extraction ---
                f1_macro = row[f1_macro_col]
                
                # Create standardized column names and add the data to the dictionary,
                # rounded to 4 decimal places.
                current_method_data[f'{game_name}_accuracy'] = round(float(accuracy), 4)
                current_method_data[f'{game_name}_f1_weighted'] = round(float(f1_weighted), 4)
                # --- Added f1_macro to the output data ---
                current_method_data[f'{game_name}_f1_macro'] = round(float(f1_macro), 4)
            
            all_methods_data.append(current_method_data)

        except Exception as e:
            print(f"Warning: Could not process file '{file_path}'. Error: {e}")

    if not all_methods_data:
        print("Error: No data could be processed. Aborting.")
        return

    # Convert the list of all method data into a DataFrame
    summary_df = pd.DataFrame(all_methods_data)

    # For better readability, move the 'Method' column to the front
    if 'Method' in summary_df.columns:
        method_col = summary_df.pop('Method')
        summary_df.insert(0, 'Method', method_col)
        
        # --- Sort columns for better readability ---
        # Get all other columns and sort them alphabetically
        other_cols = sorted([col for col in summary_df.columns if col != 'Method'])
        # Create the new column order
        new_order = ['Method'] + other_cols
        summary_df = summary_df[new_order]


    # Save the final summary DataFrame to a CSV file
    try:
        # Ensure the output directory exists
        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
        summary_df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"\nConsolidated summary has been successfully saved to: {os.path.abspath(output_file)}")

    except Exception as e:
        print(f"\nError saving the final summary file: {e}")


# --- Main execution block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Consolidates multiple result CSV files into a single summary file.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument(
        "-i", "--input_dir", 
        type=str, 
        default="results_sum", 
        help="Directory containing the input CSV files (*_results.csv).\n(Default: current working directory)"
    )
    parser.add_argument(
        "-o", "--output_file", 
        type=str, 
        default="overall_results.csv",
        help="Path for the final output CSV file.\n(Default: 'overall_results.csv' in the current directory)"
    )
    
    args = parser.parse_args()
    
    consolidate_results(args.input_dir, args.output_file)