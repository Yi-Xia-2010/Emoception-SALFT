import os
import json
import pandas as pd
import argparse

def analyze_game_results(root_folder):
    """
    Iterates through all subfolders in the specified root folder, reads the 
    'test_results.json' file in each subfolder, and extracts the 'Accuracy', 
    'Weighted avg f1', and 'f1_macro' metrics.

    Args:
        root_folder (str): The path to the root directory containing all game subfolders.

    Returns:
        pandas.DataFrame: A DataFrame containing the game name, Accuracy, 
                          Weighted avg f1, and f1_macro. Returns an empty 
                          DataFrame if no data is found.
    """
    # A list to store all game results.
    results_data = []

    # Check if the root folder exists.
    if not os.path.isdir(root_folder):
        print(f"Error: Folder '{root_folder}' does not exist or is not a valid directory.")
        return pd.DataFrame()

    # Iterate over all items in the root folder.
    for game_name in os.listdir(root_folder):
        game_folder_path = os.path.join(root_folder, game_name)

        # Ensure it is a directory.
        if os.path.isdir(game_folder_path):
            json_file_path = os.path.join(game_folder_path, 'test_results.json')

            # Check if 'test_results.json' file exists.
            if os.path.exists(json_file_path):
                try:
                    with open(json_file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)

                    accuracy = None
                    weighted_f1 = None
                    # --- Added f1_macro variable ---
                    f1_macro = None

                    # --- Updated data extraction logic to handle multiple formats ---

                    # Format 1: A dictionary containing metrics directly
                    # { "accuracy": 0.78, "f1_weighted": 0.78, "f1_macro": 0.77, ... }
                    if isinstance(data, dict):
                        accuracy = data.get('accuracy')
                        weighted_f1 = data.get('f1_weighted')
                        # --- Added f1_macro extraction ---
                        f1_macro = data.get('f1_macro')

                    # Format 2: A list containing a dictionary (old and new style)
                    # [{ "MODEL_NAME": { "Accuracy": 0.76, "Weighted avg f1": 0.76, ... } }]
                    # OR [{ "accuracy": 0.75, "f1_weighted": 0.75, "f1_macro": 0.74, ... }]
                    elif isinstance(data, list) and data and isinstance(data[0], dict):
                        # This handles both the sample you provided and the old format
                        metrics_dict = data[0]
                        
                        # Check for new format keys first within the list
                        if 'accuracy' in metrics_dict:
                            accuracy = metrics_dict.get('accuracy')
                            weighted_f1 = metrics_dict.get('f1_weighted')
                            f1_macro = metrics_dict.get('f1_macro')
                        # Fallback to old format
                        elif metrics_dict: 
                            # Get the first value of the inner dictionary, regardless of its key name
                            metrics = list(metrics_dict.values())[0]
                            if isinstance(metrics, dict):
                                accuracy = metrics.get('Accuracy')
                                weighted_f1 = metrics.get('Weighted avg f1')
                                # --- Added f1_macro extraction for old format ---
                                f1_macro = metrics.get('f1_macro')

                    # --- End of logic ---

                    # --- Updated condition to include f1_macro ---
                    if accuracy is not None and weighted_f1 is not None and f1_macro is not None:
                        # Convert values to float and round to 4 decimal places before appending.
                        results_data.append({
                            'Game': game_name,
                            'Accuracy': round(float(accuracy), 4),
                            'Weighted avg f1': round(float(weighted_f1), 4),
                            # --- Added f1_macro to the results ---
                            'f1_macro': round(float(f1_macro), 4)
                        })
                    else:
                        print(f"Warning: Required metrics not found or format not recognized in file '{json_file_path}'. Skipping.")

                except json.JSONDecodeError:
                    print(f"Error: Failed to parse file '{json_file_path}'. Please check if it is a valid JSON.")
                except Exception as e:
                    print(f"An unknown error occurred while processing file '{json_file_path}': {e}")

    # If the list is not empty, create a DataFrame using pandas.
    if results_data:
        df = pd.DataFrame(results_data)
        return df
    else:
        print("No valid result data found.")
        return pd.DataFrame()

# --- Main execution block ---
if __name__ == "__main__":
    # --- Set up command-line argument parsing ---
    parser = argparse.ArgumentParser(
        description="Scans game folders, extracts performance metrics from 'test_results.json', and saves them to a CSV file.",
        formatter_class=argparse.RawTextHelpFormatter # Keep help message formatting
    )
    
    parser.add_argument(
        "-f", "--folder_path", 
        type=str, 
        default="continue_finetuning", 
        help="Path to the root directory containing all game subfolders.\n(Default: 'continue_finetuning')"
    )
    parser.add_argument(
        "-o", "--output_dir", 
        type=str, 
        default="results_sum", 
        help="Directory path to save the CSV file.\n(Default: 'results_sum')"
    )
    
    args = parser.parse_args()

    # Pass the parsed path to the main function and get the results.
    results_df = analyze_game_results(args.folder_path)

    # If the DataFrame was generated successfully, print and save it.
    if not results_df.empty:
        print("\n--- Game Performance Metrics Summary ---")
        print(results_df.set_index('Game'))
        print("--------------------------------------\n")

        # --- Save to CSV file ---
        try:
            # Ensure the output directory exists, create it if it doesn't.
            os.makedirs(args.output_dir, exist_ok=True)

            # Get the folder name from the input path to use as the CSV filename.
            folder_name = os.path.basename(os.path.normpath(args.folder_path))
            # If the folder path is the default '.', name the file based on the current directory's name.
            if folder_name in ['.', '']:
                folder_name = os.path.basename(os.getcwd())
            csv_filename = f"{folder_name}_results.csv"
            output_path = os.path.join(args.output_dir, csv_filename)
            
            # Save DataFrame to CSV, using utf-8-sig encoding for better Excel compatibility.
            results_df.to_csv(output_path, index=False, encoding='utf-8-sig')
            
            print(f"Results have been successfully saved to: {os.path.abspath(output_path)}")

        except Exception as e:
            print(f"Error saving file: {e}")