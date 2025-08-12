import pandas as pd
import os

# --- User Configuration ---

# 1. Set the path to the folder to be scanned
#    '.' represents the current folder. You can also specify an absolute path, e.g., 'C:/Users/YourUser/Documents/GameData'
FOLDER_PATH = '.'

# 2. Define the three labels to be counted
#    Please modify according to the actual label names in your data.
LABELS_TO_COUNT = ['up', 'down', 'same']

# 3. Set the name of the output file
OUTPUT_FILENAME = 'data_statistics.csv'


def analyze_csv_files(folder_path, labels, output_file):
    """
    Iterates through CSV files, counts the occurrences of specified labels,
    and saves the results to a new CSV file.

    :param folder_path: str, The path to the folder containing the CSV files.
    :param labels: list, A list of labels to count.
    :param output_file: str, The name of the output CSV file.
    """
    # Prepare a list to store the statistics for all files
    all_statistics = []

    print(f"--- Start scanning folder: {os.path.abspath(folder_path)} ---")

    # Iterate through all files and directories in the specified path
    try:
        file_list = os.listdir(folder_path)
    except FileNotFoundError:
        print(f"Error: The specified folder '{folder_path}' was not found. Please check the path.")
        return

    for filename in file_list:
        # Check if the filename matches the 'new_{game_name}.csv' format
        if filename.startswith('new_') and filename.endswith('.csv'):
            
            # Extract game_name from the filename
            game_name = filename.replace('new_', '', 1).replace('.csv', '')
            print(f"  [Processing] File: {filename} (Game: {game_name})")
            
            file_path = os.path.join(folder_path, filename)
            
            try:
                # Read the CSV file using pandas
                df = pd.read_csv(file_path)
                
                # Check if the key column 'arousal_change' exists
                if 'arousal_change' in df.columns:
                    # Use value_counts() to efficiently count each label
                    label_counts = df['arousal_change'].value_counts()
                    
                    # Create a dictionary to store the statistics for the current file
                    stats = {'game_name': game_name}
                    for label in labels:
                        # Get the count for the label from the results, defaulting to 0 if the label is not present
                        stats[f'{label}_count'] = label_counts.get(label, 0)
                    
                    # Add the statistics for the current file to the main list
                    all_statistics.append(stats)
                else:
                    print(f"    -> Warning: File '{filename}' is missing the 'arousal_change' column. Skipping.")

            except pd.errors.EmptyDataError:
                print(f"    -> Warning: File '{filename}' is empty. Skipping.")
            except Exception as e:
                print(f"    -> Error: An unknown error occurred while processing file '{filename}': {e}")

    # Check if any data was collected
    if not all_statistics:
        print("\n--- Scan complete, but no valid data was found to process. ---")
        return

    # Convert the list of statistics into a pandas DataFrame
    summary_df = pd.DataFrame(all_statistics)
    
    # Ensure the column order is predictable (game_name, label1_count, label2_count, ...)
    column_order = ['game_name'] + [f'{label}_count' for label in labels]
    summary_df = summary_df[column_order]

    # Save the final DataFrame to a CSV file
    try:
        summary_df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"\n--- Statistics complete! Results have been successfully saved to: {output_file} ---")
        print("Generated data preview:")
        print(summary_df)
    except Exception as e:
        print(f"\nError: Failed to save results to file '{output_file}': {e}")


# --- Run Script ---
if __name__ == "__main__":
    analyze_csv_files(FOLDER_PATH, LABELS_TO_COUNT, OUTPUT_FILENAME)
