import json
import os
import re
import matplotlib.pyplot as plt

def natural_sort_key(s):
    """
    Creates a key for natural sorting to ensure layers are ordered correctly.
    e.g. "vivit.encoder.layer.10" -> ("vivit.encoder.layer.", 10)
    """
    match = re.match(r'^(.*?)(\d+)$', s)
    if match:
        prefix, number = match.groups()
        return (prefix, int(number))
    return (s, -1) # For keys without numbers like "embeddings"

def transform_legend_label(path_string):
    """
    Transforms a file path into a 'gamename_nepoch' format for the plot legend.
    Example: "./apex/0epoch/..." -> "apex_0epoch"
    """
    if not isinstance(path_string, str):
        return "N/A"

    # Normalize path separators for consistency
    path_string = path_string.replace('\\', '/')
    
    # Extract game name, which is assumed to be the directory before the epoch folder
    # e.g., in './game/1epoch/file.json', parts would be ['.', 'game', '1epoch', 'file.json']
    parts = path_string.split('/')
    gamename = parts[-3] if len(parts) >= 3 else "unknown"

    # Use regex to find the 'nepoch' string
    match = re.search(r'(\d+epoch)', path_string)
    if match:
        epoch_str = match.group(1)
        return f"{gamename}_{epoch_str}"
    
    return path_string # Fallback

def plot_from_json(file_path, output_dir):
    """
    Reads data from a specified JSON file, plots a comparison graph,
    and saves it to the specified output directory.
    """
    print(f"Processing '{file_path}'...")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return
    except json.JSONDecodeError:
        print(f"Error: The file '{file_path}' is not a valid JSON file.")
        return

    # Extract data, using .get() for safety
    shared_layers = data.get("shared_layers", [])
    param_l2_diff_file1 = data.get("param_l2_diff_file1", {})
    param_l2_diff_file2 = data.get("param_l2_diff_file2", {})
    spearman_stats = data.get("spearman", {})
    correlation = spearman_stats.get("correlation")
    p_value = spearman_stats.get("p_value")

    # The layers in the JSON are already naturally sorted by the previous script
    values1 = [param_l2_diff_file1.get(layer, 0) for layer in shared_layers]
    values2 = [param_l2_diff_file2.get(layer, 0) for layer in shared_layers]

    # Shorten the X-axis labels for better readability
    short_labels = [
        label.replace("vivit.encoder.layer.", "L").replace("embeddings", "Emb") 
        for label in shared_layers
    ]
    
    legend_label1 = transform_legend_label(data.get("file1"))
    legend_label2 = transform_legend_label(data.get("file2"))

    # Create the plot
    fig, ax = plt.subplots(figsize=(14, 8))

    # Plot the two line graphs
    ax.plot(short_labels, values1, marker='o', linestyle='--', label=legend_label1)
    ax.plot(short_labels, values2, marker='s', linestyle='-', label=legend_label2)

    # Style the plot
    ax.set_title('Comparison of Parameter L2 Difference Across Layers', fontsize=18)
    ax.set_xlabel('Model Layer', fontsize=14)
    ax.set_ylabel('Parameter L2 Difference Value', fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.xticks(rotation=45, ha='right')

    # Add Spearman correlation annotation
    if correlation is not None and p_value is not None:
        stats_text = (f"Spearman Correlation: {correlation:.4f}\n"
                      f"p-value: {p_value:.2e}")
        props = dict(boxstyle='round,pad=0.5', facecolor='aliceblue', alpha=0.8)
        ax.text(0.97, 0.97, stats_text, transform=ax.transAxes, fontsize=12,
                verticalalignment='top', horizontalalignment='right', bbox=props)

    plt.tight_layout()

    # Save the plot
    base_name = os.path.basename(file_path)
    file_name_without_ext = os.path.splitext(base_name)[0]
    output_filename = f"{file_name_without_ext}.png"
    output_path = os.path.join(output_dir, output_filename)
    
    try:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot successfully saved to: {output_path}\n")
    except Exception as e:
        print(f"Error saving plot: {e}\n")
    
    plt.close(fig) # Close the figure to free up memory

def main():
    """
    Finds all correlation JSON files in the 'spearman_results' directory,
    and generates a plot for each one.
    """
    input_dir = "spearman_results"
    output_dir = "spearman_plots"
    
    if not os.path.isdir(input_dir):
        print(f"Error: Input directory '{input_dir}' not found.")
        print("Please run the data generation script first to create the results.")
        return

    os.makedirs(output_dir, exist_ok=True)
    
    json_files = [f for f in os.listdir(input_dir) if f.endswith('_corr.json')]
    
    if not json_files:
        print(f"No JSON files found in '{input_dir}'.")
        return
        
    print(f"Found {len(json_files)} files to plot.")
    for filename in json_files:
        file_path = os.path.join(input_dir, filename)
        plot_from_json(file_path, output_dir)

if __name__ == '__main__':
    main()