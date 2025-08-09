import json
import os
import re
import matplotlib.pyplot as plt

def natural_sort_key(s):
    """Creates a key for natural sorting."""
    match = re.match(r'^(.*?)(\d+)$', s)
    if match:
        prefix, number = match.groups()
        return (prefix, int(number))
    return (s, -1)

def transform_legend_label(path_string):
    """Converts file path to gamename_nepoch format."""
    if not isinstance(path_string, str):
        return "N/A"

    path_string = path_string.replace('\\', '/')
    parts = path_string.split('/')
    gamename = parts[-3] if len(parts) >= 3 else "unknown"
    match = re.search(r'(\d+epoch)', path_string)
    if match:
        return f"{gamename}_{match.group(1)}"
    return path_string

def plot_from_json(file_path, output_dir):
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

    shared_layers = data.get("shared_layers", [])
    param_l2_diff_file1 = data.get("param_l2_diff_file1", {})
    param_l2_diff_file2 = data.get("param_l2_diff_file2", {})
    spearman_stats = data.get("spearman", {})
    correlation = spearman_stats.get("correlation")
    p_value = spearman_stats.get("p_value")

    values1 = [param_l2_diff_file1.get(layer, 0) for layer in shared_layers]
    values2 = [param_l2_diff_file2.get(layer, 0) for layer in shared_layers]

    short_labels = [
        label.replace("vivit.encoder.layer.", "L").replace("embeddings", "Emb")
        for label in shared_layers
    ]

    legend_label1 = transform_legend_label(data.get("file1"))
    legend_label2 = transform_legend_label(data.get("file2"))

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.plot(short_labels, values1, marker='o', linestyle='--', label=legend_label1, linewidth=3.5)
    ax.plot(short_labels, values2, marker='s', linestyle='-', label=legend_label2, linewidth=3.5)

    ax.set_title('Comparison of Parameter L2 Difference Across Layers', fontsize=20)
    ax.set_xlabel('Model Layer', fontsize=16)
    ax.set_ylabel('Parameter L2 Difference Value', fontsize=16)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.tick_params(axis='both', which='major', labelsize=14) 
    plt.xticks(rotation=45, ha='right')

    # Dynamic legend placement under stats box
    legend_y = 0.97
    inv = ax.transAxes.inverted()

    if correlation is not None and p_value is not None:
        if (p_value < 0.001) and (p_value!=0):
            p_val_str = f"{p_value:.2e}"
            mantissa, exponent = p_val_str.split('e')
            exponent_val = int(exponent)
            p_text = fr"${mantissa} \times 10^{{{exponent_val}}}$"
        else:
            p_text = f"{p_value:.4f}"

        stats_text = (f"Spearman Correlation: {correlation:.4f}\n"
                      f"p-value: {p_text}")
        props = dict(boxstyle='round,pad=0.5', facecolor='aliceblue', alpha=0.8)

        text_obj = ax.text(0.97, 0.97, stats_text, transform=ax.transAxes, fontsize=25,
                           verticalalignment='top', horizontalalignment='right', bbox=props)

        plt.draw()
        bbox = text_obj.get_window_extent(renderer=fig.canvas.get_renderer())
        bbox_ax = inv.transform(bbox)
        legend_y = bbox_ax[0, 1] - 0.02


        def overlaps_line(y_anchor):
            legend_height = 0.08
            y_range = (y_anchor - legend_height, y_anchor)
            x_range = (0.75, 1.0)

            x_mapping = {label: idx for idx, label in enumerate(short_labels)}

            for line in ax.lines:
                x_data = line.get_xdata()
                y_data = line.get_ydata()
                for x_val, y_val in zip(x_data, y_data):
                    if isinstance(x_val, str):
                        if x_val not in x_mapping:
                            continue
                        x_val = x_mapping[x_val]
                    x_ax, y_ax = ax.transData.transform((x_val, y_val))
                    x_ax, y_ax = inv.transform((x_ax, y_ax))
                    if x_range[0] <= x_ax <= x_range[1] and y_range[0] <= y_ax <= y_range[1]:
                        return True
            return False

        while overlaps_line(legend_y) and legend_y > 0.1:
            legend_y -= 0.05

    if legend_y <= 0.1:
        ax.legend(loc='lower right', fontsize=14)
    else:
        ax.legend(loc='upper right', bbox_to_anchor=(0.97, legend_y), fontsize=14)

    plt.tight_layout()

    base_name = os.path.basename(file_path)
    file_name_without_ext = os.path.splitext(base_name)[0]
    output_filename = f"{file_name_without_ext}.png"
    output_path = os.path.join(output_dir, output_filename)

    try:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot successfully saved to: {output_path}\n")
    except Exception as e:
        print(f"Error saving plot: {e}\n")

    plt.close(fig)

def main():
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
