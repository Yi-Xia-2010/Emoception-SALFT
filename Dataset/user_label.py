import pandas as pd
import matplotlib.pyplot as plt
import os

# ================= Configuration =================
INPUT_FILE = 'new_solid.csv'           
OUTPUT_CSV = 'arousal_pid_stats.csv'
OUTPUT_IMG = 'pid_distribution_plot.png' 
# =============================================

def extract_pid(path_str):
    """
    Extracts the Player ID (PID) from the file path.
    Logic: Takes the filename (after last /), then the part before the first _.
    """
    if not isinstance(path_str, str):
        return str(path_str)

    # Normalize separators and get filename
    filename = path_str.replace('\\', '/').split('/')[-1]
    
    # Extract PID (text before the first underscore)
    pid = filename.split('_')[0]
    
    return pid

def main():
    print(f"--- Processing {INPUT_FILE} ---")
    
    # 1. Read Data
    try:
        df = pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        print(f"Error: File {INPUT_FILE} not found.")
        return

    # 2. Data Cleaning
    df.columns = df.columns.str.strip()
    
    # Auto-detect columns
    path_col = next((c for c in df.columns if 'path' in c.lower() or 'video' in c.lower()), 'video_path')
    change_col = next((c for c in df.columns if 'change' in c.lower() or 'arousal' in c.lower()), 'arousal_change')
    
    print(f"Using columns: Path='{path_col}', Change='{change_col}'")

    df[path_col] = df[path_col].astype(str).str.strip()
    df[change_col] = df[change_col].astype(str).str.strip()

    # 3. Extract PID
    df['PID'] = df[path_col].apply(extract_pid)
    # Check if data exists before accessing index 0 to avoid errors on empty files
    example_pid = df['PID'].iloc[0] if not df.empty else "No Data"
    print(f"PID extraction complete. Example: {example_pid}")

    # 4. Calculate Distribution
    counts = pd.crosstab(df['PID'], df[change_col])
    props = pd.crosstab(df['PID'], df[change_col], normalize='index')
    
    # Save full statistics
    full_stats = pd.concat([counts, (props*100).round(2).add_suffix('_%')], axis=1)
    full_stats.to_csv(OUTPUT_CSV)
    print(f"Full PID statistics saved to: {OUTPUT_CSV}")

    # 5. Select Representative Players (Top 3 Up/Down/Neutral)
    selected_pids = []
    target_cols = ['up', 'down', 'neutral'] 
    
    available_cols = props.columns.tolist()
    
    for target in target_cols:
        # Case-insensitive match for column names
        match = next((col for col in available_cols if col.lower() == target), None)
        if match:
            # Get top 3 PIDs for this category
            top_3 = props.nlargest(3, match).index.tolist()
            selected_pids.extend(top_3)
    
    # Remove duplicates
    selected_pids = list(dict.fromkeys(selected_pids))
    
    # Fallback if selection fails
    if not selected_pids:
        print("Warning: Could not select specific players (check column names 'up'/'down'/'neutral'). Displaying first 9.")
        selected_pids = props.index[:9].tolist()

    subset = props.loc[selected_pids]
    
    # Sort by 'up' column for better visualization (if it exists)
    sort_col = next((col for col in subset.columns if 'up' in col.lower()), subset.columns[0])
    subset = subset.sort_values(by=sort_col)

    # 6. Plotting
    # Define colors: Up=Red, Neutral=Grey, Down=Blue
    color_map = {'up': '#d73027', 'neutral': '#e0e0e0', 'down': '#4575b4'}

    # Assign colors to columns dynamically
    plot_colors = []
    for col in subset.columns:
        c = '#999999' # Default grey if no match
        for k, v in color_map.items():
            if k in col.lower():
                c = v
        plot_colors.append(c)

    fig, ax = plt.subplots(figsize=(10, 6))
    subset.plot(kind='barh', stacked=True, color=plot_colors, ax=ax, edgecolor='white')

    # Chart Styling
    ax.set_title("Arousal Distribution by Player ID (Top Divergent Samples)", fontsize=13)
    ax.set_xlabel("Proportion")
    ax.set_ylabel("Player ID") 
    
    # Remove top and right borders
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Legend position
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    
    plt.savefig(OUTPUT_IMG, dpi=300)
    print(f"Chart generated: {OUTPUT_IMG}")
    plt.show()

if __name__ == "__main__":
    main()