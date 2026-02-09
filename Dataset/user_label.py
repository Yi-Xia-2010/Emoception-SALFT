import pandas as pd
import matplotlib.pyplot as plt
import os


INPUT_FILE = 'new_solid.csv'           
OUTPUT_CSV = 'arousal_pid_stats.csv'
OUTPUT_IMG = 'pid_distribution_plot.png' 


def extract_pid(path_str):

    if not isinstance(path_str, str):
        return str(path_str)

    filename = path_str.replace('\\', '/').split('/')[-1]
    
    pid = filename.split('_')[0]
    
    return pid

def main():
    print(f"--- 开始处理 {INPUT_FILE} ---")
    
    try:
        df = pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        print(f"错误：找不到文件 {INPUT_FILE}")
        return

    df.columns = df.columns.str.strip()
    
    path_col = next((c for c in df.columns if 'path' in c.lower() or 'video' in c.lower()), 'video_path')
    change_col = next((c for c in df.columns if 'change' in c.lower() or 'arousal' in c.lower()), 'arousal_change')
    
    print(f"正在使用列名: 路径列='{path_col}', 变化列='{change_col}'")

    df[path_col] = df[path_col].astype(str).str.strip()
    df[change_col] = df[change_col].astype(str).str.strip()

    df['PID'] = df[path_col].apply(extract_pid)
    print("PID 提取完成，示例:", df['PID'].iloc[0] if not df.empty else "无数据")

    counts = pd.crosstab(df['PID'], df[change_col])
    props = pd.crosstab(df['PID'], df[change_col], normalize='index')
    
    full_stats = pd.concat([counts, (props*100).round(2).add_suffix('_%')], axis=1)
    full_stats.to_csv(OUTPUT_CSV)
    print(f"全量 PID 统计表已保存至: {OUTPUT_CSV}")

    selected_pids = []
    target_cols = ['up', 'down', 'neutral'] 
    
    available_cols = props.columns.tolist()
    
    for target in target_cols:
        match = next((col for col in available_cols if col.lower() == target), None)
        if match:
            top_3 = props.nlargest(3, match).index.tolist()
            selected_pids.extend(top_3)
    
    selected_pids = list(dict.fromkeys(selected_pids))
    
    if not selected_pids:
        print("警告：未能筛选出特定玩家，可能是列名(up/down/neutral)不匹配。将随机展示 9 个。")
        selected_pids = props.index[:9].tolist()

    subset = props.loc[selected_pids]
    
    sort_col = next((col for col in subset.columns if 'up' in col.lower()), subset.columns[0])
    subset = subset.sort_values(by=sort_col)

    color_map = {'up': '#d73027', 'neutral': '#e0e0e0', 'down': '#4575b4'}

    plot_colors = []
    for col in subset.columns:
        c = '#999999' 
        for k, v in color_map.items():
            if k in col.lower():
                c = v
        plot_colors.append(c)

    fig, ax = plt.subplots(figsize=(10, 6))
    subset.plot(kind='barh', stacked=True, color=plot_colors, ax=ax, edgecolor='white')


    ax.set_title("Arousal Distribution by Player ID (Top Divergent Samples)", fontsize=13)
    ax.set_xlabel("Proportion")
    ax.set_ylabel("Player ID") 
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    
    plt.savefig(OUTPUT_IMG, dpi=300)
    print(f"图表已生成: {OUTPUT_IMG}")
    plt.show()

if __name__ == "__main__":
    main()