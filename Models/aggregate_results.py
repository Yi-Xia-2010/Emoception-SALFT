import os
import json
import glob
import numpy as np
import scipy.stats as stats

def get_class_names(data, file_type):
    """ Extract class names based on file type (RF or DL). """
    if file_type == 'rf':
        try:
            if "avg_class_metrics" in data: return list(data["avg_class_metrics"].keys())
            if "class_metrics" in data: return list(data["class_metrics"].keys())
        except: pass
    elif file_type == 'dl':
        try:
            if "fold_results" in data and len(data["fold_results"]) > 0:
                metrics = data["fold_results"][0].get("class_metrics", {})
                return list(metrics.keys())
        except: pass
    return ["down", "same", "up"]

def calculate_stats_from_list(scores):
    """
    Helper: Calculate Mean and 95% CI from a list of scores.
    Returns: (mean, ci95)
    """
    if not scores: return 0.0, 0.0
    n = len(scores)
    mean_val = np.mean(scores)
    if n < 2: return mean_val, 0.0
    
    # Use Sample Standard Deviation (ddof=1)
    std_val = np.std(scores, ddof=1)
    se = std_val / np.sqrt(n)
    # Calculate t-score for 95% confidence
    t_score = stats.t.ppf((1 + 0.95) / 2., n - 1)
    ci95 = se * t_score
    return mean_val, ci95

def calculate_ci95_from_std(std_val, n):
    """ Helper: Calculate CI95 if we only have pre-calculated Std. """
    if n < 2: return 0.0
    se = std_val / np.sqrt(n)
    t_score = stats.t.ppf((1 + 0.95) / 2., n - 1)
    return se * t_score

def format_confusion_matrix(cm_percent, class_names):
    if not cm_percent: return "No Confusion Matrix Data Found"
    col_width = 12
    row_header_width = 15
    header = " " * row_header_width
    for cls in class_names: header += f"Pred_{cls}".rjust(col_width)
    lines = ["=== Confusion Matrix (Percentages %) ===", header]
    for i, row in enumerate(cm_percent):
        actual_label = class_names[i] if i < len(class_names) else f"Class_{i}"
        line = f"Actual_{actual_label}".ljust(row_header_width)
        for val in row: line += f"{val:.2f}".rjust(col_width)
        lines.append(line)
    return "\n".join(lines)

def collect_rf_folds_cm(summary_file_path):
    base_dir = os.path.dirname(summary_file_path)
    fold_files = glob.glob(os.path.join(base_dir, "fold_*", "results.json"))
    if not fold_files: return []
    agg_cm = None
    try:
        for fpath in fold_files:
            with open(fpath, 'r', encoding='utf-8') as f:
                fold_data = json.load(f)
            if "confusion_matrix" in fold_data:
                cm = np.array(fold_data["confusion_matrix"])
                if agg_cm is None: agg_cm = cm
                else: agg_cm += cm
        if agg_cm is not None:
            with np.errstate(divide='ignore', invalid='ignore'):
                agg_cm_percent = agg_cm.astype('float') / agg_cm.sum(axis=1)[:, np.newaxis] * 100
                agg_cm_percent = np.nan_to_num(agg_cm_percent)
            return np.round(agg_cm_percent, 2).tolist()
    except: pass
    return []

def process_file(file_path):
    filename = os.path.basename(file_path)
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # --- Path Parsing ---
    try:
        game_dir = os.path.dirname(file_path)
        game_name = os.path.basename(game_dir)
        method_dir = os.path.dirname(game_dir)
        method_name = os.path.basename(method_dir)
        experiment_name = f"{method_name}/{game_name}"
    except:
        method_name = "unknown"; game_name = "unknown"; experiment_name = file_path

    # Initialize result dictionary
    res = {
        "name": experiment_name, "method": method_name, "game": game_name,
        "f1": 0.0, "f1_ci": 0.0,
        "acc": 0.0, "acc_ci": 0.0,
        "macro": 0.0, "macro_ci": 0.0,
        "cm": [], "classes": []
    }

    # =========================================
    # MODE A: Random Forest (summary.json)
    # =========================================
    if filename == "summary.json":
        n_folds = 5
        # 1. Weighted F1
        w_f1 = data.get("weighted_f1", {})
        res["f1"] = w_f1.get("mean", 0.0)
        res["f1_ci"] = calculate_ci95_from_std(w_f1.get("std", 0.0), n_folds)
        
        # 2. Accuracy
        acc_data = data.get("accuracy", {})
        res["acc"] = acc_data.get("mean", 0.0)
        res["acc_ci"] = calculate_ci95_from_std(acc_data.get("std", 0.0), n_folds)

        # 3. Macro F1
        macro_data = data.get("macro_f1", {})
        res["macro"] = macro_data.get("mean", 0.0)
        res["macro_ci"] = calculate_ci95_from_std(macro_data.get("std", 0.0), n_folds)
        
        res["classes"] = get_class_names(data, 'rf')

        # CM Extraction
        if "aggregated_confusion_matrix_percent" in data:
            res["cm"] = data["aggregated_confusion_matrix_percent"]
        elif "aggregate_confusion_matrix" in data:
            try:
                raw_cm = np.array(data["aggregate_confusion_matrix"])
                with np.errstate(divide='ignore', invalid='ignore'):
                    calc_percent = raw_cm.astype('float') / raw_cm.sum(axis=1)[:, np.newaxis] * 100
                    calc_percent = np.nan_to_num(calc_percent)
                res["cm"] = np.round(calc_percent, 2).tolist()
            except: pass
        if not res["cm"]: res["cm"] = collect_rf_folds_cm(file_path)
        
    # =========================================
    # MODE B: Deep Learning (aggregated_cv_results.json)
    # =========================================
    else:
        # CM Auto-Aggregation
        if "aggregated_confusion_matrix_percent" not in data and "fold_results" in data:
            try:
                first_cm = data["fold_results"][0]["confusion_matrix"]
                cm_shape = np.array(first_cm).shape
                agg_cm = np.zeros(cm_shape, dtype=int)
                for fold in data["fold_results"]:
                    agg_cm += np.array(fold["confusion_matrix"])
                with np.errstate(divide='ignore', invalid='ignore'):
                    agg_cm_percent = agg_cm.astype('float') / agg_cm.sum(axis=1)[:, np.newaxis] * 100
                    agg_cm_percent = np.nan_to_num(agg_cm_percent)
                data["aggregated_confusion_matrix_percent"] = np.round(agg_cm_percent, 2).tolist()
            except: pass
        res["cm"] = data.get("aggregated_confusion_matrix_percent", [])
        res["classes"] = get_class_names(data, 'dl')
            
        # 🔥 Metrics Calculation (Collecting lists for CI calculation)
        f1_list, acc_list, macro_list = [], [], []
        
        if "fold_results" in data:
            for fold in data["fold_results"]:
                # Weighted F1
                if fold.get("f1_weighted") is not None: 
                    f1_list.append(fold.get("f1_weighted"))
                
                # Accuracy
                if fold.get("accuracy") is not None: 
                    acc_list.append(fold.get("accuracy"))
                
                # Macro F1 (Auto-Repair Logic)
                m_val = fold.get("f1_macro")
                if m_val is None and "class_metrics" in fold:
                    try:
                        vals = [m["f1_score"] for m in fold["class_metrics"].values()]
                        if vals: m_val = sum(vals) / len(vals)
                    except: pass
                if m_val is not None: 
                    macro_list.append(m_val)

        # 1. Calculate Mean and CI from Lists (Primary Method)
        res["f1"], res["f1_ci"] = calculate_stats_from_list(f1_list)
        res["acc"], res["acc_ci"] = calculate_stats_from_list(acc_list)
        res["macro"], res["macro_ci"] = calculate_stats_from_list(macro_list)

        # 2. Robust Fallback: Check top-level JSON keys if lists were empty
        # Weighted F1 Fallbacks
        if res["f1"] == 0.0:
            if "mean_f1_weighted" in data: res["f1"] = data["mean_f1_weighted"]
            elif "mean_f1" in data: res["f1"] = data["mean_f1"]
        
        if res["f1_ci"] == 0.0:
            if "ci95_f1_weighted" in data: res["f1_ci"] = data["ci95_f1_weighted"]
            elif "ci95_f1" in data: res["f1_ci"] = data["ci95_f1"]
            elif "std_f1_weighted" in data: res["f1_ci"] = calculate_ci95_from_std(data["std_f1_weighted"], 5)

        # Accuracy Fallbacks
        if res["acc"] == 0.0 and "mean_accuracy" in data: res["acc"] = data["mean_accuracy"]
        if res["acc_ci"] == 0.0 and "ci95_accuracy" in data: res["acc_ci"] = data["ci95_accuracy"]
        
        # Macro F1 Fallbacks
        if res["macro"] == 0.0 and "mean_f1_macro" in data: res["macro"] = data["mean_f1_macro"]
        # (Macro CI usually calculated from list, but if summary has std, could use here if field existed)

    return res

def main():
    base_dir = "Results/new/"
    output_cm_file = "confusion_matrices_report.txt"
    output_metrics_file = "f1_scores_summary.csv"
    
    patterns = [
        os.path.join(base_dir, "**", "aggregated_cv_results.json"),
        os.path.join(base_dir, "**", "summary.json")
    ]
    
    files = []
    for p in patterns: files.extend(glob.glob(p, recursive=True))
    
    if not files: 
        print(f"No files found in {base_dir}")
        return
    print(f"Searching in: {base_dir} | Found {len(files)} files.\n")
    
    summary_data = []
    for fp in files:
        try:
            summary_data.append(process_file(fp))
        except Exception as e:
            print(f"Error processing {fp}: {e}")
            
    # Sorting: Game (Asc), Weighted F1 (Desc)
    summary_data.sort(key=lambda x: (x["game"], -x["f1"]))

    # --- Console Output ---
    # Fmt: Game | Method | W-F1 (CI) | Acc (CI) | Macro (CI)
    header_fmt = "{:<15} | {:<20} | {:<18} | {:<18} | {:<18}"
    print("-" * 105)
    print(header_fmt.format("Game", "Method", "W-F1 (±CI)", "Acc (±CI)", "Macro (±CI)"))
    print("-" * 105)
    
    current_game = None
    for item in summary_data:
        if current_game != item['game']:
            if current_game is not None: print("-" * 105)
            current_game = item['game']

        # Format each metric as "0.5555 ±0.0123"
        s_f1 = f"{item['f1']:.4f} ±{item['f1_ci']:.4f}"
        s_acc = f"{item['acc']:.4f} ±{item['acc_ci']:.4f}"
        s_macro = f"{item['macro']:.4f} ±{item['macro_ci']:.4f}"

        print(header_fmt.format(item['game'], item['method'], s_f1, s_acc, s_macro))
    print("-" * 105)

    # --- CSV Output ---
    try:
        with open(output_metrics_file, 'w', encoding='utf-8') as f:
            # Add CI columns for all metrics
            f.write("Game,Method,Mean Weighted F1,CI95(W-F1),Mean Accuracy,CI95(Acc),Mean Macro F1,CI95(Macro)\n")
            for item in summary_data:
                f.write(f"{item['game']},{item['method']},"
                        f"{item['f1']:.6f},{item['f1_ci']:.6f},"
                        f"{item['acc']:.6f},{item['acc_ci']:.6f},"
                        f"{item['macro']:.6f},{item['macro_ci']:.6f}\n")
        print(f"\n[Saved] Metrics table: {output_metrics_file}")
    except Exception as e:
        print(f"Failed to save CSV: {e}")

    # --- TXT Output ---
    try:
        with open(output_cm_file, 'w', encoding='utf-8') as f:
            f.write("Aggregated Evaluation Report (Full Metrics)\n===========================================\n\n")
            for item in summary_data:
                f.write(f"Game:       {item['game']}\n")
                f.write(f"Method:     {item['method']}\n")
                f.write(f"Full Name:  {item['name']}\n")
                f.write(f"Weighted F1: {item['f1']:.4f} (± {item['f1_ci']:.4f})\n")
                f.write(f"Accuracy:    {item['acc']:.4f} (± {item['acc_ci']:.4f})\n")
                f.write(f"Macro F1:    {item['macro']:.4f} (± {item['macro_ci']:.4f})\n")
                f.write(format_confusion_matrix(item['cm'], item['classes']))
                f.write("\n\n" + "-"*40 + "\n\n")
        print(f"[Saved] Detailed report: {output_cm_file}")
    except Exception as e:
        print(f"Failed to save TXT report: {e}")

if __name__ == "__main__":
    main()