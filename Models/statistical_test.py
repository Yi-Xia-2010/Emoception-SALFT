import os
import json
import numpy as np
import pandas as pd
from scipy import stats
import warnings

# ================= Configuration Area =================

# 1. List of games to process in batch
GAMES = ["apex", "endless", "fps", "gallery", "solid", "gun", "platform", "tiny", "topdown"] 

# 2. Default path structure for 'Ours' and 'Full FT'
# Logic: {ROOT}/{METHOD}/{GAME}/{FILENAME}
# Please ensure the folder names (case-sensitive) match the directory 
DATA_ROOT = "Results/new"
OURS_DIR  = "vivit_salft"
FULL_DIR  = "full"
DEFAULT_FILENAME = "aggregated_cv_results.json"

# 3. Baseline path configuration (Specific per game)
BASELINE_MAP = {
    "apex":   "Results/new/full/apex/aggregated_cv_results.json",
    "endless":    "Results/new/resnet_full/endless/aggregated_cv_results.json",
    "fps":    "Results/new/resnet_full/fps/aggregated_cv_results.json",
    "gallery": "Results/new/full/gallery/aggregated_cv_results.json", 
    "solid":   "Results/new/full/solid/aggregated_cv_results.json",
    "gun":    "Results/new/full/gun/aggregated_cv_results.json",
    "platform":    "Results/new/full/platform/aggregated_cv_results.json",
    "tiny":    "Results/new/full/tiny/aggregated_cv_results.json",
    "topdown": "Results/new/resnet_full/topdown/aggregated_cv_results.json", 
}

# 4. Metrics to evaluate
METRICS = ["f1_weighted", "f1_macro", "accuracy"]

# ===================================================================

def load_scores(file_path, method_label, metric_name):
    """
    Universal Loader: Supports multiple formats and automatically repairs Macro F1.
    """
    if not os.path.exists(file_path):
        # Only print warning if non-Baseline file is missing (Baselines might be optional)
        if "Baseline" not in method_label:
            print(f"  [Missing] File not found: {file_path}")
        return None

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        scores = []
        
        # --- Case A: fold_results (List of Dicts) ---
        # Applicable to: Ours, Full FT, and LoRA Baseline
        if "fold_results" in data:
            for i, fold in enumerate(data["fold_results"]):
                val = fold.get(metric_name)
                
                #  [Auto-Repair] For Full FT/LoRA missing f1_macro
                if val is None and metric_name == "f1_macro":
                    if "class_metrics" in fold:
                        try:
                            c_metrics = fold["class_metrics"]
                            f1_values = [m["f1_score"] for m in c_metrics.values()]
                            if f1_values:
                                val = sum(f1_values) / len(f1_values)
                                if i == 0: 
                                    print(f"   🔧 [Auto-Repair] {method_label}: Calculated f1_macro from class_metrics")
                        except: pass
                
                scores.append(val)
                
            if any(s is None for s in scores):
                return None
            return np.array(scores)

        # --- Case B: fold_details (Dict of Lists) ---
        # Only applicable to: RF Baseline (summary.json)
        elif "fold_details" in data:
            mapping = {
                "accuracy": "accuracy",
                "f1_weighted": "weighted_f1s",
                "f1_macro": "macro_f1s"
            }
            target_key = mapping.get(metric_name, metric_name)
            details = data["fold_details"]
            
            if target_key in details:
                return np.array(details[target_key])
            else:
                return None
        else:
            return None
            
    except Exception as e:
        print(f" [Exception] Failed to read {file_path}: {e}")
        return None

def get_sig_stars(p_value):
    if p_value < 0.001: return "***"
    if p_value < 0.01:  return "**"
    if p_value < 0.05:  return "*"
    return "ns"


def analyze_pair(scores_a, scores_b):
    if len(scores_a) != len(scores_b): 
        return np.nan, np.nan, "LenErr"
    
    diffs = scores_a - scores_b
    mean_diff = np.mean(diffs)
    
    if np.all(diffs == 0):
        return mean_diff, 1.0, "ns"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            _, p_val = stats.wilcoxon(
                scores_a, 
                scores_b, 
                alternative='two-sided', 
                method='exact',          
                zero_method='pratt'
            )
        except ValueError as e:
            print(f" [Warning] Wilcoxon failed: {e}")
            return mean_diff, 1.0, "ns"
            
    return mean_diff, p_val, get_sig_stars(p_val)

def main():
    print(f"\n=== Batch Statistical Test Report (Games: {len(GAMES)}) ===\n")
    results_list = []

    for metric in METRICS:
        print(f"\n Testing Metric: {metric.upper()}")
        print("=" * 100)
        
        for game in GAMES:
            # 1. Automatically build paths for Ours and Full FT
            path_ours = os.path.join(DATA_ROOT, OURS_DIR, game, DEFAULT_FILENAME)
            path_full = os.path.join(DATA_ROOT, FULL_DIR, game, DEFAULT_FILENAME)
            
            # 2. Get Baseline path (from MAP)
            path_base = BASELINE_MAP.get(game)

            # 3. Load Ours Data (Primary)
            scores_ours = load_scores(path_ours, f"Ours({game})", metric)
            if scores_ours is None: 
                print(f" Skipping {game}: Ours results not found")
                continue

            row = {
                "Metric": metric,
                "Game": game,
                "Ours (Mean±Std)": f"{scores_ours.mean():.4f} ± {scores_ours.std():.4f}"
            }
            print(f" {game:<10} | Ours Loaded", end="")

            # 4. Compare vs Full FT
            scores_full = load_scores(path_full, f"Full({game})", metric)
            if scores_full is not None:
                mean_diff, p_val, sig = analyze_pair(scores_ours, scores_full)
                row["vs Full (Diff)"] = f"{mean_diff:+.4f}"
                row["vs Full (p)"] = f"{p_val:.4f}"
                row["vs Full (Sig)"] = sig
                print(f" | vs Full: {sig:<3} (p={p_val:.4f})", end="")
            else:
                row["vs Full (Sig)"] = "N/A"
                print(f" | vs Full: N/A", end="")

            # 5. Compare vs Baseline
            if path_base:
                scores_base = load_scores(path_base, f"Base({game})", metric)
                if scores_base is not None:
                    mean_diff, p_val, sig = analyze_pair(scores_ours, scores_base)
                    row["vs Base (Diff)"] = f"{mean_diff:+.4f}"
                    row["vs Base (p)"] = f"{p_val:.4f}"
                    row["vs Base (Sig)"] = sig
                    print(f" | vs Base: {sig:<3} (p={p_val:.4f})", end="")
                else:
                    row["vs Base (Sig)"] = "N/A"
            else:
                row["vs Base (Sig)"] = "-"
            
            print("") # Newline
            results_list.append(row)

    if results_list:
        df = pd.DataFrame(results_list)
        
        # Adjust column order
        cols = ["Metric", "Game", "Ours (Mean±Std)"]
        extra_cols = [c for c in df.columns if c not in cols]
        # Sort: Full related columns first, Base related columns second
        extra_cols.sort(key=lambda x: ("Full" in x, "Diff" not in x), reverse=True)
        
        df = df[cols + extra_cols]
        
        print("\n" + "="*120)
        print(df.to_string(index=False))
        print("="*120)
        
        # Save CSV
        df.to_csv("statistical_report_batch.csv", index=False)
        print(f"\n Results saved to statistical_report_batch.csv")

if __name__ == "__main__":
    main()