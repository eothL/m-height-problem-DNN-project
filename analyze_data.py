"""
Comprehensive Data Analysis for M-Height Prediction Project
============================================================
This script analyzes the train and validation datasets to understand:
1. Sample counts per (n,k,m) group
2. Target (h) distribution statistics per group
3. P matrix statistics
4. Why some groups may be harder to predict

Outputs a JSON report that can be used by future agents.
"""

import os
import glob
import pickle
import json
import numpy as np
import pandas as pd
from collections import defaultdict

# Configuration - Primary dataset (LARGER: 20000_random folders)
TRAIN_FOLDER = "./split_data_train_20000_random"
VAL_FOLDER = "./split_data_validation_20000_random"
OUTPUT_REPORT = "./data_analysis_report_large.json"

# Secondary dataset (smaller split_data folders)
TRAIN_FOLDER_SMALL = "./split_data"
VAL_FOLDER_SMALL = "./split_data_validation"
OUTPUT_REPORT_SMALL = "./data_analysis_report_small.json"

def load_pickle_files(folder):
    """Load all pickle files from a folder and return combined data."""
    all_data = []
    files = sorted(glob.glob(os.path.join(folder, "*.pkl")))
    
    for fp in files:
        try:
            with open(fp, "rb") as f:
                df = pickle.load(f)
            if isinstance(df, pd.DataFrame):
                all_data.append(df)
        except Exception as e:
            print(f"Error loading {fp}: {e}")
    
    if all_data:
        return pd.concat(all_data, ignore_index=True)
    return pd.DataFrame()

def analyze_group(group_df, n, k, m):
    """Analyze a single (n,k,m) group and return statistics."""
    h_values = group_df['result'].values.astype(float)
    
    # Basic statistics
    stats = {
        "n": int(n),
        "k": int(k),
        "m": int(m),
        "count": len(h_values),
        "h_min": float(np.min(h_values)),
        "h_max": float(np.max(h_values)),
        "h_mean": float(np.mean(h_values)),
        "h_median": float(np.median(h_values)),
        "h_std": float(np.std(h_values)),
        "h_variance": float(np.var(h_values)),
    }
    
    # Log2 space statistics (since we use Log2-MSE loss)
    log2_h = np.log2(np.clip(h_values, 1e-9, None))
    stats["log2_h_min"] = float(np.min(log2_h))
    stats["log2_h_max"] = float(np.max(log2_h))
    stats["log2_h_mean"] = float(np.mean(log2_h))
    stats["log2_h_std"] = float(np.std(log2_h))
    stats["log2_h_range"] = float(np.max(log2_h) - np.min(log2_h))
    
    # Percentiles
    percentiles = [5, 10, 25, 50, 75, 90, 95]
    for p in percentiles:
        stats[f"h_p{p}"] = float(np.percentile(h_values, p))
    
    # Dynamic range ratio (max/min)
    stats["dynamic_range"] = float(np.max(h_values) / max(np.min(h_values), 1e-9))
    
    # P matrix analysis (sample a few)
    if 'P' in group_df.columns:
        p_matrices = group_df['P'].head(100).tolist()
        p_flat = []
        for p in p_matrices:
            arr = np.array(p).flatten()
            p_flat.extend(arr.tolist())
        p_flat = np.array(p_flat)
        stats["p_mean"] = float(np.mean(p_flat))
        stats["p_std"] = float(np.std(p_flat))
        stats["p_min"] = float(np.min(p_flat))
        stats["p_max"] = float(np.max(p_flat))
    
    return stats

def compute_difficulty_score(stats):
    """
    Compute a difficulty score based on target variance and dynamic range.
    Higher score = harder to predict.
    """
    # Factors that make prediction harder:
    # 1. High variance in log2 space
    # 2. Large dynamic range
    # 3. High m relative to k
    
    log2_std = stats.get("log2_h_std", 0)
    log2_range = stats.get("log2_h_range", 0)
    m_k_ratio = stats["m"] / max(stats["k"], 1)
    
    # Weighted combination
    difficulty = (log2_std * 2.0) + (log2_range * 0.5) + (m_k_ratio * 1.0)
    return round(difficulty, 4)

def main():
    print("=" * 60)
    print("M-Height Data Analysis")
    print("=" * 60)
    
    # Load data
    print("\nLoading training data...")
    train_df = load_pickle_files(TRAIN_FOLDER)
    print(f"  Loaded {len(train_df)} training samples")
    
    print("Loading validation data...")
    val_df = load_pickle_files(VAL_FOLDER)
    print(f"  Loaded {len(val_df)} validation samples")
    
    if train_df.empty or val_df.empty:
        print("ERROR: Could not load data!")
        return
    
    # Combine for overall analysis
    all_df = pd.concat([train_df, val_df], ignore_index=True)
    
    # Get unique (n,k,m) combinations
    all_df['group'] = list(zip(all_df['n'], all_df['k'], all_df['m']))
    unique_groups = sorted(all_df['group'].unique())
    
    print(f"\nFound {len(unique_groups)} unique (n,k,m) groups")
    
    # Analyze each group
    report = {
        "summary": {
            "total_train_samples": len(train_df),
            "total_val_samples": len(val_df),
            "total_samples": len(all_df),
            "num_groups": len(unique_groups),
            "groups": [list(g) for g in unique_groups],
        },
        "train_groups": {},
        "val_groups": {},
        "combined_groups": {},
        "difficulty_ranking": [],
    }
    
    print("\n" + "-" * 60)
    print("Per-Group Analysis (Combined Train+Val)")
    print("-" * 60)
    print(f"{'(n,k,m)':<12} {'Count':>8} {'h_mean':>10} {'h_std':>10} {'log2_range':>12} {'Difficulty':>12}")
    print("-" * 60)
    
    difficulty_scores = []
    
    for n, k, m in unique_groups:
        # Combined analysis
        group_mask = (all_df['n'] == n) & (all_df['k'] == k) & (all_df['m'] == m)
        group_df = all_df[group_mask]
        stats = analyze_group(group_df, n, k, m)
        difficulty = compute_difficulty_score(stats)
        stats["difficulty_score"] = difficulty
        report["combined_groups"][f"({n},{k},{m})"] = stats
        difficulty_scores.append(((n, k, m), difficulty))
        
        # Train-only analysis
        train_mask = (train_df['n'] == n) & (train_df['k'] == k) & (train_df['m'] == m)
        if train_mask.sum() > 0:
            train_stats = analyze_group(train_df[train_mask], n, k, m)
            report["train_groups"][f"({n},{k},{m})"] = train_stats
        
        # Val-only analysis
        val_mask = (val_df['n'] == n) & (val_df['k'] == k) & (val_df['m'] == m)
        if val_mask.sum() > 0:
            val_stats = analyze_group(val_df[val_mask], n, k, m)
            report["val_groups"][f"({n},{k},{m})"] = val_stats
        
        print(f"({n},{k},{m})"
              f"{stats['count']:>10}"
              f"{stats['h_mean']:>10.2f}"
              f"{stats['h_std']:>10.2f}"
              f"{stats['log2_h_range']:>12.2f}"
              f"{difficulty:>12.2f}")
    
    # Sort by difficulty
    difficulty_scores.sort(key=lambda x: x[1], reverse=True)
    report["difficulty_ranking"] = [
        {"group": list(g), "score": s} for g, s in difficulty_scores
    ]
    
    print("\n" + "=" * 60)
    print("DIFFICULTY RANKING (Hardest First)")
    print("=" * 60)
    for rank, ((n, k, m), score) in enumerate(difficulty_scores, 1):
        stats = report["combined_groups"][f"({n},{k},{m})"]
        print(f"{rank:>2}. ({n},{k},{m}) - Score: {score:.2f} | "
              f"h_std: {stats['h_std']:.2f} | "
              f"log2_range: {stats['log2_h_range']:.2f}")
    
    # Key insights
    print("\n" + "=" * 60)
    print("KEY INSIGHTS")
    print("=" * 60)
    
    # Correlation between m/k ratio and difficulty
    ratios_and_difficulties = []
    for (n, k, m), score in difficulty_scores:
        ratio = m / k
        ratios_and_difficulties.append((ratio, score))
    
    ratios = [r[0] for r in ratios_and_difficulties]
    diffs = [r[1] for r in ratios_and_difficulties]
    correlation = np.corrcoef(ratios, diffs)[0, 1]
    
    print(f"\n1. Correlation between m/k ratio and difficulty: {correlation:.3f}")
    print("   (Higher correlation means m/k ratio is a strong predictor of difficulty)")
    
    # Groups with highest variance
    high_var_groups = sorted(
        report["combined_groups"].items(),
        key=lambda x: x[1]["log2_h_std"],
        reverse=True
    )[:5]
    
    print("\n2. Groups with highest target variance (hardest to predict):")
    for group_name, stats in high_var_groups:
        print(f"   {group_name}: log2_std = {stats['log2_h_std']:.3f}, "
              f"range = [{stats['h_min']:.1f}, {stats['h_max']:.1f}]")
    
    # Groups with lowest variance
    low_var_groups = sorted(
        report["combined_groups"].items(),
        key=lambda x: x[1]["log2_h_std"]
    )[:5]
    
    print("\n3. Groups with lowest target variance (easiest to predict):")
    for group_name, stats in low_var_groups:
        print(f"   {group_name}: log2_std = {stats['log2_h_std']:.3f}, "
              f"range = [{stats['h_min']:.1f}, {stats['h_max']:.1f}]")
    
    # Save report
    print(f"\nSaving report to {OUTPUT_REPORT}...")
    with open(OUTPUT_REPORT, 'w') as f:
        json.dump(report, f, indent=2)
    print("Done!")
    
    # Also print a summary table
    print("\n" + "=" * 60)
    print("COMPLETE STATISTICS TABLE")
    print("=" * 60)
    
    headers = ["Group", "Count", "h_min", "h_max", "h_mean", "h_std", 
               "log2_std", "log2_range", "m/k", "Difficulty"]
    print(" | ".join(f"{h:>10}" for h in headers))
    print("-" * 130)
    
    for (n, k, m), score in difficulty_scores:
        stats = report["combined_groups"][f"({n},{k},{m})"]
        row = [
            f"({n},{k},{m})",
            stats['count'],
            f"{stats['h_min']:.2f}",
            f"{stats['h_max']:.2f}",
            f"{stats['h_mean']:.2f}",
            f"{stats['h_std']:.2f}",
            f"{stats['log2_h_std']:.3f}",
            f"{stats['log2_h_range']:.2f}",
            f"{m/k:.2f}",
            f"{score:.2f}"
        ]
        print(" | ".join(f"{str(v):>10}" for v in row))
    
    return report

if __name__ == "__main__":
    main()
