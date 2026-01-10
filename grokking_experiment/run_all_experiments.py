"""
Run all grokking experiments with different weight decay values.

Usage:
    python run_all_experiments.py
"""

import subprocess
import sys

WEIGHT_DECAY_VALUES = [0.01, 0.03, 0.1]
PYTHON_EXE = sys.executable

def main():
    print("=" * 60)
    print("RUNNING ALL GROKKING EXPERIMENTS")
    print("=" * 60)
    
    for wd in WEIGHT_DECAY_VALUES:
        print(f"\n{'='*60}")
        print(f"Starting experiment with weight_decay={wd}")
        print(f"{'='*60}\n")
        
        cmd = [
            PYTHON_EXE, "train_grokking.py",
            "--weight_decay", str(wd),
            "--epochs", "10000",
            "--batch_size", "512",
            "--lr", "0.001",
            "--samples_per_group", "5000",
            "--base_ch", "64",
            "--num_blocks", "5"
        ]
        
        result = subprocess.run(cmd, cwd=".")
        
        if result.returncode != 0:
            print(f"\nExperiment with wd={wd} failed!")
        else:
            print(f"\nExperiment with wd={wd} completed successfully!")
    
    print("\n" + "=" * 60)
    print("ALL EXPERIMENTS COMPLETED")
    print("View results at: https://wandb.ai")
    print("=" * 60)

if __name__ == "__main__":
    main()
