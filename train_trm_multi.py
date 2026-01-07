"""
TRM Multi-Configuration Training Script
========================================
Trains 5 different TRM configurations to find the best hyperparameters:

1. MORE_STEPS_LARGER_DIM:  num_steps=12, hidden_dim=256
2. MORE_STEPS_SAME_DIM:    num_steps=12, hidden_dim=128
3. LESS_STEPS_LARGER_DIM:  num_steps=5,  hidden_dim=256
4. LESS_STEPS_SAME_DIM:    num_steps=5,  hidden_dim=128
5. SAME_STEPS_LARGER_DIM:  num_steps=8,  hidden_dim=256

Each configuration is trained and results are saved to a JSON file.
"""

import os
import glob
import time
import json
import pickle
from collections import defaultdict
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import numpy as np

# Import TRM model
from trm_model import TRMWithParams


# ==============================================================================
# === Configurations to Test ===================================================
# ==============================================================================

# Base configuration (shared across all experiments)
BASE_CONFIG = {
    "train_folder": "./split_data_train_20000_random",
    "val_folder": "./split_data_validation_20000_random",
    "max_k": 6,
    "max_nk": 6,
    "batch_size": 256,        # RTX 3050 6GB friendly
    "epochs": 150,            # More epochs for thorough training
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "patience": 20,           # More patience for convergence
    "enforce_lower_bound": True,
    "use_deep_supervision": True,
    "final_step_weight": 0.5,
    "num_workers": 0,
    "pin_memory": True,
}

# The 5 configurations to test
EXPERIMENTS = {
    "1_more_steps_larger_dim": {
        "num_steps": 12,
        "hidden_dim": 256,
        "description": "More recursive steps (12) + Larger hidden dim (256)",
    },
    "2_more_steps_same_dim": {
        "num_steps": 12,
        "hidden_dim": 128,
        "description": "More recursive steps (12) + Same hidden dim (128)",
    },
    "3_less_steps_larger_dim": {
        "num_steps": 5,
        "hidden_dim": 256,
        "description": "Less recursive steps (5) + Larger hidden dim (256)",
    },
    "4_less_steps_same_dim": {
        "num_steps": 5,
        "hidden_dim": 128,
        "description": "Less recursive steps (5) + Same hidden dim (128)",
    },
    "5_same_steps_larger_dim": {
        "num_steps": 8,
        "hidden_dim": 256,
        "description": "Same recursive steps (8) + Larger hidden dim (256)",
    },
}


# ==============================================================================
# === Dataset ==================================================================
# ==============================================================================

class PickleFolderDataset(Dataset):
    """Loads samples from pickle files with proper padding."""
    
    def __init__(self, file_paths, max_k=6, max_nk=6):
        super().__init__()
        if not file_paths:
            raise ValueError("file_paths list cannot be empty.")
        
        self.max_k = max_k
        self.max_nk = max_nk
        
        self.P_raw = []
        self.h_vals = []
        self.n_vals = []
        self.k_vals = []
        self.m_vals = []
        
        required_cols = {'n', 'k', 'm', 'result', 'P'}
        
        for path in file_paths:
            try:
                with open(path, "rb") as f:
                    df = pickle.load(f)
            except Exception as e:
                print(f"  [SKIP] {path}: pickle.load failed - {e}")
                continue
            
            # Debug: Check what type we got
            if not hasattr(df, 'columns'):
                print(f"  [SKIP] {path}: Not a DataFrame (type: {type(df).__name__})")
                continue
                
            if not required_cols.issubset(df.columns):
                print(f"  [SKIP] {path}: Missing columns. Has: {list(df.columns)}")
                continue
            
            print(f"  [OK] {path}: {len(df)} rows")
            self.P_raw.extend(df["P"].tolist())
            self.h_vals.extend(df["result"].astype(float).tolist())
            self.n_vals.extend(df["n"].astype(int).tolist())
            self.k_vals.extend(df["k"].astype(int).tolist())
            self.m_vals.extend(df["m"].astype(int).tolist())
        
        if not self.h_vals:
            raise ValueError("No valid rows were loaded.")
        
        self.h_vals = torch.tensor(self.h_vals, dtype=torch.float32)
        self.n_vals = torch.tensor(self.n_vals, dtype=torch.float32)
        self.k_vals = torch.tensor(self.k_vals, dtype=torch.float32)
        self.m_vals = torch.tensor(self.m_vals, dtype=torch.float32)
    
    def __len__(self):
        return len(self.h_vals)
    
    def _pad(self, matrix):
        k_act, nk_act = matrix.shape
        pad = (0, self.max_nk - nk_act, 0, self.max_k - k_act)
        return F.pad(matrix, pad, value=0.0)
    
    def __getitem__(self, idx):
        n = int(self.n_vals[idx].item())
        k = int(self.k_vals[idx].item())
        
        raw = np.array(self.P_raw[idx])
        if raw.ndim == 1:
            raw = raw.reshape(k, n - k)
        P = torch.tensor(raw, dtype=torch.float32)
        P = self._pad(P)
        
        params = torch.tensor([n, k, self.m_vals[idx].item()], dtype=torch.float32)
        h = self.h_vals[idx].unsqueeze(0)
        return params, h, P


# ==============================================================================
# === Loss Functions ===========================================================
# ==============================================================================

class TRMDeepSupervisionLoss(nn.Module):
    """
    Loss function for TRM with deep supervision.
    Computes loss at each recursive step with increasing weights.
    """
    def __init__(self, num_steps, final_weight=0.5, eps=1e-9):
        super().__init__()
        self.num_steps = num_steps
        self.eps = eps
        
        # Exponentially increasing weights toward final step
        weights = torch.exp(torch.linspace(0, 2, num_steps))
        weights = weights / weights.sum()
        
        # Blend with uniform weighting
        uniform = torch.ones(num_steps) / num_steps
        self.register_buffer("step_weights", final_weight * weights + (1 - final_weight) * uniform)
    
    def forward(self, outputs, targets):
        """
        Args:
            outputs: (B, num_steps, 1) - predictions at each step
            targets: (B, 1) - true values
        """
        total_loss = 0.0
        targets_safe = torch.clamp(targets, min=self.eps)
        log2_targets = torch.log2(targets_safe)
        
        for step in range(self.num_steps):
            step_pred = outputs[:, step, :]
            step_pred_safe = torch.clamp(step_pred, min=self.eps)
            log2_pred = torch.log2(step_pred_safe)
            
            step_loss = torch.mean((log2_targets - log2_pred) ** 2)
            total_loss += self.step_weights[step] * step_loss
        
        # Final step loss for logging
        final_loss = torch.mean((log2_targets - torch.log2(torch.clamp(outputs[:, -1, :], min=self.eps))) ** 2)
        
        return total_loss, final_loss.item()


# ==============================================================================
# === Training =================================================================
# ==============================================================================

def train_one_config(exp_name, exp_config, train_loader, val_loader, device):
    """Train a single TRM configuration and return results."""
    
    num_steps = exp_config["num_steps"]
    hidden_dim = exp_config["hidden_dim"]
    
    print(f"\n{'='*70}")
    print(f"Training: {exp_name}")
    print(f"Description: {exp_config['description']}")
    print(f"num_steps={num_steps}, hidden_dim={hidden_dim}")
    print(f"{'='*70}")
    
    # Create model
    model = TRMWithParams(
        k_max=BASE_CONFIG["max_k"],
        nk_max=BASE_CONFIG["max_nk"],
        n_params=3,
        hidden_dim=hidden_dim,
        num_steps=num_steps,
        enforce_lower_bound=BASE_CONFIG["enforce_lower_bound"],
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")
    
    # Loss and optimizer
    criterion = TRMDeepSupervisionLoss(
        num_steps=num_steps,
        final_weight=BASE_CONFIG["final_step_weight"],
    ).to(device)
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=BASE_CONFIG["lr"],
        weight_decay=BASE_CONFIG["weight_decay"],
    )
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=BASE_CONFIG["patience"] // 2, factor=0.5
    )
    
    # Training loop
    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_epoch = 0
    
    start_time = time.time()
    
    for epoch in range(1, BASE_CONFIG["epochs"] + 1):
        epoch_start = time.time()
        
        # Train
        model.train()
        train_loss_sum = 0.0
        train_final_sum = 0.0
        train_samples = 0
        
        for params, targets, P in train_loader:
            params = params.to(device)
            targets = targets.to(device)
            P = P.to(device)
            
            optimizer.zero_grad()
            outputs = model(P, params)
            loss, final_loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            bs = params.size(0)
            train_loss_sum += loss.item() * bs
            train_final_sum += final_loss * bs
            train_samples += bs
        
        train_avg = train_final_sum / train_samples
        train_losses.append(train_avg)
        
        # Validate
        model.eval()
        val_loss_sum = 0.0
        val_final_sum = 0.0
        val_samples = 0
        
        with torch.no_grad():
            for params, targets, P in val_loader:
                params = params.to(device)
                targets = targets.to(device)
                P = P.to(device)
                
                outputs = model(P, params)
                loss, final_loss = criterion(outputs, targets)
                
                bs = params.size(0)
                val_loss_sum += loss.item() * bs
                val_final_sum += final_loss * bs
                val_samples += bs
        
        val_avg = val_final_sum / val_samples
        val_losses.append(val_avg)
        
        duration = time.time() - epoch_start
        
        # Check for improvement
        improved = ""
        if val_avg < best_val_loss:
            improved = f" ✓ New best! {best_val_loss:.4f} → {val_avg:.4f}"
            best_val_loss = val_avg
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save(model.state_dict(), f"best_trm_{exp_name}.pth")
        else:
            epochs_no_improve += 1
        
        print(f"Epoch {epoch:03d}: Train={train_avg:.4f} Val={val_avg:.4f} [{duration:.1f}s]{improved}")
        
        # Early stopping
        if epochs_no_improve >= BASE_CONFIG["patience"]:
            print(f"Early stopping at epoch {epoch}")
            break
        
        scheduler.step(val_avg)
    
    total_time = time.time() - start_time
    
    result = {
        "exp_name": exp_name,
        "description": exp_config["description"],
        "num_steps": num_steps,
        "hidden_dim": hidden_dim,
        "num_params": num_params,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "final_epoch": epoch,
        "total_time_seconds": total_time,
        "train_losses": train_losses,
        "val_losses": val_losses,
    }
    
    print(f"\n→ Best Val Loss: {best_val_loss:.4f} at epoch {best_epoch}")
    print(f"→ Total time: {total_time/60:.1f} min")
    
    return result


def main():
    print("\n" + "="*70)
    print("TRM Multi-Configuration Training")
    print("="*70)
    
    # Check device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"Memory: {mem:.1f} GB")
    else:
        print("WARNING: CUDA not available, using CPU (will be slow!)")
    
    # Load data
    print("\n--- Loading Data ---")
    train_files = glob.glob(os.path.join(BASE_CONFIG["train_folder"], "*.pkl"))
    val_files = glob.glob(os.path.join(BASE_CONFIG["val_folder"], "*.pkl"))
    
    print(f"Found {len(train_files)} train files, {len(val_files)} val files")
    
    train_dataset = PickleFolderDataset(train_files, BASE_CONFIG["max_k"], BASE_CONFIG["max_nk"])
    val_dataset = PickleFolderDataset(val_files, BASE_CONFIG["max_k"], BASE_CONFIG["max_nk"])
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=BASE_CONFIG["batch_size"],
        shuffle=True,
        num_workers=BASE_CONFIG["num_workers"],
        pin_memory=BASE_CONFIG["pin_memory"],
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BASE_CONFIG["batch_size"],
        shuffle=False,
        num_workers=BASE_CONFIG["num_workers"],
        pin_memory=BASE_CONFIG["pin_memory"],
    )
    
    print(f"Train: {len(train_dataset)} samples, Val: {len(val_dataset)} samples")
    
    # Run all experiments
    all_results = {}
    
    for exp_name, exp_config in EXPERIMENTS.items():
        result = train_one_config(exp_name, exp_config, train_loader, val_loader, device)
        all_results[exp_name] = result
        
        # Save intermediate results
        with open("trm_experiments_results.json", "w") as f:
            # Convert numpy/tensor types for JSON
            json_results = {}
            for name, res in all_results.items():
                json_results[name] = {
                    k: (v if not isinstance(v, (list,)) else v) 
                    for k, v in res.items()
                }
            json.dump(json_results, f, indent=2, default=str)
    
    # Print summary
    print("\n" + "="*70)
    print("FINAL RESULTS SUMMARY")
    print("="*70)
    print(f"{'Config':<30} {'Steps':>6} {'Dim':>6} {'Params':>10} {'Best Val':>10} {'Epoch':>6}")
    print("-"*70)
    
    sorted_results = sorted(all_results.items(), key=lambda x: x[1]["best_val_loss"])
    
    for exp_name, result in sorted_results:
        print(f"{exp_name:<30} {result['num_steps']:>6} {result['hidden_dim']:>6} "
              f"{result['num_params']:>10,} {result['best_val_loss']:>10.4f} {result['best_epoch']:>6}")
    
    print("\n" + "="*70)
    best_name, best_result = sorted_results[0]
    print(f"🏆 BEST: {best_name}")
    print(f"   Val Loss: {best_result['best_val_loss']:.4f}")
    print(f"   Steps: {best_result['num_steps']}, Hidden: {best_result['hidden_dim']}")
    print(f"   Model saved: best_trm_{best_name}.pth")
    print("="*70)
    
    # Save plots
    plot_all_results(all_results)


def plot_all_results(all_results):
    """Save training curves for all experiments."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plots")
        return
    
    # Plot 1: All validation curves on one plot
    plt.figure(figsize=(12, 6))
    
    for exp_name, result in all_results.items():
        epochs = range(1, len(result["val_losses"]) + 1)
        label = f"{exp_name} (best: {result['best_val_loss']:.3f})"
        plt.plot(epochs, result["val_losses"], label=label, linewidth=2)
    
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Validation Log2-MSE Loss", fontsize=12)
    plt.title("TRM Configurations Comparison", fontsize=14)
    plt.legend(loc="upper right", fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("trm_comparison_plot.png", dpi=150)
    print("Saved: trm_comparison_plot.png")
    plt.close()
    
    # Plot 2: Individual training curves (train vs val)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, (exp_name, result) in enumerate(all_results.items()):
        if idx >= 5:
            break
        ax = axes[idx]
        epochs = range(1, len(result["train_losses"]) + 1)
        
        ax.plot(epochs, result["train_losses"], 'b-', label="Train", alpha=0.7)
        ax.plot(epochs, result["val_losses"], 'r-', label="Val", alpha=0.7)
        ax.axvline(x=result["best_epoch"], color='g', linestyle='--', alpha=0.5)
        
        ax.set_title(f"{result['description']}\nBest: {result['best_val_loss']:.4f} @ ep {result['best_epoch']}", fontsize=10)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplot
    axes[5].axis('off')
    
    plt.tight_layout()
    plt.savefig("trm_individual_curves.png", dpi=150)
    print("Saved: trm_individual_curves.png")
    plt.close()
    
    # Plot 3: Bar chart comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    
    names = [r["description"].replace(" + ", "\n") for r in all_results.values()]
    losses = [r["best_val_loss"] for r in all_results.values()]
    colors = ['#2ecc71' if l == min(losses) else '#3498db' for l in losses]
    
    bars = ax.bar(range(len(names)), losses, color=colors, edgecolor='black')
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel("Best Validation Loss", fontsize=12)
    ax.set_title("TRM Configuration Comparison", fontsize=14)
    
    # Add value labels on bars
    for bar, loss in zip(bars, losses):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{loss:.4f}', ha='center', va='bottom', fontsize=10)
    
    ax.axhline(y=1.13, color='red', linestyle='--', label='ResNet baseline (1.13)')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig("trm_bar_comparison.png", dpi=150)
    print("Saved: trm_bar_comparison.png")
    plt.close()
    
    print("\nAll plots saved!")


if __name__ == "__main__":
    main()
