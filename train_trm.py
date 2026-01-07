"""
TRM (Tiny Recursive Model) Training Script for M-Height Prediction
===================================================================
This script trains the TRM model on the m-height prediction task.
TRM iteratively refines its prediction through recursive steps.

Key features:
- Deep supervision: loss computed at each recursive step
- Progressive weighting: later steps weighted more heavily
- Lower bound enforcement: h >= 1
"""

import os
import glob
import time
import pickle
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.cuda.amp import autocast, GradScaler

import numpy as np

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available, skipping plots")

# Import TRM model
from trm_model import TRMWithParams

# ==============================================================================
# === Configuration ============================================================
# ==============================================================================

CONFIG = {
    # Data
    "train_folder": "./split_data_train_20000_random",
    "val_folder": "./split_data_validation_20000_random",
    "max_k": 6,
    "max_nk": 6,
    
    # Model
    "hidden_dim": 128,      # TRM hidden dimension
    "num_steps": 8,         # Number of recursive refinement steps
    "enforce_lower_bound": True,
    
    # Training
    "batch_size": 256,      # Good for RTX 3050 6GB
    "epochs": 100,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "patience": 15,
    
    # Deep supervision weights (exponentially increasing for later steps)
    "use_deep_supervision": True,
    "final_step_weight": 0.5,  # Weight for final step vs intermediate
    
    # Mixed precision for memory efficiency
    "use_amp": True,
    
    # Misc
    "num_workers": 0,
    "pin_memory": True,
    "save_name": "best_trm_model",
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
        print(f"Loading data from {len(file_paths)} pickle files...")
        
        for path in file_paths:
            try:
                with open(path, "rb") as f:
                    df = pickle.load(f)
            except Exception as exc:
                print(f"  ! Skipping {path}: {exc}")
                continue
                
            if not required_cols.issubset(df.columns):
                print(f"  ! Skipping {path}: missing columns")
                continue
            
            self.P_raw.extend(df["P"].tolist())
            self.h_vals.extend(df["result"].astype(float).tolist())
            self.n_vals.extend(df["n"].astype(int).tolist())
            self.k_vals.extend(df["k"].astype(int).tolist())
            self.m_vals.extend(df["m"].astype(int).tolist())
        
        if not self.h_vals:
            raise ValueError("No valid rows were loaded.")
        
        print(f"Loaded {len(self.h_vals)} samples")
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

class LogMSELoss(nn.Module):
    """Log2-space MSE loss."""
    def __init__(self, eps=1e-9):
        super().__init__()
        self.eps = eps
    
    def forward(self, y_pred, y_true):
        y_pred = torch.clamp(y_pred, min=self.eps)
        y_true = torch.clamp(y_true, min=self.eps)
        return torch.mean((torch.log2(y_true) - torch.log2(y_pred)) ** 2)


class TRMDeepSupervisionLoss(nn.Module):
    """
    Loss function for TRM with deep supervision.
    Computes loss at each recursive step with increasing weights.
    """
    def __init__(self, num_steps, final_weight=0.5, eps=1e-9):
        super().__init__()
        self.num_steps = num_steps
        self.eps = eps
        
        # Create weights: exponentially increasing toward final step
        # E.g., for 5 steps: [0.1, 0.15, 0.2, 0.25, 0.3] normalized
        weights = torch.exp(torch.linspace(0, 2, num_steps))
        weights = weights / weights.sum()
        
        # Blend with uniform weighting based on final_weight
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
            step_pred = outputs[:, step, :]  # (B, 1)
            step_pred_safe = torch.clamp(step_pred, min=self.eps)
            log2_pred = torch.log2(step_pred_safe)
            
            step_loss = torch.mean((log2_targets - log2_pred) ** 2)
            total_loss += self.step_weights[step] * step_loss
        
        # Also return the final step loss for logging
        final_loss = torch.mean((log2_targets - torch.log2(torch.clamp(outputs[:, -1, :], min=self.eps))) ** 2)
        
        return total_loss, final_loss.item()


# ==============================================================================
# === Training =================================================================
# ==============================================================================

class TRMTrainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer,
                 device, patience=10, scheduler=None, use_amp=False, save_name="best_trm"):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler
        self.patience = patience
        self.use_amp = use_amp
        self.save_name = save_name
        
        self.scaler = GradScaler() if use_amp else None
        
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float("inf")
        self.epochs_no_improve = 0
    
    def _run_epoch(self, loader, training=True):
        if training:
            self.model.train()
        else:
            self.model.eval()
        
        total_loss = 0.0
        total_final_loss = 0.0
        total_samples = 0
        
        context = torch.enable_grad() if training else torch.no_grad()
        
        with context:
            for params, targets, P in loader:
                params = params.to(self.device)
                targets = targets.to(self.device)
                P = P.to(self.device)
                
                if training:
                    self.optimizer.zero_grad()
                
                if self.use_amp and training:
                    with autocast():
                        outputs = self.model(P, params)
                        loss, final_loss = self.criterion(outputs, targets)
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    outputs = self.model(P, params)
                    loss, final_loss = self.criterion(outputs, targets)
                    if training:
                        loss.backward()
                        self.optimizer.step()
                
                batch_size = params.size(0)
                total_loss += loss.item() * batch_size
                total_final_loss += final_loss * batch_size
                total_samples += batch_size
        
        return total_loss / total_samples, total_final_loss / total_samples
    
    def fit(self, epochs):
        print(f"\n{'='*60}")
        print(f"Training TRM Model")
        print(f"{'='*60}")
        print(f"  Hidden dim: {self.model.trm_block.net[0].in_features}")
        print(f"  Num steps: {self.model.num_steps}")
        print(f"  Device: {self.device}")
        print(f"  AMP: {self.use_amp}")
        print(f"{'='*60}\n")
        
        start = time.time()
        
        for epoch in range(1, epochs + 1):
            epoch_start = time.time()
            
            train_loss, train_final = self._run_epoch(self.train_loader, training=True)
            val_loss, val_final = self._run_epoch(self.val_loader, training=False)
            
            self.train_losses.append(train_final)
            self.val_losses.append(val_final)
            
            duration = time.time() - epoch_start
            
            print(f"Epoch {epoch:03d}: "
                  f"Train DS={train_loss:.4f} Final={train_final:.4f} | "
                  f"Val DS={val_loss:.4f} Final={val_final:.4f} | "
                  f"{duration:.1f}s")
            
            # Use final step loss for early stopping
            if val_final < self.best_val_loss:
                print(f"  ✓ New best! {self.best_val_loss:.4f} → {val_final:.4f}")
                self.best_val_loss = val_final
                self.epochs_no_improve = 0
                torch.save(self.model.state_dict(), f"{self.save_name}.pth")
            else:
                self.epochs_no_improve += 1
                if self.epochs_no_improve >= self.patience:
                    print(f"  Early stopping after {self.patience} epochs without improvement.")
                    break
            
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_final)
                else:
                    self.scheduler.step()
        
        total_time = time.time() - start
        print(f"\nTraining complete in {total_time/60:.1f} min. Best val loss: {self.best_val_loss:.4f}")
        
        return {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "best_val_loss": self.best_val_loss,
        }


def plot_losses(history, save_path="trm_training_curve.png"):
    """Plot training curves."""
    if not HAS_MATPLOTLIB:
        print("Skipping plot (matplotlib not available)")
        return
    
    train_losses = history.get("train_losses", [])
    val_losses = history.get("val_losses", [])
    
    if not train_losses:
        return
    
    plt.figure(figsize=(10, 5))
    epochs = range(1, len(train_losses) + 1)
    
    plt.plot(epochs, train_losses, 'b-', label='Train Loss', alpha=0.7)
    plt.plot(epochs, val_losses, 'r-', label='Val Loss', alpha=0.7)
    
    best_epoch = val_losses.index(min(val_losses)) + 1
    plt.axvline(x=best_epoch, color='g', linestyle='--', alpha=0.5, label=f'Best @ Ep {best_epoch}')
    
    plt.xlabel('Epoch')
    plt.ylabel('Log2 MSE Loss')
    plt.title(f'TRM Training - Best Val: {min(val_losses):.4f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved training curve to {save_path}")
    plt.close()


# ==============================================================================
# === Main =====================================================================
# ==============================================================================

def main():
    print("\n" + "="*60)
    print("TRM Model Training for M-Height Prediction")
    print("="*60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Load data
    print("\n--- Loading Data ---")
    train_files = glob.glob(os.path.join(CONFIG["train_folder"], "*.pkl"))
    val_files = glob.glob(os.path.join(CONFIG["val_folder"], "*.pkl"))
    
    train_dataset = PickleFolderDataset(train_files, CONFIG["max_k"], CONFIG["max_nk"])
    val_dataset = PickleFolderDataset(val_files, CONFIG["max_k"], CONFIG["max_nk"])
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=CONFIG["num_workers"],
        pin_memory=CONFIG["pin_memory"],
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=CONFIG["num_workers"],
        pin_memory=CONFIG["pin_memory"],
    )
    
    print(f"Train: {len(train_dataset)} samples, Val: {len(val_dataset)} samples")
    
    # Create model
    print("\n--- Creating TRM Model ---")
    model = TRMWithParams(
        k_max=CONFIG["max_k"],
        nk_max=CONFIG["max_nk"],
        n_params=3,
        hidden_dim=CONFIG["hidden_dim"],
        num_steps=CONFIG["num_steps"],
        enforce_lower_bound=CONFIG["enforce_lower_bound"],
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")
    
    # Loss and optimizer
    if CONFIG["use_deep_supervision"]:
        criterion = TRMDeepSupervisionLoss(
            num_steps=CONFIG["num_steps"],
            final_weight=CONFIG["final_step_weight"],
        ).to(device)
    else:
        criterion = LogMSELoss()
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=CONFIG["lr"],
        weight_decay=CONFIG["weight_decay"],
    )
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=CONFIG["patience"] // 2, factor=0.5
    )
    
    # Train
    trainer = TRMTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        patience=CONFIG["patience"],
        scheduler=scheduler,
        use_amp=CONFIG["use_amp"],
        save_name=CONFIG["save_name"],
    )
    
    history = trainer.fit(CONFIG["epochs"])
    
    # Plot results
    plot_losses(history)
    
    print(f"\n{'='*60}")
    print(f"Training Complete!")
    print(f"Best validation Log2-MSE: {history['best_val_loss']:.4f}")
    print(f"Model saved to: {CONFIG['save_name']}.pth")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
