"""
Grokking / Double Descent Experiment for m-Height Prediction

This script trains a ResNet model with LayerNorm to investigate grokking and 
double descent phenomena by:
1. Using a reduced dataset (5k samples per (n,k,m) group)
2. Training for extended epochs (10,000)
3. Using high weight decay (0.01 - 0.1)
4. Tracking with Weights & Biases

Usage:
    python train_grokking.py --weight_decay 0.01
    python train_grokking.py --weight_decay 0.03
    python train_grokking.py --weight_decay 0.1
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
import numpy as np
import os
import glob
import pickle
import argparse
import time
from collections import defaultdict

import wandb

# ==============================================================================
# === Model Definitions ===
# ==============================================================================

class ConvResBlockLayerNorm(nn.Module):
    """Residual block using LayerNorm instead of BatchNorm for stable grokking training."""
    def __init__(self, ch, spatial_size=6):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)
        # LayerNorm needs the full shape [C, H, W]
        self.ln1 = nn.LayerNorm([ch, spatial_size, spatial_size])
        self.ln2 = nn.LayerNorm([ch, spatial_size, spatial_size])

    def forward(self, x):
        h = F.relu(self.ln1(self.conv1(x)))
        h = self.ln2(self.conv2(h))
        return F.relu(x + h)


class ResNet2DWithLayerNorm(nn.Module):
    """
    ResNet2D model with LayerNorm for grokking experiments.
    
    LayerNorm is preferred over BatchNorm because:
    1. It doesn't depend on batch statistics
    2. Works consistently regardless of batch size
    3. Better suited for small datasets and long training
    """
    def __init__(self, k_max=6, nk_max=6, n_params=3, base_ch=64, num_blocks=5):
        super().__init__()
        self.spatial_size = k_max  # Assuming k_max == nk_max == 6
        
        self.stem = nn.Sequential(
            nn.Conv2d(1, base_ch, 3, padding=1),
            nn.LayerNorm([base_ch, self.spatial_size, self.spatial_size]),
            nn.ReLU(inplace=True)
        )
        
        self.blocks = nn.Sequential(
            *[ConvResBlockLayerNorm(base_ch, self.spatial_size) for _ in range(num_blocks)]
        )
        
        flat_dim = base_ch * k_max * nk_max
        self.param_proj = nn.Linear(n_params, 64)
        self.head = nn.Sequential(
            nn.Linear(flat_dim + 64, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1)
        )

    def forward(self, P, params):
        x = P.unsqueeze(1)  # [B, 1, 6, 6]
        x = self.blocks(self.stem(x))
        x = x.flatten(1)
        p = F.relu(self.param_proj(params.float()))
        x = torch.cat([x, p], dim=1)
        return self.head(x)


# ==============================================================================
# === Loss Function ===
# ==============================================================================

class LogMSELoss(nn.Module):
    """Log2 MSE Loss for m-height prediction."""
    def __init__(self, eps=1e-9):
        super().__init__()
        self.eps = eps

    def forward(self, y_pred, y_true):
        y_pred_safe = torch.clamp(y_pred, min=self.eps)
        y_true_safe = torch.clamp(y_true, min=self.eps)
        log2_pred = torch.log2(y_pred_safe)
        log2_true = torch.log2(y_true_safe)
        return torch.mean((log2_true - log2_pred) ** 2)


# ==============================================================================
# === Data Handling ===
# ==============================================================================

class PickleFolderDataset(Dataset):
    """Dataset that loads data from pickle files with P matrices and m-height values."""
    
    def __init__(self, file_paths: list, max_k: int = 6, max_nk: int = 6):
        super().__init__()
        self.max_k, self.max_nk = max_k, max_nk
        
        if not file_paths:
            raise ValueError("file_paths list is empty.")
        
        self.P_raw, self.h_vals = [], []
        self.n_vals, self.k_vals, self.m_vals = [], [], []
        
        print(f"Loading data from {len(file_paths)} files...")
        for fp in file_paths:
            try:
                with open(fp, "rb") as f:
                    df = pickle.load(f)
                if not all(col in df.columns for col in ['n', 'k', 'm', 'result', 'P']):
                    continue
                self.P_raw.extend([np.array(p) for p in df['P']])
                self.h_vals.extend(df['result'].astype(float).tolist())
                self.n_vals.extend(df['n'].astype(int).tolist())
                self.k_vals.extend(df['k'].astype(int).tolist())
                self.m_vals.extend(df['m'].astype(int).tolist())
            except Exception as e:
                print(f"Error loading {fp}: {e}")
        
        if not self.h_vals:
            raise ValueError("No valid data loaded.")
        
        print(f"Finished loading. Total samples: {len(self.h_vals)}")
        
        self.h_vals = torch.tensor(self.h_vals, dtype=torch.float32)
        self.n_vals = torch.tensor(self.n_vals, dtype=torch.float32)
        self.k_vals = torch.tensor(self.k_vals, dtype=torch.float32)
        self.m_vals = torch.tensor(self.m_vals, dtype=torch.float32)

    def __len__(self):
        return len(self.h_vals)

    def _pad(self, p2d: torch.Tensor):
        k_act, nk_act = p2d.shape
        pad = (0, self.max_nk - nk_act, 0, self.max_k - k_act)
        return F.pad(p2d, pad, value=0.)

    def __getitem__(self, idx):
        p_np = self.P_raw[idx]
        n, k = int(self.n_vals[idx]), int(self.k_vals[idx])
        target_shape = (k, n - k)
        if p_np.ndim == 1:
            p_np = p_np.reshape(*target_shape)
        p_t = self._pad(torch.tensor(p_np, dtype=torch.float32))
        params = torch.tensor([self.n_vals[idx], self.k_vals[idx], self.m_vals[idx]], 
                              dtype=torch.float32)
        h = self.h_vals[idx].unsqueeze(0)
        return params, h, p_t
    
    def get_group_indices(self):
        """Return dictionary mapping (n,k,m) -> list of indices."""
        group_indices = defaultdict(list)
        for idx in range(len(self)):
            n, k, m = int(self.n_vals[idx]), int(self.k_vals[idx]), int(self.m_vals[idx])
            group_indices[(n, k, m)].append(idx)
        return group_indices


def create_subsampled_dataset(full_dataset, samples_per_group=5000, seed=42):
    """Subsample dataset to have exactly `samples_per_group` per (n,k,m) group."""
    np.random.seed(seed)
    
    group_indices = full_dataset.get_group_indices()
    selected_indices = []
    
    print(f"\nSubsampling {samples_per_group} samples per group:")
    for group, indices in sorted(group_indices.items()):
        if len(indices) >= samples_per_group:
            sampled = np.random.choice(indices, size=samples_per_group, replace=False)
        else:
            sampled = indices
            print(f"  Warning: Group {group} has only {len(indices)} samples")
        selected_indices.extend(sampled)
        print(f"  Group {group}: {len(sampled)} samples selected")
    
    print(f"Total subsampled: {len(selected_indices)} samples")
    return Subset(full_dataset, selected_indices)


# ==============================================================================
# === Training ===
# ==============================================================================

def train_grokking(config):
    """Main training function with wandb logging."""
    
    # Initialize wandb
    run = wandb.init(
        project="m-height-grokking",
        name=f"resnet_wd{config['weight_decay']:.3f}",
        config=config
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data
    print("\n--- Loading Training Data ---")
    train_files = glob.glob(os.path.join(config['train_folder'], '*.pkl'))
    if not train_files:
        raise FileNotFoundError(f"No pickle files found in {config['train_folder']}")
    
    full_train_dataset = PickleFolderDataset(train_files)
    train_dataset = create_subsampled_dataset(full_train_dataset, 
                                               samples_per_group=config['samples_per_group'])
    
    print("\n--- Loading Validation Data ---")
    val_files = glob.glob(os.path.join(config['val_folder'], '*.pkl'))
    if not val_files:
        raise FileNotFoundError(f"No pickle files found in {config['val_folder']}")
    
    full_val_dataset = PickleFolderDataset(val_files)
    # Subsample validation set (2x training = 10k per group)
    val_dataset = create_subsampled_dataset(full_val_dataset, 
                                            samples_per_group=config['val_samples_per_group'])
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], 
                              shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], 
                            shuffle=False, num_workers=0, pin_memory=True)
    
    print(f"\nTrain batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # Create model
    model = ResNet2DWithLayerNorm(
        k_max=6, nk_max=6, n_params=3,
        base_ch=config['base_ch'],
        num_blocks=config['num_blocks']
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    wandb.log({"model_params": total_params})
    
    # Loss and optimizer
    criterion = LogMSELoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config['weight_decay']
    )
    
    # Training loop
    print(f"\n--- Starting Training for {config['epochs']} epochs ---")
    print(f"Weight Decay: {config['weight_decay']}")
    print(f"Validation frequency: every {config['val_frequency']} epochs")
    
    best_val_loss = float('inf')
    last_val_loss = None
    start_time = time.time()
    
    for epoch in range(1, config['epochs'] + 1):
        epoch_start = time.time()
        
        # Training
        model.train()
        train_loss_sum, train_samples = 0.0, 0
        for params, targets, p_matrices in train_loader:
            params = params.to(device)
            targets = targets.to(device)
            p_matrices = p_matrices.to(device)
            
            optimizer.zero_grad()
            outputs = model(p_matrices, params)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss_sum += loss.item() * params.size(0)
            train_samples += params.size(0)
        
        train_loss = train_loss_sum / train_samples
        epoch_time = time.time() - epoch_start
        
        # Validation (only every N epochs to speed up training)
        val_loss = None
        if epoch % config['val_frequency'] == 0 or epoch == 1:
            model.eval()
            val_loss_sum, val_samples = 0.0, 0
            with torch.no_grad():
                for params, targets, p_matrices in val_loader:
                    params = params.to(device)
                    targets = targets.to(device)
                    p_matrices = p_matrices.to(device)
                    
                    outputs = model(p_matrices, params)
                    loss = criterion(outputs, targets)
                    
                    val_loss_sum += loss.item() * params.size(0)
                    val_samples += params.size(0)
            
            val_loss = val_loss_sum / val_samples
            last_val_loss = val_loss
        
        # Calculate Weight Norm (Critical for analyzing Slingshot Mechanism)
        weight_norm = torch.tensor(0.0).to(device)
        for p in model.parameters():
            weight_norm += p.norm(2).pow(2)
        weight_norm = weight_norm.sqrt().item()

        # Log to wandb
        log_dict = {
            "epoch": epoch,
            "train_loss": train_loss,
            "epoch_time": epoch_time,
            "lr": optimizer.param_groups[0]['lr'],
            "weight_norm": weight_norm
        }
        if val_loss is not None:
            log_dict["val_loss"] = val_loss
        wandb.log(log_dict)
        
        # Print progress every 100 epochs or when validation runs
        if epoch % 100 == 0 or epoch == 1 or val_loss is not None:
            val_str = f", Val={val_loss:.4f}" if val_loss is not None else ""
            print(f"Epoch {epoch}/{config['epochs']}: "
                  f"Train={train_loss:.4f}{val_str}, "
                  f"Time={epoch_time:.2f}s")
        
        # Save best model (only when validation was run)
        if val_loss is not None and val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = os.path.join(
                config['output_dir'], 
                f"best_grokking_wd{config['weight_decay']:.3f}.pth"
            )
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss,
                'config': config
            }, checkpoint_path)
            wandb.log({"best_val_loss": best_val_loss, "best_epoch": epoch})
        
        # Save checkpoint every 500 epochs
        if epoch % 500 == 0:
            checkpoint_path = os.path.join(
                config['output_dir'], 
                f"checkpoint_ep{epoch}_wd{config['weight_decay']:.3f}.pth"
            )
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'config': config
            }, checkpoint_path)
    
    total_time = time.time() - start_time
    print(f"\n--- Training Complete ---")
    print(f"Total time: {total_time/60:.2f} minutes")
    print(f"Best validation loss: {best_val_loss:.4f}")
    
    wandb.log({"total_training_time_min": total_time/60})
    wandb.finish()
    
    return best_val_loss


# ==============================================================================
# === Main ===
# ==============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Grokking experiment for m-height prediction')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay for AdamW (default: 0.01)')
    parser.add_argument('--epochs', type=int, default=10000,
                        help='Number of training epochs (default: 10000)')
    parser.add_argument('--batch_size', type=int, default=4096,
                        help='Batch size (default: 4096)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate (default: 1e-3)')
    parser.add_argument('--samples_per_group', type=int, default=5000,
                        help='Samples per (n,k,m) group for training (default: 5000)')
    parser.add_argument('--val_samples_per_group', type=int, default=10000,
                        help='Samples per (n,k,m) group for validation (default: 10000)')
    parser.add_argument('--val_frequency', type=int, default=5,
                        help='Validate every N epochs (default: 5)')
    parser.add_argument('--base_ch', type=int, default=64,
                        help='Base channels for ResNet (default: 64)')
    parser.add_argument('--num_blocks', type=int, default=5,
                        help='Number of residual blocks (default: 5)')
    args = parser.parse_args()
    
    # Configuration
    config = {
        'weight_decay': args.weight_decay,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'samples_per_group': args.samples_per_group,
        'val_samples_per_group': args.val_samples_per_group,
        'val_frequency': args.val_frequency,
        'base_ch': args.base_ch,
        'num_blocks': args.num_blocks,
        'train_folder': 'split_data_train_20000_random',
        'val_folder': 'split_data_validation_20000_random',
        'output_dir': 'grokking_experiment'
    }
    
    print("=" * 60)
    print("GROKKING / DOUBLE DESCENT EXPERIMENT")
    print("=" * 60)
    print(f"\nConfiguration:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    
    os.makedirs(config['output_dir'], exist_ok=True)
    
    train_grokking(config)

