import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import time
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import pickle
import pandas as pd
from collections import defaultdict
import traceback

# --- SwiGLU Activation Module ---
class SwiGLU(nn.Module):
    """
    SwiGLU activation function.
    Assumes input 'x' has channels split into two halves for gate and main.
    """
    def forward(self, x):
        # Split along the channel dimension
        x_main, x_gate = x.chunk(2, dim=1)
        return F.silu(x_gate) * x_main

# --- Modified ConvResBlock with SwiGLU ---
class ConvResBlockSwiGLU(nn.Module):
    def __init__(self, ch):
        super().__init__()
        # conv1 outputs 2*ch for SwiGLU, which then reduces it back to ch
        self.conv1 = nn.Conv2d(ch, 2 * ch, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(2 * ch) # BatchNorm on 2*ch channels
        self.swiglu1 = SwiGLU()
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(ch)
        self.silu_residual = nn.SiLU() # Using SiLU for the final residual activation

    def forward(self, x):
        h = self.conv1(x)
        h = self.bn1(h)
        h = self.swiglu1(h)
        h = self.conv2(h)
        h = self.bn2(h)
        return self.silu_residual(x + h)

# --- Modified ResNet2DWithParams with SwiGLU ---
class ResNet2DWithParamsSwiGLU(nn.Module):
    """
    ResNet2D model interpreting padded P as a single-channel image,
    using SwiGLU in ConvResBlocks and SiLU/SwiGLU in other parts.
    """
    def __init__(self,
                 k_max=6, nk_max=6,
                 n_params=3,
                 base_ch=32,
                 num_blocks=3):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, base_ch, 3, padding=1),
            nn.BatchNorm2d(base_ch),
            nn.SiLU() # Changed from ReLU to SiLU
        )
        self.blocks = nn.Sequential(
            *[ConvResBlockSwiGLU(base_ch) for _ in range(num_blocks)] # Using new block
        )
        flat_dim = base_ch * k_max * nk_max
        self.param_proj = nn.Linear(n_params, 64)
        self.param_proj_act = nn.SiLU() # Changed from F.relu to SiLU

        # Head with SwiGLU
        self.head_fc1 = nn.Linear(flat_dim + 64, 256 * 2) # Output 2*dim for SwiGLU
        self.head_swiglu = SwiGLU()
        self.head_fc2 = nn.Linear(256, 1)

    def forward(self, P, params):
        x = P.unsqueeze(1)                  # (B, 1, k_max, nk_max)
        x = self.blocks(self.stem(x))
        x = x.flatten(1)
        p = self.param_proj_act(self.param_proj(params.float())) # Apply SiLU after param_proj
        x = torch.cat([x, p], dim=1)

        # Head forward pass with SwiGLU
        x = self.head_fc1(x)
        x = self.head_swiglu(x)
        x = self.head_fc2(x)
        return x

# --- Helper Functions (Copied from your notebook for self-containment) ---

# --- Define the custom MSE loss function on the log2 scale ---
class LogMSELoss(nn.Module):
    """
    Calculates the Mean Squared Error on the log2 scale.
    Loss = mean( (log2(y_true) - log2(y_pred))^2 )
    Adds a small epsilon to prevent log2(0).
    """
    def __init__(self, eps=1e-9):
        super().__init__()
        self.eps = eps

    def forward(self, y_pred, y_true):
        """
        Args:
            y_pred (torch.Tensor): Predictions from the model.
            y_true (torch.Tensor): Ground truth values.
        """
        # Clamp inputs to be positive to avoid log2(<=0)
        y_pred_safe = torch.clamp(y_pred, min=self.eps)
        y_true_safe = torch.clamp(y_true, min=self.eps) # Clamp true values too for safety

        # Calculate log base 2
        log2_pred = torch.log2(y_pred_safe)
        log2_true = torch.log2(y_true_safe)

        # Calculate the squared difference and mean over the batch
        loss = torch.mean((log2_true - log2_pred) ** 2)
        return loss

# --- Data Loader Class (Copied from your notebook for self-containment) ---
# --- Normalisation function ---
EPS = 1e-9
def norm_none(t):
    return t                                      # raw P

def norm_row_standard(t):
    mu  = t.mean(dim=1, keepdim=True)
    std = t.std (dim=1, keepdim=True) + EPS
    return (t - mu) / std                         # each row N(0,1)

def norm_col_minmax(t, to=(-1., 1.)):
    lo, hi = to
    col_min = t.min(dim=0, keepdim=True).values
    col_max = t.max(dim=0, keepdim=True).values
    rng = (col_max - col_min).clamp_min(EPS)
    return (t - col_min) / rng * (hi - lo) + lo   # columns -> [-1,1]

NORMALISERS = {
    "none":           norm_none,
    "row_standard":   norm_row_standard,
    "col_minmax":     norm_col_minmax,
}

class PickleFolderDataset(Dataset):
    """
    PyTorch Dataset for loading data from a **list of .pkl file paths**.
    Assumes each .pkl file contains a Pandas DataFrame with columns
    'n', 'k', 'm', 'result', 'P'.
    Handles cases where 'P' might be a 2D array OR a flattened 1D array.
    Reshapes 1D 'P' arrays to 2D using 'n' and 'k' before padding.
    Handles padding P_matrices.
    Returns data as (params, h, P).
    """
    def __init__(self, file_paths: list, max_k :int = 6, max_nk:int = 6, transform:callable=None, p_normaliser:str = "none", return_col_indices=False):
        super().__init__()
        if p_normaliser not in NORMALISERS:
            raise ValueError(f"Invalid p_normaliser: {p_normaliser}. Must be one of: {list(NORMALISERS.keys())}")

        self._normalise = NORMALISERS[p_normaliser]
        self.return_col_indices = return_col_indices
        self.max_k,self.max_nk = max_k,max_nk
        self.transform = transform

        if not file_paths:
            raise ValueError("The provided file_paths list is empty.")

        # ------------------------------------------------------------------
        # LOAD DATA
        # ------------------------------------------------------------------
        # Store data directly as loaded from pickle (P can be 1D or 2D)
        self.P_raw, self.h_vals, self.n_vals, self.k_vals, self.m_vals = [], [], [], [], []

        print(f"Loading data from {len(file_paths)} specified pickle files (expecting DataFrames)...")
        required = ['n', 'k', 'm', 'result', 'P']

        for fp in file_paths: # Iterate through the provided list
            try:
                with open(fp, "rb") as f:
                    df = pickle.load(f)
                if not all(col in df.columns for col in required):
                    continue

                self.P_raw.extend([np.array(p) for p in df['P']])
                self.h_vals.extend(df['result'].astype(float).tolist())
                self.n_vals.extend(df['n'].astype(int).tolist())
                self.k_vals.extend(df['k'].astype(int).tolist())
                self.m_vals.extend(df['m'].astype(int).tolist())

            except FileNotFoundError:
                    print(f"Error: File not found {fp}. Skipping.")
            except Exception as e:
                print(f"Error loading/processing {fp}: {type(e).__name__} - {e}. Skipping file.")

        if not self.h_vals:
                raise ValueError("No valid data loaded from any pickle files.")

        print(f"Finished initial loading. Total samples found: {len(self.h_vals)}")

        # --- Convert non-P lists to tensors ---
        self.h_vals  = torch.tensor(self.h_vals , dtype=torch.float32)
        self.n_vals  = torch.tensor(self.n_vals , dtype=torch.float32)
        self.k_vals  = torch.tensor(self.k_vals , dtype=torch.float32)
        self.m_vals  = torch.tensor(self.m_vals , dtype=torch.float32)

    def __len__(self):
        return len(self.h_vals)

    def _pad(self, p2d: torch.Tensor):
        k_act, nk_act = p2d.shape
        pad = (0, self.max_nk - nk_act, 0, self.max_k - k_act)  # (W_left,W_right,H_top,H_bottom)
        return F.pad(p2d, pad, value=0.)

    def __getitem__(self, idx):
        # --- reshape / pad ---
        p_np   = self.P_raw[idx]
        n, k   = int(self.n_vals[idx]), int(self.k_vals[idx])
        target = (k, n - k)
        if p_np.ndim == 1:
            p_np = p_np.reshape(*target)  # raises if shape mismatched
        p_t = torch.tensor(p_np, dtype=torch.float32)
        p_t = self._pad(p_t)

        # --- normalise ---
        p_t = self._normalise(p_t)

        # --- extra transform hook ---
        if self.transform:
            p_t = self.transform(p_t)

        # --- build output ---
        params  = torch.tensor([self.n_vals[idx], self.k_vals[idx], self.m_vals[idx]],
                               dtype=torch.float32)
        h       = self.h_vals[idx].unsqueeze(0)

        if self.return_col_indices:
            col_idx = torch.arange(target[1], dtype=torch.long)      # 0 … n‑k‑1
            if target[1] < self.max_nk:
                 col_idx = F.pad(col_idx, (0, self.max_nk - target[1]), value = 0)
            return params, h, p_t, col_idx

        return params, h, p_t

# --- Trainer Class ---
class ResNetTrainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer,
                 device, model_name="Model", patience=10, scheduler=None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.model_name = model_name
        self.patience = patience
        self.scheduler = scheduler

        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.epochs_no_improve = 0

    def _train_epoch(self):
        self.model.train()
        running_loss = 0.0
        total_samples = 0
        for params, targets, p_matrices in self.train_loader:
            params, targets, p_matrices = params.to(self.device), targets.to(self.device), p_matrices.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(p_matrices, params)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * params.size(0)
            total_samples += params.size(0)
        return running_loss / total_samples if total_samples > 0 else 0.0

    def _validate_epoch(self):
        self.model.eval()
        running_loss = 0.0
        total_samples = 0
        with torch.no_grad():
            for params, targets, p_matrices in self.val_loader:
                params, targets, p_matrices = params.to(self.device), targets.to(self.device), p_matrices.to(self.device)
                outputs = self.model(p_matrices, params)
                loss = self.criterion(outputs, targets)
                running_loss += loss.item() * params.size(0)
                total_samples += params.size(0)
        return running_loss / total_samples if total_samples > 0 else 0.0

    def run_training(self, epochs):
        print(f"\n--- Starting Training Loop for {self.model_name} ---")
        start_time = time.time()

        for epoch in range(1, epochs + 1):
            epoch_start = time.time()
            print(f"\nEpoch {epoch}/{epochs} ({self.model_name})")

            train_loss = self._train_epoch()
            val_loss = self._validate_epoch()

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            duration = time.time() - epoch_start
            print(f"Epoch {epoch} Summary ({self.model_name}): "
                  f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                  f"Duration: {duration:.2f}s")

            if val_loss < self.best_val_loss:
                print(f"  {self.model_name} Val loss improved "
                      f"({self.best_val_loss:.4f} -> {val_loss:.4f}). Saving checkpoint.")
                self.best_val_loss = val_loss
                self.epochs_no_improve = 0
                torch.save(self.model.state_dict(),
                           f"best_{self.model_name.lower().replace(' ', '_')}.pth")
            else:
                self.epochs_no_improve += 1
                print(f"  No improvement for {self.epochs_no_improve} epoch(s).")
                if self.epochs_no_improve >= self.patience:
                    print(f"Early stopping triggered after {self.patience} epochs with no improvement.")
                    break

            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

        total_time = time.time() - start_time
        print(f"\n{self.model_name} Training Finished. Total time: {total_time:.2f}s. "
              f"Best Val Loss: {self.best_val_loss:.4f}")
        return self.train_losses, self.val_losses

# --- Plotting Function (can remain standalone or be part of a utility class) ---
def plot_losses(results_dict):
    """
    Plots training and validation losses for multiple models.
    """
    num_models = len(results_dict)
    if num_models == 0:
        print("No results to plot.")
        return

    print("\nPlotting losses...")
    cols = 2
    rows = (num_models + cols - 1) // cols
    plt.figure(figsize=(6 * cols, 5 * rows))

    for idx, (model_name, (train_losses, val_losses)) in enumerate(results_dict.items(), start=1):
        if not train_losses or not val_losses:
            print(f"Skipping plot for {model_name} (no data).")
            continue

        epochs = list(range(1, len(train_losses) + 1))
        best_loss  = min(val_losses)
        best_epoch = val_losses.index(best_loss) + 1

        plt.subplot(rows, cols, idx)
        plt.plot(epochs,
                 train_losses,
                 label=f"{model_name} Train Loss")
        plt.plot(epochs,
                 val_losses,
                 label=(f"{model_name} Val Loss "
                        f"(best: {best_loss:.4f} @ epoch {best_epoch})"))

        plt.title(f"{model_name} Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Log2 MSE Loss")
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.show()


# --- Main Execution Block ---
if __name__ == '__main__':
    # --- Configuration ---
    TRAIN_DATA_FOLDERS = ['./split_data_train_20000_random']
    VALIDATION_DATA_FOLDERS = ['./split_data_validation_20000_random']
    max_k = 6
    max_nk = 6
    batch_size = 512
    val_split_ratio = 0.2 # 80/20 split
    NUM_WORKERS = 0 # Safer default for Windows
    PIN_MEMORY = True # Generally good if using GPU
    EPOCHS = 50
    PATIENCE = 10
    LEARNING_RATE = 3e-4
    WEIGHT_DECAY = 1e-4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Load and Split Data ---
    try:
        print("--- Preparing Data ---")
        all_files = []
        for folder in TRAIN_DATA_FOLDERS + VALIDATION_DATA_FOLDERS:
             files = glob.glob(os.path.join(folder, '*.pkl'))
             if not files:
                  print(f"Warning: No .pkl files found in folder: {folder}")
             all_files.extend(files)

        if not all_files:
            raise FileNotFoundError("No .pkl files found in any specified train/validation folders.")

        print(f"Found {len(all_files)} total .pkl files.")

        combined_dataset = PickleFolderDataset(
            file_paths=all_files,
            max_k=max_k,
            max_nk=max_nk,
            p_normaliser="none", # Use None for now  
        )

        total_samples = len(combined_dataset)
        print(f"Total samples loaded from combined files: {total_samples}")

        val_size = int(total_samples * val_split_ratio)
        train_size = total_samples - val_size

        if train_size <= 0 or val_size <= 0:
             raise ValueError(f"Calculated train ({train_size}) or validation ({val_size}) size is zero or less.")

        print(f"Splitting data: Training={train_size} ({100*(1-val_split_ratio):.1f}%), Validation={val_size} ({100*val_split_ratio:.1f}%)\n")
        train_dataset, val_dataset = random_split(combined_dataset, [train_size, val_size])

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY
        )
        print("DataLoaders created.")
        print(f"Training batches: {len(train_loader)}, Validation batches: {len(val_loader)}")

    except Exception as e:
        print(f"\nError during data loading: {e}")
        print(traceback.format_exc())
        exit() # Exit if data loading fails

    # --- Instantiate the ResNet2DWithParamsSwiGLU model ---
    print("\n--- Initializing ResNet2DWithParamsSwiGLU model ---")
    resnet_swiglu_model = ResNet2DWithParamsSwiGLU(
        k_max=max_k,
        nk_max=max_nk,
        n_params=3,
        base_ch=32,
        num_blocks=3
    ).to(device)

    # --- Loss & Optimizer ---
    criterion = LogMSELoss()
    optimizer = torch.optim.AdamW(resnet_swiglu_model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # --- Scheduler ---
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=PATIENCE//2, factor=0.2)

    # --- Instantiate and Run Trainer ---
    model_name = "ResNet2D_SwiGLU_OOP"
    trainer = ResNetTrainer(
        model=resnet_swiglu_model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        model_name=model_name,
        patience=PATIENCE,
        scheduler=scheduler
    )

    train_losses, val_losses = trainer.run_training(epochs=EPOCHS)

    # --- Record & Plot Results ---
    results = {}
    results[model_name] = (train_losses, val_losses)
    plot_losses(results)

    print(f"\nTraining of {model_name} complete. Best validation loss achieved: {min(val_losses):.4f}")
