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

# ==============================================================================
# === Architecture Components (SwiGLU, ConvResBlock) ===
# ==============================================================================

# --- SwiGLU Activation Module ---
class SwiGLU(nn.Module):
    def forward(self, x):
        x_main, x_gate = x.chunk(2, dim=1)
        return F.silu(x_gate) * x_main

# --- Modified ConvResBlock with SwiGLU ---
class ConvResBlockSwiGLU(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, 2 * ch, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(2 * ch)
        self.swiglu1 = SwiGLU()
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(ch)
        self.silu_residual = nn.SiLU()

    def forward(self, x):
        h = self.conv1(x)
        h = self.bn1(h)
        h = self.swiglu1(h)
        h = self.conv2(h)
        h = self.bn2(h)
        return self.silu_residual(x + h)

# ==============================================================================
# === Model Definitions ===
# ==============================================================================

# --- Modified ResNet2DWithParams with SwiGLU (Supports Strategy 1a) ---
class ResNet2DWithParamsSwiGLU(nn.Module):
    """
    ResNet2D model. Supports Strategy 1a: Enforcing Lower Bound (h >= 1).
    """
    def __init__(self,
                 k_max=6, nk_max=6,
                 n_params=3,
                 base_ch=32,
                 num_blocks=3,
                 enforce_lower_bound=False): # NEW: Flag for Strategy 1a
        super().__init__()
        self.enforce_lower_bound = enforce_lower_bound # NEW

        self.stem = nn.Sequential(
            nn.Conv2d(1, base_ch, 3, padding=1),
            nn.BatchNorm2d(base_ch),
            nn.SiLU()
        )
        self.blocks = nn.Sequential(
            *[ConvResBlockSwiGLU(base_ch) for _ in range(num_blocks)]
        )
        flat_dim = base_ch * k_max * nk_max
        self.param_proj = nn.Linear(n_params, 64)
        self.param_proj_act = nn.SiLU()

        self.head_fc1 = nn.Linear(flat_dim + 64, 256 * 2)
        self.head_swiglu = SwiGLU()
        self.head_fc2 = nn.Linear(256, 1)

    def forward(self, P, params):
        x = P.unsqueeze(1)
        x = self.blocks(self.stem(x))
        x = x.flatten(1)
        p = self.param_proj_act(self.param_proj(params.float()))
        x = torch.cat([x, p], dim=1)

        x = self.head_fc1(x)
        x = self.head_swiglu(x)
        x = self.head_fc2(x)

        # NEW: Strategy 1a Implementation
        if self.enforce_lower_bound:
            # Use Softplus + 1 to guarantee output > 1
            x = 1.0 + F.softplus(x)

        return x

# --- NEW: Monotonic ResNet (Strategy 1b) ---
class MonotonicResNetSwiGLU(nn.Module):
    """
    ResNet2D model designed for Strategy 1b: Enforcing Monotonicity.
    Predicts the sequence H = (h_2, ..., h_{n-k}).
    """
    def __init__(self,
                 k_max=6, nk_max=6,
                 base_ch=32,
                 num_blocks=3):
        super().__init__()
        # n_params is fixed at 2 (n, k). 'm' is not an input.
        n_params = 2
        # Output sequence length (m=2 to n-k). Max length is nk_max - 1.
        # E.g. if nk_max=6, we predict h_2, h_3, h_4, h_5, h_6 (Length 5).
        self.max_seq_len = nk_max - 1

        # Feature extraction (same architecture)
        self.stem = nn.Sequential(
            nn.Conv2d(1, base_ch, 3, padding=1),
            nn.BatchNorm2d(base_ch),
            nn.SiLU()
        )
        self.blocks = nn.Sequential(
            *[ConvResBlockSwiGLU(base_ch) for _ in range(num_blocks)]
        )
        flat_dim = base_ch * k_max * nk_max
        self.param_proj = nn.Linear(n_params, 64)
        self.param_proj_act = nn.SiLU()

        # Head modified for sequence output
        self.head_fc1 = nn.Linear(flat_dim + 64, 512 * 2) # Increased capacity
        self.head_swiglu = SwiGLU()
        # Output layer predicts raw deltas
        self.head_fc2 = nn.Linear(512, self.max_seq_len)

    def forward(self, P, params):
        # Feature extraction
        x = P.unsqueeze(1)
        x = self.blocks(self.stem(x))
        x = x.flatten(1)
        # Params here are just (n, k)
        p = self.param_proj_act(self.param_proj(params.float()))
        x = torch.cat([x, p], dim=1)

        # Head forward pass
        x = self.head_fc1(x)
        x = self.head_swiglu(x)
        raw_deltas = self.head_fc2(x) # (B, max_seq_len)

        # --- Strategy 1b Implementation: Monotonicity Enforcement ---

        # 1. Enforce non-negativity (Softplus)
        deltas = F.softplus(raw_deltas)

        # 2. Reconstruct heights using cumulative sum, starting from the base lower bound of 1:
        # h_m = 1 + sum(delta_2, ..., delta_m)
        heights = torch.cumsum(deltas, dim=1) + 1.0

        return heights # (B, max_seq_len)

# ==============================================================================
# === Loss Functions ===
# ==============================================================================

class LogMSELoss(nn.Module):
    """
    Calculates the Mean Squared Error on the log2 scale.
    """
    def __init__(self, eps=1e-9):
        super().__init__()
        self.eps = eps

    def forward(self, y_pred, y_true):
        y_pred_safe = torch.clamp(y_pred, min=self.eps)
        y_true_safe = torch.clamp(y_true, min=self.eps)

        log2_pred = torch.log2(y_pred_safe)
        log2_true = torch.log2(y_true_safe)

        loss = torch.mean((log2_true - log2_pred) ** 2)
        return loss

# --- NEW: Masked LogMSE Loss for Monotonic Model (Strategy 1b) ---
class MaskedLogMSELoss(nn.Module):
    """
    Calculates LogMSELoss for sequences, ignoring padded elements using sequence lengths.
    Uses dynamic masking for efficiency.
    """
    def __init__(self, eps=1e-9):
        super().__init__()
        self.eps = eps

    def forward(self, y_pred, y_true, lengths):
        """
        Args:
            y_pred (torch.Tensor): Predictions (B, SeqMax).
            y_true (torch.Tensor): Ground truth (B, SeqMax).
            lengths (torch.Tensor): Actual lengths of sequences (B,).
        """
        B, SeqMax = y_pred.shape

        # Calculate Log2 values
        y_pred_safe = torch.clamp(y_pred, min=self.eps)
        y_true_safe = torch.clamp(y_true, min=self.eps)
        log2_pred = torch.log2(y_pred_safe)
        log2_true = torch.log2(y_true_safe)

        # Calculate squared differences
        sq_diff = (log2_true - log2_pred) ** 2

        # Create mask dynamically
        # Create a range tensor [0, 1, ..., SeqMax-1]
        range_tensor = torch.arange(SeqMax, device=y_pred.device).expand(B, SeqMax)
        # Mask is True where range < length, False otherwise
        mask = range_tensor < lengths.unsqueeze(1)

        # Apply mask
        masked_sq_diff = sq_diff * mask.float()

        # Calculate mean loss over valid elements
        total_loss = torch.sum(masked_sq_diff)
        total_elements = torch.sum(lengths).float()

        if total_elements == 0:
            # Handle edge case where batch might have no valid elements
            return torch.tensor(0.0, device=y_pred.device, requires_grad=True)

        return total_loss / total_elements

# ==============================================================================
# === Data Handling ===
# ==============================================================================

# --- Normalisation functions (Helper) ---
EPS = 1e-9
def norm_none(t):
    return t
# (Add other normalization functions here if needed, e.g., norm_row_standard)
NORMALISERS = {
    "none": norm_none,
}

# --- Standard Data Loader Class (For Strategy 1a) ---
class PickleFolderDataset(Dataset):
    """
    PyTorch Dataset for loading individual (n, k, m, P, h_m) samples.
    Used for the standard ResNet and Strategy 1a.
    """
    def __init__(self, file_paths: list, max_k :int = 6, max_nk:int = 6, p_normaliser:str = "none"):
        super().__init__()
        if p_normaliser not in NORMALISERS:
            raise ValueError(f"Invalid p_normaliser: {p_normaliser}.")

        self._normalise = NORMALISERS[p_normaliser]
        self.max_k,self.max_nk = max_k,max_nk

        # LOAD DATA (Simplified loading using pandas for efficiency)
        print(f"Loading data from {len(file_paths)} pickle files (Standard Format)...")
        all_dfs = []
        for fp in file_paths:
            try:
                with open(fp, "rb") as f:
                    df = pickle.load(f)
                    all_dfs.append(df)
            except Exception as e:
                print(f"Error loading {fp}: {e}. Skipping.")

        if not all_dfs:
            raise ValueError("No data loaded.")

        data = pd.concat(all_dfs, ignore_index=True)

        self.P_raw = [np.array(p) for p in data['P']]
        self.h_vals = torch.tensor(data['result'].astype(float).tolist(), dtype=torch.float32)
        self.n_vals = torch.tensor(data['n'].astype(int).tolist(), dtype=torch.float32)
        self.k_vals = torch.tensor(data['k'].astype(int).tolist(), dtype=torch.float32)
        self.m_vals = torch.tensor(data['m'].astype(int).tolist(), dtype=torch.float32)

        print(f"Finished initial loading. Total samples found: {len(self.h_vals)}")


    def __len__(self):
        return len(self.h_vals)

    def _pad(self, p2d: torch.Tensor):
        k_act, nk_act = p2d.shape
        pad = (0, self.max_nk - nk_act, 0, self.max_k - k_act)
        return F.pad(p2d, pad, value=0.)

    def __getitem__(self, idx):
        # --- reshape / pad ---
        p_np   = self.P_raw[idx]
        n, k   = int(self.n_vals[idx]), int(self.k_vals[idx])
        target_shape = (k, n - k)
        if p_np.ndim == 1:
            p_np = p_np.reshape(*target_shape)
        p_t = torch.tensor(p_np, dtype=torch.float32)
        p_t = self._pad(p_t)

        # --- normalise ---
        p_t = self._normalise(p_t)

        # --- build output ---
        # Params: (n, k, m)
        params  = torch.tensor([self.n_vals[idx], self.k_vals[idx], self.m_vals[idx]],
                               dtype=torch.float32)
        h       = self.h_vals[idx].unsqueeze(0)

        return params, h, p_t

# --- NEW: Monotonic Data Loader Class (For Strategy 1b) ---
class MonotonicPickleDataset(Dataset):
    """
    PyTorch Dataset for Strategy 1b. Groups samples by P matrix to form height sequences (m>=2).
    Returns (params, H_seq, P, lengths).
    """
    def __init__(self, file_paths: list, max_k: int = 6, max_nk: int = 6, p_normaliser: str = "none"):
        super().__init__()
        self.max_k = max_k
        self.max_nk = max_nk
        # Sequence length for m=2 to n-k. Max length is nk_max - 1.
        self.max_seq_len = max_nk - 1
        self._normalise = NORMALISERS[p_normaliser]

        print(f"Loading and grouping data from {len(file_paths)} pickle files (Monotonic Format m>=2)...")

        # 1. Efficiently load all data
        all_dfs = []
        for fp in file_paths:
            try:
                with open(fp, "rb") as f:
                    df = pickle.load(f)
                    all_dfs.append(df)
            except Exception as e:
                print(f"Error loading {fp}: {e}. Skipping.")

        if not all_dfs:
            raise ValueError("No data loaded.")

        data = pd.concat(all_dfs, ignore_index=True)

        # Create a hashable representation of P for grouping
        # Ensure P is flattened consistently
        data['P_hashable'] = data['P'].apply(lambda x: tuple(np.array(x).flatten()))

        # 2. Group by P matrix (and n, k) and aggregate results
        self.samples = []
        grouped = data.groupby(['n', 'k', 'P_hashable'])

        for (n, k, p_hashable), group_df in grouped:
            nk = n - k
            # Sequence length for m=2 to n-k is (n-k) - 2 + 1 = n-k-1
            seq_len = nk - 1

            if seq_len <= 0:
                continue # Skip if n-k < 2 (no relevant m values)

            # Initialize height sequence H_seq
            H_seq = [0.0] * seq_len

            # Populate the sequence
            for index, row in group_df.iterrows():
                m = int(row['m'])
                # Ensure m is within the expected range (2 to n-k)
                if 2 <= m <= nk:
                    # Convert m to 0-based index (m-2)
                    H_seq[m-2] = float(row['result'])

            # Check if the sequence is complete (no zeros left, since h_m >= 1)
            if any(h < 1.0 for h in H_seq):
                 # Skip incomplete sequences found in the dataset
                 continue

            # Reconstruct the original P matrix shape
            P_matrix = np.array(p_hashable).reshape(k, nk)

            self.samples.append({
                'n': n,
                'k': k,
                'P': P_matrix,
                'H_seq': H_seq,
                'length': seq_len
            })

        print(f"Finished grouping. Total unique P matrices (samples): {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def _pad_P(self, p2d: torch.Tensor):
        k_act, nk_act = p2d.shape
        pad = (0, self.max_nk - nk_act, 0, self.max_k - k_act)
        return F.pad(p2d, pad, value=0.)

    def _pad_H(self, h_seq: torch.Tensor):
        # Pad H sequence to max_seq_len
        seq_len_act = h_seq.shape[0]
        pad = (0, self.max_seq_len - seq_len_act)
        return F.pad(h_seq, pad, value=0.) # Padding value will be masked

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Process P matrix
        P_t = torch.tensor(sample['P'], dtype=torch.float32)
        P_t = self._normalise(P_t)
        P_padded = self._pad_P(P_t)

        # Process H sequence
        H_t = torch.tensor(sample['H_seq'], dtype=torch.float32)
        H_padded = self._pad_H(H_t)

        # Params (n, k)
        params = torch.tensor([sample['n'], sample['k']], dtype=torch.float32)
        length = torch.tensor(sample['length'], dtype=torch.long)

        # Returns (params, H_padded, P_padded, length)
        return params, H_padded, P_padded, length

# ==============================================================================
# === Trainer Class ===
# ==============================================================================

# --- Trainer Class (Updated to handle both standard and monotonic approaches) ---
class ResNetTrainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer,
                 device, model_name="Model", patience=10, scheduler=None, is_monotonic=False):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.model_name = model_name
        self.patience = patience
        self.scheduler = scheduler
        self.is_monotonic = is_monotonic # NEW: Flag to indicate the training mode

        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.epochs_no_improve = 0

    def _train_epoch(self):
        self.model.train()
        running_loss = 0.0
        total_samples = 0

        # The loop structure depends on the dataset format (determined by is_monotonic)
        if self.is_monotonic:
            # Monotonic format: (params, H_padded, P_padded, length)
            for params, targets, p_matrices, lengths in self.train_loader:
                params, targets, p_matrices, lengths = params.to(self.device), targets.to(self.device), p_matrices.to(self.device), lengths.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(p_matrices, params)
                # Use MaskedLogMSELoss (requires lengths)
                loss = self.criterion(outputs, targets, lengths)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item() * params.size(0)
                total_samples += params.size(0)
        else:
            # Standard format: (params, h, P)
            for params, targets, p_matrices in self.train_loader:
                params, targets, p_matrices = params.to(self.device), targets.to(self.device), p_matrices.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(p_matrices, params)
                # Use standard LogMSELoss
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
            if self.is_monotonic:
                 # Monotonic format
                for params, targets, p_matrices, lengths in self.val_loader:
                    params, targets, p_matrices, lengths = params.to(self.device), targets.to(self.device), p_matrices.to(self.device), lengths.to(self.device)
                    outputs = self.model(p_matrices, params)
                    loss = self.criterion(outputs, targets, lengths)
                    running_loss += loss.item() * params.size(0)
                    total_samples += params.size(0)
            else:
                # Standard format
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

            train_loss = self._train_epoch()
            val_loss = self._validate_epoch()

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            duration = time.time() - epoch_start
            print(f"Epoch {epoch}/{epochs} ({self.model_name}): "
                  f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                  f"Duration: {duration:.2f}s")

            if val_loss < self.best_val_loss:
                print(f"  Val loss improved ({self.best_val_loss:.4f} -> {val_loss:.4f}). Saving checkpoint.")
                self.best_val_loss = val_loss
                self.epochs_no_improve = 0
                torch.save(self.model.state_dict(),
                           f"best_{self.model_name.lower().replace(' ', '_')}.pth")
            else:
                self.epochs_no_improve += 1
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

# ==============================================================================
# === Utilities ===
# ==============================================================================

# --- Plotting Function ---
def plot_losses(results_dict):
    """
    Plots training and validation losses.
    """
    if not results_dict:
        return

    print("\nPlotting losses...")
    plt.figure(figsize=(8, 6))

    for model_name, (train_losses, val_losses) in results_dict.items():
        if not train_losses or not val_losses:
            continue

        epochs = list(range(1, len(train_losses) + 1))
        best_loss  = min(val_losses)
        best_epoch = val_losses.index(best_loss) + 1

        plt.plot(epochs,
                 train_losses,
                 label=f"Train Loss")
        plt.plot(epochs,
                 val_losses,
                 label=(f"Val Loss (best: {best_loss:.4f} @ epoch {best_epoch})"))

        plt.title(f"{model_name} Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Log2 MSE Loss")
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.show()


# ==============================================================================
# === Main Execution Block ===
# ==============================================================================

if __name__ == '__main__':
    # --- Configuration ---
    # === CHOOSE STRATEGY ===
    # 0: Baseline (Original ResNet SwiGLU)
    # 1: Strategy 1a (ResNet SwiGLU + Enforce Lower Bound >= 1)
    # 2: Strategy 1b (Monotonic ResNet SwiGLU)
    STRATEGY = 1 # <-- Set the desired strategy here (e.g., 1 or 2)
    # =======================

    TRAIN_DATA_FOLDERS = ['./split_data_train_20000_random']
    VALIDATION_DATA_FOLDERS = ['./split_data_validation_20000_random']
    max_k = 6
    max_nk = 6
    batch_size = 512 # Adjust if needed, especially for the Monotonic model
    val_split_ratio = 0.2
    NUM_WORKERS = 0
    PIN_MEMORY = True
    EPOCHS = 50
    PATIENCE = 10
    LEARNING_RATE = 3e-4
    WEIGHT_DECAY = 1e-4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Determine configuration based on strategy
    if STRATEGY == 2:
        DatasetClass = MonotonicPickleDataset
        IS_MONOTONIC = True
        print("\nSelected Strategy 1b: Monotonic Enforcement (m>=2).")
    else:
        DatasetClass = PickleFolderDataset
        IS_MONOTONIC = False
        if STRATEGY == 1:
            print("\nSelected Strategy 1a: Lower Bound Enforcement.")
        else:
            print("\nSelected Baseline Strategy.")


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
            raise FileNotFoundError("No .pkl files found.")

        # Initialize the appropriate dataset
        combined_dataset = DatasetClass(
            file_paths=all_files,
            max_k=max_k,
            max_nk=max_nk,
            p_normaliser="none", # Using "none" normalization
        )

        total_samples = len(combined_dataset)
        print(f"Total samples loaded (Note: For Monotonic, this is the number of unique matrices): {total_samples}")

        val_size = int(total_samples * val_split_ratio)
        train_size = total_samples - val_size

        if train_size <= 0 or val_size <= 0:
             raise ValueError(f"Train ({train_size}) or validation ({val_size}) size is zero or less.")

        print(f"Splitting data: Training={train_size}, Validation={val_size}\n")
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

    except Exception as e:
        print(f"\nError during data loading: {e}")
        print(traceback.format_exc())
        exit()

    # --- Model Initialization, Loss & Optimizer ---
    print("\n--- Initializing Model and Components ---")

    if IS_MONOTONIC:
        # Strategy 1b: Monotonic Model
        model = MonotonicResNetSwiGLU(
            k_max=max_k,
            nk_max=max_nk,
            base_ch=32,
            num_blocks=3
        ).to(device)
        criterion = MaskedLogMSELoss()
        MODEL_NAME = "Monotonic_ResNet_SwiGLU"

    else:
        # Strategy 0 or 1a: Standard Model
        ENFORCE_LOWER_BOUND = (STRATEGY == 1)
        model = ResNet2DWithParamsSwiGLU(
            k_max=max_k,
            nk_max=max_nk,
            n_params=3,
            base_ch=32,
            num_blocks=3,
            enforce_lower_bound=ENFORCE_LOWER_BOUND
        ).to(device)
        criterion = LogMSELoss()
        MODEL_NAME = "ResNet_SwiGLU_Baseline"
        if ENFORCE_LOWER_BOUND:
            MODEL_NAME = "ResNet_SwiGLU_LowerBound"

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    # Reduce LR on Plateau is generally effective
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=PATIENCE//2, factor=0.2)

    # --- Instantiate and Run Trainer ---
    trainer = ResNetTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        model_name=MODEL_NAME,
        patience=PATIENCE,
        scheduler=scheduler,
        is_monotonic=IS_MONOTONIC # Pass the mode flag to the trainer
    )

    train_losses, val_losses = trainer.run_training(epochs=EPOCHS)

    # --- Record & Plot Results ---
    results = {}
    results[MODEL_NAME] = (train_losses, val_losses)
    plot_losses(results)

    print(f"\nTraining of {MODEL_NAME} complete. Best validation loss achieved: {min(val_losses):.4f}")