import os
import glob
import time
import pickle
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset
import matplotlib.pyplot as plt


# ==============================================================================
# === Dataset Utilities ========================================================
# ==============================================================================

EPS = 1e-9


def norm_none(t: torch.Tensor) -> torch.Tensor:
    return t


NORMALISERS = {
    "none": norm_none,
}


class PickleFolderDataset(Dataset):
    """
    Loads samples from pickle files containing DataFrames with columns
    ['n', 'k', 'm', 'result', 'P']. Handles reshaping/padding of P and returns
    (params, target, padded_P).
    """

    def __init__(self, file_paths: Sequence[str], max_k: int = 6, max_nk: int = 6, p_normaliser: str = "none"):
        super().__init__()
        if not file_paths:
            raise ValueError("file_paths list is empty.")
        if p_normaliser not in NORMALISERS:
            raise ValueError(f"Unknown p_normaliser '{p_normaliser}'. Valid options: {list(NORMALISERS.keys())}")

        self.max_k = max_k
        self.max_nk = max_nk
        self._normalise = NORMALISERS[p_normaliser]

        self.P_raw: List[np.ndarray] = []
        self.h_vals: List[float] = []
        self.n_vals: List[int] = []
        self.k_vals: List[int] = []
        self.m_vals: List[int] = []

        required = {"n", "k", "m", "result", "P"}
        print(f"Loading data from {len(file_paths)} pickle file(s)...")
        for fp in file_paths:
            try:
                with open(fp, "rb") as f:
                    df = pickle.load(f)
            except Exception as exc:
                print(f"  ! Skipping {fp}: {exc}")
                continue
            if not required.issubset(df.columns):
                print(f"  ! Skipping {fp}: missing columns {required - set(df.columns)}")
                continue

            self.P_raw.extend(df["P"].tolist())
            self.h_vals.extend(df["result"].astype(float).tolist())
            self.n_vals.extend(df["n"].astype(int).tolist())
            self.k_vals.extend(df["k"].astype(int).tolist())
            self.m_vals.extend(df["m"].astype(int).tolist())

        if not self.h_vals:
            raise ValueError("No valid samples were loaded.")

        print(f"Finished loading. Total samples: {len(self.h_vals)}")
        self.h_vals = torch.tensor(self.h_vals, dtype=torch.float32)
        self.n_vals = torch.tensor(self.n_vals, dtype=torch.float32)
        self.k_vals = torch.tensor(self.k_vals, dtype=torch.float32)
        self.m_vals = torch.tensor(self.m_vals, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.h_vals)

    def _pad(self, mat: torch.Tensor) -> torch.Tensor:
        k_act, nk_act = mat.shape
        pad = (0, self.max_nk - nk_act, 0, self.max_k - k_act)  # (W_left,W_right,H_top,H_bottom)
        return F.pad(mat, pad, value=0.0)

    def __getitem__(self, idx: int):
        n = int(self.n_vals[idx].item())
        k = int(self.k_vals[idx].item())
        m = int(self.m_vals[idx].item())

        raw = np.array(self.P_raw[idx])
        if raw.ndim == 1:
            raw = raw.reshape(k, n - k)
        P = torch.tensor(raw, dtype=torch.float32)
        P = self._normalise(self._pad(P))

        params = torch.tensor([n, k, m], dtype=torch.float32)
        target = self.h_vals[idx].unsqueeze(0)
        return params, target, P


class PredominantSpecialistSubset(Dataset):
    """
    Creates a dataset view where samples for a specific (n,k) pair are 'predominant'
    and mixed with a specified ratio of other samples.
    """
    def __init__(self, base_dataset: PickleFolderDataset, n_value: int, k_value: int, predominant_ratio: float):
        super().__init__()
        self.base_dataset = base_dataset
        
        n_all = base_dataset.n_vals.cpu().numpy().astype(int)
        k_all = base_dataset.k_vals.cpu().numpy().astype(int)

        predominant_mask = (n_all == n_value) & (k_all == k_value)
        predominant_indices = np.where(predominant_mask)[0]
        other_indices = np.where(~predominant_mask)[0]
        
        if len(predominant_indices) == 0:
            raise ValueError(f"No samples found for predominant pair n={n_value}, k={k_value}.")

        num_predominant = len(predominant_indices)
        
        # Calculate how many 'other' samples we need to match the desired ratio
        if predominant_ratio < 1.0:
            num_other_needed = int(num_predominant * (1.0 - predominant_ratio) / predominant_ratio)
            num_other_needed = min(num_other_needed, len(other_indices)) # cap at available samples
        else:
            num_other_needed = 0

        if num_other_needed > 0:
            # Randomly sample from the 'other' indices
            other_sampled_indices = np.random.choice(other_indices, num_other_needed, replace=False)
            self.indices = np.concatenate([predominant_indices, other_sampled_indices]).tolist()
        else:
            self.indices = predominant_indices.tolist()

        self.n_value = n_value
        self.k_value = k_value
        self.ratio = predominant_ratio
        
        actual_ratio = len(predominant_indices) / len(self.indices) if len(self.indices) > 0 else 0
        
        print(f"  Predominant subset (n={n_value}, k={k_value}, ratio={predominant_ratio:.2f}) -> "
              f"{len(self.indices)} samples ({len(predominant_indices)} predominant, {num_other_needed} other, actual_ratio={actual_ratio:.2f})")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, subset_idx: int):
        base_idx = self.indices[subset_idx]
        return self.base_dataset[base_idx]


# ==============================================================================
# === Models ===================================================================
# ==============================================================================


class ConvResBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(ch)
        self.bn2 = nn.BatchNorm2d(ch)

    def forward(self, x):
        h = F.relu(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        return F.relu(x + h)


class ResNet2DWithParams(nn.Module):
    """
    ResNet processing padded P plus (n,k,m) parameters. Head enforces lower bound via 1+softplus.
    """

    def __init__(self, k_max=6, nk_max=6, n_params=3, base_ch=32, num_blocks=4, dropout_p=0.1):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, base_ch, 3, padding=1),
            nn.BatchNorm2d(base_ch),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(*[ConvResBlock(base_ch) for _ in range(num_blocks)])
        flat_dim = base_ch * k_max * nk_max
        self.param_proj = nn.Linear(n_params, 64)
        self.head = nn.Sequential(
            nn.Linear(flat_dim + 64, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
            nn.Linear(256, 1),
        )

    def forward(self, P, params):
        x = P.unsqueeze(1)
        x = self.blocks(self.stem(x))
        x = x.flatten(1)
        param_embed = F.relu(self.param_proj(params.float()))
        x = torch.cat([x, param_embed], dim=1)
        z = self.head(x)
        return 1.0 + F.softplus(z)


# ==============================================================================
# === Losses & Trainers ========================================================
# ==============================================================================


class LogMSELoss(nn.Module):
    def __init__(self, eps: float = 1e-9):
        super().__init__()
        self.eps = eps

    def forward(self, y_pred, y_true):
        y_pred = torch.clamp(y_pred, min=self.eps)
        y_true = torch.clamp(y_true, min=self.eps)
        return torch.mean((torch.log2(y_true) - torch.log2(y_pred)) ** 2)


class ViolationInformedLossAccelerated(nn.Module):
    def __init__(self, lambda_violation=0.5, num_samples=10, eps=1e-9):
        super().__init__()
        self.lambda_violation = lambda_violation
        self.num_samples = num_samples
        self.eps = eps
        self.log_mse = LogMSELoss(eps=eps)

    def forward(self, y_pred, y_true, P_padded=None, params=None, calculate_violation=True):
        loss_logmse = self.log_mse(y_pred, y_true)

        if not calculate_violation or P_padded is None or params is None or self.lambda_violation == 0:
            # Return tuple to match signature, but 3rd element is 0
            return loss_logmse, loss_logmse.item(), 0.0

        B = P_padded.shape[0]
        device = y_pred.device
        total_penalty = torch.tensor(0.0, device=device)

        # Group by (n, k, m) to batch matrix operations
        unique_params_combos, inverse_indices = torch.unique(params, dim=0, return_inverse=True)

        for i in range(unique_params_combos.size(0)):
            n = int(unique_params_combos[i, 0].item())
            k = int(unique_params_combos[i, 1].item())
            m = int(unique_params_combos[i, 2].item())
            
            mask = inverse_indices == i
            
            group_P_padded = P_padded[mask]
            group_y_pred = y_pred[mask]
            group_batch_size = group_P_padded.size(0)

            # Safety checks
            if m + 1 > n or k <= 0 or n - k < 0:
                continue

            # Extract P (k x n-k) from padded (max_k x max_nk)
            # P_padded is (Batch, max_k, max_nk)
            # We want P_actual to be (Batch, k, n-k)
            # P_padded was padded: (0, max_nk - nk, 0, max_k - k) -> right and bottom padding
            # So valid data is top-left
            P_actual = group_P_padded[:, :k, :(n-k)]
            
            # Construct G = [I | P]
            I = torch.eye(k, device=device)
            # Expand I to batch: (Batch, k, k)
            I_batch = I.unsqueeze(0).expand(group_batch_size, -1, -1)
            
            # G_group: (Batch, k, n) where n = k + (n-k)
            G_group = torch.cat([I_batch, P_actual], dim=2)
            
            # Monte Carlo Sampling
            # Generate X: (Batch, num_samples, k)
            X_samples = torch.randn(group_batch_size, self.num_samples, k, device=device)
            
            # C = X * G -> (Batch, num_samples, n)
            # bmm requires (B, N, M) x (B, M, P) -> (B, N, P)
            # Here: (Batch*num_samples, 1, k) x (Batch*num_samples, k, n) ? No, that's too slow.
            # Use bmm with broadcasting? bmm doesn't broadcast.
            # Reshape X to (Batch * num_samples, 1, k) ? 
            # Let's keep it grouped.
            # G_group is (Batch, k, n). X_samples is (Batch, num_samples, k).
            # We want C: (Batch, num_samples, n).
            # C[b] = X[b] @ G[b]. 
            # X[b] is (num_samples, k). G[b] is (k, n).
            # Result is (num_samples, n). Correct.
            
            C_samples = torch.bmm(X_samples, G_group)

            magnitudes = torch.abs(C_samples) # (Batch, num_samples, n)
            
            if m + 1 > magnitudes.shape[2]:
                 continue
            
            # We need m-th height.
            # Sort magnitudes along dimension 2 (n) descending
            # We need flat view for topk? No, topk works on last dim by default.
            top_magnitudes, _ = torch.topk(magnitudes, k=m+1, dim=2, largest=True)
            
            c_max = top_magnitudes[:, :, 0]     # 1st largest
            c_m   = top_magnitudes[:, :, m]     # (m+1)-th largest (index m)
            
            # h(x) = c_max / c_m
            hm_samples = c_max / (c_m + self.eps) # (Batch, num_samples)
            
            # Max over samples -> approximate m-height
            max_hm_sample_group, _ = torch.max(hm_samples, dim=1) # (Batch,)
            
            # Penalty: if predicted < max_hm_sample_group, penalize
            # We want pred >= approximate_lower_bound
            # So penalty = relu(approx_lb - pred)
            group_penalty = F.relu(max_hm_sample_group - group_y_pred.squeeze())
            
            total_penalty += torch.sum(group_penalty)

        loss_violation = total_penalty / B
        total_loss = loss_logmse + self.lambda_violation * loss_violation

        return total_loss, loss_logmse.item(), loss_violation.item()


class ResNetTrainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, device,
                 scheduler=None, patience=10, model_name="Specialist"):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.patience = patience
        self.model_name = model_name
        self.best_val = float("inf")
        self.epochs_no_improve = 0
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []

    def _run_epoch(self, loader, training: bool):
        self.model.train(training)
        total_loss, n_samples = 0.0, 0
        context = torch.enable_grad() if training else torch.no_grad()
        
        is_violation_loss = isinstance(self.criterion, ViolationInformedLossAccelerated)

        with context:
            for params, targets, P in loader:
                params = params.to(self.device)
                targets = targets.to(self.device)
                P = P.to(self.device)

                if training:
                    self.optimizer.zero_grad()
                
                preds = self.model(P, params)
                
                if is_violation_loss:
                    # Calculate violation only during training
                    loss_obj, _, _ = self.criterion(preds, targets, P_padded=P, params=params, calculate_violation=training)
                    loss = loss_obj
                else:
                    loss = self.criterion(preds, targets)
                
                if training:
                    loss.backward()
                    self.optimizer.step()
                
                batch = params.size(0)
                total_loss += loss.item() * batch
                n_samples += batch
        return total_loss / max(n_samples, 1)

    def fit(self, epochs: int, checkpoint_path: str):
        print(f"--- Training {self.model_name} for {epochs} epoch(s) ---")
        for epoch in range(1, epochs + 1):
            start = time.time()
            train_loss = self._run_epoch(self.train_loader, training=True)
            val_loss = self._run_epoch(self.val_loader, training=False)
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            duration = time.time() - start
            print(f"Epoch {epoch:03d} | Train {train_loss:.4f} | Val {val_loss:.4f} | {duration:.1f}s")

            if val_loss < self.best_val:
                print(f"  Improvement detected ({self.best_val:.4f} -> {val_loss:.4f}). Saving checkpoint.")
                self.best_val = val_loss
                self.epochs_no_improve = 0
                torch.save(self.model.state_dict(), checkpoint_path)
            else:
                self.epochs_no_improve += 1
                if self.epochs_no_improve >= self.patience:
                    print(f"  Early stopping triggered after {self.epochs_no_improve} epochs with no improvement.")
                    break

            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

        print(f"Best validation loss for {self.model_name}: {self.best_val:.4f}")
        return self.best_val


# ==============================================================================
# === Helper Functions =========================================================
# ==============================================================================


def build_dataloaders(dataset: Dataset, batch_size: int, val_ratio: float, num_workers: int = 0,
                      pin_memory: bool = True):
    val_size = max(1, int(len(dataset) * val_ratio))
    train_size = len(dataset) - val_size
    if train_size <= 0:
        if len(dataset) > 0:
             # If dataset is very small, use it all for training and validation
            train_ds, val_ds = dataset, dataset
        else:
            raise ValueError("Train split is empty. Reduce val_ratio or provide more samples.")
    else:
        train_ds, val_ds = random_split(dataset, [train_size, val_size])
        
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin_memory)
    return train_loader, val_loader


def ensure_dir(path: str):
    dirpath = os.path.dirname(path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)


def plot_training_curves(train_losses: List[float], val_losses: List[float], title: str, output_path: str):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Log MSE Loss')
    plt.legend()
    plt.grid(True)
    ensure_dir(output_path)
    plt.savefig(output_path)
    plt.close()


def train_predominant_specialist(
    pair: Tuple[int, int], 
    ratio: float,
    base_dataset: PickleFolderDataset, 
    device,
    *,
    batch_size: int, 
    epochs: int, 
    patience: int, 
    lr: float, 
    weight_decay: float,
    val_ratio: float, 
    num_workers: int, 
    pin_memory: bool,
    model_kwargs: Dict, 
    checkpoint_dir: str,
    plots_dir: str,
    scheduler_patience: int = 5,
    lambda_violation: float = 0.0,
    num_samples: int = 10
):
    n_val, k_val = pair
    ckpt_name = f"predominant_specialist_n{n_val}_k{k_val}_ratio{ratio:.2f}.pth"
    checkpoint_path = os.path.join(checkpoint_dir, ckpt_name)
    
    # Check if checkpoint exists
    if os.path.exists(checkpoint_path):
        print(f"\n--- Checkpoint found for (n={n_val}, k={k_val}, ratio={ratio:.2f}). Skipping training. ---")
        print(f"  Loading from {checkpoint_path}...")
        
        # We still need the val_loader to evaluate the score for the summary plot
        subset = PredominantSpecialistSubset(base_dataset, n_val, k_val, ratio)
        _, val_loader = build_dataloaders(
            subset, batch_size=batch_size, val_ratio=val_ratio,
            num_workers=num_workers, pin_memory=pin_memory
        )
        
        model = ResNet2DWithParams(**model_kwargs).to(device)
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.eval()
        
        criterion = LogMSELoss()
        # Quick evaluation
        total_loss = 0.0
        n_samples = 0
        with torch.no_grad():
            for params, targets, P in val_loader:
                params = params.to(device)
                targets = targets.to(device)
                P = P.to(device)
                preds = model(P, params)
                loss = criterion(preds, targets)
                batch = params.size(0)
                total_loss += loss.item() * batch
                n_samples += batch
        
        val_loss = total_loss / max(n_samples, 1)
        print(f"  Loaded model validation loss: {val_loss:.4f}")
        return val_loss

    print(f"\n--- Preparing specialist for (n={n_val}, k={k_val}) with predominant ratio {ratio:.2f} ---")
    
    subset = PredominantSpecialistSubset(base_dataset, n_val, k_val, ratio)
    
    train_loader, val_loader = build_dataloaders(
        subset, batch_size=batch_size, val_ratio=val_ratio,
        num_workers=num_workers, pin_memory=pin_memory
    )
    
    model = ResNet2DWithParams(**model_kwargs).to(device)
    
    if lambda_violation > 0:
        criterion = ViolationInformedLossAccelerated(lambda_violation=lambda_violation, num_samples=num_samples)
    else:
        criterion = LogMSELoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=max(1, scheduler_patience), factor=0.3
    )
    
    ensure_dir(checkpoint_path)

    model_name = f"ResNet (n={n_val},k={k_val},r={ratio:.2f})"
    trainer = ResNetTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        patience=patience,
        model_name=model_name
    )
    
    best_val = trainer.fit(epochs=epochs, checkpoint_path=checkpoint_path)
    
    # Plot training curve
    plot_title = f"Training Curve: n={n_val}, k={k_val}, ratio={ratio:.2f}"
    plot_filename = f"curve_n{n_val}_k{k_val}_ratio{ratio:.2f}.png"
    plot_path = os.path.join(plots_dir, plot_filename)
    plot_training_curves(trainer.train_losses, trainer.val_losses, plot_title, plot_path)
    
    return best_val

def plot_ratio_results(results: Dict[Tuple[float, int, int], float], pairs: Sequence[Tuple[int, int]], ratios: Sequence[float], output_path: str):
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 8))

    for pair in pairs:
        n_val, k_val = pair
        losses = []
        for ratio in ratios:
            loss = results.get((ratio, n_val, k_val), None)
            losses.append(loss)
        
        ax.plot(ratios, losses, marker='o', linestyle='-', label=f'n={n_val}, k={k_val}')

    ax.set_xlabel("Predominant Pair Ratio")
    ax.set_ylabel("Best Validation Log2 MSE")
    ax.set_title("Specialist Performance vs. Predominant Data Ratio")
    ax.legend(title="(n,k) Pair")
    ax.invert_xaxis() # Ratios from high (more pure) to low (more mixed)
    fig.tight_layout()
    
    ensure_dir(output_path)
    fig.savefig(output_path)
    print(f"\nSaved results plot to {output_path}")
    plt.close(fig)

# ==============================================================================
# === Main =====================================================================
# ==============================================================================


if __name__ == "__main__":
    # --- Configuration --------------------------------------------------------
    TRAIN_DATA_FOLDERS = ["./split_data_train_20000_random"]
    VALIDATION_DATA_FOLDERS = ["./split_data_validation_20000_random"]
    
    # Pairs to train specialists for
    SPECIALIST_PAIRS = [(9, 4), (9, 5), (9, 6), (10, 4), (10, 5), (10, 6)]
    # Ratios to test for predominant training
    PREDOMINANT_RATIOS = [0.8, 0.6, 0.5, 0.4, 0.2]

    DATA_MAX_K = 6
    DATA_MAX_NK = 6
    NUM_WORKERS = 0
    PIN_MEMORY = True

    SPECIALIST_BATCH_SIZE = 256
    SPECIALIST_VAL_RATIO = 0.15
    SPECIALIST_EPOCHS = 25
    SPECIALIST_PATIENCE = 8
    SPECIALIST_LR = 3e-4
    SPECIALIST_WEIGHT_DECAY = 1e-4
    SPECIALIST_DROPOUT = 0.2
    
    SPECIALIST_LAMBDA_VIOLATION = 0.3
    SPECIALIST_NUM_SAMPLES = 15

    CHECKPOINT_DIR = "./predominant_specialist/checkpoints"
    PLOTS_DIR = "./predominant_specialist/plots"
    RESULTS_PLOT_PATH = "./predominant_specialist/ratio_performance.png"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Load pickle files once ----------------------------------------------
    file_list = []
    for folder in TRAIN_DATA_FOLDERS + VALIDATION_DATA_FOLDERS:
        folder_files = glob.glob(os.path.join(folder, "*.pkl"))
        if not folder_files:
            print(f"Warning: no pickle files found in {folder}")
        file_list.extend(folder_files)
    if not file_list:
        raise FileNotFoundError("No pickle files found. Check TRAIN_DATA_FOLDERS/VALIDATION_DATA_FOLDERS.")

    full_dataset = PickleFolderDataset(
        file_paths=file_list,
        max_k=DATA_MAX_K,
        max_nk=DATA_MAX_NK,
        p_normaliser="none",
    )

    resnet_kwargs = {
        "k_max": DATA_MAX_K,
        "nk_max": DATA_MAX_NK,
        "n_params": 3,
        "base_ch": 64,
        "num_blocks": 5,
        "dropout_p": SPECIALIST_DROPOUT,
    }

    # --- Stage 1: Train per-(n,k) specialists with different ratios ----------
    experiment_results: Dict[Tuple[float, int, int], float] = {}

    for ratio in PREDOMINANT_RATIOS:
        for pair in SPECIALIST_PAIRS:
            best_val_loss = train_predominant_specialist(
                pair=pair,
                ratio=ratio,
                base_dataset=full_dataset,
                device=device,
                batch_size=SPECIALIST_BATCH_SIZE,
                epochs=SPECIALIST_EPOCHS,
                patience=SPECIALIST_PATIENCE,
                lr=SPECIALIST_LR,
                weight_decay=SPECIALIST_WEIGHT_DECAY,
                val_ratio=SPECIALIST_VAL_RATIO,
                num_workers=NUM_WORKERS,
                pin_memory=PIN_MEMORY,
                model_kwargs=resnet_kwargs,
                checkpoint_dir=CHECKPOINT_DIR,
                plots_dir=PLOTS_DIR,
                scheduler_patience=max(1, SPECIALIST_PATIENCE // 2),
                lambda_violation=SPECIALIST_LAMBDA_VIOLATION,
                num_samples=SPECIALIST_NUM_SAMPLES,
            )
            experiment_results[(ratio, pair[0], pair[1])] = best_val_loss
            
    # --- Stage 2: Plot results for analysis ----------------------------------
    plot_ratio_results(experiment_results, SPECIALIST_PAIRS, PREDOMINANT_RATIOS, RESULTS_PLOT_PATH)

    # --- Save numerical results to CSV ---
    results_csv_path = os.path.join(os.path.dirname(RESULTS_PLOT_PATH), "experiment_results.csv")
    import csv
    with open(results_csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['ratio', 'n', 'k', 'val_loss'])
        for (ratio, n, k), loss in experiment_results.items():
            writer.writerow([ratio, n, k, loss])
    print(f"Saved numerical results to {results_csv_path}")

    print("\n--- Predominant Specialist Training Experiment Complete ---")
    # Find and print the best ratio overall
    if experiment_results:
        best_combo = min(experiment_results, key=experiment_results.get)
        best_loss = experiment_results[best_combo]
        print(f"Best overall performance: {best_loss:.4f} at ratio={best_combo[0]} for pair (n={best_combo[1]}, k={best_combo[2]})")
