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

# --- Squeeze-and-Excitation Block ---
class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

# --- Modified ConvResBlock with SE (accepts reduction ratio) ---
class ConvResBlockSE(nn.Module):
    def __init__(self, ch, reduction=16):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(ch)
        self.bn2 = nn.BatchNorm2d(ch)
        self.se = SEBlock(ch, reduction)

    def forward(self, x):
        h = F.relu(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        h = self.se(h)
        return F.relu(x + h)

# --- ResNet with SE blocks (accepts reduction ratio) ---
class ResNet2DWithParamsSE(nn.Module):
    def __init__(self,
                 k_max=6, nk_max=6,
                 n_params=3,
                 base_ch=32,
                 num_blocks=3,
                 reduction=16):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, base_ch, 3, padding=1),
            nn.BatchNorm2d(base_ch),
            nn.ReLU(inplace=True)
        )
        self.blocks = nn.Sequential(
            *[ConvResBlockSE(base_ch, reduction) for _ in range(num_blocks)]
        )
        flat_dim = base_ch * k_max * nk_max
        self.param_proj = nn.Linear(n_params, 64)
        self.head = nn.Sequential(
            nn.Linear(flat_dim + 64, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1)
        )

    def forward(self, P, params):
        x = P.unsqueeze(1)
        x = self.blocks(self.stem(x))
        x = x.flatten(1)
        p = F.relu(self.param_proj(params.float()))
        x = torch.cat([x, p], dim=1)
        return self.head(x)

# --- Helper Functions ---
class LogMSELoss(nn.Module):
    def __init__(self, eps=1e-9):
        super().__init__()
        self.eps = eps
    def forward(self, y_pred, y_true):
        y_pred_safe = torch.clamp(y_pred, min=self.eps)
        y_true_safe = torch.clamp(y_true, min=self.eps)
        log2_pred = torch.log2(y_pred_safe)
        log2_true = torch.log2(y_true_safe)
        return torch.mean((log2_true - log2_pred) ** 2)

EPS = 1e-9
def norm_none(t): return t
NORMALISERS = {"none": norm_none}

class PickleFolderDataset(Dataset):
    def __init__(self, file_paths: list, max_k :int = 6, max_nk:int = 6, p_normaliser:str = "none", **kwargs):
        super().__init__()
        if p_normaliser not in NORMALISERS:
            raise ValueError(f"Invalid p_normaliser: {p_normaliser}")
        self._normalise = NORMALISERS[p_normaliser]
        self.max_k, self.max_nk = max_k, max_nk
        if not file_paths: raise ValueError("file_paths list is empty.")
        self.P_raw, self.h_vals, self.n_vals, self.k_vals, self.m_vals = [], [], [], [], []
        print(f"Loading data from {len(file_paths)} files...")
        for fp in file_paths:
            try:
                with open(fp, "rb") as f: df = pickle.load(f)
                if not all(col in df.columns for col in ['n', 'k', 'm', 'result', 'P']): continue
                self.P_raw.extend([np.array(p) for p in df['P']])
                self.h_vals.extend(df['result'].astype(float).tolist())
                self.n_vals.extend(df['n'].astype(int).tolist())
                self.k_vals.extend(df['k'].astype(int).tolist())
                self.m_vals.extend(df['m'].astype(int).tolist())
            except Exception as e: print(f"Error loading {fp}: {e}")
        if not self.h_vals: raise ValueError("No valid data loaded.")
        print(f"Finished loading. Total samples: {len(self.h_vals)}")
        self.h_vals = torch.tensor(self.h_vals, dtype=torch.float32)
        self.n_vals = torch.tensor(self.n_vals, dtype=torch.float32)
        self.k_vals = torch.tensor(self.k_vals, dtype=torch.float32)
        self.m_vals = torch.tensor(self.m_vals, dtype=torch.float32)
    def __len__(self): return len(self.h_vals)
    def _pad(self, p2d: torch.Tensor):
        k_act, nk_act = p2d.shape
        pad = (0, self.max_nk - nk_act, 0, self.max_k - k_act)
        return F.pad(p2d, pad, value=0.)
    def __getitem__(self, idx):
        p_np, n, k = self.P_raw[idx], int(self.n_vals[idx]), int(self.k_vals[idx])
        target = (k, n - k)
        if p_np.ndim == 1: p_np = p_np.reshape(*target)
        p_t = self._pad(torch.tensor(p_np, dtype=torch.float32))
        p_t = self._normalise(p_t)
        params = torch.tensor([self.n_vals[idx], self.k_vals[idx], self.m_vals[idx]], dtype=torch.float32)
        h = self.h_vals[idx].unsqueeze(0)
        return params, h, p_t

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
        for params, targets, p_matrices in self.train_loader:
            params, targets, p_matrices = params.to(self.device), targets.to(self.device), p_matrices.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(p_matrices, params)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item() * params.size(0)
        return running_loss / len(self.train_loader.dataset)
    def _validate_epoch(self):
        self.model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for params, targets, p_matrices in self.val_loader:
                params, targets, p_matrices = params.to(self.device), targets.to(self.device), p_matrices.to(self.device)
                outputs = self.model(p_matrices, params)
                loss = self.criterion(outputs, targets)
                running_loss += loss.item() * params.size(0)
        return running_loss / len(self.val_loader.dataset)
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
            print(f"Epoch {epoch}/{epochs} ({self.model_name}): Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Duration: {duration:.2f}s")
            if val_loss < self.best_val_loss:
                print(f"  Val loss improved ({self.best_val_loss:.4f} -> {val_loss:.4f}). Saving checkpoint.")
                self.best_val_loss = val_loss
                self.epochs_no_improve = 0
                torch.save(self.model.state_dict(), f"best_{self.model_name.lower().replace(' ', '_')}.pth")
            else:
                self.epochs_no_improve += 1
                print(f"  No improvement for {self.epochs_no_improve} epoch(s).")
                if self.epochs_no_improve >= self.patience:
                    print(f"Early stopping triggered after {self.patience} epochs.")
                    break
            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau): self.scheduler.step(val_loss)
                else: self.scheduler.step()
        total_time = time.time() - start_time
        print(f"\n{self.model_name} Training Finished. Total time: {total_time:.2f}s. Best Val Loss: {self.best_val_loss:.4f}")
        return self.train_losses, self.val_losses

def plot_losses(results_dict):
    if not results_dict: print("No results to plot."); return
    print("\nPlotting losses...")
    num_models = len(results_dict)
    cols = min(2, num_models); rows = (num_models + cols - 1) // cols
    plt.figure(figsize=(7 * cols, 5 * rows))
    for idx, (model_name, (train_losses, val_losses)) in enumerate(results_dict.items()):
        if not train_losses or not val_losses: continue
        ax = plt.subplot(rows, cols, idx + 1)
        epochs = range(1, len(train_losses) + 1)
        best_loss = min(val_losses); best_epoch = val_losses.index(best_loss) + 1
        ax.plot(epochs, train_losses, label=f"{model_name} Train Loss")
        ax.plot(epochs, val_losses, label=f"{model_name} Val Loss (Best: {best_loss:.4f} @ Ep {best_epoch})")
        ax.scatter([best_epoch], [best_loss], color='red', s=50, zorder=5)
        ax.set_title(f"{model_name} Loss"); ax.set_xlabel("Epoch"); ax.set_ylabel("Log2 MSE Loss")
        ax.legend(); ax.grid(True)
    plt.tight_layout(); plt.show()

if __name__ == '__main__':
    # --- Configuration ---
    TRAIN_DATA_FOLDERS = ['./split_data_train_20000_random']
    VALIDATION_DATA_FOLDERS = ['./split_data_validation_20000_random']
    max_k, max_nk, batch_size, val_split_ratio = 6, 6, 512, 0.2
    NUM_WORKERS, PIN_MEMORY, EPOCHS, PATIENCE = 0, True, 100, 10
    
    # --- Best Hyperparameters from Optuna ---
    BEST_PARAMS = {
        "lr": 0.006306265160645964,
        "weight_decay": 5.787391032694206e-05,
        "base_ch": 32,
        "num_blocks": 4,
        "reduction": 16
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Load Data ---
    try:
        print("--- Preparing Data for Final Training ---")
        all_files = [f for folder in TRAIN_DATA_FOLDERS + VALIDATION_DATA_FOLDERS for f in glob.glob(os.path.join(folder, '*.pkl'))]
        if not all_files: raise FileNotFoundError("No data files found.")
        
        combined_dataset = PickleFolderDataset(file_paths=all_files, max_k=max_k, max_nk=max_nk)
        train_size = int(len(combined_dataset) * (1 - val_split_ratio))
        val_size = len(combined_dataset) - train_size
        train_dataset, val_dataset = random_split(combined_dataset, [train_size, val_size])

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
        print("DataLoaders created.")
    except Exception as e: 
        print(f"\nError during data loading: {e}"); traceback.print_exc(); exit()

    # --- Instantiate Model with Best Params ---
    print("\n--- Initializing SE-ResNet with Best Hyperparameters ---")
    final_model = ResNet2DWithParamsSE(
        k_max=max_k,
        nk_max=max_nk,
        n_params=3,
        base_ch=BEST_PARAMS["base_ch"],
        num_blocks=BEST_PARAMS["num_blocks"],
        reduction=BEST_PARAMS["reduction"]
    ).to(device)

    # --- Loss, Optimizer, Scheduler ---
    criterion = LogMSELoss()
    optimizer = optim.AdamW(final_model.parameters(), lr=BEST_PARAMS["lr"], weight_decay=BEST_PARAMS["weight_decay"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=PATIENCE//2, factor=0.2)

    # --- Instantiate and Run Trainer ---
    model_name = "Final_SE_ResNet"
    trainer = ResNetTrainer(
        model=final_model,
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

    # --- Plot Results ---
    results = {model_name: (train_losses, val_losses)}
    plot_losses(results)

    if val_losses:
        print(f"\nFinal training of {model_name} complete. Best validation loss: {min(val_losses):.4f}")
    else:
        print(f"\nTraining of {model_name} did not produce validation results.")
