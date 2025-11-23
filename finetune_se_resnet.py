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
import optuna

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

# --- Optuna Objective Function ---
def objective(trial: optuna.Trial, train_loader, val_loader, device):
    # --- Hyperparameter Search Space ---
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    base_ch = trial.suggest_categorical("base_ch", [32, 64])
    num_blocks = trial.suggest_int("num_blocks", 3, 6)
    reduction = trial.suggest_categorical("reduction", [8, 16, 32])
    
    # --- Setup for this trial ---
    model = ResNet2DWithParamsSE(
        base_ch=base_ch,
        num_blocks=num_blocks,
        reduction=reduction
    ).to(device)
    
    criterion = LogMSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # --- Training Loop for Trial ---
    best_val_loss = float('inf')
    for epoch in range(15): # Train for a fixed number of epochs for each trial
        model.train()
        for params, targets, p_matrices in train_loader:
            params, targets, p_matrices = params.to(device), targets.to(device), p_matrices.to(device)
            optimizer.zero_grad()
            outputs = model(p_matrices, params)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        # --- Validation ---
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for params, targets, p_matrices in val_loader:
                params, targets, p_matrices = params.to(device), targets.to(device), p_matrices.to(device)
                outputs = model(p_matrices, params)
                val_loss += criterion(outputs, targets).item() * params.size(0)
        
        val_loss /= len(val_loader.dataset)
        trial.report(val_loss, epoch)

        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
            
        best_val_loss = min(best_val_loss, val_loss)

    return best_val_loss

# --- Main Execution Block ---
if __name__ == '__main__':
    # --- Configuration ---
    TRAIN_DATA_FOLDERS = ['./split_data_train_20000_random']
    VALIDATION_DATA_FOLDERS = ['./split_data_validation_20000_random']
    max_k = 6
    max_nk = 6
    batch_size = 512
    val_split_ratio = 0.2
    NUM_WORKERS = 0
    PIN_MEMORY = True
    N_TRIALS = 25 # Number of Optuna trials to run

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Load and Split Data ---
    try:
        print("--- Preparing Data for Hyperparameter Tuning ---")
        all_files = []
        for folder in TRAIN_DATA_FOLDERS + VALIDATION_DATA_FOLDERS:
             files = glob.glob(os.path.join(folder, '*.pkl'))
             all_files.extend(files)
        
        if not all_files: raise FileNotFoundError("No data files found.")

        combined_dataset = PickleFolderDataset(file_paths=all_files, max_k=max_k, max_nk=max_nk)
        total_samples = len(combined_dataset)
        val_size = int(total_samples * val_split_ratio)
        train_size = total_samples - val_size
        train_dataset, val_dataset = random_split(combined_dataset, [train_size, val_size])

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
        print("DataLoaders created.")

    except Exception as e:
        print(f"\nError during data loading: {e}")
        print(traceback.format_exc())
        exit()

    # --- Run Optuna Study ---
    study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner())
    study.optimize(lambda trial: objective(trial, train_loader, val_loader, device), n_trials=N_TRIALS)

    # --- Print Results ---
    print("\n--- Hyperparameter Tuning Complete ---")
    print(f"Number of finished trials: {len(study.trials)}")
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value (Best Val Loss): {trial.value:.4f}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
