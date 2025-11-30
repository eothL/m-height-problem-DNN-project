import os
import glob
import time
import pickle
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
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


class SpecialistSubset(Dataset):
    """
    Lightweight view over PickleFolderDataset restricted to a single (n,k) pair.
    """

    def __init__(self, base_dataset: PickleFolderDataset, n_value: int, k_value: int):
        super().__init__()
        self.base_dataset = base_dataset
        n_all = base_dataset.n_vals.cpu().numpy().astype(int)
        k_all = base_dataset.k_vals.cpu().numpy().astype(int)
        idx = np.where((n_all == n_value) & (k_all == k_value))[0].tolist()
        if not idx:
            raise ValueError(f"No samples found for n={n_value}, k={k_value}.")
        self.indices = idx
        self.n_value = n_value
        self.k_value = k_value
        print(f"  Specialist subset (n={n_value}, k={k_value}) -> {len(self.indices)} samples.")

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


class SparseTopKGating(nn.Module):
    """
    Small gating network that keeps only the top-k mixture weights.
    """

    def __init__(self, input_dim: int, hidden_dim: int, num_experts: int, top_k: int = 2):
        super().__init__()
        self.top_k = top_k
        self.num_experts = num_experts
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_experts),
        )

    def forward(self, inputs):
        logits = self.net(inputs)
        probs = torch.softmax(logits, dim=-1)
        if self.top_k >= self.num_experts:
            return probs
        top_vals, top_idx = torch.topk(probs, self.top_k, dim=-1)
        mask = torch.zeros_like(probs)
        mask.scatter_(1, top_idx, top_vals)
        return mask / mask.sum(dim=-1, keepdim=True).clamp_min(EPS)


class SpecialistMoEResNet(nn.Module):
    """
    Sparse MoE container whose experts correspond to fixed (n,k) pairs. The order
    of `specialist_pairs` defines which checkpoint loads into which expert.
    """

    def __init__(self, specialist_pairs: Sequence[Tuple[int, int]], gate_hidden_dim=32,
                 top_k=2, expert_kwargs=None):
        super().__init__()
        if expert_kwargs is None:
            expert_kwargs = {}
        self.specialist_pairs = list(specialist_pairs)
        self.experts = nn.ModuleList([ResNet2DWithParams(**expert_kwargs) for _ in self.specialist_pairs])
        self.gate = SparseTopKGating(
            input_dim=3,  # (n,k,m)
            hidden_dim=gate_hidden_dim,
            num_experts=len(self.specialist_pairs),
            top_k=top_k,
        )

    def forward(self, P, params):
        gate_inputs = params[:, :3]
        weights = self.gate(gate_inputs).unsqueeze(1)  # (B,1,E)
        expert_outputs = torch.cat([expert(P, params).unsqueeze(-1) for expert in self.experts], dim=-1)
        return torch.sum(expert_outputs * weights, dim=-1)


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
        with context:
            for params, targets, P in loader:
                params = params.to(self.device)
                targets = targets.to(self.device)
                P = P.to(self.device)

                if training:
                    self.optimizer.zero_grad()
                preds = self.model(P, params)
                loss = self.criterion(preds, targets)
                if training:
                    loss.backward()
                    self.optimizer.step()
                batch = params.size(0)
                total_loss += loss.item() * batch
                n_samples += batch
        return total_loss / max(n_samples, 1)

    def fit(self, epochs: int, checkpoint_path: str):
        print(f"\n--- Training {self.model_name} for {epochs} epoch(s) ---")
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
                print(f"  No improvement for {self.epochs_no_improve} epoch(s).")
                if self.epochs_no_improve >= self.patience:
                    print("  Early stopping triggered.")
                    break

            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

        print(f"Best validation loss for {self.model_name}: {self.best_val:.4f}")
        return self.best_val


class MoETrainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, device,
                 scheduler=None, patience=10, model_name="MoE"):
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
        self.history = {"train": [], "val": []}

    def _run_epoch(self, loader, training: bool):
        self.model.train(training)
        total_loss, n_samples = 0.0, 0
        context = torch.enable_grad() if training else torch.no_grad()
        with context:
            for params, targets, P in loader:
                params = params.to(self.device)
                targets = targets.to(self.device)
                P = P.to(self.device)
                if training:
                    self.optimizer.zero_grad()
                outputs = self.model(P, params)
                loss = self.criterion(outputs, targets)
                if training:
                    loss.backward()
                    self.optimizer.step()
                batch = params.size(0)
                total_loss += loss.item() * batch
                n_samples += batch
        return total_loss / max(n_samples, 1)

    def fit(self, epochs: int, checkpoint_path: str):
        print(f"\n--- Fine-tuning {self.model_name} for {epochs} epoch(s) ---")
        for epoch in range(1, epochs + 1):
            start = time.time()
            train_loss = self._run_epoch(self.train_loader, training=True)
            val_loss = self._run_epoch(self.val_loader, training=False)
            self.history["train"].append(train_loss)
            self.history["val"].append(val_loss)
            duration = time.time() - start
            print(f"Epoch {epoch:03d} | Train {train_loss:.4f} | Val {val_loss:.4f} | {duration:.1f}s")

            if val_loss < self.best_val:
                print(f"  Validation improved ({self.best_val:.4f} -> {val_loss:.4f}). Saving checkpoint.")
                self.best_val = val_loss
                self.epochs_no_improve = 0
                torch.save(self.model.state_dict(), checkpoint_path)
            else:
                self.epochs_no_improve += 1
                print(f"  No improvement for {self.epochs_no_improve} epoch(s).")
                if self.epochs_no_improve >= self.patience:
                    print("  Early stopping triggered.")
                    break

            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

        print(f"Best validation loss for {self.model_name}: {self.best_val:.4f}")
        return self.history


def plot_losses(loss_dict: Dict[str, Tuple[List[float], List[float]]], title: str = "Training Curves"):
    if not loss_dict:
        print("No loss data to plot.")
        return
    cols = 2
    rows = (len(loss_dict) + cols - 1) // cols
    plt.figure(figsize=(6 * cols, 4.5 * rows))
    for idx, (label, losses) in enumerate(loss_dict.items(), start=1):
        train_losses, val_losses = losses
        if not train_losses or not val_losses:
            print(f"Skipping plot for {label}: empty loss history.")
            continue
        epochs = range(1, len(train_losses) + 1)
        plt.subplot(rows, cols, idx)
        plt.plot(epochs, train_losses, label=f"{label} Train")
        plt.plot(epochs, val_losses, label=f"{label} Val")
        plt.xlabel("Epoch")
        plt.ylabel("Log2 MSE")
        plt.title(label)
        plt.grid(True)
        plt.legend()
    plt.suptitle(title)
    plt.tight_layout()
    filename = title.replace(" ", "_").lower() + ".png"
    plt.savefig(filename)
    print(f"Saved plot to {filename}")
    plt.close()


# ==============================================================================
# === Helper Functions =========================================================
# ==============================================================================


def build_dataloaders(dataset: Dataset, batch_size: int, val_ratio: float, num_workers: int = 0,
                      pin_memory: bool = True):
    val_size = max(1, int(len(dataset) * val_ratio))
    train_size = len(dataset) - val_size
    if train_size <= 0:
        raise ValueError("Train split is empty. Reduce val_ratio or provide more samples.")
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin_memory)
    return train_loader, val_loader


def ensure_dir(path: str):
    dirpath = os.path.dirname(path) or "."
    os.makedirs(dirpath, exist_ok=True)


def train_specialist(pair: Tuple[int, int], base_dataset: PickleFolderDataset, device, *,
                     batch_size: int, epochs: int, patience: int, lr: float, weight_decay: float,
                     val_ratio: float, num_workers: int, pin_memory: bool,
                     model_kwargs: Dict, checkpoint_dir: str, scheduler_patience: int = 5):
    n_val, k_val = pair
    subset = SpecialistSubset(base_dataset, n_val, k_val)
    train_loader, val_loader = build_dataloaders(
        subset, batch_size=batch_size, val_ratio=val_ratio,
        num_workers=num_workers, pin_memory=pin_memory
    )
    model = ResNet2DWithParams(**model_kwargs).to(device)
    criterion = LogMSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=max(1, scheduler_patience), factor=0.3
    )
    ckpt_name = f"specialist_resnet_n{n_val}_k{k_val}.pth"
    checkpoint_path = os.path.join(checkpoint_dir, ckpt_name)
    ensure_dir(checkpoint_path)
    trainer = ResNetTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        patience=patience,
        model_name=f"ResNet (n={n_val}, k={k_val})"
    )
    best_val = trainer.fit(epochs=epochs, checkpoint_path=checkpoint_path)
    return checkpoint_path, best_val, trainer.train_losses, trainer.val_losses


def load_specialists_into_moe(moe_model: SpecialistMoEResNet, checkpoint_paths: Sequence[str], device):
    if len(checkpoint_paths) != len(moe_model.experts):
        raise ValueError("Number of checkpoints does not match number of experts.")
    for expert, ckpt_path in zip(moe_model.experts, checkpoint_paths):
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        state_dict = torch.load(ckpt_path, map_location=device)
        missing, unexpected = expert.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            print(f"  Warning when loading {ckpt_path}: missing={missing}, unexpected={unexpected}")
    print(f"Loaded {len(checkpoint_paths)} specialist checkpoint(s) into the MoE experts.")


# ==============================================================================
# === Main =====================================================================
# ==============================================================================


if __name__ == "__main__":
    # --- Configuration --------------------------------------------------------
    TRAIN_DATA_FOLDERS = ["./split_data_train_20000_random"]
    VALIDATION_DATA_FOLDERS = ["./split_data_validation_20000_random"]
    SPECIALIST_PAIRS = [(9, 4), (9, 5), (9, 6), (10, 4), (10, 5), (10, 6)]

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

    MOE_BATCH_SIZE = 512
    MOE_VAL_RATIO = 0.2
    MOE_EPOCHS = 40
    MOE_PATIENCE = 10
    MOE_LR = 1e-4
    MOE_WEIGHT_DECAY = 1e-4
    MOE_TOP_K = 2
    MOE_GATE_HIDDEN = 48

    CHECKPOINT_DIR = "./specialist_checkpoints"
    MOE_CHECKPOINT = "./best_specialist_sparse_moe.pth"
    FORCE_RETRAIN_SPECIALISTS = True

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

    # --- Stage 1: Train per-(n,k) specialists --------------------------------
    specialist_paths: Dict[Tuple[int, int], str] = {}
    specialist_curves: Dict[str, Tuple[List[float], List[float]]] = {}
    for pair in SPECIALIST_PAIRS:
        n_val, k_val = pair
        ckpt_name = f"specialist_resnet_n{n_val}_k{k_val}.pth"
        checkpoint_path = os.path.join(CHECKPOINT_DIR, ckpt_name)
        if os.path.exists(checkpoint_path) and not FORCE_RETRAIN_SPECIALISTS:
            print(f"Skipping training for (n={n_val}, k={k_val}); checkpoint exists at {checkpoint_path}")
            specialist_paths[pair] = checkpoint_path
            specialist_curves[f"ResNet n{n_val} k{k_val}"] = ([], [])
            continue

        path, best_val, train_losses, val_losses = train_specialist(
            pair=pair,
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
            scheduler_patience=max(1, SPECIALIST_PATIENCE // 2),
        )
        specialist_paths[pair] = path
        specialist_curves[f"ResNet n{n_val} k{k_val}"] = (train_losses, val_losses)
        print(f"Finished specialist (n={n_val}, k={k_val}) with best val {best_val:.4f}")

    if specialist_curves:
        plot_losses(specialist_curves, title="Specialist Training Curves")

    ordered_paths = [specialist_paths[pair] for pair in SPECIALIST_PAIRS]

    # --- Stage 2: Build dataloaders for MoE fine-tuning ----------------------
    moe_train_loader, moe_val_loader = build_dataloaders(
        full_dataset,
        batch_size=MOE_BATCH_SIZE,
        val_ratio=MOE_VAL_RATIO,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    moe_model = SpecialistMoEResNet(
        specialist_pairs=SPECIALIST_PAIRS,
        gate_hidden_dim=MOE_GATE_HIDDEN,
        top_k=MOE_TOP_K,
        expert_kwargs=resnet_kwargs,
    ).to(device)
    load_specialists_into_moe(moe_model, ordered_paths, device=device)

    moe_criterion = LogMSELoss()
    moe_optimizer = torch.optim.AdamW(moe_model.parameters(), lr=MOE_LR, weight_decay=MOE_WEIGHT_DECAY)
    moe_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        moe_optimizer, mode="min", patience=max(1, MOE_PATIENCE // 2), factor=0.3
    )

    moe_trainer = MoETrainer(
        model=moe_model,
        train_loader=moe_train_loader,
        val_loader=moe_val_loader,
        criterion=moe_criterion,
        optimizer=moe_optimizer,
        scheduler=moe_scheduler,
        device=device,
        patience=MOE_PATIENCE,
        model_name="SparseMoE_SpecialistResNet",
    )
    moe_history = moe_trainer.fit(epochs=MOE_EPOCHS, checkpoint_path=MOE_CHECKPOINT)
    plot_losses({"MoE": (moe_history["train"], moe_history["val"])}, title="MoE Fine-tuning Curve")

    print(f"\nPipeline complete. Best MoE validation loss: {moe_trainer.best_val:.4f}")
