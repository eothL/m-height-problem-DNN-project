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

import numpy as np
import matplotlib.pyplot as plt


# ==============================================================================
# === Dataset Utilities ========================================================
# ==============================================================================

EPS = 1e-9


def norm_none(t):
    return t


NORMALISERS = {
    "none": norm_none,
}


class PickleFolderDataset(Dataset):
    """
    Loads samples from a list of pickle files. Each file stores a DataFrame with
    columns ['n', 'k', 'm', 'result', 'P']. Matrices may be flattened; we reshape
    to (k, n-k), pad to (max_k, max_nk), and return (params, target, padded_P).
    """

    def __init__(self, file_paths, max_k=6, max_nk=6, p_normaliser="none"):
        super().__init__()
        if not file_paths:
            raise ValueError("file_paths list cannot be empty.")
        if p_normaliser not in NORMALISERS:
            raise ValueError(f"Unknown p_normaliser '{p_normaliser}'.")

        self.max_k = max_k
        self.max_nk = max_nk
        self._normalise = NORMALISERS[p_normaliser]

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
                print(f"  ! Skipping {path}: missing columns {required_cols - set(df.columns)}")
                continue

            self.P_raw.extend(df["P"].tolist())
            self.h_vals.extend(df["result"].astype(float).tolist())
            self.n_vals.extend(df["n"].astype(int).tolist())
            self.k_vals.extend(df["k"].astype(int).tolist())
            self.m_vals.extend(df["m"].astype(int).tolist())

        if not self.h_vals:
            raise ValueError("No valid rows were loaded from the pickle files.")

        print(f"Finished loading. Total samples: {len(self.h_vals)}")
        self.h_vals = torch.tensor(self.h_vals, dtype=torch.float32)
        self.n_vals = torch.tensor(self.n_vals, dtype=torch.float32)
        self.k_vals = torch.tensor(self.k_vals, dtype=torch.float32)
        self.m_vals = torch.tensor(self.m_vals, dtype=torch.float32)

    def __len__(self):
        return len(self.h_vals)

    def _pad(self, matrix: torch.Tensor):
        k_act, nk_act = matrix.shape
        pad_h = self.max_k - k_act
        pad_w = self.max_nk - nk_act
        pad = (0, pad_w, 0, pad_h)
        return F.pad(matrix, pad, value=0.0)

    def __getitem__(self, idx):
        n = int(self.n_vals[idx].item())
        k = int(self.k_vals[idx].item())
        m = int(self.m_vals[idx].item())

        raw = np.array(self.P_raw[idx])
        if raw.ndim == 1:
            raw = raw.reshape(k, n - k)
        P = torch.tensor(raw, dtype=torch.float32)
        P = self._normalise(self._pad(P))

        params = torch.tensor([n, k, m], dtype=torch.float32)
        h = self.h_vals[idx].unsqueeze(0)
        return params, h, P


# ==============================================================================
# === Model Definitions ========================================================
# ==============================================================================


class ConvResBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(ch)
        self.bn2 = nn.BatchNorm2d(ch)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(residual + x)


class ResNet2DWithParams(nn.Module):
    """
    Standard ReLU ResNet used throughout the project. Accepts padded P matrices
    and a 3D parameter vector (n, k, m). Optional lower-bound enforcement.
    """

    def __init__(self, k_max=6, nk_max=6, n_params=3, base_ch=64, num_blocks=5, enforce_lower_bound=True):
        super().__init__()
        self.enforce_lower_bound = enforce_lower_bound
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
            nn.Linear(256, 1),
        )

    def forward(self, P, params):
        x = P.unsqueeze(1)
        x = self.blocks(self.stem(x))
        x = x.flatten(1)
        param_embed = F.relu(self.param_proj(params.float()))
        x = torch.cat([x, param_embed], dim=1)
        z = self.head(x)
        if self.enforce_lower_bound:
            return 1.0 + F.softplus(z)
        return z


class SparseTopKGating(nn.Module):
    """
    Produces sparse mixture weights by selecting the top-k logits from a gating
    network and renormalising them.
    """

    def __init__(self, input_dim, hidden_dim, num_experts, top_k=2):
        super().__init__()
        self.top_k = top_k
        self.num_experts = num_experts
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_experts),
        )

    def forward(self, gate_inputs):
        logits = self.net(gate_inputs)
        weights = torch.softmax(logits, dim=-1)
        if self.top_k >= self.num_experts:
            return weights
        top_values, top_idx = torch.topk(weights, self.top_k, dim=-1)
        mask = torch.zeros_like(weights)
        mask.scatter_(1, top_idx, top_values)
        mask_sum = mask.sum(dim=-1, keepdim=True).clamp_min(EPS)
        return mask / mask_sum


class SparseMoEResNet(nn.Module):
    """
    Mixture-of-experts container with sparse soft gating. Each expert is the
    ReLU ResNet with lower-bound enforcement. Forward pass aggregates expert
    outputs using the sparse gate weights.
    """

    def __init__(self, num_experts=6, top_k=2, gate_hidden_dim=16, gate_input_dim=3,
                 expert_kwargs=None):
        super().__init__()
        if expert_kwargs is None:
            expert_kwargs = {}
        self.num_experts = num_experts
        self.experts = nn.ModuleList([
            ResNet2DWithParams(**expert_kwargs) for _ in range(num_experts)
        ])
        self.gate = SparseTopKGating(
            input_dim=gate_input_dim,
            hidden_dim=gate_hidden_dim,
            num_experts=num_experts,
            top_k=top_k,
        )

    def forward(self, P, params):
        expert_outputs = []
        for expert in self.experts:
            expert_outputs.append(expert(P, params).unsqueeze(-1))
        expert_outputs = torch.cat(expert_outputs, dim=-1)  # (B, 1, num_experts)
        gate_inputs = params[:, :3]  # n, k, m
        gate_weights = self.gate(gate_inputs)  # (B, num_experts)
        gate_weights = gate_weights.unsqueeze(1)
        mixture = torch.sum(expert_outputs * gate_weights, dim=-1)
        return mixture


# ==============================================================================
# === Loss Function ============================================================
# ==============================================================================


class LogMSELoss(nn.Module):
    def __init__(self, eps=1e-9):
        super().__init__()
        self.eps = eps

    def forward(self, y_pred, y_true):
        y_pred = torch.clamp(y_pred, min=self.eps)
        y_true = torch.clamp(y_true, min=self.eps)
        return torch.mean((torch.log2(y_true) - torch.log2(y_pred)) ** 2)


class ViolationInformedLossAccelerated(nn.Module):
    """
    Same accelerated violation-aware loss from the final ResNet training script.
    Penalises predictions that fall below sampled harmonic mean ratios derived
    from random codewords of each generator matrix.
    """

    def __init__(self, lambda_violation=0.5, num_samples=20, eps=1e-9):
        super().__init__()
        self.lambda_violation = lambda_violation
        self.num_samples = num_samples
        self.eps = eps
        self.log_mse = LogMSELoss(eps=eps)

    def forward(self, y_pred, y_true, P_padded=None, params=None, calculate_violation=True):
        logmse = self.log_mse(y_pred, y_true)
        if (not calculate_violation) or P_padded is None or params is None or self.lambda_violation == 0:
            return logmse, logmse.item(), 0.0

        device = y_pred.device
        total_penalty = torch.tensor(0.0, device=device)
        B = P_padded.shape[0]

        unique_combos, inverse_indices = torch.unique(params, dim=0, return_inverse=True)
        for idx in range(unique_combos.size(0)):
            n, k, m = unique_combos[idx].tolist()
            n = int(n)
            k = int(k)
            m = int(m)
            mask = inverse_indices == idx
            group_P = P_padded[mask]
            group_pred = y_pred[mask]
            group_bs = group_P.size(0)
            if group_bs == 0 or m + 1 > n or k <= 0 or n - k < 0:
                continue

            P_actual = group_P[:, :k, :(n - k)]
            I = torch.eye(k, device=device).unsqueeze(0).expand(group_bs, -1, -1)
            G_group = torch.cat([I, P_actual], dim=2)
            samples = torch.randn(group_bs, self.num_samples, k, device=device)
            codewords = torch.bmm(samples, G_group)
            magnitudes = torch.abs(codewords)
            magnitudes = magnitudes.view(-1, n)
            if m + 1 > magnitudes.shape[1]:
                continue
            top_vals, _ = torch.topk(magnitudes, k=m + 1, dim=1, largest=True)
            c_max = top_vals[:, 0]
            c_m = top_vals[:, m]
            hm_samples = (c_max / (c_m + self.eps)).view(group_bs, self.num_samples)
            max_hm, _ = torch.max(hm_samples, dim=1)
            penalty = F.relu(max_hm - group_pred.squeeze())
            total_penalty += penalty.sum()

        violation = total_penalty / max(B, 1)
        total_loss = logmse + self.lambda_violation * violation
        return total_loss, logmse.item(), violation.item()


# ==============================================================================
# === Training Helper ==========================================================
# ==============================================================================


class MoETrainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer,
                 device, patience=10, scheduler=None, model_name="MoE_ResNet_SparseGate"):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler
        self.model_name = model_name
        self.patience = patience

        self.train_losses = []
        self.val_losses = []
        self.train_logmse = []
        self.train_violation = []
        self.best_val_loss = float("inf")
        self.epochs_no_improve = 0

    def _run_epoch(self, loader, training=True):
        if training:
            self.model.train()
        else:
            self.model.eval()

        total_loss = 0.0
        total_logmse = 0.0
        total_violation = 0.0
        total_samples = 0

        context = torch.enable_grad() if training else torch.no_grad()
        with context:
            for params, targets, P in loader:
                params = params.to(self.device)
                targets = targets.to(self.device)
                P = P.to(self.device)

                if training:
                    self.optimizer.zero_grad()

                outputs = self.model(P, params)
                loss, logmse, violation = self.criterion(
                    outputs, targets, P_padded=P, params=params, calculate_violation=training
                )

                if training:
                    loss.backward()
                    self.optimizer.step()

                batch_size = params.size(0)
                total_loss += loss.item() * batch_size
                total_logmse += logmse * batch_size
                total_violation += violation * batch_size
                total_samples += batch_size

        denom = max(total_samples, 1)
        return (
            total_loss / denom,
            total_logmse / denom,
            total_violation / denom,
        )

    def fit(self, epochs):
        print(f"\n--- Training {self.model_name} for {epochs} epochs ---")
        start = time.time()
        for epoch in range(1, epochs + 1):
            epoch_start = time.time()
            train_loss, train_logmse, train_violation = self._run_epoch(self.train_loader, training=True)
            val_loss, _, _ = self._run_epoch(self.val_loader, training=False)

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_logmse.append(train_logmse)
            self.train_violation.append(train_violation)

            duration = time.time() - epoch_start
            print(f"Epoch {epoch:03d}: Train {train_loss:.4f} | Val {val_loss:.4f} | "
                  f"LogMSE {train_logmse:.4f} | Viol {train_violation:.4f} | {duration:.1f}s")

            if val_loss < self.best_val_loss:
                print(f"  Validation improved {self.best_val_loss:.4f} -> {val_loss:.4f}. Saving checkpoint.")
                self.best_val_loss = val_loss
                self.epochs_no_improve = 0
                torch.save(self.model.state_dict(), f"best_{self.model_name.lower()}.pth")
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

        total_time = time.time() - start
        print(f"\nTraining finished in {total_time/60:.1f} minutes. "
              f"Best validation loss: {self.best_val_loss:.4f}")
        return {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "train_logmse": self.train_logmse,
            "train_violation": self.train_violation,
        }


def plot_losses(history, title="Training vs Validation Loss", save_path=None):
    train_losses = history.get("train_losses", [])
    val_losses = history.get("val_losses", [])
    if not train_losses:
        return
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(7, 4))
    plt.plot(epochs, train_losses, label="Train")
    plt.plot(epochs, val_losses, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Saved loss plot to {save_path}")
    plt.close()


def load_pretrained_experts(moe_model, checkpoint_path, device):
    """
    Loads the provided checkpoint into each expert. All experts share the same
    initialization weights before fine-tuning inside the MoE.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Pretrained checkpoint not found: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location=device)
    for idx, expert in enumerate(moe_model.experts):
        missing, unexpected = expert.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            print(f"Warning: Expert {idx} missing keys: {missing}, unexpected: {unexpected}")
    print(f"Loaded pretrained weights into {len(moe_model.experts)} experts from {checkpoint_path}.")


# ==============================================================================
# === Main =====================================================================
# ==============================================================================


if __name__ == "__main__":
    # --- Configuration ---
    TRAIN_DATA_FOLDERS = ["./split_data_train_20000_random"]
    VALIDATION_DATA_FOLDERS = ["./split_data_validation_20000_random"]
    PRETRAINED_EXPERT_PATH = "./best_relu_resnet_lowerbound_violationloss.pth"

    max_k = 6
    max_nk = 6
    batch_size = 256
    val_split_ratio = 0.2
    NUM_WORKERS = 0
    PIN_MEMORY = True

    NUM_EPOCHS = 50
    PATIENCE = 10
    LR = 1e-4
    WEIGHT_DECAY = 1e-4
    TOP_K = 2
    NUM_EXPERTS = 6
    GATE_HIDDEN = 32
    LAMBDA_VIOLATION = 0.55
    NUM_SAMPLES = 20

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Prepare dataset ---
    all_files = []
    for folder in TRAIN_DATA_FOLDERS + VALIDATION_DATA_FOLDERS:
        folder_files = glob.glob(os.path.join(folder, "*.pkl"))
        all_files.extend(folder_files)

    if not all_files:
        raise FileNotFoundError("No .pkl files found for training or validation.")

    dataset = PickleFolderDataset(
        file_paths=all_files,
        max_k=max_k,
        max_nk=max_nk,
        p_normaliser="none",
    )
    train_size = int(len(dataset) * (1 - val_split_ratio))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # --- Instantiate model ---
    expert_cfg = {
        "k_max": max_k,
        "nk_max": max_nk,
        "n_params": 3,
        "base_ch": 64,
        "num_blocks": 5,
        "enforce_lower_bound": True,
    }
    moe_model = SparseMoEResNet(
        num_experts=NUM_EXPERTS,
        top_k=TOP_K,
        gate_hidden_dim=GATE_HIDDEN,
        gate_input_dim=3,
        expert_kwargs=expert_cfg,
    ).to(device)

    load_pretrained_experts(moe_model, PRETRAINED_EXPERT_PATH, device=device)

    criterion = ViolationInformedLossAccelerated(
        lambda_violation=LAMBDA_VIOLATION,
        num_samples=NUM_SAMPLES,
    )
    optimizer = optim.AdamW(moe_model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=PATIENCE // 2, factor=0.3
    )

    trainer = MoETrainer(
        model=moe_model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        patience=PATIENCE,
        scheduler=scheduler,
        model_name="Sparse_MoE_ReLU_ResNet_LB_Violation",
    )

    history = trainer.fit(NUM_EPOCHS)
    plot_path = "sparse_moe_resnet_lb_violation_losses.png"
    plot_losses(history, title="Sparse MoE ResNet Training", save_path=plot_path)

    if history["val_losses"]:
        best_val = min(history["val_losses"])
        print(f"Best validation LogMSE: {best_val:.4f}")
    else:
        print("Training did not log any validation losses.")
