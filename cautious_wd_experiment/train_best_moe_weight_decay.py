import argparse
import csv
import glob
import math
import os
import pickle
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset

import wandb

class PickleFolderDataset(Dataset):
    """
    Loads pickled DataFrames with columns ['n', 'k', 'm', 'result', 'P'].
    P is reshaped to (k, n-k), padded to (max_k, max_nk), and returned with params.
    """

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

        required_cols = {"n", "k", "m", "result", "P"}
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
        P = self._pad(P)

        params = torch.tensor([n, k, m], dtype=torch.float32)
        h = self.h_vals[idx].unsqueeze(0)
        return params, h, P


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
    def __init__(
        self,
        k_max=6,
        nk_max=6,
        n_params=3,
        base_ch=64,
        num_blocks=5,
        enforce_lower_bound=False,
        dropout_p=0.0,
    ):
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
        if self.enforce_lower_bound:
            return 1.0 + F.softplus(z)
        return z


class LogMSELoss(nn.Module):
    def __init__(self, eps=1e-9):
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
            return loss_logmse, loss_logmse.item(), 0.0

        batch = P_padded.shape[0]
        device = y_pred.device
        total_penalty = torch.tensor(0.0, device=device)
        unique_params, inverse_indices = torch.unique(params, dim=0, return_inverse=True)

        for i in range(unique_params.size(0)):
            n = int(unique_params[i, 0].item())
            k = int(unique_params[i, 1].item())
            m = int(unique_params[i, 2].item())
            mask = inverse_indices == i
            group_P = P_padded[mask]
            group_pred = y_pred[mask]
            group_batch = group_P.size(0)
            if m + 1 > n or k <= 0 or n - k < 0:
                continue

            P_actual = group_P[:, :k, :(n - k)]
            I = torch.eye(k, device=device)
            G_group = torch.cat([I.unsqueeze(0).expand(group_batch, -1, -1), P_actual], dim=2)
            X_samples = torch.randn(group_batch, self.num_samples, k, device=device)
            C_samples = torch.bmm(X_samples, G_group)
            magnitudes = torch.abs(C_samples)
            if m + 1 > magnitudes.shape[2]:
                continue
            magnitudes_flat = magnitudes.view(-1, n)
            top_magnitudes, _ = torch.topk(magnitudes_flat, k=m + 1, dim=1, largest=True)
            c_max, c_m = top_magnitudes[:, 0], top_magnitudes[:, m]
            hm_samples = (c_max / (c_m + self.eps)).view(group_batch, self.num_samples)
            max_hm_sample_group, _ = torch.max(hm_samples, dim=1)
            group_penalty = F.relu(max_hm_sample_group - group_pred.squeeze())
            total_penalty += torch.sum(group_penalty)

        loss_violation = total_penalty / batch
        total_loss = loss_logmse + self.lambda_violation * loss_violation
        return total_loss, loss_logmse.item(), loss_violation.item()


class CautiousAdamW(optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        if lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("CautiousAdamW does not support sparse gradients.")

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                state["step"] += 1
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]
                step_size = lr * math.sqrt(bias_correction2) / bias_correction1

                denom = exp_avg_sq.sqrt().add_(eps)
                update = exp_avg / denom
                update.mul_(-step_size)

                if weight_decay != 0.0:
                    mask = (p * update) > 0
                    p.add_(p * mask, alpha=-lr * weight_decay)

                p.add_(update)

        return loss


def build_decay_param_groups(model, weight_decay):
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or name.endswith(".bias"):
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    return [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]


def resolve_device(device_arg: str) -> torch.device:
    if device_arg.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available. Use --device cpu to override.")
        return torch.device(device_arg)
    if device_arg == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Unsupported device value: {device_arg}")


def run_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    training=True,
    use_violation_loss=False,
    use_amp=False,
    scaler=None,
    non_blocking=False,
):
    if training:
        model.train()
        context = torch.enable_grad()
    else:
        model.eval()
        context = torch.no_grad()

    total_loss = 0.0
    total_logmse = 0.0
    total_violation = 0.0
    total_samples = 0

    with context:
        for params, targets, P in loader:
            params = params.to(device, non_blocking=non_blocking)
            targets = targets.to(device, non_blocking=non_blocking)
            P = P.to(device, non_blocking=non_blocking)

            if training:
                optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=use_amp):
                outputs = model(P, params)
                if use_violation_loss:
                    loss, logmse, violation = criterion(
                        outputs,
                        targets,
                        P_padded=P,
                        params=params,
                        calculate_violation=training,
                    )
                    total_logmse += logmse * params.size(0)
                    total_violation += violation * params.size(0)
                else:
                    loss = criterion(outputs, targets)

            if training:
                if use_amp:
                    if scaler is None:
                        raise RuntimeError("AMP requested but scaler is None.")
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

            batch_size = params.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

    denom = max(total_samples, 1)
    avg_loss = total_loss / denom
    avg_logmse = total_logmse / denom if use_violation_loss else None
    avg_violation = total_violation / denom if use_violation_loss else None
    return avg_loss, avg_logmse, avg_violation


def save_history_csv(path, history):
    fieldnames = ["epoch", "train_loss", "val_loss"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in history:
            writer.writerow(row)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train ResNet with cautious weight decay and optional violation loss.",
    )
    parser.add_argument("--train-dir", default="split_data_train_20000_random")
    parser.add_argument("--val-dir", default="split_data_validation_20000_random")
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--persistent-workers", dest="persistent_workers", action="store_true")
    parser.add_argument("--no-persistent-workers", dest="persistent_workers", action="store_false")
    parser.set_defaults(persistent_workers=True)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--base-ch", type=int, default=64)
    parser.add_argument("--num-blocks", type=int, default=5)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--enforce-lower-bound", dest="enforce_lower_bound", action="store_true")
    parser.add_argument("--no-enforce-lower-bound", dest="enforce_lower_bound", action="store_false")
    parser.set_defaults(enforce_lower_bound=True)
    parser.add_argument("--lambda-violation", type=float, default=0.5508697436585597)
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--disable-violation-loss", action="store_true")
    parser.add_argument("--max-k", type=int, default=6)
    parser.add_argument("--max-nk", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-every", type=int, default=0)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", dest="amp", action="store_true")
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.set_defaults(amp=True)
    parser.add_argument("--tf32", dest="tf32", action="store_true")
    parser.add_argument("--no-tf32", dest="tf32", action="store_false")
    parser.set_defaults(tf32=True)
    parser.add_argument("--cudnn-benchmark", dest="cudnn_benchmark", action="store_true")
    parser.add_argument("--no-cudnn-benchmark", dest="cudnn_benchmark", action="store_false")
    parser.set_defaults(cudnn_benchmark=True)
    return parser.parse_args()


def main():
    args = parse_args()

    if args.seed:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    run_name = args.run_name or f"resnet_cautious_wd{args.weight_decay:.4f}_rerun"
    run = wandb.init(
        project="m-height-grokking",
        name=run_name,
        config={
            "optimizer": "CautiousAdamW",
            **vars(args),
        },
    )

    device = resolve_device(args.device)
    print(f"Using device: {device}")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = args.tf32
        torch.backends.cudnn.allow_tf32 = args.tf32
        torch.backends.cudnn.benchmark = args.cudnn_benchmark
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high" if args.tf32 else "highest")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = args.output_dir or os.path.join(script_dir, "outputs")
    os.makedirs(output_dir, exist_ok=True)

    train_files = sorted(glob.glob(os.path.join(args.train_dir, "*.pkl")))
    val_files = sorted(glob.glob(os.path.join(args.val_dir, "*.pkl")))
    if not train_files:
        raise FileNotFoundError(f"No .pkl files found in train dir: {args.train_dir}")
    if not val_files:
        raise FileNotFoundError(f"No .pkl files found in val dir: {args.val_dir}")

    train_dataset = PickleFolderDataset(
        file_paths=train_files,
        max_k=args.max_k,
        max_nk=args.max_nk,
    )
    val_dataset = PickleFolderDataset(
        file_paths=val_files,
        max_k=args.max_k,
        max_nk=args.max_nk,
    )

    pin_memory = device.type == "cuda"
    non_blocking = pin_memory
    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": pin_memory,
    }
    if args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = args.prefetch_factor
        loader_kwargs["persistent_workers"] = args.persistent_workers
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        **loader_kwargs,
    )

    model_cfg = {
        "k_max": args.max_k,
        "nk_max": args.max_nk,
        "n_params": 3,
        "base_ch": args.base_ch,
        "num_blocks": args.num_blocks,
        "enforce_lower_bound": args.enforce_lower_bound,
        "dropout_p": args.dropout,
    }
    model = ResNet2DWithParams(**model_cfg).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    wandb.log({"model_params": total_params})

    use_violation_loss = not args.disable_violation_loss
    if use_violation_loss:
        criterion = ViolationInformedLossAccelerated(
            lambda_violation=args.lambda_violation,
            num_samples=args.num_samples,
        )
    else:
        criterion = LogMSELoss()
    param_groups = build_decay_param_groups(model, args.weight_decay)
    optimizer = CautiousAdamW(param_groups, lr=args.lr)
    use_amp = args.amp and device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)

    best_val = float("inf")
    history = []
    best_path = os.path.join(output_dir, "best_resnet_cautious_wd.pth")
    last_path = os.path.join(output_dir, "last_resnet_cautious_wd.pth")
    history_path = os.path.join(output_dir, "training_history.csv")

    print(
        "Config:"
        f" epochs={args.epochs}, batch={args.batch_size}, lr={args.lr}, wd={args.weight_decay},"
        f" base_ch={args.base_ch}, blocks={args.num_blocks},"
        f" lower_bound={args.enforce_lower_bound}, violation_loss={use_violation_loss},"
        f" amp={use_amp}, tf32={args.tf32}, cudnn_benchmark={args.cudnn_benchmark}"
    )

    start = time.time()
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        train_loss, train_logmse, train_violation = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            training=True,
            use_violation_loss=use_violation_loss,
            use_amp=use_amp,
            scaler=scaler,
            non_blocking=non_blocking,
        )
        val_loss, _, _ = run_epoch(
            model,
            val_loader,
            criterion,
            optimizer,
            device,
            training=False,
            use_violation_loss=use_violation_loss,
            use_amp=use_amp,
            scaler=scaler,
            non_blocking=non_blocking,
        )

        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        save_history_csv(history_path, history)

        elapsed = time.time() - epoch_start
        if use_violation_loss:
            print(
                f"Epoch {epoch:05d}: train={train_loss:.4f} (logmse={train_logmse:.4f}, "
                f"viol={train_violation:.4f}) val={val_loss:.4f} ({elapsed:.1f}s)"
            )
        else:
            print(
                f"Epoch {epoch:05d}: train={train_loss:.4f} val={val_loss:.4f} "
                f"({elapsed:.1f}s)"
            )
        wandb.log(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "epoch_time": elapsed,
                "lr": optimizer.param_groups[0]["lr"],
            }
        )
        if use_violation_loss:
            wandb.log(
                {
                    "train_logmse": train_logmse,
                    "train_violation": train_violation,
                }
            )

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), best_path)
            print(f"  New best val {best_val:.4f}. Saved {best_path}.")
            wandb.log({"best_val_loss": best_val, "best_epoch": epoch})

        if args.save_every and epoch % args.save_every == 0:
            checkpoint_path = os.path.join(output_dir, f"checkpoint_epoch_{epoch}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  Checkpoint saved: {checkpoint_path}")

        torch.save(model.state_dict(), last_path)

    total_minutes = (time.time() - start) / 60.0
    print(f"Training complete in {total_minutes:.1f} minutes. Best val loss: {best_val:.4f}")
    wandb.log({"total_training_time_min": total_minutes})
    wandb.finish()


if __name__ == "__main__":
    main()
