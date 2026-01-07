#!/usr/bin/env python3
"""
Utility to score the predominant specialist checkpoints for every (n, k, ratio)
combination and verify that the dataset mixing ratio matches the intended
definition (predominant samples comprise `ratio` fraction of the subset).
"""

import argparse
import glob
import math
import os
import re
import sys
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

# Ensure the project root is on sys.path so we can import sibling modules.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from predominant_specialist.train_predominant_specialists import (
    LogMSELoss,
    PickleFolderDataset,
    ResNet2DWithParams,
)


class PairOnlySubset(Dataset):
    """Views a PickleFolderDataset but only keeps samples that match (n, k)."""

    def __init__(self, base_dataset: PickleFolderDataset, n_value: int, k_value: int):
        super().__init__()
        n_all = base_dataset.n_vals.cpu().numpy().astype(int)
        k_all = base_dataset.k_vals.cpu().numpy().astype(int)
        mask = (n_all == n_value) & (k_all == k_value)
        self.indices = np.where(mask)[0].tolist()
        if not self.indices:
            raise ValueError(f"No samples available for (n={n_value}, k={k_value}) in evaluation data.")
        self.base_dataset = base_dataset

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        return self.base_dataset[self.indices[idx]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate predominant specialist checkpoints.")
    parser.add_argument(
        "--data-folders",
        nargs="+",
        default=["./split_data_validation_20000_random"],
        help="Folder(s) containing *.pkl shards for evaluation.",
    )
    parser.add_argument(
        "--ratio-folders",
        nargs="+",
        default=["./split_data_train_20000_random", "./split_data_validation_20000_random"],
        help="Folder(s) used to verify ratio definitions (defaults to train+validation).",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default="./predominant_specialist/checkpoints",
        help="Directory that stores the per-ratio specialist checkpoints.",
    )
    parser.add_argument(
        "--results-csv",
        default="./predominant_specialist/evaluation_results.csv",
        help="Where to save the aggregate table.",
    )
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--no-pin-memory", action="store_true", help="Disable pin_memory in dataloaders.")
    parser.add_argument("--device", default=None, help="Override torch device string.")
    parser.add_argument("--k-max", type=int, default=6)
    parser.add_argument("--nk-max", type=int, default=6)
    parser.add_argument("--base-ch", type=int, default=64)
    parser.add_argument("--num-blocks", type=int, default=5)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--n-params", type=int, default=3)
    return parser.parse_args()


def collect_pickle_files(folders: Sequence[str]) -> List[str]:
    file_paths: List[str] = []
    for folder in folders:
        pattern = os.path.join(folder, "*.pkl")
        matched = sorted(glob.glob(pattern))
        if not matched:
            print(f"Warning: no pickle files found under {folder}")
        file_paths.extend(matched)
    if not file_paths:
        raise FileNotFoundError("No pickle files could be found with the supplied --data-folders.")
    return file_paths


def collect_checkpoints(checkpoint_dir: str) -> List[Tuple[str, int, int, float]]:
    pattern = os.path.join(checkpoint_dir, "predominant_specialist_n*_k*_ratio*.pth")
    regex = re.compile(r"n(?P<n>\d+)_k(?P<k>\d+)_ratio(?P<ratio>[\d.]+)\.pth$")
    checkpoints: List[Tuple[str, int, int, float]] = []
    for path in sorted(glob.glob(pattern)):
        name = os.path.basename(path)
        match = regex.search(name)
        if not match:
            print(f"Skipping checkpoint with unexpected name: {name}")
            continue
        n_val = int(match.group("n"))
        k_val = int(match.group("k"))
        ratio = float(match.group("ratio"))
        checkpoints.append((path, n_val, k_val, ratio))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints matched {pattern}")
    return checkpoints


def ratio_stats(base_dataset: PickleFolderDataset, n_value: int, k_value: int, ratio: float) -> Dict[str, float]:
    n_all = base_dataset.n_vals.cpu().numpy().astype(int)
    k_all = base_dataset.k_vals.cpu().numpy().astype(int)
    mask = (n_all == n_value) & (k_all == k_value)
    num_predominant = int(mask.sum())
    if num_predominant == 0:
        raise ValueError(f"No samples for (n={n_value}, k={k_value}) were found.")
    num_other_available = len(n_all) - num_predominant
    if ratio >= 1.0:
        num_other_needed = 0
    else:
        ideal_other = num_predominant * (1.0 - ratio) / ratio
        num_other_needed = min(int(ideal_other), num_other_available)
    total = num_predominant + num_other_needed
    actual_ratio = num_predominant / total if total > 0 else 0.0
    return {
        "num_predominant": num_predominant,
        "num_other_for_ratio": num_other_needed,
        "actual_ratio": actual_ratio,
        "num_other_available": num_other_available,
    }


def build_pair_loaders(
    pairs: Iterable[Tuple[int, int]],
    dataset: PickleFolderDataset,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> Dict[Tuple[int, int], DataLoader]:
    loaders: Dict[Tuple[int, int], DataLoader] = {}
    for n_val, k_val in sorted(set(pairs)):
        subset = PairOnlySubset(dataset, n_val, k_val)
        loader = DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        loaders[(n_val, k_val)] = loader
        print(
            f"Prepared evaluation loader for (n={n_val}, k={k_val}) "
            f"with {len(subset)} samples (pure pair subset)."
        )
    return loaders


def evaluate_checkpoint(
    model: ResNet2DWithParams,
    loader: DataLoader,
    criterion: LogMSELoss,
    device: torch.device,
) -> Tuple[float, int]:
    model.eval()
    total_logmse = 0.0
    total = 0
    with torch.no_grad():
        for params, targets, P in loader:
            params = params.to(device)
            targets = targets.to(device)
            P = P.to(device)
            preds = model(P, params)
            batch = params.size(0)
            loss = criterion(preds, targets)
            total_logmse += loss.item() * batch
            total += batch
    if total == 0:
        raise RuntimeError("Evaluation loader produced zero samples.")
    logmse = total_logmse / total
    return logmse, total


def main():
    args = parse_args()
    pin_memory = not args.no_pin_memory
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")

    eval_files = collect_pickle_files(args.data_folders)
    print(f"Loaded {len(eval_files)} pickle shard(s) for evaluation.")
    dataset = PickleFolderDataset(
        file_paths=eval_files,
        max_k=args.k_max,
        max_nk=args.nk_max,
        p_normaliser="none",
    )

    ratio_files = collect_pickle_files(args.ratio_folders)
    print(f"Loaded {len(ratio_files)} pickle shard(s) for ratio verification.")
    ratio_dataset = PickleFolderDataset(
        file_paths=ratio_files,
        max_k=args.k_max,
        max_nk=args.nk_max,
        p_normaliser="none",
    )

    checkpoints = collect_checkpoints(args.checkpoint_dir)
    checkpoint_pairs = [(n_val, k_val) for _, n_val, k_val, _ in checkpoints]
    loaders = build_pair_loaders(
        checkpoint_pairs,
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    model_kwargs = {
        "k_max": args.k_max,
        "nk_max": args.nk_max,
        "n_params": args.n_params,
        "base_ch": args.base_ch,
        "num_blocks": args.num_blocks,
        "dropout_p": args.dropout,
    }
    criterion = LogMSELoss()

    results: List[Dict[str, float]] = []

    for ckpt_path, n_val, k_val, ratio in checkpoints:
        print(f"\nEvaluating checkpoint {os.path.basename(ckpt_path)}...")
        stats = ratio_stats(ratio_dataset, n_val, k_val, ratio)
        print(
            f"  Ratio target {ratio:.2f} -> predominant={stats['num_predominant']} | "
            f"other_for_ratio={stats['num_other_for_ratio']} (actual_ratio={stats['actual_ratio']:.3f})"
        )

        loader = loaders.get((n_val, k_val))
        if loader is None:
            print(f"  ! No evaluation loader available for (n={n_val}, k={k_val}); skipping.")
            continue

        model = ResNet2DWithParams(**model_kwargs).to(device)
        state_dict = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state_dict)

        logmse, total = evaluate_checkpoint(model, loader, criterion, device)
        print(f"  Metrics -> log2-MSE: {logmse:.4f}, samples: {total}")

        results.append(
            {
                "n": n_val,
                "k": k_val,
                "ratio": ratio,
                "log2_mse": logmse,
                "num_eval_samples": total,
                "num_predominant_samples_available": stats["num_predominant"],
                "num_other_samples_for_ratio": stats["num_other_for_ratio"],
                "num_other_samples_available": stats["num_other_available"],
                "actual_ratio_from_definition": stats["actual_ratio"],
            }
        )

    if not results:
        raise RuntimeError("No evaluation metrics were produced.")

    results.sort(key=lambda r: (r["n"], r["k"], -r["ratio"]))

    os.makedirs(os.path.dirname(args.results_csv), exist_ok=True)
    header = [
        "n",
        "k",
        "ratio",
        "log2_mse",
        "num_eval_samples",
        "num_predominant_samples_available",
        "num_other_samples_for_ratio",
        "num_other_samples_available",
        "actual_ratio_from_definition",
    ]
    with open(args.results_csv, "w", newline="") as f:
        import csv

        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for row in results:
            writer.writerow(row)
    print(f"\nSaved evaluation table with {len(results)} rows to {args.results_csv}")

    best_entry = min(results, key=lambda r: r["log2_mse"])
    print(
        "Best checkpoint overall: "
        f"(n={best_entry['n']}, k={best_entry['k']}, ratio={best_entry['ratio']:.2f}) "
        f"with log2-MSE={best_entry['log2_mse']:.4f}"
    )


if __name__ == "__main__":
    main()
