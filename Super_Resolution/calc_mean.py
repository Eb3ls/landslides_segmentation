import os
import sys
import json
from datetime import datetime
import argparse
from typing import Literal, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Super_Resolution.config import load_config
from data_utils import ComuneType, SuperResolutionDataset


def _compute_stats_from_loader_on_tensor(
    loader: DataLoader, device: torch.device
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute per-channel mean/std over batches for High-Resolution (HR) tensors.

    Args:
        loader: DataLoader yielding (low_res, high_res) or just high_res
        device: torch.device to use
    Returns:
        (mean, std) as numpy arrays of shape (C,)
    """
    sums = None
    sq_sums = None
    total_pixels = 0.0

    with torch.no_grad():
        for _, (_, hr) in enumerate(loader):
            x = hr.to(device)
            b, _, h, w = x.shape
            n = float(b * h * w)
            s = x.sum(dim=(0, 2, 3)).cpu().numpy()
            sq = (x * x).sum(dim=(0, 2, 3)).cpu().numpy()

            if sums is None:
                sums = s
                sq_sums = sq
            else:
                sums += s
                sq_sums += sq

            total_pixels += n

    if sums is None or sq_sums is None or total_pixels == 0:
        raise RuntimeError("Empty DataLoader or zero pixels while computing stats")

    mean = sums / total_pixels
    var = (sq_sums / total_pixels) - (mean**2)
    std = np.sqrt(np.maximum(var, 1e-12))
    return mean, std


def main():
    parser = argparse.ArgumentParser(
        description="Compute per-channel mean/std for SR training data"
    )
    parser.add_argument(
        "--dataset_size",
        type=int,
        default=1000,
        help="Number of patches for the temporary dataset (defaults to train.dataset_size)",
    )
    parser.add_argument(
        "--synthetic_data",
        default=False,
        help="Use synthetic data for stats computation",
    )
    parser.add_argument(
        "--excluded_comune",
        type=str,
        default="Brisighella",
        help="Comune to exclude",
    )
    args = parser.parse_args()

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    img_size = 128

    # Crea un dataset analogo a quello di training ma SENZA augmentazione
    train_dataset = SuperResolutionDataset(
        comune=args.excluded_comune,
        scale=5,
        patch_size=img_size,
        num_patches=args.dataset_size,
        for_training=True,
        to_augment=False,
        synthetic_data=args.synthetic_data,
    )

    loader = DataLoader(
        train_dataset,
        batch_size=16,
        num_workers=4,
        persistent_workers=True,
        pin_memory=torch.cuda.is_available(),
    )

    print(f"Computing stats on: {'High-Res (HR)'}")
    mean, std = _compute_stats_from_loader_on_tensor(loader, device)
    print("Computed channel stats")

    # Prepare output directory and file
    payload = {
        "timestamp": datetime.now().isoformat(),
        "mean": mean.tolist(),
        "std": std.tolist(),
        "details": {
            "dataset_size": train_dataset.num_patches,
            "img_size": img_size,
            "synthetic_data": args.synthetic_data,
            "training_comuni": train_dataset.set_comuni,
        },
    }

    with open("Super_Resolution/channel_stats.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4)


if __name__ == "__main__":
    main()
