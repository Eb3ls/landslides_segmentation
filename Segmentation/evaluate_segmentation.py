import os
import torch
import argparse

from .sperimental import (
    UNet,
    SegmentationSingleDataset,
    evaluate_model,
    visualize_results,
    seed_everything,
)

import torch.nn as nn
from torch.utils.data import DataLoader


def build_parser():
    p = argparse.ArgumentParser(description="Valutazione modello frane")
    p.add_argument("--comune", default="Casola-Valsenio", help="Nome del comune da valutare")
    return p


def main():
    args = build_parser().parse_args()
    seed_everything(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    comune = args.comune

    # Parametri
    MODEL_PATH = "seg_augment.pth"
    PATCH_SIZE = 256
    NUM_PATCHES = 1000
    BATCH_SIZE = 4
    THRESHOLD = 0.5
    VIS_SAMPLES = 5

    # Dataset di sola valutazione
    eval_dataset = SegmentationSingleDataset(
        comune=comune,
        patch_size=PATCH_SIZE,
        num_patches=NUM_PATCHES,
    )

    sample_in, sample_out = eval_dataset[0]
    n_in, n_out = sample_in.shape[0], sample_out.shape[0]
    print(f"Canali input: {n_in}  | Canali output: {n_out}")

    # Modello
    model = UNet(n_channels=n_in, n_classes=n_out).to(device)

    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(f"Pesi non trovati: {MODEL_PATH}")

    state = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state)
    model.eval()
    print(f"Pesi caricati da: {MODEL_PATH}")

    # Dataloader
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True if device.type == "cuda" else False,
    )

    # Loss coerente con training (logits) e pos_weight
    criterion = nn.BCEWithLogitsLoss()

    # Valutazione
    val_loss, val_iou, val_oa = evaluate_model(
        model=model,
        dataloader=eval_loader,
        device=device,
        criterion=criterion,
        threshold=THRESHOLD,
    )

    print(f"Risultati valutazione su {comune}:")
    print(f"  Loss media : {val_loss:.6f}")
    print(f"  IoU        : {val_iou:.4f}")
    print(f"  OA         : {val_oa:.4f}")

    print("Apro visualizzazione napari...")
    visualize_results(
        model=model,
        dataset=eval_dataset,
        device=device,
        num_samples=min(VIS_SAMPLES, len(eval_dataset))
        )

if __name__ == "__main__":
    main()
