import os
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.utils.data import DataLoader

from .seg_unet import (
    UNet,
    evaluate_model,
    visualize_results,
    seed_everything,
)

from data_utils import SegmentationSingleDataset

MODEL_PATH = "attention_ndvi_model.pth"
PATCH_SIZE = 256
NUM_PATCHES = 1000
BATCH_SIZE = 4
FIXED_THRESHOLD = 0.42  # Non utilizzato per soglia variabile
VIS_SAMPLES = 5


def collect_eval_probs_and_labels(model, dataloader, device):
    model.eval()
    probs_list, labels_list = [], []
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            logits = model(x)
            probs = torch.sigmoid(logits).cpu()
            probs_list.append(probs)
            labels_list.append(y.cpu())
    probs_all = torch.cat(probs_list, dim=0).numpy()  # (N,1,H,W)
    labels_all = torch.cat(labels_list, dim=0).numpy()  # (N,1,H,W)
    return probs_all, labels_all


def build_parser():
    p = argparse.ArgumentParser(description="Valutazione modello frane")
    p.add_argument(
        "--comune", default="Casola-Valsenio", help="Nome del comune da valutare"
    )
    p.add_argument(
        "--variable-threshold",
        dest="variable_threshold",
        action="store_true",
        help="Usa soglia variabile invece di una fissa",
    )
    return p


def main():
    args = build_parser().parse_args()
    if args.variable_threshold:
        threshold = None
    else:
        threshold = FIXED_THRESHOLD

    seed_everything(420)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    comune = args.comune

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

    if threshold is not None:
        print(f"Utilizzo soglia fissa: {threshold}")

        # Valutazione
        val_loss, val_iou, val_oa, val_dice, confusion_matrix = evaluate_model(
            model=model,
            dataloader=eval_loader,
            device=device,
            criterion=criterion,
            threshold=threshold,
            return_confusion_matrix=True,
        )

        print(f"Risultati valutazione su {comune}:")
        print(f"  Loss media : {val_loss:.6f}")
        print(f"  IoU        : {val_iou:.4f}")
        print(f"  OA         : {val_oa:.4f}")
        print(f"  Dice       : {val_dice:.4f}")
        print(f"  Confusion Matrix:\n{confusion_matrix}")

        print("Apro visualizzazione napari...")
        visualize_results(
            model=model,
            dataset=eval_dataset,
            device=device,
            num_samples=min(VIS_SAMPLES, len(eval_dataset)),
        )

    else:
        print("Utilizzo soglia variabile.")

        iou_values = []
        oa_values = []

        thresholds = np.arange(0.2, 0.8 + 1e-9, 0.02)

        y_prob, y_true = collect_eval_probs_and_labels(model, eval_loader, device)

        iou_values, oa_values = [], []
        for th in thresholds:
            preds = (y_prob > th).astype(np.float32)
            inter = (preds * y_true).sum()
            union = ((preds + y_true) > 0).sum()
            iou = inter / union if union > 0 else 1.0
            oa = (preds == y_true).mean()
            iou_values.append(iou)
            oa_values.append(oa)

        max_iou = np.max(iou_values)
        max_oa = np.max(oa_values)
        max_iou_index = np.argmax(iou_values)
        max_oa_index = np.argmax(oa_values)

        print(f"Best IoU: {max_iou:.4f} (Threshold: {thresholds[max_iou_index]:.2f})")
        print(f"Best OA : {max_oa:.4f} (Threshold: {thresholds[max_oa_index]:.2f})")

        # Creazione dir per plot
        base_dir = os.path.dirname(os.path.abspath(__file__))
        plots_dir = os.path.join(base_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)

        # Plot metriche
        plt.subplot(1, 2, 1)
        plt.plot(np.arange(0.2, 0.8 + 1e-9, 0.02), iou_values, marker="o")
        plt.title("IoU")
        plt.xlabel("Threshold")
        plt.ylabel("IoU")

        plt.subplot(1, 2, 2)
        plt.plot(np.arange(0.2, 0.8 + 1e-9, 0.02), oa_values, marker="o")
        plt.title("OA")
        plt.xlabel("Threshold")
        plt.ylabel("OA")

        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "threshold_metrics.png"), dpi=150)
        plt.show()


if __name__ == "__main__":
    main()
