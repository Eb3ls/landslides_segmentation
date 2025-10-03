import os
import torch
import argparse
import napari
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Tuple

from .unet import UNet, AttentionUNet, log_attention_stats
from .swin_unet import SwinUnet
from .swin_config import get_config
from data_utils import SegmentationSingleDataset

MODEL_PATH = "best_model.pth"
PATCH_SIZE = 256
NUM_PATCHES = 1000
BATCH_SIZE = 8
FIXED_THRESHOLD = 0.5  # Non utilizzato per soglia variabile
VIS_SAMPLES = 3


def seed_everything(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def compute_confusion_matrix(
    preds: torch.Tensor, labels: torch.Tensor
) -> list[list[float]]:
    """Calcola la matrice di confusione."""
    preds = preds.view(-1)
    labels = labels.view(-1)
    tp = ((preds == 1) & (labels == 1)).sum().item()
    fp = ((preds == 1) & (labels == 0)).sum().item()
    tn = ((preds == 0) & (labels == 0)).sum().item()
    fn = ((preds == 0) & (labels == 1)).sum().item()
    return [[tn, fp], [fn, tp]]


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    threshold: float = 0.5,
    return_confusion_matrix: bool = False,
) -> Tuple[float, float, float, float, np.ndarray | None]:
    """Valuta il modello restituendo (loss media, IoU, OA, Dice). Ritorna opzionalmente anche la matrice di confusione."""

    model.eval()
    total_loss_sum = 0.0
    num_samples = 0

    # Accumulatori per metriche
    intersection_sum = 0.0
    union_sum = 0.0
    correct_sum = 0.0
    total_pixels = 0

    # Inizializzazione matrice di confusione
    confusion_matrix_total = (
        np.zeros((2, 2), dtype=np.float32) if return_confusion_matrix else None
    )

    with torch.no_grad():
        for input, labels in dataloader:
            input = input.to(device)
            labels = labels.to(device)
            outputs = model(input)
            loss = criterion(outputs, labels)

            batch_size = input.size(0)
            total_loss_sum += loss.item() * batch_size
            num_samples += batch_size

            # Calcolo metriche batch
            probs = torch.sigmoid(outputs)
            preds = (probs > threshold).float()

            if confusion_matrix_total is not None:
                # Calcolo matrice di confusione
                confusion_matrix = compute_confusion_matrix(preds, labels)
                confusion_matrix_total += confusion_matrix

            # Calcolo intersezione, unione e accuratezza -> ricordare che i valori sono binari
            inter = (preds * labels).sum().item()
            union = ((preds + labels) > 0).float().sum().item()
            intersection_sum += inter
            union_sum += union

            correct = (preds == labels).sum().item()
            correct_sum += correct
            total = preds.numel()
            total_pixels += total

    val_loss = total_loss_sum / max(1, num_samples)
    iou = intersection_sum / union_sum if union_sum > 0 else 1.0
    oa = correct_sum / max(1, total_pixels)
    # Dice = 2 * intersezione / (ground truth + predizioni)
    # ground truth + predizioni si ottiene anche come intersezione + unione di esse
    dice = (
        (2.0 * intersection_sum) / (intersection_sum + union_sum)
        if (intersection_sum + union_sum) > 0
        else 1.0
    )

    return (
        val_loss,
        iou,
        oa,
        dice,
        (
            confusion_matrix_total / total_pixels
            if confusion_matrix_total is not None
            else None
        ),
    )  # len(dataloader) è il numero di batch


def visualize_results(
    model: nn.Module,
    dataset: SegmentationSingleDataset,
    device: torch.device,
    num_samples: int = 5,
) -> None:
    """Visualizza i risultati del modello. Mostra le immagini rgb, le predizioni e le ground truth."""

    model.eval()
    viewer = napari.Viewer()

    with torch.no_grad():
        for i in range(num_samples):
            # Ottieni un campione
            data, landslide = dataset[i]
            data_batch = data.unsqueeze(0).to(device)

            # Genera predizione
            pred = model(data_batch).squeeze(0).cpu()

            # Applica la sigmoide per ottenere probabilità e binarizza
            pred_prob = torch.sigmoid(pred)
            pred_mask = (pred_prob > 0.5).float()

            # Converti a numpy per visualizzazione
            data_np = data.cpu().numpy()
            landslide_np = landslide.cpu().numpy().squeeze()
            pred_mask_np = pred_mask.numpy().squeeze()

            # Estrai i canali di input
            pre_rgb = data_np[:3, :, :].transpose(1, 2, 0)
            post_rgb = data_np[4:7, :, :].transpose(1, 2, 0)

            # Aggiungiamo i layer a napari
            viewer.add_image(pre_rgb, name=f"Pre RGB {i+1}", rgb=True)
            viewer.add_image(post_rgb, name=f"Post RGB {i+1}", rgb=True)
            viewer.add_image(
                landslide_np,
                name=f"True Landslide Mask {i+1}",
                colormap="green",
                blending="additive",
            )
            viewer.add_image(
                pred_mask_np,
                name=f"Predicted Landslide Mask {i+1}",
                colormap="magenta",
                blending="additive",
            )

    napari.run()


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

    seed_everything(200)  # 42, 200, 420

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
    cfg = get_config()
    model = SwinUnet(config=cfg).to(device)

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
