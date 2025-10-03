import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import numpy as np
from typing import Tuple
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import cast
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
try:
    from data_utils import ComuneType, SegmentationMultiDataset, SegmentationSingleDataset
except Exception as e:
    raise ImportError(
        f"Impossibile importare data_utils. Verifica che esista '{PROJECT_ROOT / 'data_utils.py'}' "
        f"e avvia lo script dalla root del progetto. Dettagli: {e}"
    )

from .unet import UNet, AttentionUNet, log_attention_stats
from .swin_unet import SwinUnet
from .swin_config import get_config
from .evaluate import evaluate_model, seed_everything
from .losses import DiceLoss, SquaredDiceLoss, BCEDiceLoss

PATCH_SIZE = 256
NUM_PATCHES = 4000
BATCH_SIZE = 32
NUM_EPOCHS = 120  # 120
LR = 5e-4
SWIN_WEIGHT_DECAY = 5e-2  # WD per SwinUnet (richiesto 1e-2–1e-3)
UNET_WEIGHT_DECAY = 1e-4  # WD per UNet e AttentionUNet

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    eval_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    scheduler: ReduceLROnPlateau,
    num_epochs: int = 20,
    early_stop_patience: int = 15,
    prefix: str = "",
    weights_dir: str = "weights",
) -> Tuple[list[float], list[float], list[float], list[float], list[float]]:
    """Addestra il modello di segmentazione."""

    model.train()
    train_losses = []
    val_losses = []
    oa_values = []
    iou_values = []
    dice_values = []

    best_iou = 0
    epochs_without_improvement = 0

    for epoch in range(num_epochs):
        num_samples = 0
        epoch_loss_sum = 0.0

        with tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"
        ) as pbar:  # Si genera la barra di avanzamento
            for batch_idx, (data, landslide) in enumerate(pbar):
                data = data.to(device)
                landslide = landslide.to(device)

                # Azzeramento dei gradienti
                optimizer.zero_grad()

                # Forward pass
                outputs = model(data)  # Generazione delle predizioni
                loss = criterion(outputs, landslide)

                # Backward pass
                loss.backward()
                optimizer.step()

                batch_size = data.size(0)
                epoch_loss_sum += (
                    loss.item() * batch_size
                )  # Loss sul batch corrente * dimensione del batch
                num_samples += batch_size
                pbar.set_postfix({"Loss": f"{loss.item():.6f}"})

        train_loss = epoch_loss_sum / max(1, num_samples)
        train_losses.append(train_loss)

        val_loss, val_iou, oa, dice_score, _ = evaluate_model(
            model, eval_loader, device, criterion
        )

        model.train()

        val_losses.append(val_loss)
        iou_values.append(val_iou)
        dice_values.append(dice_score)
        oa_values.append(oa)

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {train_loss:.6f}, Validation Loss: {val_loss:.6f}, Validation IoU: {val_iou:.6f}, Validation Dice: {dice_score:.6f}, Overall Accuracy: {oa:.6f}, LR: {current_lr:.2e}"
        )

        log_attention_stats(model, prefix=f"Epoch {epoch+1}: ")

        # Salvataggio del modello migliore + Early Stopping
        if val_iou > best_iou:
            best_iou = val_iou
            epochs_without_improvement = 0
            print(
                f"New best model found at epoch {epoch+1} with average IoU {best_iou:.6f}"
            )
            save_path = os.path.join(weights_dir, f"{prefix}_model.pth")
            torch.save(model.state_dict(), save_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= early_stop_patience and epoch >= 30:
                print(
                    f"Early stopping: nessun miglioramento di IoU per {early_stop_patience} epoche (epoch {epoch+1})."
                )
                break

    return train_losses, val_losses, iou_values, oa_values, dice_values


def build_prefix(model_name: str, loss_name: str, include_slope_ndvi: bool) -> str:
    """Costruisce il prefisso per il salvataggio dei modelli."""
    prefix = f"{model_name}_{loss_name}"
    if include_slope_ndvi:
        prefix += "_slope_ndvi"
    return prefix


def save_model(model: nn.Module, path: str) -> None:
    """Salva il modello."""
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")


def seed_workers(worker_id: int) -> None:
    """Imposta il seed per i worker di PyTorch."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed + worker_id)
    random.seed(worker_seed + worker_id)
    torch.manual_seed(worker_seed + worker_id)


def main():
    """Funzione principale per addestrare e valutare il modello di segmentazione."""
    # Parser argomenti CLI per selezionare il modello
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=["swin", "unet", "attunet"],
        default="unet",
        help="Seleziona il modello da usare per il training.",
    )
    # Selezione funzione di loss
    parser.add_argument(
        "--loss",
        choices=["bce", "dice", "sqdice", "bce_dice"],
        default="bce",
        help="Loss da usare: bce, dice, sqdice (squared dice), bce_dice.",
    )
    parser.add_argument(
        "--include_slope_ndvi",
        action="store_true",
        help="Includi i dati di Slope e Ndvi nel training.",
    )
    args = parser.parse_args()
    prefix = build_prefix(args.model, args.loss, args.include_slope_ndvi)

    # Parametri di configurazione
    train_comuni = cast(list[ComuneType], ["Predappio", "Modigliana", "Brisighella"])
    eval_comune = cast(ComuneType, "Casola-Valsenio")

    # Dispositivo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Seed per riproducibilità
    seed_everything(42)

    try:
        # Creazione del dataset
        train_dataset = SegmentationMultiDataset(
            train_comuni,
            PATCH_SIZE,
            NUM_PATCHES,
            include_slope_ndvi=args.include_slope_ndvi,
        )
        eval_dataset = SegmentationSingleDataset(
            eval_comune,
            PATCH_SIZE,
            NUM_PATCHES,
            include_slope_ndvi=args.include_slope_ndvi,
        )

        # Printiamo shape in input e output
        sample_input, sample_landslide = train_dataset[0]
        n_channels_in = sample_input.shape[0]
        n_channels_out = sample_landslide.shape[0]

        print(f"Input channels: {n_channels_in}, Output channels: {n_channels_out}")

        # Creiamo i data loader
        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            drop_last=True,
            # Parallelizziamo la generazione dei batch
            num_workers=0,
            worker_init_fn=seed_workers,
            # Velocizziamo il caricamento dei dati su GPU se disponibile
            pin_memory=True if device.type == "cuda" else False,
        )

        eval_loader = DataLoader(
            eval_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=0,
            worker_init_fn=seed_workers,
            pin_memory=True if device.type == "cuda" else False,
        )

        # Creazione modello
        print(f"Creating model ({args.model})...")
        if args.model == "swin":
            cfg = get_config()
            model = SwinUnet(config=cfg).to(device)
        elif args.model == "unet":
            model = UNet(n_channels=n_channels_in, n_classes=n_channels_out).to(device)
        elif args.model == "attunet":
            model = AttentionUNet(
                n_channels=n_channels_in, n_classes=n_channels_out
            ).to(device)
        else:
            raise ValueError(f"Modello non supportato: {args.model}")

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Parametri addestrabili: {trainable_params:,}")

        # Loss e optimizer
        if args.loss == "bce":
            criterion: nn.Module = nn.BCEWithLogitsLoss()
        elif args.loss == "dice":
            criterion = DiceLoss()
        elif args.loss == "sqdice":
            criterion = SquaredDiceLoss()
        elif args.loss == "bce_dice":
            criterion = BCEDiceLoss(bce_weight=0.5)
        else:
            raise ValueError(f"Loss non supportata: {args.loss}")
        criterion = criterion.to(device)

        if isinstance(model, SwinUnet):
            # Semplificazione: un solo gruppo di parametri con AdamW, LR fisso e WD fisso
            optimizer = optim.AdamW(
                model.parameters(),
                lr=LR,
                weight_decay=SWIN_WEIGHT_DECAY,
            )
        else:
            optimizer = optim.AdamW(
                model.parameters(),
                lr=LR,
                weight_decay=UNET_WEIGHT_DECAY,
            )

        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=5,
            threshold=1e-4,
            min_lr=1e-7,
        )

        base_dir = os.path.dirname(os.path.abspath(__file__))
        weights_dir = os.path.join(base_dir, "weights")
        os.makedirs(weights_dir, exist_ok=True)

        # Allenamento
        print("Starting training...")
        train_losses, val_losses, iou_values, oa_values, dice_values = train_model(
            model,
            train_loader,
            eval_loader,
            criterion,
            optimizer,
            device,
            scheduler,
            num_epochs=NUM_EPOCHS,
            early_stop_patience=15,
            prefix=prefix,
            weights_dir=weights_dir
        )

        # Crea cartella plots se non esistente
        base_dir = os.path.dirname(os.path.abspath(__file__))
        plots_dir = os.path.join(base_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)

        # Plot training losses
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label="Training Loss")
        plt.plot(val_losses, label="Validation Loss")
        plt.title("Losses")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"{prefix}_losses.png"), dpi=150)
        plt.show()

        # Plot OA
        plt.figure(figsize=(10, 5))
        plt.plot(oa_values, label="Validation OA")
        plt.title("OA")
        plt.xlabel("Epoch")
        plt.ylabel("Score")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"{prefix}_oa.png"), dpi=150)
        plt.show()

        # Plot Dice and IoU
        plt.figure(figsize=(10, 5))
        plt.plot(dice_values, label="Validation Dice")
        plt.plot(iou_values, label="Validation IoU")
        plt.title("Dice")
        plt.xlabel("Epoch")
        plt.ylabel("Score")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"{prefix}_iou.png"), dpi=150)
        plt.show()

    except Exception as e:
        print(f"Error occurred during training: {e}")
        import traceback

        traceback.print_exc()
        pass

if __name__ == "__main__":
    main()
