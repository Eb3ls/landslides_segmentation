import os
import random
import napari
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Tuple
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import cast

# TODO: early stopping, data augmentation modificata, sperimentazioni sul modello

from data_utils import (
    ComuneType,
    get_segmentation_stack,
    get_random_patch,
    generate_dataset_mask,
    augment_data,
)


class DoubleConv(nn.Module):
    """Blocco di doppia convoluzione utilizzato in U-Net."""

    def __init__(self, in_channels: int, out_channels: int):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(max(4, out_channels // 32), out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels, out_channels, kernel_size=3, padding=1
            ),  # La seconda convoluzione mantiene la dimensione dei canali
            nn.GroupNorm(max(4, out_channels // 32), out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling con maxpool seguito da doppia convoluzione."""

    def __init__(self, in_channels: int, out_channels: int):
        super(Down, self).__init__()
        self.pool_conv = nn.Sequential(
            nn.MaxPool2d(2), DoubleConv(in_channels, out_channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool_conv(x)


class Up(nn.Module):
    """Upscaling seguito da doppia convoluzione."""

    def __init__(self, in_channels: int, out_channels: int):
        super(Up, self).__init__()

        self.up = nn.ConvTranspose2d(
            in_channels,
            in_channels // 2,
            kernel_size=2,
            stride=2,  # I canali nel decoder dimezzano a ogni up
        )

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)

        # Calcola eventuale differenza tra upsampling e skip (tipicamente tra 0 e 1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        # Aggiunge padding per gestire eventuali differenze di dimensione
        x1 = nn.functional.pad(
            x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2]
        )

        # Concatenazione della skip connection
        x = torch.cat([x2, x1], dim=1)

        return self.conv(x)


class OutConv(nn.Module):
    """Layer di convoluzione di output."""

    def __init__(self, in_channels: int, out_channels: int):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNet(nn.Module):
    """Modello U-Net per segmentazione."""

    def __init__(self, n_channels: int, n_classes: int):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.in_conv = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)
        self.outc = OutConv(64, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.down4(x4)
        x = self.up1(x, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)  # Il logit è il valore grezzo di output al modello
        return logits


class SegmentationSingleDataset(Dataset):
    """Dataset per il training della segmentazione con un comune."""

    def __init__(
        self, comune: ComuneType, patch_size: int = 256, num_patches: int = 1000
    ):
        self.comune = comune
        self.patch_size = patch_size
        self.num_patches = num_patches

        print(f"Loading data for {comune}...")

        # Generiamo la maschera del dataset
        self.mask = generate_dataset_mask(comune)

        # Carichiamo gli stack di dati
        self.stack_input, self.stack_landslide = get_segmentation_stack(comune)

    def __len__(self) -> int:
        return self.num_patches

    def __getitem__(self, _) -> Tuple[torch.Tensor, torch.Tensor]:

        input_patch, landslide_patch, _ = get_random_patch(
            self.stack_input, self.stack_landslide, self.patch_size, self.mask
        )

        # Convertiamo a tensori e assicuriamo il formato corretto
        input_tensor = torch.from_numpy(input_patch).float()
        landslide_tensor = torch.from_numpy(landslide_patch).float()

        # Settiamo i valori NaN a 0
        input_tensor = torch.nan_to_num(input_tensor, nan=0.0)
        landslide_tensor = torch.nan_to_num(landslide_tensor, nan=0.0)

        return input_tensor, landslide_tensor


class SegmentationMultiDataset(Dataset):
    """Dataset per il training della segmentazione con più comuni."""

    def __init__(
        self, comuni: list[ComuneType], patch_size: int = 256, num_patches: int = 1000
    ):
        self.comuni = comuni
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.masks = []
        self.stack_input = []
        self.stack_landslide = []

        print(f"Loading data for {comuni}...")

        for comune in comuni:
            # Generiamo la maschera del dataset
            self.masks.append(generate_dataset_mask(comune))
            # Carichiamo gli stack di dati
            input, landslide = get_segmentation_stack(comune)
            self.stack_input.append(input)
            self.stack_landslide.append(landslide)

    def __len__(self) -> int:
        return self.num_patches

    def __getitem__(self, _) -> Tuple[torch.Tensor, torch.Tensor]:
        # Selezioniamo un comune casuale
        comune_idx = np.random.choice(len(self.comuni))

        input_patch, landslide_patch, patch_mask = get_random_patch(
            self.stack_input[comune_idx],
            self.stack_landslide[comune_idx],
            self.patch_size,
            self.masks[comune_idx],
        )

        # Convertiamo a tensori e assicuriamo il formato corretto
        input_tensor = torch.from_numpy(input_patch).float()
        landslide_tensor = torch.from_numpy(landslide_patch).float()
        patch_mask_tensor = torch.from_numpy(patch_mask)

        # Settiamo i valori NaN a 0
        input_tensor = torch.nan_to_num(input_tensor, nan=0.0)
        landslide_tensor = torch.nan_to_num(landslide_tensor, nan=0.0)

        # Aggiungiamo eventuali augmentazioni
        input_tensor, landslide_tensor = augment_data(
            input_tensor, landslide_tensor, patch_mask_tensor, prob=0.5
        )

        return input_tensor, landslide_tensor


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    eval_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    scheduler: ReduceLROnPlateau,
    num_epochs: int = 20,
) -> Tuple[list[float], list[float], list[float], list[float]]:
    """Addestra il modello di segmentazione."""

    model.train()
    train_losses = []
    val_losses = []
    oa_values = []
    iou_values = []
    best_iou = 0

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
                loss = criterion(outputs, landslide)  # Calcolo della loss

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

        val_loss, val_iou, oa = evaluate_model(
                model, eval_loader, device, criterion
        )

        model.train()

        val_losses.append(val_loss)
        iou_values.append(val_iou)
        oa_values.append(oa)

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {train_loss:.6f}, Validation Loss: {val_loss:.6f}, Validation IoU: {val_iou:.6f}, Overall Accuracy: {oa:.6f}, LR: {current_lr:.2e}"
        )

        # Salvataggio del modello migliore
        if val_iou > best_iou and epoch > 50:  # Non salviamo il primo modello
            best_iou = val_iou
            print(f"New best model found at epoch {epoch+1} with average IoU {best_iou:.6f}")
            torch.save(model.state_dict(), "best_model.pth")

    return train_losses, val_losses, iou_values, oa_values


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    threshold: float = 0.5,
) -> Tuple[float, float, float]:
    """Valuta il modello restituendo (loss media, IoU, OA)."""

    model.eval()
    total_loss_sum = 0.0
    num_samples = 0

    # Accumulatori per metriche
    intersection_sum = 0.0
    union_sum = 0.0
    correct_sum = 0.0
    total_pixels = 0

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

            # Calcolo intersezione, unione e accuratezza -> ricordare che i valori sono binari
            inter = (preds * labels).sum().item()
            union = ((preds + labels) > 0).float().sum().item()
            intersection_sum += inter
            union_sum += union
            correct = (preds == labels).sum().item()
            total = preds.numel()

            correct_sum += correct
            total_pixels += total


    val_loss = total_loss_sum / max(1, num_samples)
    iou = intersection_sum / union_sum if union_sum > 0 else 1.0
    oa = correct_sum / max(1, total_pixels)
    return val_loss, iou, oa


def visualize_results(
    model: nn.Module,
    dataset: SegmentationSingleDataset,
    device: torch.device,
    num_samples: int = 5,
) -> None:
    """Visualizza i risultati del modello."""

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


def seed_everything(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def main():
    """Funzione principale per addestrare e valutare il modello di segmentazione."""

    # Parametri di configurazione
    train_comuni = cast(list[ComuneType], ["Predappio", "Modigliana", "Brisighella"])
    eval_comune = cast(ComuneType, "Casola-Valsenio")

    patch_size = 256
    num_patches = 2000
    batch_size = 8
    num_epochs = 170

    # Dispositivo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Seed per riproducibilità
    seed_everything(42)

    try:
        # Creazione del dataset
        train_dataset = SegmentationMultiDataset(train_comuni, patch_size, num_patches)
        eval_dataset = SegmentationSingleDataset(eval_comune, patch_size, num_patches)

        # Printiamo shape in input e output
        sample_input, sample_landslide = train_dataset[0]
        n_channels_in = sample_input.shape[0]
        n_channels_out = sample_landslide.shape[0]

        print(f"Input channels: {n_channels_in}, Output channels: {n_channels_out}")

        # Creiamo i data loader
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            # Parallelizziamo la generazione dei batch
            num_workers=0,
            worker_init_fn=seed_workers,
            # Velocizziamo il caricamento dei dati su GPU se disponibile
            pin_memory=True if device.type == "cuda" else False,
        )

        eval_loader = DataLoader(
            eval_dataset,
            batch_size=batch_size,
            num_workers=0,
            worker_init_fn=seed_workers,
            pin_memory=True if device.type == "cuda" else False,
        )

        # Creazione
        print("Creating model...")
        model = UNet(n_channels=n_channels_in, n_classes=n_channels_out).to(device)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Parametri totali: {total_params:,}")

        # Loss and optimizer
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.AdamW(model.parameters())
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,  # moltiplica il lr
            patience=5,  # epoche senza miglioramento prima del drop
            threshold=1e-4,  # miglioramento minimo considerato
            min_lr=1e-7,
        )

        # Allenamento
        print("Starting training...")
        train_losses, val_losses, iou_values, oa_values = train_model(
            model, train_loader, eval_loader, criterion, optimizer, device, scheduler, num_epochs
        )

        # Valutazione con metriche
        print("Evaluating model...")
        val_loss, val_iou, val_oa = evaluate_model(
            model, eval_loader, device, criterion
        )
        print(f"Validation Loss: {val_loss:.6f}")
        print(f"Validation IoU: {val_iou:.4f}")
        print(f"Validation OA: {val_oa:.4f}")

        # Plot training losses
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label="Training Loss")
        plt.plot(val_losses, label="Validation Loss")
        plt.title("Losses")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        plt.show()

        # Plot IoU and OA
        plt.figure(figsize=(10, 5))
        plt.plot(iou_values, label="Validation IoU")
        plt.plot(oa_values, label="Validation OA")
        plt.title("IoU and OA")
        plt.xlabel("Epoch")
        plt.ylabel("Score")
        plt.legend()
        plt.grid(True)
        plt.show()

        # Risultati
        visualize_results(model, eval_dataset, device)

    except Exception as e:
        print(f"Error occurred during training: {e}")
        import traceback

        traceback.print_exc()
        pass


if __name__ == "__main__":
    main()
