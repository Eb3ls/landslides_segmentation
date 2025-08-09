import os
import random
import napari
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Tuple
import matplotlib.pyplot as plt
from tqdm import tqdm

# TODO: capire che è dice e se aggiungere pos weight

from data_utils import (
    ComuneType,
    get_segmentation_stack,
    get_random_patch,
    generate_dataset_mask,
)

# Configurazione ambiente
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class DoubleConv(nn.Module):
    """Blocco di doppia convoluzione utilizzato in U-Net."""

    def __init__(self, in_channels: int, out_channels: int):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels, out_channels, kernel_size=3, padding=1
            ),  # La seconda convoluzione mantiene la dimensione dei canali
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


class SegmentationDataset(Dataset):
    """Dataset per il training della segmentazione."""

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


def train_model(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    num_epochs: int = 20,
) -> list:
    """Addestra il modello di segmentazione."""

    model.train()
    losses = []

    for epoch in range(num_epochs):
        num_samples = 0
        epoch_loss_sum = 0.0

        with tqdm(
            dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"
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

        avg_loss = epoch_loss_sum / max(1, num_samples)  # Evita divisione per zero
        losses.append(avg_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.6f}")

    return losses


def evaluate_model(
    model: nn.Module, dataloader: DataLoader, device: torch.device
) -> float:
    """Valuta il modello."""

    model.eval()
    total_loss_sum = 0.0
    num_samples = 0
    criterion = (
        nn.BCEWithLogitsLoss()
    )  # Combinazione numericamente stabile di Sigmoid + Binary Cross Entropy (calcola direttamente la loss)

    with torch.no_grad():
        for data, landslide in dataloader:
            data = data.to(device)
            landslide = landslide.to(device)
            outputs = model(data)
            loss = criterion(outputs, landslide)
            batch_size = data.size(0)
            total_loss_sum += loss.item() * batch_size
            num_samples += batch_size

    return total_loss_sum / max(1, num_samples)


def visualize_results(
    model: nn.Module,
    dataset: SegmentationDataset,
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


def main():
    """Funzione principale per addestrare e valutare il modello di segmentazione."""

    # Parametri di configurazione
    train_comune = "Predappio"
    eval_comune = "Predappio"

    patch_size = 256
    num_patches = 2000
    batch_size = 4
    num_epochs = 10

    # Dispositivo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Seed per riproducibilità
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    try:
        # Creazione del dataset
        train_dataset = SegmentationDataset(train_comune, patch_size, num_patches)
        eval_dataset = SegmentationDataset(eval_comune, patch_size, num_patches)

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
            num_workers=2,
            worker_init_fn=seed_workers,
            # Velocizziamo il caricamento dei dati su GPU se disponibile
            pin_memory=True if device.type == "cuda" else False,
        )

        # Si differenziano i dataloader per il futuro anche ora se identici
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=batch_size,
            num_workers=2,
            worker_init_fn=seed_workers,
            pin_memory=True if device.type == "cuda" else False,
        )

        # Creazione
        print("Creating model...")
        model = UNet(n_channels=n_channels_in, n_classes=n_channels_out).to(device)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Parametri totali: {total_params:,}  (allenabili: {trainable_params:,})")

        # Loss and optimizer con miglioramenti
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.AdamW(model.parameters())

        # Allenamento
        print("Starting training...")
        losses = train_model(
            model, train_loader, criterion, optimizer, device, num_epochs
        )

        # Valutazione
        print("Evaluating model...")
        val_loss = evaluate_model(model, eval_loader, device)
        print(f"Validation Loss: {val_loss:.6f}")

        # Plot training losses
        plt.figure(figsize=(12, 8))
        plt.plot(losses)
        plt.title("Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.show()

        # Risultati
        visualize_results(model, eval_dataset, device)

        # Salvataggio del modello
        save_model(model, "segmentation_model.pth")

    except Exception as e:
        print(f"Error occurred during training: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
