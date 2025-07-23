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

from data_utils import (
    ComuneType,
    get_super_resolution_stack,
    get_random_patch,
    generate_dataset_mask,
)


class DoubleConv(nn.Module):
    """Blocco di doppia convoluzione utilizzato in U-Net."""

    def __init__(self, in_channels: int, out_channels: int):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling con maxpool seguita da doppia convoluzione."""

    def __init__(self, in_channels: int, out_channels: int):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2), DoubleConv(in_channels, out_channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling seguita da doppia convoluzione."""

    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True):
        super(Up, self).__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)

        # Input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = nn.functional.pad(
            x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2]
        )

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
    """Modello U-Net per super risoluzione."""

    def __init__(self, n_channels: int, n_classes: int, bilinear: bool = True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


# TODO: Usare dataset di piú comuni
class SuperResolutionDataset(Dataset):
    """Dataset per il training della super risoluzione."""

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
        _, self.stack_post = get_super_resolution_stack(comune)

    def __len__(self) -> int:
        return self.num_patches

    def __getitem__(self, _) -> Tuple[torch.Tensor, torch.Tensor]:

        (low_res_patch, high_res_patch), _ = get_random_patch(
            self.stack_post, self.patch_size, self.mask
        )

        # Convertiamo a tensori e assicuriamo il formato corretto
        low_res_tensor = torch.from_numpy(low_res_patch).float()
        high_res_tensor = torch.from_numpy(high_res_patch).float()

        # Settiamo i valori NaN a 0
        low_res_tensor = torch.nan_to_num(low_res_tensor, nan=0.0)
        high_res_tensor = torch.nan_to_num(high_res_tensor, nan=0.0)

        return low_res_tensor, high_res_tensor


def train_model(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    num_epochs: int = 10,
) -> list:
    """Addestra il modello di super risoluzione."""

    model.train()
    losses = []

    for epoch in range(num_epochs):
        epoch_loss = 0.0

        with tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}") as pbar:
            for batch_idx, (low_res, high_res) in enumerate(pbar):
                low_res = low_res.to(device)
                high_res = high_res.to(device)

                # Zero gradients
                optimizer.zero_grad()

                # Forward pass
                outputs = model(low_res)
                loss = criterion(outputs, high_res)

                # Backward pass
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                pbar.set_postfix({"Loss": f"{loss.item():.6f}"})

        avg_loss = epoch_loss / len(dataloader)
        losses.append(avg_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.6f}")

    return losses


def evaluate_model(
    model: nn.Module, dataloader: DataLoader, device: torch.device
) -> float:
    """Valuta il modello."""

    model.eval()
    total_loss = 0.0
    criterion = nn.MSELoss()

    with torch.no_grad():
        for low_res, high_res in dataloader:
            low_res = low_res.to(device)
            high_res = high_res.to(device)

            outputs = model(low_res)
            loss = criterion(outputs, high_res)
            total_loss += loss.item()

    return total_loss / len(dataloader)


def visualize_results(
    model: nn.Module,
    dataset: SuperResolutionDataset,
    device: torch.device,
    num_samples: int = 3,
) -> None:
    """Visualizza i risultati del modello con miglioramenti di contrasto e gestione RGB."""

    model.eval()
    viewer = napari.Viewer()

    with torch.no_grad():
        for i in range(num_samples):
            # Ottieni un campione
            low_res, high_res = dataset[i]
            low_res_batch = low_res.unsqueeze(0).to(device)

            # Genera predizione
            pred = model(low_res_batch).squeeze(0).cpu()

            # Converti a numpy per visualizzazione
            low_res_np = low_res.cpu().numpy()
            high_res_np = high_res.cpu().numpy()
            pred_np = pred.numpy()

            # Aggiungiamo i layer a napari RGB+NIR separati
            low_rgb = low_res_np[:3, :, :]  # RGB
            high_rgb = high_res_np[:3, :, :]  # RGB
            pred_rgb = pred_np[:3, :, :]  # RGB
            low_rgb = low_rgb.transpose(1, 2, 0)
            high_rgb = high_rgb.transpose(1, 2, 0)
            pred_rgb = pred_rgb.transpose(1, 2, 0)

            low_nir = low_res_np[3:, :, :]  # NIR
            high_nir = high_res_np[3:, :, :]  # NIR
            pred_nir = pred_np[3:, :, :]  # NIR

            viewer.add_image(low_rgb, name=f"Low RGB")
            viewer.add_image(high_rgb, name=f"High RGB")
            viewer.add_image(pred_rgb, name=f"Predicted RGB")
            viewer.add_image(low_nir, name=f"Low NIR")
            viewer.add_image(high_nir, name=f"High NIR")
            viewer.add_image(pred_nir, name=f"Predicted NIR")

    napari.run()


def save_model(model: nn.Module, path: str) -> None:
    """Salva il modello su disco."""
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")


def seed_workers(worker_id: int) -> None:
    """Imposta il seed per i worker di PyTorch."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed + worker_id)
    random.seed(worker_seed + worker_id)
    torch.manual_seed(worker_seed + worker_id)


def main():
    """Funzione principale per addestrare e valutare il modello di super risoluzione."""

    # Parametri di configurazione
    train_comune = "Predappio"
    eval_comune = "Predappio"

    patch_size = 128
    num_patches = 500
    batch_size = 16
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
        train_dataset = SuperResolutionDataset(train_comune, patch_size, num_patches)
        eval_dataset = SuperResolutionDataset(eval_comune, patch_size, num_patches)

        # Printiamo shape in input e output
        sample_low, sample_high = train_dataset[0]
        n_channels_in = sample_low.shape[0]
        n_channels_out = sample_high.shape[0]

        print(f"Input channels: {n_channels_in}, Output channels: {n_channels_out}")

        # Creiamo i data loader
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            # Parallelizziamo la generazione dei batch
            num_workers=4,
            worker_init_fn=seed_workers,
            # Velocizziamo il caricamento dei dati su GPU se disponibile
            pin_memory=True if device.type == "cuda" else False,
            persistent_workers=True,
        )

        # Creazione del modello
        print("Creating model...")
        model = UNet(n_channels=n_channels_in, n_classes=n_channels_out).to(device)

        # Loss and optimizer con miglioramenti
        criterion = nn.MSELoss()
        optimizer = optim.AdamW(model.parameters())

        # Allenamento
        print("Starting training...")
        losses = train_model(
            model, train_loader, criterion, optimizer, device, num_epochs
        )

        # Valutazione
        print("Evaluating model...")
        val_loss = evaluate_model(model, train_loader, device)
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
        save_model(model, "super_resolution_model.pth")

    except Exception as e:
        print(f"Error occurred during training: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
