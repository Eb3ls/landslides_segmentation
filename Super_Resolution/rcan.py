import random
import napari
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_utils import (
    SuperResolutionDataset,
)

# Directory per salvare immagini e modelli
save_dir = "Super_Resolution/Trash/"
# Nome del modello da salvare e caricare
save_name = "rcan.pth"


class RCAB(nn.Module):
    """Residual Channel Attention Block"""

    def __init__(self, n_channels: int, reduction: int = 16):
        super(RCAB, self).__init__()
        self.conv1 = nn.Conv2d(n_channels, n_channels, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(n_channels, n_channels, kernel_size=3, padding=1)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv3 = nn.Conv2d(n_channels, n_channels // reduction, kernel_size=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(n_channels // reduction, n_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        # Primo blocco
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)

        # Channel attention
        attention = self.global_avg_pool(out)
        attention = self.conv3(attention)
        attention = self.relu2(attention)
        attention = self.conv4(attention)
        attention = self.sigmoid(attention)

        # Applichiamo il peso
        out = out * attention

        # Residual connection
        out = out + residual

        return out


class ResidualGroup(nn.Module):
    """Gruppo di blocchi Residual Channel Attention Block"""

    def __init__(self, n_channels: int, reduction: int = 16, n_blocks: int = 4):
        super(ResidualGroup, self).__init__()
        self.blocks = nn.Sequential(
            *[RCAB(n_channels, reduction) for _ in range(n_blocks)]
        )
        self.conv = nn.Conv2d(n_channels, n_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.blocks(x)
        out = self.conv(out)
        # Residual connection
        out = out + residual
        return out


class RCAN(nn.Module):
    """Residual Channel Attention Network"""

    def __init__(
        self,
        in_channels: int,
        n_channels: int,
        reduction: int = 16,
        # Numero di blocchi Residual Group
        n_groups: int = 10,
    ):
        super(RCAN, self).__init__()
        self.in_channels = in_channels
        self.n_channels = n_channels
        self.reduction = reduction

        scale = 5

        self.conv1 = nn.Conv2d(in_channels, n_channels, kernel_size=3, padding=1)
        self.groups = nn.ModuleList(
            [ResidualGroup(n_channels, reduction) for _ in range(n_groups)]
        )
        self.conv2 = nn.Conv2d(n_channels, n_channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(
            n_channels, n_channels * (scale**2), kernel_size=3, padding=1
        )
        self.pixel_shuffle = nn.PixelShuffle(scale)
        self.conv4 = nn.Conv2d(n_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # Feature extraction
        out = self.conv1(x)
        feat = out

        # RIR (Residual In Residual)
        for group in self.groups:
            out = group(out)
        out = self.conv2(out)

        # Residual connection
        out = out + feat

        out = self.conv3(out)
        out = self.pixel_shuffle(out)
        out = self.conv4(out)

        return out


def train_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    num_epochs: int = 10,
    show_progress: bool = True,
) -> list:
    """Addestra il modello di super risoluzione."""

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=1e-6
    )

    model.train()
    losses = []

    for epoch in range(num_epochs):
        epoch_loss = 0.0

        with tqdm(
            dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", disable=not show_progress
        ) as pbar:
            for _, (low_res, high_res) in enumerate(pbar):
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

        scheduler.step()

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
    run_napari: bool = True,
) -> None:
    """Visualizza i risultati del modello con miglioramenti di contrasto e gestione RGB."""

    model.eval()

    images = []
    with torch.no_grad():
        for i in range(num_samples):
            # Ottieni un campione
            low_res, high_res = dataset[i]
            low_res_batch = low_res.unsqueeze(0).to(device)

            # Genera predizione
            pred = model(low_res_batch).squeeze(0).cpu()

            # Reupscaliamo l'immagine a bassa risoluzione per avere un confronto diretto
            low_res = torch.nn.functional.interpolate(
                low_res.unsqueeze(0),
                scale_factor=5,
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

            # Converti a numpy per visualizzazione
            low_res_np = low_res.cpu().numpy()
            high_res_np = high_res.cpu().numpy()
            pred_np = pred.numpy()

            # Clippiamo i valori per evitare valori fuori range
            # Gestirlo nel modello puó peggiorare la qualitá?
            pred_np = np.clip(pred_np, 0, 1)

            # Separiamo RGB+NIR

            # RGB
            low_rgb = low_res_np[:3, :, :]
            high_rgb = high_res_np[:3, :, :]
            pred_rgb = pred_np[:3, :, :]
            low_rgb = low_rgb.transpose(1, 2, 0)
            high_rgb = high_rgb.transpose(1, 2, 0)
            pred_rgb = pred_rgb.transpose(1, 2, 0)

            # NIR
            low_nir = low_res_np[3, :, :]
            high_nir = high_res_np[3, :, :]
            pred_nir = pred_np[3, :, :]

            images.append((low_rgb, high_rgb, pred_rgb, low_nir, high_nir, pred_nir))

    if run_napari:
        viewer = napari.Viewer()
        for img_set in images:
            low_rgb, high_rgb, pred_rgb, low_nir, high_nir, pred_nir = img_set
            viewer.add_image(low_rgb, name="Low Resolution RGB")
            viewer.add_image(high_rgb, name="High Resolution RGB")
            viewer.add_image(pred_rgb, name="Predicted RGB")
            viewer.add_image(low_nir, name="Low Resolution NIR")
            viewer.add_image(high_nir, name="High Resolution NIR")
            viewer.add_image(pred_nir, name="Predicted NIR")
        napari.run()

    # Salviamo le immagini come file PNG
    for i, img_set in enumerate(images):
        low_rgb, high_rgb, pred_rgb, low_nir, high_nir, pred_nir = img_set
        plt.figure(figsize=(15, 10))

        plt.subplot(2, 3, 1)
        plt.imshow(low_rgb)
        plt.title("Low Resolution RGB")
        plt.axis("off")

        plt.subplot(2, 3, 2)
        plt.imshow(high_rgb)
        plt.title("High Resolution RGB")
        plt.axis("off")

        plt.subplot(2, 3, 3)
        plt.imshow(pred_rgb)
        plt.title("Predicted RGB")
        plt.axis("off")

        plt.subplot(2, 3, 4)
        plt.imshow(low_nir, cmap="gray")
        plt.title("Low Resolution NIR")
        plt.axis("off")

        plt.subplot(2, 3, 5)
        plt.imshow(high_nir, cmap="gray")
        plt.title("High Resolution NIR")
        plt.axis("off")

        plt.subplot(2, 3, 6)
        plt.imshow(pred_nir, cmap="gray")
        plt.title("Predicted NIR")
        plt.axis("off")

        plt.tight_layout()
        plt.savefig(f"{save_dir}/super_resolution_sample_{i}.png")


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


def load_model(path: str, model_class: nn.Module, device: torch.device) -> nn.Module:
    """Carica un modello salvato da file."""
    # Inizializziamo il modello
    model = model_class.to(device)
    # Carichiamo i pesi
    model.load_state_dict(torch.load(path, map_location=device))

    return model


# TODO: algoritmo di valutazione differenza tra input e output
# Alcune immagini non sono coerenti, quindi dobbiamo capire come gestirle
def main():
    """Funzione principale per addestrare e valutare il modello di super risoluzione."""

    # Puliamo la memoria CUDA
    torch.cuda.empty_cache()

    # Impostare True per caricare il modello esistente e valutarlo
    to_load_model = True

    # Parametri di configurazione
    train_comune = "Predappio"
    eval_comune = "Modigliana"

    scale = 5
    patch_size = 128
    num_patches = 100
    batch_size = 8
    num_epochs = 5
    num_groups = 20
    run_napari = True
    show_progress = True

    # Dispositivo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Seed per riproducibilità
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    try:

        # Dataset di valutazione
        eval_dataset = SuperResolutionDataset(
            eval_comune, scale, patch_size, num_patches
        )

        # Creazione del modello
        model = RCAN(
            in_channels=4,
            n_channels=64,
            n_groups=num_groups,
        ).to(device)

        print(sum(p.numel() for p in model.parameters()))

        if to_load_model:
            # Caricamento del modello esistente
            model = load_model(save_dir + save_name, model, device)
            visualize_results(model, eval_dataset, device)
            return

        # Altrimenti alleniamo il modello

        # Dataset di addestramento
        train_dataset = SuperResolutionDataset(
            train_comune, scale, patch_size, num_patches, to_augment=True
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            num_workers=4,
            persistent_workers=True,
        )

        # Allenamento
        print("Starting training...")
        losses = train_model(model, train_loader, device, num_epochs, show_progress)

        # Plottiamo la loss del training logaritmica
        plt.figure(figsize=(12, 8))
        plt.plot(losses)
        plt.title("Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.yscale("log")
        plt.grid(True)
        plt.savefig(f"{save_dir}super_resolution_training_loss.png")
        plt.show()

        print("Evaluating model...")
        val_loss = evaluate_model(model, train_loader, device)
        print(f"Validation Loss: {val_loss:.6f}")

        visualize_results(model, eval_dataset, device, run_napari=run_napari)

        # Salvataggio del modello
        save_model(model, save_dir + save_name)

    except Exception as e:
        print(f"Error occurred during training: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
