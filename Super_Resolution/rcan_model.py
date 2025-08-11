import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from Super_Resolution.config import ConfigRCAN
from Super_Resolution.model_utils import (
    evaluate_model,
    load_model,
    save_model,
    train_model,
    visualize_predictions,
    save_metrics,
)
from data_utils import (
    SuperResolutionDataset,
)


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

    def __init__(self, config: ConfigRCAN):
        super(RCAN, self).__init__()
        self.in_channels = 4
        self.n_channels = config.model.feature_extraction_channels
        self.reduction = config.model.reduction_channels

        scale = config.model.scale

        self.conv1 = nn.Conv2d(
            self.in_channels, self.n_channels, kernel_size=3, padding=1
        )
        self.groups = nn.ModuleList(
            [
                ResidualGroup(self.n_channels, self.reduction)
                for _ in range(config.model.residual_groups)
            ]
        )
        self.conv2 = nn.Conv2d(
            self.n_channels, self.n_channels, kernel_size=3, padding=1
        )
        self.conv3 = nn.Conv2d(
            self.n_channels, self.n_channels * (scale**2), kernel_size=3, padding=1
        )
        self.pixel_shuffle = nn.PixelShuffle(scale)
        self.conv4 = nn.Conv2d(
            self.n_channels, self.in_channels, kernel_size=3, padding=1
        )

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


# TODO: algoritmo di valutazione differenza tra input e output
# Alcune immagini non sono coerenti, quindi dobbiamo capire come gestirle
def main():
    """Funzione principale per addestrare e valutare il modello di super risoluzione."""

    # Puliamo la memoria CUDA
    torch.cuda.empty_cache()

    config = ConfigRCAN()

    # Dispositivo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Seed per riproducibilità
    torch.manual_seed(config.train.seed)
    np.random.seed(config.train.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.train.seed)

    try:

        # Dataset di valutazione
        test_dataset = SuperResolutionDataset(
            config.test.comune,
            config.model.scale,
            config.model.patch_size,
            config.test.dataset_size,
        )

        # Creazione del modello
        model = RCAN(config).to(device)

        print(f"Parametri: {sum(p.numel() for p in model.parameters())}")

        if config.test.load_model:
            # Caricamento del modello esistente
            model = load_model(config, model, device)
            visualize_predictions(model, test_dataset, device, config)
            return

        # Altrimenti alleniamo il modello

        # Dataset di addestramento
        train_dataset = SuperResolutionDataset(
            config.train.comune,
            config.model.scale,
            config.model.patch_size,
            config.train.dataset_size,
            config.train.augment_data,
        )

        train_loader = DataLoader(
            train_dataset,
            config.train.batch_size,
            num_workers=config.train.workers,
            persistent_workers=True,
        )

        # Allenamento
        print("Starting training...")
        losses = train_model(model, train_loader, device, config)

        # Plottiamo la loss del training logaritmica
        plt.figure(figsize=(12, 8))
        plt.plot(losses)
        plt.title("Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.yscale("log")
        plt.grid(True)
        plt.savefig(f"{config.model.dir_path}{config.model.name}/loss.png")
        plt.show()

        print("Evaluating model...")
        metrics = evaluate_model(model, train_loader, device)
        save_metrics(metrics, config)

        visualize_predictions(model, test_dataset, device, config)

        # Salvataggio del modello
        save_model(model, config)

    except Exception as e:
        print(f"Error occurred during training: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
