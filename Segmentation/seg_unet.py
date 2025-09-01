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

from data_utils import ComuneType, SegmentationMultiDataset, SegmentationSingleDataset

PATCH_SIZE = 256
NUM_PATCHES = 4000
BATCH_SIZE = 16
NUM_EPOCHS = 60


class DoubleConv(nn.Module):
    """Blocco di doppia convoluzione utilizzato in U-Net."""

    def __init__(self, in_channels: int, out_channels: int):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels, out_channels, kernel_size=3, padding=1
            ),  # La seconda convoluzione mantiene la dimensione dei canali
            nn.BatchNorm2d(out_channels),
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


class AttentionGate(nn.Module):
    """Gate di attenzione additiva per filtrare le skip connections."""

    def __init__(
        self, skip_channels: int, gate_channels: int, inter_channels: int
    ):  # inter_channels è il numero di canali in cui proietto skip e gate
        super().__init__()
        self.W_g = nn.Conv2d(
            gate_channels,
            inter_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )
        self.W_x = nn.Conv2d(  # ottengo i pesi processati della skip connection
            skip_channels,
            inter_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

        self.psi = nn.Sequential(  # funzione psi dello scoring dell'attention gate
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(inter_channels, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Sigmoid(),
        )

        self._init_open()
        # Accumulatori aggregati
        self._alpha_sum: float = 0.0
        self._alpha_count: int = 0
        self._alpha_min: float = float("inf")
        self._alpha_max: float = float("-inf")

    def _init_open(self):
        # Piccole proiezioni e bias della psi alto → alpha≈1 all’inizio
        with torch.no_grad():  # per inizializzare i pesi si può disattivare il tracciamento dei gradienti
            nn.init.normal_(self.W_g.weight, std=1e-3)
            if self.W_g.bias is not None:
                nn.init.zeros_(self.W_g.bias)

            nn.init.normal_(self.W_x.weight, std=1e-3)
            if self.W_x.bias is not None:
                nn.init.zeros_(self.W_x.bias)

            # self.psi[1] è la Conv2d(1x1) -> esplicito il tipo per il type checker
            conv = cast(nn.Conv2d, self.psi[1])
            nn.init.zeros_(conv.weight)
            assert conv.bias is not None
            nn.init.constant_(conv.bias, 1.0)

    def forward(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        # x: feature da encoder (skip), g: gating dal decoder (upsampled)
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        alpha = self.psi(g1 + x1)  # output: coefficiente di attenzione in [0,1]
        with torch.no_grad():
            a = alpha.detach()
            self._alpha_stats = (
                float(a.mean().item()),
                float(a.min().item()),
                float(a.max().item()),
            )
            # Accumulo aggregato per tutta la validazione (se abilitato)
            if getattr(self, "_collect_alpha", False):
                self._alpha_sum += float(a.sum().item())
                self._alpha_count += int(a.numel())
                self._alpha_min = min(self._alpha_min, float(a.min().item()))
                self._alpha_max = max(self._alpha_max, float(a.max().item()))
        return x * alpha  # applica il gate

    def reset_alpha_aggregate(self) -> None:
        self._alpha_sum = 0.0
        self._alpha_count = 0
        self._alpha_min = float("inf")
        self._alpha_max = float("-inf")


class Up(nn.Module):
    """Upscaling seguito da doppia convoluzione."""

    def __init__(self, in_channels: int, out_channels: int, attention: bool = False):
        super(Up, self).__init__()

        self.up = nn.ConvTranspose2d(
            in_channels,
            in_channels // 2,
            kernel_size=2,
            stride=2,  # I canali nel decoder dimezzano a ogni up
        )

        self.attention = attention

        if self.attention:
            # Skip e gating hanno canali = in_channels//2 a questo livello di decoder
            inter = max(in_channels // 4, 1)
            self.att_gate = AttentionGate(
                skip_channels=in_channels // 2,
                gate_channels=in_channels // 2,
                inter_channels=inter,
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

        if self.attention:
            x2 = self.att_gate(x2, x1)

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


class AttentionUNet(nn.Module):
    """Modello U-Net per segmentazione."""

    def __init__(self, n_channels: int, n_classes: int):
        super(AttentionUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.in_conv = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        self.up1 = Up(1024, 512, True)
        self.up2 = Up(512, 256, True)
        self.up3 = Up(256, 128, True)
        self.up4 = Up(128, 64, True)
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


def log_attention_stats(model: nn.Module, prefix: str = "") -> None:
    """Stampa statistiche delle mappe di attenzione (mean/min/max) per ogni gate."""
    idx = 0
    for m in model.modules():
        if isinstance(m, AttentionGate):
            # Preferisci statistiche aggregate se presenti
            if hasattr(m, "_alpha_count") and getattr(m, "_alpha_count") > 0:
                mean_ = m._alpha_sum / max(1, m._alpha_count)
                min_ = m._alpha_min
                max_ = m._alpha_max
            elif hasattr(m, "_alpha_stats"):
                mean_, min_, max_ = m._alpha_stats
            else:
                continue
            print(f"{prefix}AttnGate[{idx}]: mean={mean_:.3f} min={min_:.3f} max={max_:.3f}")
            idx += 1


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    eval_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    scheduler: ReduceLROnPlateau,
    num_epochs: int = 20,
) -> Tuple[list[float], list[float], list[float], list[float], list[float]]:
    """Addestra il modello di segmentazione."""

    model.train()
    train_losses = []
    val_losses = []
    oa_values = []
    iou_values = []
    dice_values = []

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

        # Rimosso: logging non aggregato post-epoch (si logga già in evaluate_model)
        # log_attention_stats(model, prefix=f"Epoch {epoch+1}: ")

        # Salvataggio del modello migliore
        if val_iou > best_iou and epoch > 30:
            best_iou = val_iou
            print(
                f"New best model found at epoch {epoch+1} with average IoU {best_iou:.6f}"
            )
            torch.save(model.state_dict(), "best_model.pth")

    return train_losses, val_losses, iou_values, oa_values, dice_values


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

    # Abilita raccolta aggregata alpha sugli AttentionGate e resetta accumulatori
    for m in model.modules():
        if isinstance(m, AttentionGate):
            m.reset_alpha_aggregate()

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

    # Log statistiche aggregate su tutta la validazione e disabilita raccolta
    log_attention_stats(model, prefix="Alpha stats (val) - ")
    for m in model.modules():
        if isinstance(m, AttentionGate):
            m.reset_alpha_aggregate()

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

    # Dispositivo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Seed per riproducibilità
    seed_everything(42)

    try:
        # Creazione del dataset
        train_dataset = SegmentationMultiDataset(train_comuni, PATCH_SIZE, NUM_PATCHES)
        eval_dataset = SegmentationSingleDataset(eval_comune, PATCH_SIZE, NUM_PATCHES)

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

        # Creazione
        print("Creating model...")
        model = AttentionUNet(n_channels=n_channels_in, n_classes=n_channels_out).to(
            device
        )
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
        train_losses, val_losses, iou_values, oa_values, dice_values = train_model(
            model,
            train_loader,
            eval_loader,
            criterion,
            optimizer,
            device,
            scheduler,
            NUM_EPOCHS,
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
        plt.savefig(os.path.join(plots_dir, "losses.png"), dpi=150)
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
        plt.savefig(os.path.join(plots_dir, "oa.png"), dpi=150)
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
        plt.savefig(os.path.join(plots_dir, "dice_iou.png"), dpi=150)
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
