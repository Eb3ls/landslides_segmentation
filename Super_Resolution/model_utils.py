import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import napari
from torch.utils.data import DataLoader
from tqdm import tqdm

# LPIPS genera warning sul modo in cui carica i pesi
from piq import ssim, psnr, LPIPS, SSIMLoss
from datetime import datetime
from Super_Resolution.config import Config
from data_utils import SuperResolutionDataset


def save_metrics(metrics_dict: dict, config: Config) -> None:
    """Salva le metriche di valutazione in un file JSON.

    Args:
        metrics_dict: Dizionario contenente le metriche
        config: Configurazione del modello
    """

    metrics_data = {
        "timestamp": datetime.now().isoformat(),
        "model_name": config.model.name,
        "metrics": metrics_dict,
        "model_config": str(config.model),
    }

    with open(f"{config.model.dir_path}{config.model.name}/metrics.json", "w") as f:
        json.dump(metrics_data, f, indent=4)

    print(f"Metrics saved to {config.model.dir_path}{config.model.name}/metrics.json")


def _cc_single_torch(
    raw_tensor: torch.Tensor, dst_tensor: torch.Tensor
) -> torch.Tensor:
    """
    Compute the Cross-Correlation (CC) metric between two input tensors representing images.

    CC measures the similarity between two images by calculating the cross-correlation coefficient between spectral bands.

    Args:
        raw_tensor (torch.Tensor): The image tensor to be compared.
        dst_tensor (torch.Tensor): The reference image tensor.

    Returns:
        CC (torch.Tensor): The Cross-Correlation (CC) metric score.

    """
    N_spectral = raw_tensor.shape[1]

    # Reshaping fused and reference data
    raw_tensor_reshaped = raw_tensor.view(N_spectral, -1)
    dst_tensor_reshaped = dst_tensor.view(N_spectral, -1)

    # Calculating mean value
    mean_raw = torch.mean(raw_tensor_reshaped, 1).unsqueeze(1)
    mean_dst = torch.mean(dst_tensor_reshaped, 1).unsqueeze(1)

    CC = torch.sum(
        (raw_tensor_reshaped - mean_raw) * (dst_tensor_reshaped - mean_dst), 1
    ) / torch.sqrt(
        torch.sum((raw_tensor_reshaped - mean_raw) ** 2, 1)
        * torch.sum((dst_tensor_reshaped - mean_dst) ** 2, 1)
    )

    CC = torch.mean(CC)

    return CC


def ncc_loss(sr: torch.Tensor, hr: torch.Tensor) -> torch.Tensor:
    """
    Calculate the Normalized Cross-Correlation (NCC) loss between super-resolved and high-resolution images.

    Args:
        sr (torch.Tensor): Super-resolved image tensor.
        hr (torch.Tensor): High-resolution image tensor.

    Returns:
        torch.Tensor: NCC loss value.
    """
    cc_value = _cc_single_torch(sr, hr)
    # Normalizziamo il valore di CC per ottenere una loss
    return 1 - ((cc_value + 1) * 0.5)


def ncc_ssim_loss(sr: torch.Tensor, hr: torch.Tensor) -> torch.Tensor:
    ncc_value = ncc_loss(sr, hr)

    # Clampiamo sr per evitare valori fuori range
    sr = torch.clamp(sr, 0, 1)

    ssim = SSIMLoss()
    ssim_value = ssim(sr, hr)
    return ncc_value + ssim_value


def train_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    config: Config,
) -> list[float]:
    """Addestra il modello di super risoluzione."""

    # Loss and optimizer
    criterion = ncc_ssim_loss
    # TODO guardare possibili optimizers
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.train.epochs, eta_min=1e-6
    )

    model.train()
    losses = []

    for epoch in range(config.train.epochs):
        epoch_loss = 0.0

        with tqdm(
            dataloader,
            desc=f"Epoch {epoch+1}/{config.train.epochs}",
            disable=not config.train.show_progress,
        ) as pbar:
            for _, (low_res, high_res) in enumerate(pbar):
                # Spopstiamo i tensori sul dispositivo per velocizzare il training
                low_res = low_res.to(device, non_blocking=True)
                high_res = high_res.to(device, non_blocking=True)

                outputs = model(low_res)

                # Allenamento
                loss = criterion(outputs, high_res)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                pbar.set_postfix({"Loss": f"{loss.item():.6f}"})

        scheduler.step()

        avg_loss = epoch_loss / len(dataloader)
        losses.append(avg_loss)
        print(f"Epoch [{epoch+1}/{config.train.epochs}], Average Loss: {avg_loss:.6f}")

    return losses


def evaluate_model(
    model: nn.Module, dataloader: DataLoader, device: torch.device
) -> dict:
    """Valuta il modello con PSNR, SSIM, LPIPS e NCC"""

    model.eval()
    psnr_total = 0.0
    ssim_total = 0.0
    lpips_total = 0.0
    ncc_total = 0.0
    num_batches = 0

    with torch.no_grad():
        for low_res, high_res in dataloader:
            low_res = low_res.to(device)
            high_res = high_res.to(device)

            outputs = model(low_res)

            # Clamp dei valori per SSIM (deve essere in [0,1])
            print(
                f"Under 0 values: {torch.sum(outputs < 0)}, Over 1 values: {torch.sum(outputs > 1)}"
            )
            outputs_clamped = torch.clamp(outputs, 0, 1)

            # PSNR: range [0, +inf]
            psnr_total += psnr(outputs_clamped, high_res).item()

            # SSIM: range [0, 1]
            ssim_total += ssim(outputs_clamped, high_res).item()  # type: ignore

            # LPIPS: range [0, 1], RGB only
            lpips_total += LPIPS()(
                outputs_clamped[:, :3, :, :], high_res[:, :3, :, :]
            ).item()

            # NCC: range [-1, 1]
            cc_value = _cc_single_torch(outputs_clamped, high_res).item()
            ncc_total += (cc_value + 1) * 0.5

            num_batches += 1

    return {
        "psnr": psnr_total / num_batches,
        "ssim": ssim_total / num_batches,
        "lpips": lpips_total / num_batches,
        "ncc_loss": ncc_total / num_batches,
    }


def visualize_predictions(
    model: nn.Module,
    dataset: SuperResolutionDataset,
    device: torch.device,
    config: Config,
) -> None:
    """Visualizza i risultati del modello con miglioramenti di contrasto e gestione RGB."""

    model.eval()

    images = []
    with torch.no_grad():
        for i in range(config.test.image_samples):
            # Ottieni un campione
            low_res, high_res = dataset[i]
            low_res_batch = low_res.unsqueeze(0).to(device)

            # Genera predizione
            pred = model(low_res_batch).squeeze(0).cpu()

            # Reupscaliamo l'immagine a bassa risoluzione per avere un confronto diretto
            low_res = torch.nn.functional.interpolate(
                low_res.unsqueeze(0),
                scale_factor=config.model.scale,
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

    if config.test.run_napari:
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
        plt.savefig(config.model.dir_path + config.model.name + f"/sample_{i}.png")


def save_model(model: nn.Module, config: Config) -> None:
    """Salva il modello su disco."""
    path = config.model.dir_path + config.model.name + "/model.pth"
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")


def seed_workers(worker_id: int) -> None:
    """Imposta il seed per i worker di PyTorch."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed + worker_id)
    random.seed(worker_seed + worker_id)
    torch.manual_seed(worker_seed + worker_id)


def load_model(
    config: Config, model_class: nn.Module, device: torch.device
) -> nn.Module:
    """Carica un modello salvato da file."""
    # Inizializziamo il modello
    model = model_class.to(device)
    # Carichiamo i pesi
    path = config.model.dir_path + config.model.name + "/model.pth"
    model.load_state_dict(torch.load(path, map_location=device))

    return model
