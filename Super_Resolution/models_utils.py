import json
import random
from typing import Literal
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import napari
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataclasses import asdict

# LPIPS genera warning sul modo in cui carica i pesi
from piq import ssim, psnr, LPIPS, SSIMLoss
from datetime import datetime
from Super_Resolution.config import Config, ConfigRCAN, ConfigSwin2Mose
from Super_Resolution.rcan.rcan_model import RCAN
from Super_Resolution.swin2mose.swin2mose_model import Swin2MoSE
from data_utils import SuperResolutionDataset


def _config_to_dict(config: Config) -> dict:
    """Converte la Config in un dizionario annidato pronto per JSON."""
    return {
        "model": asdict(config.model),
        "train": asdict(config.train),
        "test": asdict(config.test),
    }


def _json_default(o):
    """Gestione di tipi non serializzabili (es. numpy) in JSON."""
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    return str(o)


def save_metrics(metrics_dict: dict, config: Config) -> None:
    """Salva le metriche di valutazione in un file JSON formattato e leggibile."""

    metrics_data = {
        "timestamp": datetime.now().isoformat(),
        "model_name": config.model.name,
        "metrics": metrics_dict,
        "model_config": _config_to_dict(config),
    }

    with open(
        f"{config.model.dir_path}{config.model.name}/metrics.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metrics_data,
            f,
            indent=4,
            sort_keys=True,
            ensure_ascii=False,
            default=_json_default,
        )

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


def ncc_ssim_loss(
    sr: torch.Tensor,
    hr: torch.Tensor,
    config: Config,
    moe_loss: float | torch.Tensor = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Calcola le componenti di loss (NCC, SSIM, MoE) e la loss totale.

    Ritorna:
        total_loss, ncc_component, ssim_component, moe_component
    """
    # NCC come loss (0 buono, 1 cattivo)
    ncc_value = ncc_loss(sr, hr)

    # Clampiamo sr per evitare valori fuori range per SSIM
    sr = torch.clamp(sr, 0, 1)

    # SSIMLoss restituisce (1 - SSIM), quindi è già una loss
    ssim_loss_val = SSIMLoss()(sr, hr)

    # MoE può essere float o tensor: convertiamo a tensor su device corretto
    if not torch.is_tensor(moe_loss):
        moe_loss_tensor = torch.tensor(moe_loss, device=sr.device, dtype=sr.dtype)
    else:
        moe_loss_tensor = moe_loss.to(device=sr.device, dtype=sr.dtype)

    # Applichiamo i pesi
    ncc_component = config.train.loss_weights["ncc"] * ncc_value
    ssim_component = config.train.loss_weights["ssim"] * ssim_loss_val
    moe_component = config.train.loss_weights["moe"] * moe_loss_tensor

    total_loss = ncc_component + ssim_component + moe_component

    return total_loss, ncc_component, ssim_component, moe_component


def train_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    config: Config,
) -> dict[str, list[float]]:
    """Addestra il modello di super risoluzione.

    Ritorna un dizionario con le curve per-epoca: total, ncc, ssim, moe
    """

    # Loss and optimizer
    criterion = ncc_ssim_loss
    # TODO guardare possibili optimizers
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.train.epochs, eta_min=1e-6
    )

    use_amp = device.type == "cuda" and torch.cuda.get_device_capability()[0] >= 7
    if use_amp:
        print("Using Automatic Mixed Precision (AMP) for training.")
    else:
        print("AMP is not supported on this device, using full precision.")
    scaler = torch.GradScaler(enabled=use_amp)

    model.train()
    losses_epoch = {"total": [], "ncc": [], "ssim": [], "moe": []}

    for epoch in range(config.train.epochs):
        epoch_total = 0.0
        epoch_ncc = 0.0
        epoch_ssim = 0.0
        epoch_moe = 0.0

        with tqdm(
            dataloader,
            desc=f"Epoch {epoch+1}/{config.train.epochs}",
            disable=not config.train.show_progress,
        ) as pbar:
            for _, (low_res, high_res) in enumerate(pbar):
                # Spostiamo i tensori sul dispositivo per velocizzare il training
                low_res = low_res.to(device, non_blocking=True)
                high_res = high_res.to(device, non_blocking=True)

                with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
                    outputs = model(low_res)
                    if isinstance(outputs, tuple):
                        # Se il modello restituisce anche la loss di MoE
                        outputs, moe_loss_val = outputs
                    else:
                        moe_loss_val = 0.0

                    total_loss, ncc_c, ssim_c, moe_c = criterion(
                        outputs,
                        high_res,
                        config,
                        moe_loss_val,
                    )

                optimizer.zero_grad()
                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
                scaler.update()

                # Aggiorna metriche batch
                epoch_total += float(total_loss.detach().item())
                epoch_ncc += float(ncc_c.detach().item())
                epoch_ssim += float(ssim_c.detach().item())
                epoch_moe += float(moe_c.detach().item())

                pbar.set_postfix(
                    {
                        "tot": f"{total_loss.item():.6f}",
                        "scale": f"{scaler.get_scale():.2f}",
                    }
                )

        scheduler.step()

        denom = max(1, len(dataloader))
        losses_epoch["total"].append(epoch_total / denom)
        losses_epoch["ncc"].append(epoch_ncc / denom)
        losses_epoch["ssim"].append(epoch_ssim / denom)
        losses_epoch["moe"].append(epoch_moe / denom)

        print(
            f"Epoch [{epoch+1}/{config.train.epochs}] | "
            f"avg tot: {losses_epoch['total'][-1]:.6f} | "
        )

    return losses_epoch


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
            if isinstance(outputs, tuple):
                # Se il modello restituisce anche la loss di Moe
                outputs, _ = outputs

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
            outputs = model(low_res_batch)
            if isinstance(outputs, tuple):
                # Se il modello restituisce anche la loss di Moe
                pred, _ = outputs
            else:
                pred = outputs
            pred = pred.squeeze(0).cpu()

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


def launch_all(model_type: Literal["rcan", "swin2mose"]) -> None:
    """Funzione principale per addestrare e valutare il modello di super risoluzione."""

    # Puliamo la memoria CUDA
    torch.cuda.empty_cache()

    if model_type == "rcan":
        config = ConfigRCAN()
    elif model_type == "swin2mose":
        config = ConfigSwin2Mose()

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
        if model_type == "rcan":
            model = RCAN(config).to(device)  # type: ignore
        elif model_type == "swin2mose":
            model = Swin2MoSE(config).to(device)  # type: ignore

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

        # Plottiamo le loss del training (totale, ncc, ssim, moe)
        fig, axs = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
        keys = ["total", "ncc", "ssim", "moe"]
        titles = ["Total Loss", "NCC Loss", "SSIM Loss", "MoE Loss"]
        for ax, k, title in zip(axs.ravel(), keys, titles):
            ax.plot(losses[k])
            ax.set_title(title)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.set_yscale("log")
            ax.grid(True)
        plt.tight_layout()
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
