import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import napari
import os
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataclasses import asdict
from typing import Optional

from pytorch_msssim import ms_ssim
from torchmetrics.image import (
    PeakSignalNoiseRatio,
)
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from datetime import datetime
from Super_Resolution.config import (
    Config,
    RCANModelConfig,
    Swin2MoseModelConfig,
    MyModelConfig,
    DRCTModelConfig,
)
from Super_Resolution.rcan.rcan_model import RCAN
from Super_Resolution.swin2mose.swin2mose_model import Swin2MoSE
from Super_Resolution.mymodel.mymodel_model import MyModel
from Super_Resolution.myDRCT.DRCT import DRCT
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


def charbonnier_loss(
    x: torch.Tensor, y: torch.Tensor, eps: float = 1e-3
) -> torch.Tensor:
    """Charbonnier (pseudo L1) loss: sqrt((x-y)^2 + eps^2)."""
    diff = x - y
    # Per stabilitá numerica in AMP cast a float32
    if diff.dtype == torch.float16:
        diff32 = diff.float()
        loss = torch.sqrt(diff32 * diff32 + eps * eps).mean()
        return loss.to(diff.dtype)
    return torch.sqrt(diff * diff + eps * eps).mean()


# Cache dei kernel Sobel per evitare ricreazioni continue
_SOBEL_KERNELS: dict[str, torch.Tensor] | None = None


def _get_sobel_kernels(
    device: torch.device, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    global _SOBEL_KERNELS
    if (
        _SOBEL_KERNELS is None
        or _SOBEL_KERNELS["dx"].device != device
        or _SOBEL_KERNELS["dx"].dtype != dtype
    ):
        kx = (
            torch.tensor(
                [[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=dtype, device=device
            )
            / 8.0
        )
        ky = (
            torch.tensor(
                [[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=dtype, device=device
            )
            / 8.0
        )
        _SOBEL_KERNELS = {"dx": kx.view(1, 1, 3, 3), "dy": ky.view(1, 1, 3, 3)}
    return _SOBEL_KERNELS["dx"], _SOBEL_KERNELS["dy"]


def sobel_gradient(t: torch.Tensor) -> torch.Tensor:
    dx_k, dy_k = _get_sobel_kernels(t.device, t.dtype)
    c = t.shape[1]
    t_pad = torch.nn.functional.pad(t, (1, 1, 1, 1), mode="reflect")
    dx = torch.nn.functional.conv2d(t_pad, dx_k.repeat(c, 1, 1, 1), groups=c)
    dy = torch.nn.functional.conv2d(t_pad, dy_k.repeat(c, 1, 1, 1), groups=c)
    grad = torch.sqrt(dx * dx + dy * dy + 1e-12)
    return grad


def gradient_loss(sr: torch.Tensor, hr: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.l1_loss(sobel_gradient(sr), sobel_gradient(hr))


# High-Frequency loss via blur-subtraction: HF(x) = x - (x * b)
_GAUSS_KERNELS: dict[tuple, torch.Tensor] | None = None


def _get_gaussian_kernel2d(
    device: torch.device, dtype: torch.dtype, ksize: int = 5, sigma: float = 1.0
) -> torch.Tensor:
    """Crea o recupera un kernel gaussiano 2D normalizzato (ksize x ksize).
    Usato per separare le componenti a bassa frequenza (blur)."""
    global _GAUSS_KERNELS
    if _GAUSS_KERNELS is None:
        _GAUSS_KERNELS = {}
    key = (device, dtype, int(ksize), float(sigma))
    if key in _GAUSS_KERNELS:
        return _GAUSS_KERNELS[key]

    ax = torch.arange(ksize, device=device, dtype=dtype) - (ksize - 1) / 2
    xx, yy = torch.meshgrid(ax, ax, indexing="ij")
    kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma * sigma))
    kernel = kernel / (kernel.sum() + 1e-12)
    kernel = kernel.view(1, 1, ksize, ksize)
    _GAUSS_KERNELS[key] = kernel
    return kernel


def _gaussian_blur(x: torch.Tensor, ksize: int = 5, sigma: float = 2) -> torch.Tensor:
    k = _get_gaussian_kernel2d(x.device, x.dtype, ksize, sigma)
    c = x.shape[1]
    # Applichiamo lo stesso kernel a ogni canale
    padding = ksize // 2
    x_padded = torch.nn.functional.pad(
        x, (padding, padding, padding, padding), mode="reflect"
    )
    return torch.nn.functional.conv2d(
        x_padded, k.repeat(c, 1, 1, 1), padding=0, groups=c
    )


def high_frequency_map(
    x: torch.Tensor, ksize: int = 5, sigma: float = 1.0
) -> torch.Tensor:
    """Restituisce la componente ad alta frequenza: HF(x) = x - blur(x)."""
    return x - _gaussian_blur(x, ksize, sigma)


def high_frequency_loss(
    sr: torch.Tensor, hr: torch.Tensor, ksize: int = 5, sigma: float = 1.0
) -> torch.Tensor:
    """|| HF(y) - HF(x) ||_1. Enfatizza bordi e texture.
    Args:
        sr: predizione (B,C,H,W)
        hr: ground truth (B,C,H,W)
        ksize: dimensione kernel gaussiano
        sigma: deviazione standard gaussiana
    """
    sr_hf = high_frequency_map(sr, ksize, sigma)
    hr_hf = high_frequency_map(hr, ksize, sigma)
    return torch.nn.functional.l1_loss(sr_hf, hr_hf)


# LPIPS per la loss (cache per device)
_LPIPS_LOSS: LearnedPerceptualImagePatchSimilarity | None = None


def _get_lpips_loss(device: torch.device) -> LearnedPerceptualImagePatchSimilarity:
    """Ottieni o crea LPIPS (VGG) sul device corretto per usarlo come loss."""
    global _LPIPS_LOSS
    if _LPIPS_LOSS is None or _LPIPS_LOSS.device != device:
        _LPIPS_LOSS = LearnedPerceptualImagePatchSimilarity(
            net_type="vgg", normalize=True
        ).to(device)
    return _LPIPS_LOSS


def total_variation_loss(x: torch.Tensor) -> torch.Tensor:
    loss_h = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]).mean()
    loss_w = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]).mean()
    return loss_h + loss_w


def composite_loss(
    sr: torch.Tensor,
    hr: torch.Tensor,
    config: Config,
    moe_loss: float | torch.Tensor = 0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Calcola dinamicamente le componenti di loss in base ai pesi > 0.
    Ritorna total_loss e dizionario componenti (già pesate)."""
    weights = config.train.loss_weights

    comps: dict[str, torch.Tensor] = {}

    # Charbonnier
    if weights.get("charb", 0.0) > 0:
        charb_v_rgb = charbonnier_loss(sr[:, :3], hr[:, :3])
        charb_v_nir = charbonnier_loss(sr[:, 3:], hr[:, 3:])
        avg = 0.7 * charb_v_rgb + 0.3 * charb_v_nir
        comps["charb"] = weights["charb"] * avg

    # High-Frequency loss (blur-subtraction)
    if weights.get("hf", 0.0) > 0:
        hf_v = high_frequency_loss(sr, hr)
        comps["hf"] = weights["hf"] * hf_v

    # Gradient HF (Sobel)
    if weights.get("sobel", 0.0) > 0:
        grad_v = gradient_loss(sr, hr)
        comps["sobel"] = weights["sobel"] * grad_v

    if weights.get("ssim", 0.0) > 0:
        # MS-SSIM richiede [0,1]; clamp solo per questo termine
        ssim_v = 1 - ms_ssim(
            torch.clamp(sr[:, :3, :, :], 0.0, 1.0),
            torch.clamp(hr[:, :3, :, :], 0.0, 1.0),
            data_range=1.0,
            weights=[0.5, 0.5],
        )
        comps["ssim"] = weights["ssim"] * ssim_v

    if weights.get("tv", 0.0) > 0:
        sr_clamp = torch.clamp(sr, 0.0, 1.0)
        comps["tv"] = weights["tv"] * total_variation_loss(sr_clamp)

    # LPIPS (solo RGB, clamped [0,1])
    if weights.get("lpips", 0.0) > 0:
        lpips_module = _get_lpips_loss(sr.device)
        # Usa solo i primi 3 canali (RGB), clamp in [0,1]
        sr_rgb = sr[:, :3, :, :]
        hr_rgb = hr[:, :3, :, :]
        sr_rgb = torch.clamp(sr_rgb, 0.0, 1.0)
        hr_rgb = torch.clamp(hr_rgb, 0.0, 1.0)
        lpips_v = lpips_module(sr_rgb, hr_rgb)
        comps["lpips"] = weights["lpips"] * lpips_v

    # MoE
    if weights.get("moe", 0.0) > 0:
        if not torch.is_tensor(moe_loss):
            moe_loss_tensor = torch.tensor(moe_loss, device=sr.device, dtype=sr.dtype)
        else:
            moe_loss_tensor = moe_loss.to(device=sr.device, dtype=sr.dtype)
        comps["moe"] = weights["moe"] * moe_loss_tensor

    total = (
        torch.stack([v for v in comps.values()]).sum()
        if comps
        else torch.tensor(0.0, device=sr.device, dtype=sr.dtype)
    )

    return total, comps


def train_model(
    model: nn.Module,
    dataloader: DataLoader,
    eval_dataloader: DataLoader,
    device: torch.device,
    config: Config,
    params_override: Optional[list[nn.Parameter]] = None,
) -> dict[str, list[float]]:
    """Addestra il modello di super risoluzione.

    Ritorna un dizionario con le curve per-epoca
    """

    # Gradient Accumulation
    accumulation_steps = config.train.accumulation_steps

    # Loss and optimizer
    criterion = composite_loss
    # Optimizer: Adam senza weight decay
    opt_params = (
        params_override
        if params_override is not None
        else [p for p in model.parameters() if p.requires_grad]
    )
    optimizer = optim.AdamW(opt_params, lr=2e-4, weight_decay=0, betas=(0.9, 0.999))

    total_steps = config.train.epochs * len(dataloader) // accumulation_steps
    milestones = [
        int(0.3 * total_steps),
        int(0.5 * total_steps),
        int(0.75 * total_steps),
        int(0.9 * total_steps),
    ]
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=milestones,
        gamma=0.5,
    )
    steps_per_epoch = len(dataloader) // accumulation_steps

    # scheduler = optim.lr_scheduler.OneCycleLR(
    #     optimizer,
    #     max_lr=2e-4,
    #     epochs=config.train.epochs,
    #     steps_per_epoch=steps_per_epoch,
    #     pct_start=0.2,
    #     anneal_strategy="cos",
    #     div_factor=25.0,  # Lr iniziale = max_lr / div_factor
    #     final_div_factor=1000,  # Lr finale = max_lr / final_div_factor
    #     three_phase=False,  # solo salita e discesa
    #     last_epoch=-1,  # inizia da 0
    #     cycle_momentum=False,  # Gestito da AdamW
    # )

    # warmup_epochs = 0.1 * config.train.epochs
    # warmup_steps = int(warmup_epochs * steps_per_epoch)

    # warmup = optim.lr_scheduler.LinearLR(
    #     optimizer, start_factor=1 / 10, end_factor=1.0, total_iters=warmup_steps
    # )
    # cosine = optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer,
    #     T_max=total_steps - warmup_steps,
    #     eta_min=1e-5,
    # )
    # scheduler = optim.lr_scheduler.SequentialLR(
    #     optimizer,
    #     schedulers=[warmup, cosine],
    #     milestones=[warmup_steps],
    # )

    print("Total updates:", total_steps, "and each epoch:", steps_per_epoch)
    # print("Warmup steps:", warmup_steps, "cosine steps:", total_steps - warmup_steps)

    # Prepara dizionario dinamico delle curve
    active_losses = [k for k, w in config.train.loss_weights.items() if w > 0]
    tracking: dict[str, list[float]] = {"total": []}
    # Aggiungiamo le loss al tracking
    for k in active_losses:
        tracking[k] = []
    tracking["lr"] = []
    tracking["psnr_total"] = []
    tracking["psnr_rgb"] = []
    tracking["psnr_nir"] = []

    model.train()

    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)

    for epoch in range(config.train.epochs):
        accumulators = {k: 0.0 for k in ["total", *active_losses]}
        batch_count = 0

        optimizer.zero_grad(set_to_none=True)

        with tqdm(
            dataloader,
            desc=f"Epoch {epoch+1}/{config.train.epochs}",
            disable=not config.train.show_progress,
        ) as pbar:
            for step, (low_res, high_res) in enumerate(pbar):
                low_res = low_res.to(device, non_blocking=True)
                high_res = high_res.to(device, non_blocking=True)

                # Forward
                outputs = model(low_res)
                if isinstance(outputs, tuple):
                    outputs, moe_loss_val = outputs
                else:
                    moe_loss_val = 0.0

                total_loss, comps = criterion(outputs, high_res, config, moe_loss_val)
                (total_loss / accumulation_steps).backward()

                batch_count += 1

                accumulators["total"] += float(total_loss.detach().item())
                for k in active_losses:
                    accumulators[k] += float(comps[k].detach().item())

                # Se è l'ultimo step di accumulazione, esegui l'update
                do_step = ((step + 1) % accumulation_steps == 0) or (
                    (step + 1) == len(dataloader)
                )
                if do_step:
                    # Clip dei gradienti per stabilitá
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()

                pbar.set_postfix({"tot": f"{total_loss.item():.6f}"})

        # Fine epoca

        tracking["total"].append(accumulators["total"] / batch_count)
        for k in active_losses:
            tracking[k].append(accumulators[k] / batch_count)
        tracking["lr"].append(optimizer.param_groups[0]["lr"])

        comps_log = " | ".join(f"{k}:{tracking[k][-1]:.5f}" for k in active_losses)
        print(
            f"Epoch [{epoch+1}/{config.train.epochs}] | avg tot: {tracking['total'][-1]:.6f} "
            f"| lr: {optimizer.param_groups[0]['lr']:.6e} | {comps_log}"
        )

        # Valutazione ad ogni epoca psnr only
        psnr_total = 0.0
        psnr_rgb = 0.0
        psnr_nir = 0.0

        model.eval()
        with torch.no_grad():
            for low_res, high_res in eval_dataloader:
                low_res = low_res.to(device)
                high_res = high_res.to(device)

                outputs = model(low_res)
                if isinstance(outputs, tuple):
                    outputs, _ = outputs

                psnr_total += psnr_metric(outputs, high_res).item()
                psnr_rgb += psnr_metric(
                    outputs[:, :3, :, :], high_res[:, :3, :, :]
                ).item()
                psnr_nir += psnr_metric(
                    outputs[:, 3:, :, :], high_res[:, 3:, :, :]
                ).item()
        num_batches = len(eval_dataloader)
        tracking["psnr_total"].append(psnr_total / num_batches)
        tracking["psnr_rgb"].append(psnr_rgb / num_batches)
        tracking["psnr_nir"].append(psnr_nir / num_batches)

    return tracking


def evaluate_model(
    model: nn.Module, dataloader: DataLoader, device: torch.device
) -> dict:
    """Valuta il modello con PSNR, SSIM e LPIPS"""

    model.eval()
    psnr_total = 0.0
    psnr_total_no_nir = 0.0
    psnr_only_nir = 0.0
    ssim_total = 0.0
    lpips_total = 0.0
    num_batches = 0

    bicubic_list = []
    nearest_list = []

    # Inizializza le metriche
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_metric = ms_ssim
    lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type="vgg").to(device)

    with torch.no_grad():
        for low_res, high_res in dataloader:
            low_res = low_res.to(device)
            high_res = high_res.to(device)

            outputs = model(low_res)
            if isinstance(outputs, tuple):
                outputs, _ = outputs

            # Clamp solo per metriche
            outputs_c = torch.clamp(outputs, 0.0, 1.0)
            target_c = torch.clamp(high_res, 0.0, 1.0)

            psnr_total += psnr_metric(outputs_c, target_c).item()
            psnr_total_no_nir += psnr_metric(
                outputs_c[:, :3, :, :], target_c[:, :3, :, :]
            ).item()
            psnr_only_nir += psnr_metric(
                outputs_c[:, 3:, :, :], target_c[:, 3:, :, :]
            ).item()
            ssim_total += ssim_metric(
                outputs_c, target_c, data_range=1.0, weights=[0.5, 0.5]
            ).item()

            lpips_total += lpips_metric(
                outputs_c[:, :3, :, :], target_c[:, :3, :, :]
            ).item()

            num_batches += 1

            # Bicubica/nearest di baseline
            low_res_tensor = torch.nn.functional.interpolate(
                low_res, scale_factor=5, mode="bicubic", align_corners=False
            )
            bicubic_list.append(
                psnr_metric(torch.clamp(low_res_tensor, 0, 1), target_c).item()
            )

            low_res_tensor = torch.nn.functional.interpolate(
                low_res,
                scale_factor=5,
                mode="nearest",
            )
            nearest_list.append(
                psnr_metric(torch.clamp(low_res_tensor, 0, 1), target_c).item()
            )

    print(f"Avg Bicubic PSNR: {np.mean(bicubic_list):.4f}")
    print(f"Avg Nearest PSNR: {np.mean(nearest_list):.4f}")
    return {
        "psnr": psnr_total / num_batches,
        "psnr_no_nir": psnr_total_no_nir / num_batches,
        "psnr_only_nir": psnr_only_nir / num_batches,
        "ssim": ssim_total / num_batches,
        "lpips": lpips_total / num_batches,
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
            low_res_upscale = torch.nn.functional.interpolate(
                low_res.unsqueeze(0),
                scale_factor=config.model.scale,
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

            # Converti a numpy per visualizzazione
            low_res_np = low_res.cpu().numpy()
            low_res_upscale_np = low_res_upscale.cpu().numpy()
            high_res_np = high_res.cpu().numpy()
            pred_np = pred.numpy()

            # Clippiamo i valori per evitare valori fuori range
            # Gestirlo nel modello puó peggiorare la qualitá?
            pred_np = np.clip(pred_np, 0, 1)

            # Separiamo RGB+NIR

            # RGB
            low_rgb = low_res_np[:3, :, :]
            low_upscale_rgb = low_res_upscale_np[:3, :, :]
            high_rgb = high_res_np[:3, :, :]
            pred_rgb = pred_np[:3, :, :]
            low_rgb = low_rgb.transpose(1, 2, 0)
            low_upscale_rgb = low_upscale_rgb.transpose(1, 2, 0)
            high_rgb = high_rgb.transpose(1, 2, 0)
            pred_rgb = pred_rgb.transpose(1, 2, 0)

            # NIR
            low_nir = low_res_np[3, :, :]
            low_upscale_nir = low_res_upscale_np[3, :, :]
            high_nir = high_res_np[3, :, :]
            pred_nir = pred_np[3, :, :]

            images.append(
                (
                    low_rgb,
                    low_upscale_rgb,
                    high_rgb,
                    pred_rgb,
                    low_nir,
                    low_upscale_nir,
                    high_nir,
                    pred_nir,
                )
            )

    if config.test.run_napari:
        viewer = napari.Viewer()
        for img_set in images:
            (
                low_rgb,
                low_upscale_rgb,
                high_rgb,
                pred_rgb,
                low_nir,
                low_upscale_nir,
                high_nir,
                pred_nir,
            ) = img_set
            viewer.add_image(low_rgb, name="Low Resolution RGB")
            viewer.add_image(low_upscale_rgb, name="Low Upscale RGB")
            viewer.add_image(high_rgb, name="High Resolution RGB")
            viewer.add_image(pred_rgb, name="Predicted RGB")
            viewer.add_image(low_nir, name="Low Resolution NIR")
            viewer.add_image(low_upscale_nir, name="Low Upscale NIR")
            viewer.add_image(high_nir, name="High Resolution NIR")
            viewer.add_image(pred_nir, name="Predicted NIR")
        napari.run()

    # Salviamo le immagini come file PNG
    for i, img_set in enumerate(images):
        (
            low_rgb,
            low_upscale_rgb,
            high_rgb,
            pred_rgb,
            low_nir,
            low_upscale_nir,
            high_nir,
            pred_nir,
        ) = img_set
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


def _freeze_module(m: nn.Module) -> None:
    for p in m.parameters():
        p.requires_grad = False


def _enable_module(m: nn.Module) -> None:
    for p in m.parameters():
        p.requires_grad = True


def _prepare_finetune_head_only(model: nn.Module) -> list[nn.Parameter]:
    """Freeze everything except the SR tail for DRCT.
    Returns the list of trainable parameters.
    """
    # Freeze everything first
    for p in model.parameters():
        p.requires_grad = False

    if isinstance(model, DRCT) and getattr(model, "upsampler", "") == "test":
        if hasattr(model, "tail") and isinstance(model.tail, nn.Module):
            _enable_module(model.tail)
            if hasattr(model.tail, "antialias") and isinstance(
                model.tail.antialias, nn.Module
            ):
                _freeze_module(model.tail.antialias)
    else:
        if hasattr(model, "conv_last") and isinstance(model.conv_last, nn.Module):
            _enable_module(model.conv_last)

    return [p for p in model.parameters() if p.requires_grad]


def _load_finetune_checkpoint(
    model: nn.Module, ckpt_path: str, device: torch.device
) -> None:
    if ckpt_path and os.path.isfile(ckpt_path):
        state = torch.load(ckpt_path, map_location=device)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            print(f"[Finetune] Missing keys: {len(missing)} (ok if tail changed)")
        if unexpected:
            print(f"[Finetune] Unexpected keys: {len(unexpected)}")
    else:
        print("[Finetune] No valid checkpoint provided; training from current init")


def launch_all(
    config: Config,
) -> None:
    """Funzione principale per addestrare e valutare il modello di super risoluzione."""

    # Puliamo la memoria CUDA
    torch.cuda.empty_cache()

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
            comune=config.test.comune,
            scale=config.model.scale,
            patch_size=config.model.img_size,
            num_patches=config.test.dataset_size,
            for_training=False,
            to_augment=False,
            synthetic_data=config.train.synthetic_data,
        )

        test_loader = DataLoader(
            test_dataset,
            config.test.batch_size,
            num_workers=config.train.workers,
            persistent_workers=True,
            pin_memory=torch.cuda.is_available(),
            worker_init_fn=seed_workers,
        )

        # Creazione del modello
        if isinstance(config.model, RCANModelConfig):
            model = RCAN(config).to(device)  # type: ignore
        elif isinstance(config.model, Swin2MoseModelConfig):
            model = Swin2MoSE(config).to(device)  # type: ignore
        elif isinstance(config.model, MyModelConfig):
            model = MyModel(config).to(device)  # type: ignore
        elif isinstance(config.model, DRCTModelConfig):
            model = DRCT(config).to(device)  # type: ignore
        else:
            raise ValueError(f"Unsupported config type: {type(config.model)}")

        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Parametri: {params}")

        if config.test.load_model:
            # Caricamento del modello esistente
            model = load_model(config, model, device)
            visualize_predictions(model, test_dataset, device, config)
            return

        # Fine-tuning setup
        finetune = getattr(config.train, "finetune", False)
        finetune_scope = getattr(config.train, "finetune_scope", "head")
        finetune_ckpt = getattr(config.train, "finetune_from", "")
        if finetune:
            # Rinomina la cartella di output aggiungendo _finetune
            base_name = config.model.name
            if not str(base_name).endswith("_finetune"):
                config.model.name = f"{base_name}_finetune"
            os.makedirs(f"{config.model.dir_path}{config.model.name}", exist_ok=True)
            print(f"[Finetune] Output dir: {config.model.dir_path}{config.model.name}")

            _load_finetune_checkpoint(model, finetune_ckpt, device)
            if finetune_scope == "head":
                trainable_params = _prepare_finetune_head_only(model)
            else:
                trainable_params = [p for p in model.parameters() if p.requires_grad]
        else:
            # Assicura la cartella esista comunque
            os.makedirs(f"{config.model.dir_path}{config.model.name}", exist_ok=True)
            trainable_params = [p for p in model.parameters() if p.requires_grad]

        # Dataset di addestramento
        train_dataset = SuperResolutionDataset(
            comune=config.test.comune,
            scale=config.model.scale,
            patch_size=config.model.img_size,
            num_patches=config.train.dataset_size,
            for_training=True,
            to_augment=config.train.augment_data,
            synthetic_data=config.train.synthetic_data,
        )

        train_loader = DataLoader(
            train_dataset,
            config.train.batch_size,
            num_workers=config.train.workers,
            persistent_workers=True,
            pin_memory=torch.cuda.is_available(),
            worker_init_fn=seed_workers,
        )

        # Allenamento
        print("Starting training...")
        losses = train_model(
            model,
            train_loader,
            test_loader,
            device,
            config,
            params_override=trainable_params,
        )

        # Plot dinamico solo delle componenti attive
        active_components = [k for k, w in config.train.loss_weights.items() if w > 0]
        n_plots = 1 + len(active_components)
        cols = 3
        rows = (n_plots + cols - 1) // cols
        _, axs = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), sharex=True)
        axs = np.atleast_1d(axs).ravel()

        axs[0].plot(losses["total"])
        axs[0].set_title("Total Loss")
        axs[0].set_xlabel("Epoch")
        axs[0].set_ylabel("Loss")
        axs[0].set_yscale("log")
        axs[0].grid(True)

        last_filled = 0
        for idx, k in enumerate(active_components, start=1):
            axs[idx].plot(losses[k])
            axs[idx].set_title(f"{k.upper()} Loss")
            axs[idx].set_xlabel("Epoch")
            axs[idx].set_ylabel("Loss")
            axs[idx].set_yscale("log")
            axs[idx].grid(True)
            last_filled = idx

        for j in range(last_filled + 1, len(axs)):
            axs[j].axis("off")

        plt.tight_layout()
        plt.savefig(f"{config.model.dir_path}{config.model.name}/loss.png")
        if config.test.run_napari:
            plt.show()

        n_plots = 3
        cols = 3
        rows = (n_plots + cols - 1) // cols
        _, axs = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), sharex=True)
        axs = np.atleast_1d(axs).ravel()
        axs[0].set_title("PSNR")
        axs[0].set_xlabel("Epoch")
        axs[0].set_ylabel("PSNR (dB)")
        axs[0].grid(True)
        axs[0].legend()

        axs[0].plot(losses["psnr_total"], label="Total (RGB+NIR)")
        axs[1].plot(losses["psnr_rgb"], label="RGB")
        axs[1].set_title("PSNR RGB")
        axs[1].set_xlabel("Epoch")
        axs[1].set_ylabel("PSNR (dB)")
        axs[1].grid(True)
        axs[1].legend()

        axs[2].plot(losses["psnr_nir"], label="NIR")
        axs[2].set_title("PSNR NIR")
        axs[2].set_xlabel("Epoch")
        axs[2].set_ylabel("PSNR (dB)")
        axs[2].grid(True)
        axs[2].legend()

        for j in range(3, len(axs)):
            axs[j].axis("off")

        plt.tight_layout()
        plt.savefig(f"{config.model.dir_path}{config.model.name}/psnr.png")
        if config.test.run_napari:
            plt.show()

        print("Evaluating model...")
        metrics = evaluate_model(model, test_loader, device)
        # Salviamo anche le curve di loss attive nel JSON metrics
        metrics["training_curves"] = losses
        metrics["params"] = params
        save_metrics(metrics, config)

        visualize_predictions(model, test_dataset, device, config)

        save_model(model, config)

    except Exception as e:
        print(f"Error occurred during training: {e}")
        import traceback

        traceback.print_exc()
