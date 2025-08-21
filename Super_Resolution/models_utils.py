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
from dataclasses import asdict

# LPIPS genera warning sul modo in cui carica i pesi
from piq import ssim, psnr, LPIPS, SSIMLoss
from datetime import datetime
from Super_Resolution.config import Config, ConfigMyModel, ConfigRCAN, ConfigSwin2Mose
from Super_Resolution.rcan.rcan_model import RCAN
from Super_Resolution.swin2mose.swin2mose_model import Swin2MoSE
from Super_Resolution.mymodel.mymodel_model import MyModel
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
    """Calcola il modulo del gradiente Sobel per ciascun canale e li concatena.
    Ritorna tensore con stessa shape dell'input (approssimando: modulo normalizzato)."""
    dx_k, dy_k = _get_sobel_kernels(t.device, t.dtype)
    c = t.shape[1]
    # group conv per applicare stesso kernel a ogni canale
    dx = torch.nn.functional.conv2d(t, dx_k.repeat(c, 1, 1, 1), padding=1, groups=c)
    dy = torch.nn.functional.conv2d(t, dy_k.repeat(c, 1, 1, 1), padding=1, groups=c)
    grad = torch.sqrt(torch.clamp(dx * dx + dy * dy, min=0.0) + 1e-12)
    return grad


def gradient_loss(sr: torch.Tensor, hr: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.l1_loss(sobel_gradient(sr), sobel_gradient(hr))


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

    # Controlli range sr/hr (prevenzione valori fuori scala che degradano SSIM/NCC)
    if torch.isnan(sr).any() or torch.isinf(sr).any():
        raise ValueError("NaN/Inf rilevati in sr")
    if torch.isnan(hr).any() or torch.isinf(hr).any():
        raise ValueError("NaN/Inf rilevati in hr")

    # NCC
    if weights.get("ncc", 0.0) > 0:
        ncc_v = ncc_loss(sr, hr)
        comps["ncc"] = weights["ncc"] * ncc_v

    # Clamp per SSIM / Charbonnier (input deve essere [0,1])
    sr_clamped = torch.clamp(sr, 0, 1)

    # SSIM
    if weights.get("ssim", 0.0) > 0:
        ssim_v = SSIMLoss()(sr_clamped, hr)
        comps["ssim"] = weights["ssim"] * ssim_v

    # Charbonnier
    if weights.get("charb", 0.0) > 0:
        charb_v = charbonnier_loss(sr_clamped, hr)
        comps["charb"] = weights["charb"] * charb_v

    # Gradient HF
    if weights.get("grad", 0.0) > 0:
        grad_v = gradient_loss(sr_clamped, hr)
        comps["grad"] = weights["grad"] * grad_v

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
    device: torch.device,
    config: Config,
) -> dict[str, list[float]]:
    """Addestra il modello di super risoluzione.

    Ritorna un dizionario con le curve per-epoca: total, ncc, ssim, moe
    """

    # Loss and optimizer
    criterion = composite_loss
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=4,
        threshold=1e-4,
        min_lr=1e-6,
    )

    use_amp = False  # Disabilitiamo AMP per semplicità/stabilità con nuove loss
    if use_amp:
        print("Using Automatic Mixed Precision (AMP) for training.")
    else:
        print("Training in full precision (AMP disabilitato).")
    scaler = torch.GradScaler(enabled=use_amp)

    # Prepara dizionario dinamico delle curve
    active_components = [k for k, w in config.train.loss_weights.items() if w > 0]
    losses_epoch: dict[str, list[float]] = {"total": []}
    for k in active_components:
        losses_epoch[k] = []

    model.train()

    for epoch in range(config.train.epochs):
        accumulators = {k: 0.0 for k in ["total", *active_components]}

        with tqdm(
            dataloader,
            desc=f"Epoch {epoch+1}/{config.train.epochs}",
            disable=not config.train.show_progress,
        ) as pbar:
            for _, (low_res, high_res) in enumerate(pbar):
                low_res = low_res.to(device, non_blocking=True)
                high_res = high_res.to(device, non_blocking=True)

                with torch.autocast(
                    device_type=device.type, dtype=torch.float16, enabled=use_amp
                ):
                    outputs = model(low_res)
                    if isinstance(outputs, tuple):
                        outputs, moe_loss_val = outputs
                    else:
                        moe_loss_val = 0.0
                    total_loss, comps = criterion(
                        outputs, high_res, config, moe_loss_val
                    )

                optimizer.zero_grad()
                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
                scaler.update()

                accumulators["total"] += float(total_loss.detach().item())
                for k in active_components:
                    if k in comps:
                        accumulators[k] += float(comps[k].detach().item())

                pbar.set_postfix({"tot": f"{total_loss.item():.6f}"})

        denom = max(1, len(dataloader))
        losses_epoch["total"].append(accumulators["total"] / denom)
        for k in active_components:
            losses_epoch[k].append(
                accumulators[k] / denom if accumulators[k] != 0 else 0.0
            )

        scheduler.step(losses_epoch["total"][-1])

        comps_log = " | ".join(
            f"{k}:{losses_epoch[k][-1]:.5f}" for k in active_components
        )
        print(
            f"Epoch [{epoch+1}/{config.train.epochs}] | avg tot: {losses_epoch['total'][-1]:.6f} | lr: {optimizer.param_groups[0]['lr']:.6e} | {comps_log}"
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
        "ncc": ncc_total / num_batches,
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


def launch_all(
    config: ConfigRCAN | ConfigSwin2Mose | ConfigMyModel,
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
        )

        test_loader = DataLoader(
            test_dataset,
            config.test.batch_size,
            num_workers=config.train.workers,
            persistent_workers=True,
        )

        # Creazione del modello
        if isinstance(config, ConfigRCAN):
            model = RCAN(config).to(device)  # type: ignore
        elif isinstance(config, ConfigSwin2Mose):
            model = Swin2MoSE(config).to(device)  # type: ignore
        elif isinstance(config, ConfigMyModel):
            model = MyModel(config).to(device)  # type: ignore

        print(f"Parametri: {sum(p.numel() for p in model.parameters())}")

        if config.test.load_model:
            # Caricamento del modello esistente
            model = load_model(config, model, device)
            visualize_predictions(model, test_dataset, device, config)
            return

        # Altrimenti alleniamo il modello

        # Dataset di addestramento
        train_dataset = SuperResolutionDataset(
            comune=config.train.comune,
            scale=config.model.scale,
            patch_size=config.model.img_size,
            num_patches=config.train.dataset_size,
            to_augment=config.train.augment_data,
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

        # Plot dinamico solo delle componenti attive
        active_components = [k for k, w in config.train.loss_weights.items() if w > 0]
        n_plots = 1 + len(active_components)
        cols = 3
        rows = (n_plots + cols - 1) // cols
        fig, axs = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), sharex=True)
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

        print("Evaluating model...")
        metrics = evaluate_model(model, test_loader, device)
        # Salviamo anche le curve di loss attive nel JSON metrics
        metrics["training_curves"] = losses
        save_metrics(metrics, config)

        visualize_predictions(model, test_dataset, device, config)

        save_model(model, config)

    except Exception as e:
        print(f"Error occurred during training: {e}")
        import traceback

        traceback.print_exc()
