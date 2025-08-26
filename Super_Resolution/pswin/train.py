"""
Esempio di integrazione del modello multi-scale nel sistema esistente.

Questo script mostra come utilizzare il modello progressivo con il framework esistente.
"""

import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import napari
from torch.utils.data import Dataset, DataLoader
from torch import Tensor
from typing import Dict, List, Tuple
from tqdm import tqdm
from torchmetrics.image import (
    PeakSignalNoiseRatio,
    StructuralSimilarityIndexMeasure,
)
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

# Ensure repo root is in path
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from data_utils import SuperResolutionDataset
from Super_Resolution.config import ConfigPSWin
from Super_Resolution.models_utils import (
    save_metrics,
    save_model,
    seed_workers,
    composite_loss,
)
from Super_Resolution.pswin.pswin_model import MultiScaleProgressiveModel


class MultiScaleTargetsDataset(Dataset):
    """Wrapper per dataset che genera target multi-scale."""

    def __init__(self, base_dataset):
        super().__init__()
        self.base_dataset = base_dataset

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        low_res, high_res = self.base_dataset[idx]
        # Generiamo target multi-scale secondo la logica del modello
        targets = self.prepare_multi_scale_targets(high_res.unsqueeze(0))
        # Rimuovi la dimensione del batch dai target
        targets = {k: v.squeeze(0) for k, v in targets.items()}
        return low_res, targets

    @staticmethod
    def prepare_multi_scale_targets(hr_image: Tensor) -> Dict[str, Tensor]:
        """
        Prepare multi-scale targets from HR image.

        Args:
            hr_image: High resolution image [B, C, H, W]

        Returns:
            Dictionary with downscaled targets for each supervision stage
        """
        # Target for stage 1 (2x): downsample HR by 2.5x
        target_stage_1 = F.interpolate(
            hr_image, scale_factor=1 / 2.5, mode="bicubic", align_corners=False
        )

        # Clamp nel range [0, 1]
        target_stage_1 = torch.clamp(target_stage_1, 0, 1)

        # Target for stage 2 (4x): downsample HR by 1.25x
        target_stage_2 = F.interpolate(
            hr_image, scale_factor=1 / 1.25, mode="bicubic", align_corners=False
        )

        # Clamp nel range [0, 1]
        target_stage_2 = torch.clamp(target_stage_2, 0, 1)

        # Target for final stage (5x): original HR
        target_final = hr_image

        return {
            "stage_1": target_stage_1,
            "stage_2": target_stage_2,
            "final": target_final,
        }


def multi_scale_progressive_loss(
    predictions: Dict[str, Tensor],
    targets: Dict[str, Tensor],
    config: ConfigPSWin,
    stage_weights: List[float],
) -> Tuple[Tensor, Dict[str, Tensor]]:
    """
    Compute multi-scale progressive loss usando il sistema composite_loss.

    Args:
        predictions: Dictionary with model outputs for each stage
        targets: Dictionary with ground truth targets for each stage
        stage_weights: Weights for each supervision stage [stage_1, stage_2, final]
        config: Configuration object for loss weights

    Returns:
        total_loss: Weighted sum of all stage losses
        loss_components: Dictionary with individual stage losses per stage
    """
    loss_components: Dict[str, Tensor] = {}
    total_loss = torch.tensor(0.0, device=next(iter(predictions.values())).device)

    # Compute losses for each stage
    stages = ["stage_1", "stage_2", "final"]

    for i, stage in enumerate(stages):
        if stage in predictions and stage in targets:
            # Usa composite_loss per ogni stage
            stage_loss, stage_comps = composite_loss(
                predictions[stage], targets[stage], config
            )

            loss_components[f"{stage}"] = stage_loss
            total_loss += stage_weights[i] * stage_loss

            # Aggiungi anche le componenti individuali per logging dettagliato
            for comp_name, comp_value in stage_comps.items():
                loss_components[f"{stage}_{comp_name}"] = comp_value

    return total_loss, loss_components


def train_multi_scale_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    config: ConfigPSWin,
) -> dict[str, list[float]]:
    """
    Addestra il modello multi-scale progressivo.

    Args:
        model: MultiScaleProgressiveModel
        dataloader: DataLoader con MultiScaleTargetsDataset
        device: Device di computazione
        config: Configurazione del modello

    Returns:
        Dictionary con le curve di loss per epoca: total, stage_1, stage_2, final
    """

    # Optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=4,
        threshold=1e-4,
        min_lr=1e-6,
    )

    use_amp = False  # Disabilitiamo AMP per stabilità con multi-scale loss
    scaler = torch.GradScaler(enabled=use_amp)

    # Training curves per ogni stage
    losses_epoch = {"total": [], "stage_1": [], "stage_2": [], "final": []}

    model.train()

    for epoch in range(config.train.epochs):
        epoch_losses = {"total": 0.0, "stage_1": 0.0, "stage_2": 0.0, "final": 0.0}

        with tqdm(
            dataloader,
            desc=f"Epoch {epoch+1}/{config.train.epochs}",
            disable=not config.train.show_progress,
        ) as pbar:
            for _, (low_res, targets) in enumerate(pbar):
                low_res = low_res.to(device, non_blocking=True)

                targets = {
                    k: v.to(device, non_blocking=True) for k, v in targets.items()
                }

                with torch.autocast(
                    device_type=device.type, dtype=torch.float16, enabled=use_amp
                ):
                    # Forward pass con output multi-scale
                    predictions = model(low_res, return_intermediates=True)

                    # Compute multi-scale loss
                    total_loss, loss_components = multi_scale_progressive_loss(
                        predictions,
                        targets,
                        config,
                        config.model.multiscale_weights,
                    )

                # Backward pass
                optimizer.zero_grad()
                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
                scaler.update()

                # Accumula losses
                epoch_losses["total"] += float(total_loss.detach().item())
                for stage in ["stage_1", "stage_2", "final"]:
                    if stage in loss_components:
                        epoch_losses[stage] += float(
                            loss_components[stage].detach().item()
                        )

                pbar.set_postfix({"tot": f"{total_loss.item():.6f}"})

        # Average losses for epoch
        num_batches = len(dataloader)
        for key in epoch_losses:
            losses_epoch[key].append(epoch_losses[key] / num_batches)

        scheduler.step(losses_epoch["total"][-1])

        # Log delle componenti
        print(
            f"Epoch [{epoch+1}/{config.train.epochs}] | "
            f"Total: {losses_epoch['total'][-1]:.6f} | "
            f"Stage1: {losses_epoch['stage_1'][-1]:.6f} | "
            f"Stage2: {losses_epoch['stage_2'][-1]:.6f} | "
            f"Final: {losses_epoch['final'][-1]:.6f} | "
            f"LR: {optimizer.param_groups[0]['lr']:.6e}"
        )

    return losses_epoch


def evaluate_multi_scale_model(
    model: nn.Module, dataloader: DataLoader, device: torch.device
) -> dict:
    """
    Valuta il modello multi-scale con PSNR, SSIM, LPIPS per ogni stage.

    Args:
        model: MultiScaleProgressiveModel
        dataloader: DataLoader con MultiScaleTargetsDataset
        device: Device di computazione

    Returns:
        Dictionary con metriche per ogni stage
    """

    model.eval()
    # Importiamo qui per evitare problemi di import

    # Metrics for each stage
    metrics = {
        "stage_1_psnr": 0.0,
        "stage_1_ssim": 0.0,
        "stage_2_psnr": 0.0,
        "stage_2_ssim": 0.0,
        "final_psnr": 0.0,
        "final_ssim": 0.0,
        "final_lpips": 0.0,
    }

    num_batches = 0

    # Inizializza le metriche
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_metric = StructuralSimilarityIndexMeasure().to(device)
    lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type="vgg").to(device)

    with torch.no_grad():
        for low_res, targets in dataloader:
            low_res = low_res.to(device)

            targets = {k: v.to(device) for k, v in targets.items()}

            # Forward pass con output multi-scale
            predictions = model(low_res, return_intermediates=True)

            # Compute metrics for each stage
            stages = ["stage_1", "stage_2", "final"]
            for stage in stages:
                if stage in predictions and stage in targets:
                    pred_clamped = torch.clamp(predictions[stage], 0, 1)
                    target = targets[stage]

                    # PSNR per tutti gli stage
                    metrics[f"{stage}_psnr"] += psnr_metric(pred_clamped, target).item()

                    # SSIM per tutti gli stage
                    ssim_val = ssim_metric(pred_clamped, target).item()
                    metrics[f"{stage}_ssim"] += float(ssim_val)

                    # LPIPS solo per lo stage finale (RGB channels)
                    if stage == "final":
                        lpips_val = lpips_metric(
                            pred_clamped[:, :3, :, :], target[:, :3, :, :]
                        )
                        metrics[f"{stage}_lpips"] += lpips_val.item()

            num_batches += 1

    # Average metrics
    for key in metrics:
        metrics[key] /= num_batches

    return metrics


def visualize_multi_scale_predictions(
    model: nn.Module,
    dataset: MultiScaleTargetsDataset,
    device: torch.device,
    config: ConfigPSWin,
) -> None:
    """Visualizza i risultati del modello multi-scale per tutti gli stage."""

    model.eval()

    images = []
    with torch.no_grad():
        for i in range(config.test.image_samples):
            # Ottieni un campione
            low_res, targets = dataset[i]
            low_res_batch = low_res.unsqueeze(0).to(device)

            # Genera predizioni multi-scale
            predictions = model(low_res_batch, return_intermediates=True)

            # Prendi le predizioni per ogni stage
            pred_stage_1 = predictions.get("stage_1", None)
            pred_stage_2 = predictions.get("stage_2", None)
            pred_final = predictions.get("final", None)

            # Usa i target corrispondenti
            target_stage_1 = targets.get("stage_1", None)
            target_stage_2 = targets.get("stage_2", None)
            target_final = targets.get("final", None)

            # Converti a CPU e numpy per visualizzazione
            def to_numpy(tensor):
                if tensor is not None:
                    return (
                        tensor.squeeze(0).cpu().numpy()
                        if tensor.dim() == 4
                        else tensor.cpu().numpy()
                    )
                return None

            # Converti tutti i tensori
            low_res_np = to_numpy(low_res)
            pred_1_np = to_numpy(pred_stage_1)
            pred_2_np = to_numpy(pred_stage_2)
            pred_final_np = to_numpy(pred_final)
            target_1_np = to_numpy(target_stage_1)
            target_2_np = to_numpy(target_stage_2)
            target_final_np = to_numpy(target_final)

            # Clipping per evitare valori fuori range
            if pred_final_np is not None:
                pred_final_np = np.clip(pred_final_np, 0, 1)

            images.append(
                {
                    "low_res": low_res_np,
                    "pred_stage_1": pred_1_np,
                    "pred_stage_2": pred_2_np,
                    "pred_final": pred_final_np,
                    "target_stage_1": target_1_np,
                    "target_stage_2": target_2_np,
                    "target_final": target_final_np,
                }
            )

    # Visualizzazione con napari se abilitata
    if config.test.run_napari:
        viewer = napari.Viewer()
        for i, img_data in enumerate(images):
            # Solo RGB (primi 3 canali) per ogni immagine
            if img_data["low_res"] is not None:
                viewer.add_image(
                    img_data["low_res"][:3].transpose(1, 2, 0),
                    name=f"Sample {i} - LR RGB",
                )
            if img_data["pred_stage_1"] is not None:
                viewer.add_image(
                    np.clip(img_data["pred_stage_1"][:3].transpose(1, 2, 0), 0, 1),
                    name=f"Sample {i} - Pred Stage 1 RGB",
                )
            if img_data["target_stage_1"] is not None:
                viewer.add_image(
                    img_data["target_stage_1"][:3].transpose(1, 2, 0),
                    name=f"Sample {i} - Target Stage 1 RGB",
                )
            if img_data["pred_stage_2"] is not None:
                viewer.add_image(
                    np.clip(img_data["pred_stage_2"][:3].transpose(1, 2, 0), 0, 1),
                    name=f"Sample {i} - Pred Stage 2 RGB",
                )
            if img_data["target_stage_2"] is not None:
                viewer.add_image(
                    img_data["target_stage_2"][:3].transpose(1, 2, 0),
                    name=f"Sample {i} - Target Stage 2 RGB",
                )
            if img_data["pred_final"] is not None:
                viewer.add_image(
                    img_data["pred_final"][:3].transpose(1, 2, 0),
                    name=f"Sample {i} - Pred Final RGB",
                )
            if img_data["target_final"] is not None:
                viewer.add_image(
                    img_data["target_final"][:3].transpose(1, 2, 0),
                    name=f"Sample {i} - Target Final RGB",
                )
        napari.run()

    # Salva immagini come file PNG
    for i, img_data in enumerate(images):
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))

        # Prima riga: Low Res + Predizioni per ogni stage (solo RGB)
        if img_data["low_res"] is not None:
            axes[0, 0].imshow(img_data["low_res"][:3].transpose(1, 2, 0))
            axes[0, 0].set_title("Low Resolution RGB")
            axes[0, 0].axis("off")

        if img_data["pred_stage_1"] is not None:
            axes[0, 1].imshow(
                np.clip(img_data["pred_stage_1"][:3].transpose(1, 2, 0), 0, 1)
            )
            axes[0, 1].set_title("Pred Stage 1 (2x) RGB")
            axes[0, 1].axis("off")

        if img_data["pred_stage_2"] is not None:
            axes[0, 2].imshow(
                np.clip(img_data["pred_stage_2"][:3].transpose(1, 2, 0), 0, 1)
            )
            axes[0, 2].set_title("Pred Stage 2 (4x) RGB")
            axes[0, 2].axis("off")

        if img_data["pred_final"] is not None:
            axes[0, 3].imshow(
                np.clip(img_data["pred_final"][:3].transpose(1, 2, 0), 0, 1)
            )
            axes[0, 3].set_title("Pred Final (5x) RGB")
            axes[0, 3].axis("off")

        # Seconda riga: Target per ogni stage (solo RGB)
        # Primo posto vuoto (non c'è target per LR)
        axes[1, 0].axis("off")

        if img_data["target_stage_1"] is not None:
            axes[1, 1].imshow(img_data["target_stage_1"][:3].transpose(1, 2, 0))
            axes[1, 1].set_title("Target Stage 1 (2x) RGB")
            axes[1, 1].axis("off")

        if img_data["target_stage_2"] is not None:
            axes[1, 2].imshow(img_data["target_stage_2"][:3].transpose(1, 2, 0))
            axes[1, 2].set_title("Target Stage 2 (4x) RGB")
            axes[1, 2].axis("off")

        if img_data["target_final"] is not None:
            axes[1, 3].imshow(img_data["target_final"][:3].transpose(1, 2, 0))
            axes[1, 3].set_title("Target Final (5x) RGB")
            axes[1, 3].axis("off")

        plt.tight_layout()
        plt.savefig(
            f"{config.model.dir_path}{config.model.name}/multiscale_sample_{i}.png",
            dpi=150,
        )
        plt.close()


def launch_all_multiscale(config: ConfigPSWin) -> None:
    """Funzione principale per addestrare e valutare il modello multi-scale."""

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
        base_test_dataset = SuperResolutionDataset(
            comune=config.test.comune,
            scale=config.model.scale,
            patch_size=config.model.img_size,
            num_patches=config.test.dataset_size,
            for_training=False,
            to_augment=False,
            syntetic_data=config.train.syntetic_data,
        )
        test_dataset = MultiScaleTargetsDataset(base_test_dataset)

        test_loader = DataLoader(
            test_dataset,
            config.test.batch_size,
            num_workers=config.train.workers,
            persistent_workers=True,
        )

        # Creazione del modello
        model = MultiScaleProgressiveModel(config).to(device)
        print(f"Parametri: {sum(p.numel() for p in model.parameters())}")

        if config.test.load_model:
            # Caricamento del modello esistente
            path = config.model.dir_path + config.model.name + "/model.pth"
            model.load_state_dict(torch.load(path, map_location=device))
            visualize_multi_scale_predictions(model, test_dataset, device, config)
            return

        # Altrimenti alleniamo il modello
        # Dataset di addestramento
        base_train_dataset = SuperResolutionDataset(
            comune=config.test.comune,
            scale=config.model.scale,
            patch_size=config.model.img_size,
            num_patches=config.train.dataset_size,
            for_training=True,
            to_augment=config.train.augment_data,
            syntetic_data=config.train.syntetic_data,
        )
        train_dataset = MultiScaleTargetsDataset(base_train_dataset)

        train_loader = DataLoader(
            train_dataset,
            config.train.batch_size,
            shuffle=True,
            num_workers=config.train.workers,
            persistent_workers=True,
            worker_init_fn=seed_workers,
        )

        # Allenamento
        print("\n=== Starting Training ===")
        losses = train_multi_scale_model(model, train_loader, device, config)

        # Plot dinamico per gli stage multi-scale
        stage_names = ["total", "stage_1", "stage_2", "final"]
        n_plots = len(stage_names)
        cols = 2
        rows = (n_plots + cols - 1) // cols
        _, axs = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), sharex=True)
        axs = np.atleast_1d(axs).ravel()

        for idx, stage in enumerate(stage_names):
            if stage in losses and losses[stage]:
                axs[idx].plot(losses[stage])
                axs[idx].set_title(f"{stage.replace('_', ' ').title()} Loss")
                axs[idx].set_xlabel("Epoch")
                axs[idx].set_ylabel("Loss")
                axs[idx].set_yscale("log")
                axs[idx].grid(True)

        # Hide unused subplots
        for j in range(len(stage_names), len(axs)):
            axs[j].axis("off")

        plt.tight_layout()
        plt.savefig(f"{config.model.dir_path}{config.model.name}/loss.png")
        if config.test.run_napari:
            plt.show()

        print("\n=== Evaluation ===")
        metrics = evaluate_multi_scale_model(model, test_loader, device)
        # Salviamo anche le curve di loss attive nel JSON metrics
        metrics["training_curves"] = losses
        save_metrics(metrics, config)

        visualize_multi_scale_predictions(model, test_dataset, device, config)

        save_model(model, config)

    except Exception as e:
        print(f"Error occurred during training: {e}")
        import traceback

        traceback.print_exc()


def launch_multiscale_training():
    """Lancia il training del modello multi-scale."""
    print("=== Multi-Scale Progressive Super-Resolution Training ===")
    config = ConfigPSWin()
    launch_all_multiscale(config)


if __name__ == "__main__":
    launch_multiscale_training()
