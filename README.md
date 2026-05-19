# landslides_segmentation

Deep learning pipeline for super-resolution and segmentation of multispectral Sentinel-2 satellite imagery, applied to landslide monitoring in Italian municipalities.

Bachelor's thesis project — University of Bologna, 2024/2025.

## What this does

Sentinel-2 imagery has a spatial resolution of 10m for RGB bands and 20m for NIR. This project upscales those images by a factor of 5× using super-resolution models (CNN and Transformer architectures), then feeds the enhanced imagery into a segmentation pipeline to improve landslide detection.

Two main components:

- **Super_Resolution/** — trains and evaluates RCAN, DRCT, and Swin2MoSE models on paired LR/HR satellite patches
- **Segmentation/** — UNet and Swin-UNet models for landslide segmentation on the super-resolved output

## Results

All metrics evaluated on the validation set. **Bold** = best per column.

### Architecture comparison (100 epochs, 3 comuni)

| Model | PSNR | PSNR-RGB | PSNR-NIR | SSIM | LPIPS |
|-------|------|----------|----------|------|-------|
| RCAN | **21.48** | **22.07** | **20.19** | **0.574** | 0.592 |
| DRCT | 21.15 | 21.64 | 20.03 | 0.539 | 0.607 |
| Swin2MoSE | 21.11 | 21.61 | 20.00 | 0.538 | **0.614** |

RCAN (CNN-based) outperforms transformer models on this dataset size. Transformer models need larger datasets and longer training to surpass CNN performance.

### Training duration (RCAN vs DRCT, 100 vs 300 epochs)

| Model | Epochs | PSNR | PSNR-RGB | PSNR-NIR | SSIM | LPIPS |
|-------|--------|------|----------|----------|------|-------|
| RCAN | 100 | 21.48 | 22.07 | 20.19 | 0.574 | 0.592 |
| RCAN | 300 | **21.62** | **22.14** | **20.45** | **0.586** | 0.586 |
| DRCT | 100 | 21.15 | 21.64 | 20.03 | 0.539 | **0.607** |
| DRCT | 300 | 21.19 | 21.72 | 20.00 | 0.563 | 0.591 |

### Dataset size (1 vs 3 comuni)

| Model | Comuni | PSNR | PSNR-RGB | PSNR-NIR | SSIM | LPIPS |
|-------|--------|------|----------|----------|------|-------|
| RCAN | 1 | 21.38 | 21.83 | **20.35** | **0.576** | 0.589 |
| RCAN | 3 | **21.48** | **22.07** | 20.19 | 0.574 | 0.592 |
| DRCT | 1 | 21.05 | 21.52 | 19.98 | 0.548 | 0.600 |
| DRCT | 3 | 21.15 | 21.64 | 20.03 | 0.539 | **0.607** |

### Synthetic data and fine-tuning (RCAN)

| Training | PSNR | PSNR-RGB | PSNR-NIR | SSIM | LPIPS |
|----------|------|----------|----------|------|-------|
| Synthetic only | 24.25 | 24.79 | 22.98 | 0.657 | 0.503 |
| Fine-tuned | 21.36 | 21.90 | 20.18 | 0.547 | **0.619** |
| Standard | **21.48** | **22.07** | **20.19** | **0.574** | 0.592 |

Pre-training on bicubic-downsampled synthetic data did not transfer well to real Sentinel-2 imagery.

### Loss function combinations (RCAN)

| Loss | PSNR | PSNR-RGB | PSNR-NIR | SSIM | LPIPS |
|------|------|----------|----------|------|-------|
| Charbonnier | 21.48 | **22.07** | 20.19 | 0.574 | **0.592** |
| Ch. + SSIM | 21.50 | 22.03 | 20.34 | **0.589** | 0.576 |
| Ch. + LPIPS | 21.40 | 21.84 | **20.37** | 0.547 | 0.357 |
| Ch. + HF | **21.52** | **22.07** | 20.29 | 0.578 | **0.592** |
| NCC + SSIM | 21.51 | 21.97 | 20.46 | 0.585 | 0.581 |

Charbonnier + High-Frequency loss gave the best overall results.

## Project structure

```
Super_Resolution/
├── launch_model.py       # entry point: train/evaluate a model
├── config.py             # config dataclasses and factory
├── models_utils.py       # training loop and evaluation
├── rcan/                 # RCAN model and config
├── myDRCT/               # DRCT model and config
├── swin2mose/            # Swin2MoSE model and config
└── calc_mean.py          # compute per-channel dataset stats

Segmentation/
├── train.py              # segmentation training
├── evaluate.py           # segmentation evaluation
├── unet.py               # UNet architecture
└── swin_unet.py          # Swin-UNet architecture

Cloud_Generation/         # synthetic cloud augmentation utilities
data_utils.py             # dataset loading and preprocessing
```

## Setup

Requires a CUDA-capable GPU and Python 3.10+.

```bash
pip install -r requirements.txt
```

Before training, compute per-channel normalization statistics on your dataset:

```bash
python Super_Resolution/calc_mean.py
```

This generates `Super_Resolution/channel_stats.json`, which is required by the config loader.

## Usage

Train a super-resolution model:

```bash
python Super_Resolution/launch_model.py --model rcan
python Super_Resolution/launch_model.py --model drct
python Super_Resolution/launch_model.py --model swin2mose
```

Fine-tune from a checkpoint:

```bash
python Super_Resolution/launch_model.py --model rcan --finetune --ckpt path/to/checkpoint.pth
```

Model hyperparameters and training settings are configured in each model's `config.yml`.

## Data

The models were trained on paired LR/HR Sentinel-2 patches from 4 Italian municipalities (comuni) with known landslide history. The LR images are the native Sentinel-2 10m/20m resolution; HR images are aerial orthophotos used as ground truth.

The dataset is not included in this repository.

## Author

Leonardo Berselli — Bachelor's in Computer Science, University of Bologna (2024/2025)
