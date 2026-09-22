# Multispectral super-resolution for landslide research

Bachelor's thesis project — University of Bologna, 2024/2025.

PyTorch experiments for **5× super-resolution of four-channel satellite imagery (RGB + NIR)** using RCAN, DRCT and Swin2MoSE, with aerial orthophotos as high-resolution targets. The study covers four Italian municipalities with landslide history.

The repository also contains separate segmentation experiments and synthetic cloud utilities. The segmentation loader currently uses aerial imagery directly; an end-to-end super-resolution-to-segmentation pipeline is not implemented here.

## Setup

Use **Python 3.11 or 3.12** and run commands from the repository root. CPU execution is supported; a compatible CUDA GPU is recommended for substantial training. The core environment does not require a desktop or Qt.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# CPU wheels (suitable for the tests below):
python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

For CUDA 12.8, use `https://download.pytorch.org/whl/cu128` instead of the CPU index in the PyTorch installation command. Choose a build compatible with your driver using the [official PyTorch installation instructions](https://pytorch.org/get-started/previous-versions/#v280). On Windows, activate the environment with `.venv\Scripts\Activate.ps1` in PowerShell.

The optional desktop viewers require `python -m pip install -r requirements-viewer.txt`. Keep `test.run_napari: false` for headless training.

## Data and preprocessing

The dataset and pretrained project checkpoints are **not distributed**. A fresh clone can run the regression tests and model checks, but training on the original imagery requires obtaining and preparing the source data separately.

The loader expects a `Comuni/` directory containing `Brisighella`, `Casola-Valsenio`, `Modigliana` and `Predappio`. Each municipality needs co-registered GeoTIFFs on the **same grid, spatial extent, pixel size and coordinate reference system**:

```text
Comuni/<municipality>/
├── Cgr_2023_2m.tif       # four-channel post-event aerial target; also defines valid area
├── Agea_reference.tif    # four-channel pre-event aerial reference
├── sentinel2_pre.tif    # four-channel pre-event satellite image
└── sentinel2_post.tif   # four-channel post-event satellite image
```

The example names show the loader's file roles; `Cgr_2023_2m.tif` is the required mask filename. Store the four image bands in **RGB, NIR** order. Satellite values are normalized from `[0, 10000]`, aerial values from `[0, 255]`; nodata must be declared in the GeoTIFF. The loader does not reproject or align raw satellite products for you.

Super-resolution training uses the post-event pair. It extracts a common-grid HR patch and downsamples the satellite patch by 5 to form the LR input; synthetic training instead downsamples the aerial target. A config `img_size` is the **LR** patch side, so the source rasters must support valid patches of side `5 × img_size`.

Sentinel-2 resolution depends on the band: RGB and broad NIR B8 are native 10 m, while narrow NIR B8A is 20 m. Record which NIR band and resampling procedure your prepared dataset uses; the original metadata is unavailable here. See [Copernicus band specifications](https://sentiwiki.copernicus.eu/web/s2-mission).

## Super-resolution usage

The three tracked `config.yml` files are **new runnable examples, not recovered thesis experiment settings**. Inspect their model sizes, training budgets and validation municipality before running an experiment.

First compute channel statistics from the training municipalities. The excluded municipality must match `test.comune` in the chosen config:

```bash
python Super_Resolution/calc_mean.py --excluded_comune Brisighella --patch_size 40
```

This creates `Super_Resolution/channel_stats.json`; use statistics computed from your own training data. If a stats file records its training municipalities, the loader checks that they exclude the validation municipality. For legacy stats without provenance, verify the split yourself. Then run:

```bash
python Super_Resolution/launch_model.py --model rcan
python Super_Resolution/launch_model.py --model drct
python Super_Resolution/launch_model.py --model swin2mose
```

Pass `--config path/to/config.yml` for a custom experiment. Fine-tuning requires a compatible, existing checkpoint:

```bash
python Super_Resolution/launch_model.py --model rcan --finetune --ckpt path/to/model.pth
```

Head-only fine-tuning freezes the feature extractor and trains the supported reconstruction head. Checkpoints must contain the complete matching model state. Outputs include the model checkpoint, training plots, evaluation metrics and sample comparisons in the directory specified by the config. JSON exports represent non-finite values as the strings `"Infinity"`, `"-Infinity"` or `"NaN"`; perfect reconstruction has infinite PSNR.

Validation uses fixed patches selected by the configured seed. `train.synthetic_data` controls training inputs; `test.synthetic_data` independently controls evaluation inputs and defaults to **false** (real satellite inputs). Keep the validation municipality, seed, patch count and input source identical when comparing runs. LPIPS may download pretrained VGG weights on first use and needs network access or an existing Torch cache.

## Tests

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

Tests use synthetic tensors and temporary GeoTIFFs. They check training behavior, model shapes, configuration handling and shared data loading without the research dataset. They do not reproduce the historical benchmark scores.

## Historical results

The following numbers are preserved from the original README. The original configs, checkpoints and raw metric artifacts are unavailable, so these are **historical reported results, not independently reproduced benchmarks**. Subsequent fixes to loss weighting, training mode and evaluation sampling mean new runs should not be presented as reproductions of these tables.

Within each comparable table, **bold** marks the best reported value: higher PSNR/structural similarity, lower [LPIPS distance](https://github.com/richzhang/PerceptualSimilarity). `SSIM*` preserves the original column label; the checked-in evaluator uses two-scale MS-SSIM on RGB, but the precise historical metric settings cannot be verified. PSNR includes both RGB and NIR unless the column specifies otherwise.

### Architecture comparison (100 epochs, 3 comuni)

| Model | PSNR ↑ | PSNR-RGB ↑ | PSNR-NIR ↑ | SSIM* ↑ | LPIPS ↓ |
|-------|------|----------|----------|------|-------|
| RCAN | **21.48** | **22.07** | **20.19** | **0.574** | **0.592** |
| DRCT | 21.15 | 21.64 | 20.03 | 0.539 | 0.607 |
| Swin2MoSE | 21.11 | 21.61 | 20.00 | 0.538 | 0.614 |

RCAN has the strongest reported scores in this comparison. These runs do not establish how the architectures compare at larger dataset sizes or training budgets.

### Training duration (RCAN vs DRCT, 100 vs 300 epochs)

| Model | Epochs | PSNR ↑ | PSNR-RGB ↑ | PSNR-NIR ↑ | SSIM* ↑ | LPIPS ↓ |
|-------|--------|------|----------|----------|------|-------|
| RCAN | 100 | 21.48 | 22.07 | 20.19 | 0.574 | 0.592 |
| RCAN | 300 | **21.62** | **22.14** | **20.45** | **0.586** | **0.586** |
| DRCT | 100 | 21.15 | 21.64 | 20.03 | 0.539 | 0.607 |
| DRCT | 300 | 21.19 | 21.72 | 20.00 | 0.563 | 0.591 |

### Dataset size (1 vs 3 comuni)

| Model | Comuni | PSNR ↑ | PSNR-RGB ↑ | PSNR-NIR ↑ | SSIM* ↑ | LPIPS ↓ |
|-------|--------|------|----------|----------|------|-------|
| RCAN | 1 | 21.38 | 21.83 | **20.35** | **0.576** | **0.589** |
| RCAN | 3 | **21.48** | **22.07** | 20.19 | 0.574 | 0.592 |
| DRCT | 1 | 21.05 | 21.52 | 19.98 | 0.548 | 0.600 |
| DRCT | 3 | 21.15 | 21.64 | 20.03 | 0.539 | 0.607 |

### Synthetic data and fine-tuning (RCAN)

| Training | PSNR ↑ | PSNR-RGB ↑ | PSNR-NIR ↑ | SSIM* ↑ | LPIPS ↓ |
|----------|------|----------|----------|------|-------|
| Synthetic only | 24.25 | 24.79 | 22.98 | 0.657 | 0.503 |
| Fine-tuned | 21.36 | 21.90 | 20.18 | 0.547 | 0.619 |
| Standard | 21.48 | 22.07 | 20.19 | 0.574 | 0.592 |

No cross-row winner is highlighted here: the historical launcher used synthetic validation inputs when synthetic training was enabled. Without the original run configs, these rows cannot establish a controlled comparison of transfer to real Sentinel-2 inputs.

### Loss function combinations (RCAN)

| Loss | PSNR ↑ | PSNR-RGB ↑ | PSNR-NIR ↑ | SSIM* ↑ | LPIPS ↓ |
|------|------|----------|----------|------|-------|
| Charbonnier | 21.48 | **22.07** | 20.19 | 0.574 | 0.592 |
| Ch. + SSIM | 21.50 | 22.03 | 20.34 | **0.589** | 0.576 |
| Ch. + LPIPS | 21.40 | 21.84 | 20.37 | 0.547 | **0.357** |
| Ch. + HF | **21.52** | **22.07** | 20.29 | 0.578 | 0.592 |
| NCC + SSIM | 21.51 | 21.97 | **20.46** | 0.585 | 0.581 |

The reported metrics favor different losses: Charbonnier + HF has the highest overall PSNR, while Charbonnier + LPIPS has the lowest perceptual distance. There is no single winner across all metrics.

## Project structure

```text
Super_Resolution/
├── launch_model.py       # training / checkpoint visualization entry point
├── config.py             # typed configs and validation
├── models_utils.py       # losses, training, evaluation, checkpoint handling
├── rcan/                 # RCAN and example config
├── myDRCT/               # adapted DRCT and example config
├── swin2mose/            # adapted Swin2MoSE and example config
└── calc_mean.py          # training-data channel statistics

data_utils.py             # shared raster loading, patch sampling and augmentation
tests/                    # CPU regression tests
Segmentation/             # separate, legacy segmentation experiments
Cloud_Generation/         # synthetic cloud augmentation utilities
```

The legacy segmentation scripts are outside the super-resolution validation described above. For their command-line help, use module execution (`python -m Segmentation.train --help`) after installing the optional viewer dependencies.

## Author

Leonardo Berselli — Bachelor's in Computer Science, University of Bologna (2024/2025)
