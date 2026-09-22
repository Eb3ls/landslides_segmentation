import os
import re
import numpy as np
from typing import Literal, Tuple, cast
import rasterio
from torch import Tensor
import torch
from torch.utils.data import Dataset
from skimage.metrics import structural_similarity


ComuneType = Literal["Brisighella", "Casola-Valsenio", "Modigliana", "Predappio"]

MAIN_DIR = "Comuni/"
RASTER_EXTENSIONS = {".tif", ".tiff"}


def _raster_filenames(directory: str) -> list[str]:
    """Return regular GeoTIFF files in a filesystem-independent order."""
    return sorted(
        (
            filename
            for filename in os.listdir(directory)
            if os.path.isfile(os.path.join(directory, filename))
            and os.path.splitext(filename)[1].lower() in RASTER_EXTENSIONS
        ),
        key=str.casefold,
    )


def _filename_tokens(filename: str) -> set[str]:
    stem = os.path.splitext(filename)[0].casefold()
    return {token for token in re.split(r"[^a-z0-9]+", stem) if token}


def _read_masked_raster(
    path: str,
    mask: np.ndarray,
    reference_crs,
    reference_transform,
) -> list[np.ndarray]:
    with rasterio.open(path) as src:
        if src.shape != mask.shape:
            raise ValueError(
                f"Raster '{os.path.basename(path)}' has spatial dimensions "
                f"{src.shape}, expected {mask.shape}"
            )
        if src.crs != reference_crs or src.transform != reference_transform:
            raise ValueError(
                f"Raster '{os.path.basename(path)}' is not aligned with "
                "Cgr_2023_2m.tif (CRS or transform differs)"
            )
        data = get_data(src, True)

    data[:, ~mask] = 0
    return list(data)


def normalize(data: np.ndarray, min_val: float, max_val: float) -> np.ndarray:
    """Normalizza i dati in un array numpy in base al tipo di immagine.

    Args:
        data: Array numpy da normalizzare (almeno 2D, dtype float32)
        min_val: Valore minimo per la normalizzazione
        max_val: Valore massimo per la normalizzazione

    Returns:
        Array normalizzato con valori tra 0 e 1

    Raises:
        ValueError: Se data non è almeno 2D o non è float32
    """
    if len(data.shape) < 2:
        raise ValueError("Data must be at least 2D array")
    if data.dtype != np.float32:
        raise ValueError("Data must be of type float32")

    # Creiamo una maschera per i valori validi
    mask = ~np.isnan(data)

    # Clippiamo i dati per rimuovere i valori fuori range
    data[mask] = np.clip(data[mask], min_val, max_val)

    # Normalizziamo i dati
    data[mask] = (data[mask] - min_val) / (max_val - min_val)

    return data


def print_info(dataset: rasterio.DatasetReader) -> None:
    """Stampa le informazioni di un file raster.

    Args:
        dataset: Dataset rasterio aperto da cui stampare le informazioni
    """
    print(f"Dataset info:")
    print(f"  Name: {dataset.name}")
    print(f"  Shape: {dataset.shape}")
    print(f"  Bands: {dataset.count}")
    print(f"  CRS: {dataset.crs}")
    print(f"  Dtype: {dataset.dtypes}")
    print(f"  Resolution: {dataset.res}")
    print(f"  Nodata: {dataset.nodata}")
    print(f"----- \n")


def get_data(src: rasterio.DatasetReader, to_norm: bool) -> np.ndarray:
    """Legge e normalizza i dati da un file raster.

    Args:
        src: Dataset rasterio aperto
        to_norm: Se True, normalizza i dati in base al tipo di file

    Returns:
        Array numpy con i dati del raster

    Raises:
        ValueError: Se i dati non hanno il numero di bande atteso
    """
    # Read GDAL masks and nodata together. Casting before filling is required for
    # integer rasters, which cannot represent NaN.
    data = src.read(masked=True).astype(np.float32).filled(np.nan)
    data[~np.isfinite(data)] = np.nan

    if to_norm:
        name = os.path.basename(src.name).lower()
        if "change" in name:
            return normalize(data, -2, 2)
        elif "ndvi" in name:
            return normalize(data, -1, 1)
        elif "slope" in name:
            return normalize(data, 0, 90)
        elif "frane" in name:
            return normalize(data, 0, 8)
        elif "cgr" in name or "agea" in name:
            if data.shape[0] != 4:
                raise ValueError("Cgr and Agea data must have 4 bands")
            return normalize(data, 0, 255)
        else:
            if data.shape[0] != 4:
                raise ValueError("Satellite data must have 4 bands")
            return normalize(data, 0, 10000)

    return data


def generate_dataset_mask(comune: ComuneType) -> np.ndarray:
    """Dato un comune prende l'immagine Cgr e ritorna una maschera 2D per indicarne l'area valida.

    Args:
        comune: Nome del comune da processare

    Returns:
        Array booleano 2D che indica i pixel validi

    Raises:
        FileNotFoundError: Se la directory o il file Cgr non esistono
    """

    directory = os.path.join(MAIN_DIR, comune)
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Directory '{directory}' does not exist")

    file_path = os.path.join(directory, "Cgr_2023_2m.tif")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File '{file_path}' does not exist")

    with rasterio.open(file_path) as src:
        if src.count < 4:
            raise ValueError(f"Cgr raster '{file_path}' must have at least 4 bands")
        # Leggiamo il quarto canale (NIR) per la maschera
        data = src.read(4, masked=True)
        mask = ~np.ma.getmaskarray(data) & np.isfinite(np.asarray(data))

    return mask


def get_super_resolution_stack(
    comune: ComuneType,
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """Genera due coppie di stack satellite-drone, una pre e una post evento, per un comune specifico.

    Args:
        comune: Nome del comune da processare

    Returns:
        Tupla contenente ((sentinel_pre, agea), (sentinel_post, cgr)) senza NaN

    Raises:
        FileNotFoundError: Se la directory del comune non esiste
        ValueError: Se non vengono trovati tutti i tipi di dati richiesti o se le dimensioni non corrispondono
    """
    directory = os.path.join(MAIN_DIR, comune)

    if not os.path.exists(directory):
        raise FileNotFoundError(f"Directory '{directory}' does not exist")

    stacks: dict[str, list[np.ndarray]] = {
        "agea": [],
        "cgr": [],
        "sentinel_pre": [],
        "sentinel_post": [],
    }

    # Prendiamo la maschera del comune
    mask = generate_dataset_mask(comune)
    with rasterio.open(os.path.join(directory, "Cgr_2023_2m.tif")) as reference:
        reference_crs = reference.crs
        reference_transform = reference.transform

    files_by_kind: dict[str, list[str]] = {kind: [] for kind in stacks}
    for filename in _raster_filenames(directory):
        tokens = _filename_tokens(filename)
        if "agea" in tokens:
            kind = "agea"
        elif "cgr" in tokens:
            kind = "cgr"
        elif tokens & {"change", "frane", "slope", "ndvi"}:
            continue
        elif "pre" in tokens:
            kind = "sentinel_pre"
        elif "post" in tokens:
            kind = "sentinel_post"
        else:
            continue
        files_by_kind[kind].append(filename)

    # Category order is part of the channel contract. Files within a category
    # are already case-insensitively sorted by _raster_filenames.
    for kind in ("agea", "cgr", "sentinel_pre", "sentinel_post"):
        for filename in files_by_kind[kind]:
            stacks[kind].extend(
                _read_masked_raster(
                    os.path.join(directory, filename),
                    mask,
                    reference_crs,
                    reference_transform,
                )
            )

    if not stacks["agea"]:
        raise ValueError("No Agea data found")
    if not stacks["cgr"]:
        raise ValueError("No Cgr data found")
    if not stacks["sentinel_pre"]:
        raise ValueError("No pre-event Sentinel data found")
    if not stacks["sentinel_post"]:
        raise ValueError("No post-event Sentinel data found")
    for kind, bands in stacks.items():
        if len(bands) != 4:
            raise ValueError(
                f"SR role '{kind}' must contain exactly 4 channels; "
                f"found {len(bands)} from {files_by_kind[kind]}"
            )

    # Convertiamo le liste in array numpy, stack di default sul primo asse
    agea_stack = np.stack(stacks["agea"])
    cgr_stack = np.stack(stacks["cgr"])
    sentinel_pre_stack = np.stack(stacks["sentinel_pre"])
    sentinel_post_stack = np.stack(stacks["sentinel_post"])

    if agea_stack.shape[1:] != sentinel_pre_stack.shape[1:]:
        raise ValueError(
            "Agea and Sentinel pre-event stacks must have the same spatial dimensions"
        )
    if cgr_stack.shape[1:] != sentinel_post_stack.shape[1:]:
        raise ValueError(
            "Cgr and Sentinel post-event stacks must have the same spatial dimensions"
        )

    return (sentinel_pre_stack, agea_stack), (sentinel_post_stack, cgr_stack)


def get_landslide_mask(landslides: list) -> np.ndarray:
    """Genera una maschera booleana per le frane a partire dalla mappa delle frane

    Args:
        landslides: Array numpy (CHW) contenente i dati delle frane

    Returns:
        Maschera delle frane (CHW) con valori booleani
    """
    landslide_mask = landslides[0] > 0  # Soglia arbitraria
    landslide_mask = landslide_mask.astype(np.bool_)
    if landslide_mask.ndim == 2:
        landslide_mask = np.expand_dims(landslide_mask, axis=0)

    return landslide_mask


def get_segmentation_stack(
    comune: ComuneType,
    include_slope_ndvi: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:

    directory = os.path.join(MAIN_DIR, comune)

    if not os.path.exists(directory):
        raise FileNotFoundError(f"Directory '{directory}' does not exist")

    stacks: dict[str, list[np.ndarray]] = {
        "agea": [],
        "cgr": [],
        "slope": [],
        "ndvi": [],
        "frane": [],
    }

    # Prendiamo la maschera del comune
    mask = generate_dataset_mask(comune)
    with rasterio.open(os.path.join(directory, "Cgr_2023_2m.tif")) as reference:
        reference_crs = reference.crs
        reference_transform = reference.transform

    files_by_kind: dict[str, list[str]] = {kind: [] for kind in stacks}
    for filename in _raster_filenames(directory):
        tokens = _filename_tokens(filename)
        if "sentinel2" in tokens or "s2" in tokens:
            continue
        if "agea" in tokens:
            kind = "agea"
        elif "cgr" in tokens:
            kind = "cgr"
        elif "slope" in tokens:
            kind = "slope"
        elif "ndvi" in tokens:
            kind = "ndvi"
        elif "frane" in tokens:
            kind = "frane"
        else:
            continue
        files_by_kind[kind].append(filename)

    input_kinds = (
        ("agea", "cgr", "slope", "ndvi")
        if include_slope_ndvi
        else ("agea", "cgr")
    )
    for kind in (*input_kinds, "frane"):
        for filename in files_by_kind[kind]:
            bands = _read_masked_raster(
                os.path.join(directory, filename),
                mask,
                reference_crs,
                reference_transform,
            )
            if kind == "frane":
                stacks[kind].extend(get_landslide_mask(bands))
            else:
                stacks[kind].extend(bands)

    input_stack = [band for kind in input_kinds for band in stacks[kind]]
    output_stack = stacks["frane"]

    for kind in input_kinds:
        if not stacks[kind]:
            raise ValueError(f"No {kind} data found")
    if not output_stack:
        raise ValueError("No output data found")

    # Convertiamo le liste in array numpy, stack di default sul primo asse
    input_stack = np.stack(input_stack)
    output_stack = np.stack(output_stack)

    if input_stack.shape[1:] != output_stack.shape[1:]:
        raise ValueError(
            "Input and output stacks must have the same spatial dimensions"
        )

    return (input_stack, output_stack)


def check_similarity(
    first_img: np.ndarray,
    second_img: np.ndarray,
    mask: np.ndarray,
    threshold: float = 0.9,
) -> bool:
    """Controlla se una immagine non ha errori o distorsioni significative.

    Args:
        first_img: Primo array numpy (CHW o HW) senza NaN
        second_img: Secondo array numpy (CHW o HW) senza NaN
        mask: Maschera della stessa dimensione delle immmagini che indica i pixel validi da considerare
        threshold: Soglia di similarità (0-1, default 0.9)

    Returns:
        True se la similarità è >= threshold, False altrimenti

    Raises:
        ValueError: Se i dati non hanno la stessa forma o se non sono almeno 2D o se la maschera non è valida
    """

    if first_img.ndim not in (2, 3):
        raise ValueError("Data must be a 2D or 3D array")
    if first_img.shape != second_img.shape or first_img.shape[-2:] != mask.shape:
        raise ValueError("Data must have the same shape")

    _, ssim_img = structural_similarity(  # type: ignore
        first_img,
        second_img,
        data_range=1,
        channel_axis=0 if len(first_img.shape) == 3 else None,
        full=True,
    )

    valid_pixels = np.sum(mask)
    if valid_pixels == 0:
        raise ValueError("No valid pixels in the mask")

    if len(ssim_img.shape) == 2:
        # Caso 2D: applichiamo direttamente la maschera
        ssim_masked = ssim_img[mask]
    else:
        # Caso 3D: la maschera si applica alle ultime due dimensioni
        ssim_masked = ssim_img[..., mask]
    mssim = ssim_masked.mean()

    # SSIM can be negative for anti-correlated images.
    return mssim >= threshold


def get_random_patch(
    first_img: np.ndarray,
    second_img: np.ndarray,
    patch_size: int,
    mask: np.ndarray,
    return_similar: bool = True,
    max_attempts: int = 100,
    rng: np.random.Generator | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estrae la stessa patch casuale di dimensioni patch_size x patch_size da due stack.

    Args:
        first_img: Primo array numpy (CHW o HW) senza NaN
        second_img: Secondo array numpy (CHW o HW) senza NaN
        patch_size: Dimensione della patch quadrata da estrarre
        mask: Maschera 2D che indica i pixel validi
        max_attempts: Numero massimo di posizioni casuali da provare
        rng: Generatore locale opzionale per il campionamento deterministico

    Returns:
        first_patch, second_patch, patch_mask

    Raises:
        ValueError: Se gli input non rispettano i requisiti di forma o dimensione
    """

    if first_img.ndim not in (2, 3) or second_img.ndim not in (2, 3):
        raise ValueError("Data must be a 2D or 3D array")
    if first_img.shape[-2:] != second_img.shape[-2:]:
        raise ValueError("Data must have the same spatial dimensions")
    if patch_size <= 0:
        raise ValueError("Patch size must be positive")
    if mask.ndim != 2:
        raise ValueError("Mask must be a 2D array")
    if first_img.shape[-2:] != mask.shape:
        raise ValueError("Data and mask must have the same spatial dimensions")
    if min(first_img.shape[-2:]) < patch_size:
        raise ValueError("Data must be larger than patch size")
    if max_attempts <= 0:
        raise ValueError("Maximum attempts must be positive")

    height, width = first_img.shape[-2:]
    max_y = height - patch_size
    max_x = width - patch_size

    for _ in range(max_attempts):
        if rng is None:
            start_y = np.random.randint(0, max_y + 1)
            start_x = np.random.randint(0, max_x + 1)
        else:
            start_y = int(rng.integers(0, max_y + 1))
            start_x = int(rng.integers(0, max_x + 1))

        end_y = start_y + patch_size
        end_x = start_x + patch_size

        first_patch = first_img[..., start_y:end_y, start_x:end_x]
        second_patch = second_img[..., start_y:end_y, start_x:end_x]
        patch_mask = mask[start_y:end_y, start_x:end_x].copy()

        first_finite = np.isfinite(first_patch)
        second_finite = np.isfinite(second_patch)
        if first_patch.ndim == 3:
            first_finite = first_finite.all(axis=0)
        if second_patch.ndim == 3:
            second_finite = second_finite.all(axis=0)
        patch_mask &= first_finite & second_finite

        if patch_mask.sum() < ((patch_size**2) * 0.8):
            continue

        first_patch = np.nan_to_num(
            first_patch, copy=True, nan=0.0, posinf=0.0, neginf=0.0
        )
        second_patch = np.nan_to_num(
            second_patch, copy=True, nan=0.0, posinf=0.0, neginf=0.0
        )
        return first_patch, second_patch, patch_mask

    raise ValueError(
        "Could not find a finite valid patch with at least 80% valid pixels "
        f"after {max_attempts} attempts"
    )


def augment_data(
    in_data: Tensor,
    out_data: Tensor,
    patch_mask: Tensor,
    prob: float = 0.5,
) -> Tuple[Tensor, Tensor]:
    """Applica augmentazioni casuali ai dati di input e output.

    Args:
        in_data: Tensore di input da augmentare
        out_data: Tensore di output da augmentare (con le stesse trasformazioni)
        patch_mask: Maschera 2D che indica i pixel validi
        prob: Probabilità di applicare ogni augmentazione (default 0.5)

    Returns:
        Tupla di tensori augmentati (in_data_aug, out_data_aug)

    Raises:
        ValueError: Se tutti i pixel in almeno un canale sono invalidi
    """
    # Flip orizzontale
    if np.random.rand() < prob:
        in_data = torch.flip(in_data, dims=[2])
        out_data = torch.flip(out_data, dims=[2])
        patch_mask = torch.flip(patch_mask, dims=[1])

    # Flip verticale
    if np.random.rand() < prob:
        in_data = torch.flip(in_data, dims=[1])
        out_data = torch.flip(out_data, dims=[1])
        patch_mask = torch.flip(patch_mask, dims=[0])

    # Rotazione casuale di 0, 90, 180 o 270 gradi
    if np.random.rand() < prob:
        k = np.random.randint(0, 4)
        in_data = torch.rot90(in_data, k=k, dims=[1, 2])
        out_data = torch.rot90(out_data, k=k, dims=[1, 2])
        patch_mask = torch.rot90(patch_mask, k=k, dims=[0, 1])

    # Scegliamo se applicare luminosità o contrasto
    do_brightness = np.random.rand() < 0.5

    # Maschera necessaria per non considerare i pixel che erano NaN nella media

    if do_brightness and np.random.rand() < prob:
        # Luminosità random tra 0 e 0.2
        offset = np.random.uniform(0, 0.2)
        in_data = torch.where(patch_mask, in_data + offset, in_data)
        in_data = torch.clamp(in_data, 0, 1)

    if not do_brightness and np.random.rand() < prob:
        # Contrasto random tra 0.7 e 1.3
        factor = np.random.uniform(0.7, 1.3)

        # Conta pixel validi per canale
        mask_expanded = patch_mask.unsqueeze(0).expand_as(in_data)
        counts = mask_expanded.sum(dim=[1, 2], keepdim=True)
        if counts.min() <= 0:
            raise ValueError("All pixels in at least one channel are invalid")

        # Somma solo pixel validi per canale
        sums = (in_data * mask_expanded).sum(dim=[1, 2], keepdim=True)

        # Media = somma / count
        channel_means = sums / counts

        # Applica contrast solo ai pixel validi
        in_data = torch.where(
            patch_mask,  # CONDIZIONE
            (in_data - channel_means) * factor + channel_means,  # SE VERO
            in_data,  # SE FALSO
        )

        in_data = torch.clamp(in_data, 0, 1)

    return in_data, out_data


class SuperResolutionDataset(Dataset):
    """Dataset per il training della super risoluzione.

    Args:
        comune: Nome del comune da usare se for_training è False altrimenti i comuni rimanenti
        scale: Fattore di scala per la super risoluzione
        patch_size: Dimensione delle patch quadrate da estrarre
        num_patches: Numero di patch da generare per epoch
        to_augment: Se True, applica augmentazioni casuali ai dati
        evaluation_seed: Seed opzionale per rendere ogni indice di valutazione stabile
    """

    def __init__(
        self,
        comune: ComuneType,
        scale: int,
        patch_size: int = 256,
        num_patches: int = 1000,
        for_training: bool = False,
        synthetic_data: bool = False,
        to_augment: bool = True,
        evaluation_seed: int | None = None,
    ) -> None:
        self.comune = comune
        self.scale = scale
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.to_augment = to_augment
        self.synthetic_data = synthetic_data
        if evaluation_seed is not None and evaluation_seed < 0:
            raise ValueError("Evaluation seed must be non-negative")
        if evaluation_seed is not None and to_augment:
            raise ValueError("evaluation_seed requires to_augment=False")
        self.evaluation_seed = evaluation_seed
        if for_training:
            self.set_comuni = [
                c
                for c in ["Brisighella", "Casola-Valsenio", "Modigliana", "Predappio"]
                if c != comune
            ]
            print(f"Using comuni {self.set_comuni} for training.")
        else:
            print(f"Using comune {comune} for validation/testing.")
            self.set_comuni = [comune]

        self.mask = {}
        self.stack_post = {}
        for single_comune in self.set_comuni:
            print(f"Loading data for {single_comune}...")
            self.mask[single_comune] = generate_dataset_mask(
                cast(ComuneType, single_comune)
            )
            # Carichiamo gli stack di dati
            _, self.stack_post[single_comune] = get_super_resolution_stack(
                cast(ComuneType, single_comune)
            )

    def __len__(self) -> int:
        return self.num_patches

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.evaluation_seed is None:
            rng = None
            random_comune = np.random.choice(self.set_comuni)
        else:
            rng = np.random.default_rng(
                np.random.SeedSequence([self.evaluation_seed, int(index)])
            )
            random_comune = rng.choice(self.set_comuni)

        low_res_patch, high_res_patch, patch_mask = get_random_patch(
            self.stack_post[random_comune][1 if self.synthetic_data else 0],
            self.stack_post[random_comune][1],
            self.patch_size * self.scale,
            self.mask[random_comune],
            rng=rng,
        )

        # Convertiamo a tensori e assicuriamo il formato corretto
        low_res_tensor = torch.from_numpy(low_res_patch)
        high_res_tensor = torch.from_numpy(high_res_patch)
        patch_mask = torch.from_numpy(patch_mask)

        # Augmentiamo i dati
        if self.to_augment:
            low_res_tensor, high_res_tensor = augment_data(
                low_res_tensor, high_res_tensor, patch_mask
            )

        if self.scale > 1:
            low_res_tensor = torch.nn.functional.interpolate(
                # Necessario aggiungere una dimensione batch
                low_res_tensor.unsqueeze(0),
                scale_factor=1 / self.scale,
                mode="bicubic",
                align_corners=False,
            ).squeeze(0)

        return low_res_tensor, high_res_tensor


class SegmentationSingleDataset(Dataset):
    """Dataset per il training della segmentazione con un comune."""

    def __init__(
        self, comune: ComuneType, patch_size: int = 256, num_patches: int = 1000, include_slope_ndvi: bool = False
    ):
        self.comune = comune
        self.patch_size = patch_size
        self.num_patches = num_patches

        print(f"Loading data for {comune}...")

        # Generiamo la maschera del dataset
        self.mask = generate_dataset_mask(comune)

        # Carichiamo gli stack di dati
        self.stack_input, self.stack_landslide = get_segmentation_stack(comune, include_slope_ndvi)

    def __len__(self) -> int:
        return self.num_patches

    def __getitem__(self, _) -> Tuple[torch.Tensor, torch.Tensor]:

        input_patch, landslide_patch, _ = get_random_patch(
            self.stack_input, self.stack_landslide, self.patch_size, self.mask
        )

        # Convertiamo a tensori e assicuriamo il formato corretto
        input_tensor = torch.from_numpy(input_patch).float()
        landslide_tensor = torch.from_numpy(landslide_patch).float()

        # Settiamo i valori NaN a 0
        input_tensor = torch.nan_to_num(input_tensor, nan=0.0)
        landslide_tensor = torch.nan_to_num(landslide_tensor, nan=0.0)

        return input_tensor, landslide_tensor


class SegmentationMultiDataset(Dataset):
    """Dataset per il training della segmentazione con più comuni."""

    def __init__(
        self, comuni: list[ComuneType], patch_size: int = 256, num_patches: int = 1000, augment: bool = False, include_slope_ndvi: bool = False
    ):
        self.comuni = comuni
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.masks = []
        self.stack_input = []
        self.stack_landslide = []
        self.augment = augment

        print(f"Loading data for {comuni}...")

        for comune in comuni:
            # Generiamo la maschera del dataset
            self.masks.append(generate_dataset_mask(comune))
            # Carichiamo gli stack di dati
            input, landslide = get_segmentation_stack(comune, include_slope_ndvi)
            self.stack_input.append(input)
            self.stack_landslide.append(landslide)

    def __len__(self) -> int:
        return self.num_patches

    def __getitem__(self, _) -> Tuple[torch.Tensor, torch.Tensor]:
        # Selezioniamo un comune casuale
        comune_idx = np.random.choice(len(self.comuni))

        input_patch, landslide_patch, patch_mask = get_random_patch(
            self.stack_input[comune_idx],
            self.stack_landslide[comune_idx],
            self.patch_size,
            self.masks[comune_idx],
        )

        # Convertiamo a tensori e assicuriamo il formato corretto
        input_tensor = torch.from_numpy(input_patch).float()
        landslide_tensor = torch.from_numpy(landslide_patch).float()
        patch_mask_tensor = torch.from_numpy(patch_mask)

        # Settiamo i valori NaN a 0
        input_tensor = torch.nan_to_num(input_tensor, nan=0.0)
        landslide_tensor = torch.nan_to_num(landslide_tensor, nan=0.0)

        # Aggiungiamo eventuali augmentazioni
        if self.augment:
            input_tensor, landslide_tensor = augment_data(
                input_tensor, landslide_tensor, patch_mask_tensor
            )

        return input_tensor, landslide_tensor
