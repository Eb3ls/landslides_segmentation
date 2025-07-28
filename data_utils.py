import os
import numpy as np
from typing import Literal, Tuple
import rasterio
from torch import Tensor
import torch
from torch.utils.data import Dataset


ComuneType = Literal["Brisighella", "Casola-Valsenio", "Modigliana", "Predappio"]

MAIN_DIR = "Comuni/"


def normalize(data: np.ndarray, min_val: float, max_val: float) -> np.ndarray:
    """Normalizza i dati in un array numpy in base al tipo di immagine."""

    assert len(data.shape) >= 2, "Data must be at least 2D array."
    assert data.dtype == np.float32, "Data must be of type float32."

    # Creiamo una maschera per i valori validi
    mask = ~np.isnan(data)

    # Clippiamo i dati per rimuovere i valori fuori range
    data[mask] = np.clip(data[mask], min_val, max_val)

    # Normalizziamo i dati
    data[mask] = (data[mask] - min_val) / (max_val - min_val)

    return data


def print_info(dataset) -> None:
    """Stampa le informazioni di un file raster."""
    print(f"Dataset info:")
    print(f"  Name: {dataset.name}")
    print(f"  Shape: {dataset.shape}")
    print(f"  Bands: {dataset.count}")
    print(f"  CRS: {dataset.crs}")
    print(f"  Dtype: {dataset.dtypes}")
    print(f"  Resolution: {dataset.res}")
    print(f"  Nodata: {dataset.nodata}")
    print(f"----- \n")


def get_data(src, to_norm: bool) -> np.ndarray:
    """Legge e normalizza i dati da un file raster."""
    data = src.read()

    # Impostiamo il tipo comune a float32
    if data.dtype != np.float32:
        data = data.astype(np.float32)

    # Impostiamo a NaN i valori di nodata
    if src.nodata is not None:
        data[data == src.nodata] = np.nan

    if to_norm:
        name = src.name.lower()
        if "change" in name:
            return normalize(data, -2, 2)
        elif "ndvi" in name:
            return normalize(data, -1, 1)
        elif "slope" in name:
            return normalize(data, 0, 90)
        elif "frane" in name:
            return normalize(data, 0, 8)
        elif "cgr" in name or "agea" in name:
            assert data.shape[0] == 4, "Cgr and Agea data must have 4 bands."
            return normalize(data, 0, 255)
        else:
            assert data.shape[0] == 4, "Satellite data must have 4 bands."
            return normalize(data, 0, 10000)

    return data


def generate_dataset_mask(comune: ComuneType) -> np.ndarray:
    """Dato un comune prende l'immagine Cgr e ritorna una maschera 2D per indicarne l'area valida."""

    directory = os.path.join(MAIN_DIR, comune)
    assert os.path.exists(directory), f"Directory '{directory}' does not exist."

    file_path = os.path.join(directory, "Cgr_2023_2m.tif")
    assert os.path.exists(file_path), f"File '{file_path}' does not exist."

    with rasterio.open(file_path) as src:
        # Leggiamo il quarto canale (NIR) per la maschera
        data = src.read(4)
        mask = data != src.nodata

    return mask


def get_super_resolution_stack(
    comune: ComuneType,
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """Genera due coppie di stack satellite-drone, una pre e una post evento, per un comune specifico."""
    directory = os.path.join(MAIN_DIR, comune)

    assert os.path.exists(directory), f"Directory '{directory}' does not exist."

    # Creiamo ndarray (C, H, W) stackiamo su i canali
    agea_stack = []
    cgr_stack = []
    sentinel_pre_stack = []
    sentinel_post_stack = []

    for filename in os.listdir(directory):
        # Saltiamo i file non rilevanti
        if any(
            substr in filename.lower()
            for substr in ["change", "frane", "slope", "ndvi"]
        ):
            continue

        path = os.path.join(directory, filename)

        with rasterio.open(path) as src:
            data = get_data(src, True)

            # Otteniamo i canali come liste di array 2D
            bands = list(data)

            if "Agea" in filename:
                agea_stack.extend(bands)
            elif "Cgr" in filename:
                cgr_stack.extend(bands)
            elif "pre" in filename:
                sentinel_pre_stack.extend(bands)
            elif "post" in filename:
                sentinel_post_stack.extend(bands)
            else:
                print(f"File '{filename}' not recognized, skipping.")

    assert agea_stack, "No Agea data found."
    assert cgr_stack, "No Cgr data found."
    assert sentinel_pre_stack, "No pre-event Sentinel data found."
    assert sentinel_post_stack, "No post-event Sentinel data found."

    # Convertiamo le liste in array numpy, stack di default sul primo asse
    agea_stack = np.stack(agea_stack)
    cgr_stack = np.stack(cgr_stack)
    sentinel_pre_stack = np.stack(sentinel_pre_stack)
    sentinel_post_stack = np.stack(sentinel_post_stack)

    assert (
        agea_stack.shape[1:] == sentinel_pre_stack.shape[1:]
    ), "Agea and Sentinel pre-event stacks must have the same spatial dimensions."
    assert (
        cgr_stack.shape[1:] == sentinel_post_stack.shape[1:]
    ), "Cgr and Sentinel post-event stacks must have the same spatial dimensions."

    return (sentinel_pre_stack, agea_stack), (sentinel_post_stack, cgr_stack)


def get_random_patch(
    data: Tuple[np.ndarray, np.ndarray], patch_size: int, mask: np.ndarray
) -> Tuple[Tuple[np.ndarray, np.ndarray], np.ndarray]:
    """
    Estrae la stessa patch casuale di dimensioni patch_size x patch_size da due stack.
    Altezza e larghezza sono gli ultimi due assi dell'array.
    """

    first, second = data
    assert (
        first.shape == second.shape
    ), "Sentinel and drone data must have the same shape."
    assert patch_size > 0, "Patch size must be positive."
    assert len(mask.shape) == 2, "Mask must be a 2D array."
    assert len(first.shape) >= 2, "Data must be at least a 2D array."
    assert (
        first.shape[-2:] == mask.shape
    ), "Data and mask must have the same spatial dimensions."
    assert min(first.shape[-2:]) >= patch_size, "Data must be larger than patch size."

    height, width = first.shape[-2:]
    max_y = height - patch_size
    max_x = width - patch_size

    start_y = np.random.randint(0, max_y + 1)
    start_x = np.random.randint(0, max_x + 1)

    end_y = start_y + patch_size
    end_x = start_x + patch_size

    first_patch = first[..., start_y:end_y, start_x:end_x]
    second_patch = second[..., start_y:end_y, start_x:end_x]
    patch_mask = mask[start_y:end_y, start_x:end_x]

    # Se il patch non ha almeno l'80% di area valida, lo scartiamo
    if patch_mask.sum() < ((patch_size**2) * 0.8):
        # Troviamo un nuovo patch
        return get_random_patch(data, patch_size, mask)

    return (first_patch, second_patch), patch_mask


def augment_data(
    in_data: Tensor,
    out_data: Tensor,
    prob: float = 0.5,
) -> Tuple[Tensor, Tensor]:

    # Maschera per i valori validi
    valid_mask = ~torch.isnan(in_data)
    in_data = torch.nan_to_num(in_data, nan=0.0)

    # Flip orizzontale
    if np.random.rand() < prob:
        in_data = torch.flip(in_data, dims=[2])
        out_data = torch.flip(out_data, dims=[2])

    # Flip verticale
    if np.random.rand() < prob:
        in_data = torch.flip(in_data, dims=[1])
        out_data = torch.flip(out_data, dims=[1])

    # Rotazione casuale di 0, 90, 180 o 270 gradi
    if np.random.rand() < prob:
        k = np.random.randint(0, 4)
        in_data = torch.rot90(in_data, k=k, dims=[1, 2])
        out_data = torch.rot90(out_data, k=k, dims=[1, 2])

    # Scegliamo se applicare luminosità o contrasto
    do_brightness = False

    if do_brightness and np.random.rand() < prob:
        # Luminosità random tra 0 e 0.2
        offset = np.random.uniform(0, 0.2)
        in_data = torch.where(valid_mask, in_data + offset, in_data)
        in_data = torch.clamp(in_data, 0, 1)

    if not do_brightness and np.random.rand() < prob:
        # Contrasto random tra 0.7 e 1.3
        factor = np.random.uniform(0.7, 1.3)

        # Conta pixel validi per canale
        counts = valid_mask.sum(dim=[1, 2], keepdim=True).float()
        assert counts.min() > 0, "All pixels in at least one channel are invalid."

        # Somma solo pixel validi per canale
        sums = (in_data * valid_mask.float()).sum(dim=[1, 2], keepdim=True)

        # Media = somma / count (evita divisione per zero)
        channel_means = sums / (counts)

        # Applica contrast solo ai pixel validi
        in_data = torch.where(
            valid_mask,
            (in_data - channel_means) * factor + channel_means,
            in_data,
        )
        in_data = torch.clamp(in_data, 0, 1)

    # Ripristiniamo i NaN
    in_data[~valid_mask] = float("nan")

    return in_data, out_data


# TODO: Usare dataset di piú comuni
class SuperResolutionDataset(Dataset):
    """Dataset per il training della super risoluzione."""

    def __init__(
        self,
        comune: ComuneType,
        scale: int,
        patch_size: int = 256,
        num_patches: int = 1000,
        to_augment: bool = False,
    ):
        self.comune = comune
        self.scale = scale
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.to_augment = to_augment

        print(f"Loading data for {comune}...")

        # Generiamo la maschera del dataset
        self.mask = generate_dataset_mask(comune)

        # Carichiamo gli stack di dati
        _, self.stack_post = get_super_resolution_stack(comune)

    def __len__(self) -> int:
        return self.num_patches

    def __getitem__(self, _) -> Tuple[torch.Tensor, torch.Tensor]:

        (low_res_patch, high_res_patch), _ = get_random_patch(
            self.stack_post, self.patch_size * self.scale, self.mask
        )

        # Convertiamo a tensori e assicuriamo il formato corretto
        low_res_tensor = torch.from_numpy(low_res_patch).float()
        high_res_tensor = torch.from_numpy(high_res_patch).float()

        # Augmentiamo i dati
        if self.to_augment:
            low_res_patch, high_res_patch = augment_data(
                low_res_tensor, high_res_tensor
            )

        # Settiamo i valori NaN a 0
        low_res_tensor = torch.nan_to_num(low_res_tensor, nan=0.0)
        high_res_tensor = torch.nan_to_num(high_res_tensor, nan=0.0)

        if self.scale > 1:
            low_res_tensor = torch.nn.functional.interpolate(
                # Necessario aggiungere una dimensione batch
                low_res_tensor.unsqueeze(0),
                scale_factor=1 / self.scale,
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

        return low_res_tensor, high_res_tensor
