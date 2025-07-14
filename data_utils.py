import os
import numpy as np
from typing import Literal, Tuple
import rasterio


ComuneType = Literal["Brisighella", "Casola-Valsenio", "Modigliana", "Predappio"]

MAIN_DIR = "Comuni"


def normalize(data: np.ndarray, filename) -> np.ndarray:
    """Normalizza i dati in un array numpy in base al tipo di immagine."""

    assert len(data.shape) >= 2, "Data must be at least 2D array."
    assert data.dtype == np.float32, "Data must be of type float32."

    min_val, max_val = 0, 10000
    if "change" in filename.lower():
        min_val, max_val = -2, 2
    elif "ndvi" in filename.lower():
        min_val, max_val = -1, 1
    elif "slope" in filename.lower():
        min_val, max_val = 0, 90
    elif "frane" in filename.lower():
        min_val, max_val = 0, 8

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


def get_normalized_data(src) -> np.ndarray:
    """Legge e normalizza i dati da un file raster."""
    data = src.read()

    # Impostiamo il tipo comune a float32
    if data.dtype != np.float32:
        data = data.astype(np.float32)

    # Impostiamo a NaN i valori di nodata
    if src.nodata is not None:
        data[data == src.nodata] = np.nan

    # Normalizziamo i dati
    data_norm = normalize(data, src.name)

    return data_norm


def generate_dataset_mask(comune: ComuneType) -> np.ndarray:
    """Dato un comune prende l'immagine Cgr e ritorna una maschera 0-1 per indicarne l'area valida."""

    directory = os.path.join(MAIN_DIR, comune)
    assert os.path.exists(directory), f"Directory '{directory}' does not exist."

    file_path = os.path.join(directory, "Cgr_2023_2m.tif")
    assert os.path.exists(file_path), f"File '{file_path}' does not exist."

    with rasterio.open(file_path) as src:
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
        # TODO: L'NDVI é da prendere da copernicus
        if any(
            substr in filename.lower()
            for substr in ["change", "frane", "slope", "ndvi"]
        ):
            continue

        path = os.path.join(directory, filename)

        with rasterio.open(path) as src:
            data = get_normalized_data(src)

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
    data: np.ndarray, patch_size: int, mask: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Estrae un patch casuale di dimensioni patch_size x patch_size da un array 2D. Altezza e larghezza sono gli ultimi due assi dell'array."""
    assert patch_size > 0, "Patch size must be positive."
    assert len(mask.shape) == 2, "Mask must be a 2D array."
    assert len(data.shape) >= 2, "Data must be at least a 2D array."
    assert (
        data.shape[-2:] == mask.shape
    ), "Data and mask must have the same spatial dimensions."
    assert min(data.shape[-2:]) >= patch_size, "Data must be larger than patch size."

    height, width = data.shape[-2:]
    max_y = height - patch_size
    max_x = width - patch_size

    start_y = np.random.randint(0, max_y + 1)
    start_x = np.random.randint(0, max_x + 1)

    end_y = start_y + patch_size
    end_x = start_x + patch_size

    patch = data[..., start_y:end_y, start_x:end_x]
    patch_mask = mask[start_y:end_y, start_x:end_x]

    # Se il patch non ha almeno l'80% di area valida, lo scartiamo
    if patch_mask.sum() < ((patch_size**2) * 0.8):
        # Troviamo un nuovo patch
        return get_random_patch(data, patch_size, mask)

    return patch, patch_mask
