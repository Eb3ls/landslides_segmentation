

from enum import Enum
import os
import numpy as np
from typing import Literal

import rasterio


def normalize(data: np.ndarray, filename) -> np.ndarray:
    """Normalizza i dati in un array numpy in base al tipo di immagine."""

    assert len(data.shape) >= 2, "Data must be at least 2D array."
    assert data.dtype == np.float32, "Data must be of type float32."

    min_val, max_val = 0, 10000
    if 'change' in filename.lower():
        min_val, max_val = -2, 2
    elif 'ndvi' in filename.lower():
        min_val, max_val = -1, 1
    elif 'slope' in filename.lower():
        min_val, max_val = 0, 90
    elif 'frane' in filename.lower():
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


ComuneType = Literal[
    'Brisighella',
    'Casola-Valsenio',
    'Modigliana',
    'Predappio'
]


def generate_dataset_mask(comune: ComuneType) -> np.ndarray:
    """Dato un comune prende l'immagine Cgr e ritorna una maschera 0-1 per indicarne l'area valida."""

    directory = f'Comuni/{comune}'
    assert os.path.exists(
        directory), f"Directory '{directory}' does not exist."

    file_path = os.path.join(directory, 'Cgr_2023_2m.tif')
    assert os.path.exists(file_path), f"File '{file_path}' does not exist."

    src = rasterio.open(file_path)

    data = src.read(4)
    mask = data != src.nodata

    src.close()

    return mask
