

import numpy as np


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
    print(f"  REsolution: {dataset.res}")
    print(f"  Nodata: {dataset.nodata}")
    print(f"----- \n")
