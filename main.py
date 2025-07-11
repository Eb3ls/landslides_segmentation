import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import napari
import os
import tifffile as tf
from io import BytesIO

viewer = napari.Viewer()

if not os.path.exists('Brisighella'):
    print("Directory 'Brisighella' does not exist. Please check the path.")
    exit()


def save_to_napari(fig: Figure, name="") -> None:
    """Salva l'immagine corrente su napari."""
    buffer = BytesIO()
    fig.savefig(buffer, format='png')
    buffer.seek(0)

    # Convertiamo in un array numpy
    img_array = plt.imread(buffer)
    viewer.add_image(img_array, name=name)

    plt.close(fig)
    buffer.close()


def plot_histogram(data: np.ndarray, title='Histogram') -> Figure:
    """Plotta l'istogramma del valore dei pixel di un'immagine."""
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.hist(data.ravel(), bins=50, color='blue', alpha=0.7)
    ax.set_title(title)
    ax.set_xlabel('Pixel Value')
    ax.set_ylabel('Frequency')
    ax.grid(True)
    return fig


def normalize(data: np.ndarray, filename) -> np.ndarray:
    """Normalizza i dati in un array numpy in base al tipo di immagine."""
    if len(data.shape) < 2:
        raise ValueError("Data must be at least 2D array.")

    if data.dtype == np.uint8:
        print(
            f"{filename} is already in uint8 format, skipping normalization.")
        return data

    min_val, max_val = 0, 10000
    if 'ndvi' in filename.lower():
        min_val, max_val = -1, 1
    elif 'slope' in filename.lower():
        min_val, max_val = 0, 90
    elif 'change' in filename.lower():
        min_val, max_val = -2, 2
    elif 'frane' in filename.lower():
        min_val, max_val = 0, 8

    # Creiamo una maschera per i valori validi
    mask = np.isfinite(data) & (data >= min_val) & (data <= max_val * 1.5)

    # Clippiamo i dati per rimuovere i valori fuori range
    data[~mask] = np.nan
    data[mask] = np.clip(data[mask], min_val, max_val)

    # Normalizziamo i dati
    print(f"Normalizing with range: {min_val} - {max_val}")
    data[mask] = (data[mask] - min_val) / (max_val - min_val)

    return data


def get_colormap(filename: str) -> str:
    """Determina la colormap da usare in base al nome del file."""
    if 'slope' in filename.lower():
        return 'gray_r'
    elif any(keyword in filename.lower() for keyword in ['ndvi', 'frane']):
        return 'viridis'
    else:
        return 'gray'


for filename in os.listdir('Brisighella'):
    path = os.path.join('Brisighella', filename)

    print(f"\nProcessing file: {path}")
    tiff = tf.TiffFile(path)

    for page in tiff.pages:
        if (not isinstance(page, tf.TiffPage)):
            raise TypeError(
                f"Expected TiffPage, got {type(page)} for file {filename}")

        data = page.asarray()

        print(f"Page {page.index}: shape={page.shape}, dtype={page.dtype}")
        print(f"Data range: {data.min():.2f} - {data.max():.2f}")

        # Normalizziamo i dati
        data_norm = normalize(data, filename)

        # Se immagine RGB + NIR separiamo
        if len(page.shape) == 3 and page.shape[2] == 4:
            RGB = data_norm[:, :, :3]
            NIR = data_norm[:, :, 3]

            # Aggiungiamo l'immagine RGB e il suo istogramma a napari
            layer = viewer.add_image(RGB, name=f"{filename} RGB")
            fig = plot_histogram(RGB, title=f"Histogram of {filename} RGB")
            save_to_napari(fig, name=f"{filename} RGB Histogram")

            # Aggiungiamo l'immagine NIR e il suo istogramma a napari
            viewer.add_image(NIR, name=f"{filename} NIR", colormap='gray')
            fig = plot_histogram(NIR, title=f"Histogram of {filename} NIR")
            save_to_napari(fig, name=f"{filename} NIR Histogram")

        else:
            # Aggiungiamo l'immagine a napari
            viewer.add_image(
                data_norm, name=f"{filename}", colormap=get_colormap(filename))
            fig = plot_histogram(data_norm, title=f"Histogram of {filename}")
            save_to_napari(fig, name=f"{filename} Histogram")

    tiff.close()

# Nascondiamo tutte le layer inizialmente
for layer in viewer.layers:
    layer.visible = False

napari.run()
