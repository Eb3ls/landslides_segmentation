
from io import BytesIO
import os
from typing import Literal
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
import napari
import numpy as np
import rasterio

from data_utils import normalize, print_info


def save_to_napari(fig: Figure, viewer: napari.Viewer,  name="") -> None:
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


def get_colormap(filename: str) -> str:
    """Determina la colormap da usare in base al nome del file."""
    if 'slope' in filename.lower():
        return 'gray_r'
    elif any(keyword in filename.lower() for keyword in ['ndvi', 'frane']):
        return 'viridis'
    else:
        return 'gray'


def add_image_and_histogram(viewer: napari.Viewer, data: np.ndarray, filename: str, ty: Literal[' RGB', ' NIR', ''] = '') -> None:
    """Aggiunge un'immagine e il suo istogramma a napari."""
    # Aggiungiamo l'immagine a napari
    viewer.add_image(data, name=filename + ty, colormap=get_colormap(filename))

    # Creiamo e salviamo l'istogramma
    fig = plot_histogram(data, title=f"Histogram of {filename + ty}")
    save_to_napari(fig, viewer, name=f"{filename + ty} Histogram")


def view_dataset(viewer: napari.Viewer, directory: str) -> None:
    """Visualizza tutti i file tiff in una directory con napari. Per ognuno di essi, crea un'immagine e il suo istogramma."""
    for filename in os.listdir(directory):
        path = os.path.join(directory, filename)

        with rasterio.open(path) as src:

            print_info(src)
            data = src.read()

            # Impostiamo il tipo comune a float32
            if data.dtype != np.float32:
                data = data.astype(np.float32)

            # Impostiamo a NaN i valori di nodata
            if src.nodata is not None:
                data[data == src.nodata] = np.nan

            # Normalizziamo i dati
            data_norm = normalize(data, filename)

            # Se immagine RGB + NIR separiamo
            if len(data_norm.shape) == 3 and data_norm.shape[0] == 4:
                rgb_image = data_norm[:3, :, :]
                rgb_image = rgb_image.transpose(1, 2, 0)

                nir_image = data_norm[3, :, :]

                # Aggiungiamo l'immagine RGB e il suo istogramma a napari
                add_image_and_histogram(
                    viewer, rgb_image, filename, ty=' RGB')

                # Aggiungiamo l'immagine NIR e il suo istogramma a napari
                add_image_and_histogram(
                    viewer, nir_image, filename, ty=' NIR')

            else:
                # Aggiungiamo l'immagine a napari
                add_image_and_histogram(viewer, data_norm, filename)

            src.close()

    # Nascondiamo tutte le immagini inizialmente
    for layer in viewer.layers:
        layer.visible = False
