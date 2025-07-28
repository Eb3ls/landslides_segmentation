from io import BytesIO
from typing import Literal
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
import napari
import numpy as np

from data_utils import augment_data


def save_to_napari(fig: Figure, viewer: napari.Viewer, name="") -> None:
    """Salva l'immagine corrente su napari."""
    buffer = BytesIO()
    fig.savefig(buffer, format="png")
    buffer.seek(0)

    # Convertiamo in un array numpy
    img_array = plt.imread(buffer)
    viewer.add_image(img_array, name=name)

    plt.close(fig)
    buffer.close()


def plot_histogram(data: np.ndarray, title="Histogram") -> Figure:
    """Plotta l'istogramma del valore dei pixel di un'immagine."""
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.hist(data.ravel(), bins=50, color="blue", alpha=0.7)
    ax.set_title(title)
    ax.set_xlabel("Pixel Value")
    ax.set_ylabel("Frequency")
    ax.grid(True)
    return fig


def get_colormap(filename: str) -> str:
    """Determina la colormap da usare in base al nome del file.

    Args:
        filename: Nome del file o path per cui determinare la colormap

    Returns:
        Nome della colormap da utilizzare
    """
    fl = filename.lower()

    if "slope" in fl:
        return "terrain"
    elif "change" in fl:
        return "RdBu_r"
    elif "ndvi" in fl:
        return "RdYlGn"
    elif "frane" in fl:
        return "viridis"
    else:
        return "gray"


def add_image_and_histogram(
    viewer: napari.Viewer,
    data: np.ndarray,
    filename: str,
    ty: Literal[" RGB", " NIR", ""] = "",
) -> None:
    """Aggiunge un'immagine e il suo istogramma a napari."""
    # Aggiungiamo l'immagine a napari, lasciamo che calcoli automaticamente i limiti di contrasto, la visualizzazione é migliore
    viewer.add_image(
        data,
        name=filename + ty,
        colormap=get_colormap(filename),
    )

    # Creiamo e salviamo l'istogramma
    fig = plot_histogram(data, title=f"Histogram of {filename + ty}")
    save_to_napari(fig, viewer, name=f"{filename + ty} Histogram")


def view_data(
    data: np.ndarray,
    filename: str,
    viewer: napari.Viewer,
) -> None:
    # Se immagine RGB + NIR separiamo
    if len(data.shape) == 3 and data.shape[0] == 4:
        rgb_image = data[:3, :, :]
        # Napari richiede l'ordine dei canali come (H, W, C) per le immagini RGB
        rgb_image = rgb_image.transpose(1, 2, 0)

        nir_image = data[3, :, :]

        # Aggiungiamo l'immagine RGB e il suo istogramma a napari
        viewer.add_image(rgb_image, name=filename + " RGB", contrast_limits=[0, 1])

        # Aggiungiamo l'immagine NIR e il suo istogramma a napari
        viewer.add_image(nir_image, name=filename + " NIR", contrast_limits=[0, 1])

    else:
        # Aggiungiamo l'immagine a napari
        add_image_and_histogram(viewer, data, filename)
        pass
