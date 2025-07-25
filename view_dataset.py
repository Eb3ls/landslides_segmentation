import os
from typing import cast
import napari
import numpy as np
import rasterio
from data_utils import (
    MAIN_DIR,
    ComuneType,
    generate_dataset_mask,
    get_data,
    print_info,
)
from view_utils import view_data


def main():

    if not os.path.exists(MAIN_DIR):
        print("Directory does not exist.")
        exit()

    viewer = napari.Viewer()
    for comune in os.listdir(MAIN_DIR):
        comune = cast(ComuneType, comune)
        mask = generate_dataset_mask(comune)

        for filename in os.listdir(MAIN_DIR + comune):
            path = os.path.join(MAIN_DIR + comune, filename)

            with rasterio.open(path) as src:
                print_info(src)
                data = get_data(src, True)

                # I sentinel non hanno nodata quindi applichiamo la maschera
                if "sentinel" in filename.lower():
                    data[:, ~mask] = np.nan

                view_data(data, path, viewer)

    # Nascondiamo tutte le immagini inizialmente
    for layer in viewer.layers:
        layer.visible = False
    napari.run()


if __name__ == "__main__":
    main()
