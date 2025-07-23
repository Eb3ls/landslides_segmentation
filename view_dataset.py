import os
import napari
import rasterio
from data_utils import (
    get_data,
    print_info,
)
from view_utils import view_data


def main():
    directory = "Comuni/"

    if not os.path.exists(directory):
        print("Directory does not exist.")
        exit()

    viewer = napari.Viewer()
    for comune in os.listdir(directory):
        for filename in os.listdir(directory + comune):
            path = os.path.join(directory + comune, filename)

            with rasterio.open(path) as src:
                print_info(src)
                data = get_data(src, True)
                view_data(data, path, viewer)

    # Nascondiamo tutte le immagini inizialmente
    for layer in viewer.layers:
        layer.visible = False
    napari.run()


if __name__ == "__main__":
    main()
