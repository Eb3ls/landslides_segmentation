import os
import napari
from data_utils import get_super_resolution_stack
from view_utils import view_dataset


def main():
    directory = "Comuni/Casola-Valsenio"

    if not os.path.exists(directory):
        print("Directory 'Brisighella' does not exist. Please check the path.")
        exit()

    viewer = napari.Viewer()
    view_dataset(viewer, directory)
    napari.run()

    get_super_resolution_stack("Casola-Valsenio")


if __name__ == "__main__":
    main()
