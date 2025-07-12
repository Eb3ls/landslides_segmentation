import napari
import os

from view_utils import view_dataset


def main():
    directory = 'Comuni/Casola-Valsenio'

    if not os.path.exists(directory):
        print("Directory 'Brisighella' does not exist. Please check the path.")
        exit()

    viewer = napari.Viewer()
    view_dataset(viewer, directory)
    napari.run()


if __name__ == "__main__":
    main()
