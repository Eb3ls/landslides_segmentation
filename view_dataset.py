import os
from typing import List
import napari
import numpy as np
import rasterio
from data_utils import (
    MAIN_DIR,
    ComuneType,
    generate_dataset_mask,
    get_data,
    get_random_patch,
    get_super_resolution_stack,
    print_info,
)
from view_utils import view_data


def get_user_choice(num_options: int, prompt: str) -> int:
    """Chiede all'utente di scegliere tra le opzioni disponibili."""
    while True:
        try:
            choice = int(input(prompt).strip())
            if 1 <= choice <= num_options:
                return choice
            else:
                print(f"Non é tra le scelte")
        except ValueError:
            print("Inserisci un numero valido.")


def get_patch_size() -> int:
    """Chiede all'utente la dimensione delle patch da visualizzare."""
    while True:
        try:
            size = int(input("Inserisci la dimensione delle patch: ").strip())
            if size > 0:
                return size
            else:
                print("La dimensione deve essere positiva.")
        except ValueError:
            print("Inserisci un numero valido.")


def select_comuni() -> List[ComuneType]:
    """Permette all'utente di selezionare quali comuni vedere."""
    comuni_disponibili: List[ComuneType] = [
        "Brisighella",
        "Casola-Valsenio",
        "Modigliana",
        "Predappio",
    ]

    print("Vuoi visualizzare:")
    print("1. Tutti i comuni")
    print("2. Un comune specifico")

    choice = get_user_choice(2, "Scegli un'opzione: ")
    if choice == 1:
        return comuni_disponibili
    else:
        print("Comuni disponibili:")
        for i, comune in enumerate(comuni_disponibili, start=1):
            print(f"{i}. {comune}")

        comune_idx = get_user_choice(
            len(comuni_disponibili), "Scegli il numero del comune: "
        )
        return [comuni_disponibili[comune_idx - 1]]


def view_from_comuni(comuni_list: List[ComuneType]) -> None:
    """Visualizza le immagini dei comuni specificati in napari."""
    viewer = napari.Viewer()

    for comune in comuni_list:
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


def view_patches(comune: ComuneType, show_similar: bool, patch_size: int) -> None:
    """Visualizza patch con controllo di similarità."""

    mask = generate_dataset_mask(comune)
    _, stack_post = get_super_resolution_stack(comune)

    while True:
        print(f"Dataset shape: {stack_post[0].shape}, {stack_post[1].shape}")
        viewer = napari.Viewer()
        low_res_patch, high_res_patch, _ = get_random_patch(
            stack_post[0], stack_post[1], patch_size, mask, show_similar
        )
        view_data(high_res_patch, f"High Res Patch", viewer)
        view_data(low_res_patch, f"Low Res Patch", viewer)
        napari.run()


def main():
    if not os.path.exists(MAIN_DIR):
        print("Directory does not exist.")
        exit()

    # Chiediamo se vuole vedere tutti i comuni o uno specifico
    comuni_list = select_comuni()
    if len(comuni_list) == 1:
        choice = get_user_choice(2, "Vuoi vedere le immagini (1) o le patch (2)?")
        if choice == 1:
            view_from_comuni(comuni_list)
        else:
            choice = get_user_choice(2, "Vuoi vedere le patch simili (1) o meno (2)?")
            size = get_patch_size()
            if choice == 1:
                view_patches(comuni_list[0], True, size)
            else:
                view_patches(comuni_list[0], False, size)
    else:
        view_from_comuni(comuni_list)


if __name__ == "__main__":
    main()
