# TODO: algoritmo di valutazione differenza tra input e output

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from Super_Resolution.config import load_config
from Super_Resolution.models_utils import launch_all


def main():
    parser = argparse.ArgumentParser(description="Launch Super Resolution Model")
    parser.add_argument(
        "--model",
        type=str,
        choices=["rcan", "swin2mose", "mymodel", "drct"],
    )
    args = parser.parse_args()
    model_type = args.model

    try:
        config = load_config(model_type)
    except ValueError as e:
        print(f"Errore: {e}")
        return

    launch_all(config)


if __name__ == "__main__":
    main()
