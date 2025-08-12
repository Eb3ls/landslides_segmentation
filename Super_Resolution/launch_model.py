# TODO: algoritmo di valutazione differenza tra input e output

import os
import sys


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from Super_Resolution.models_utils import launch_all


def main():
    # Prendiamo in input arg swin2mose o rcan
    parser = argparse.ArgumentParser(description="Launch Super Resolution Model")
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["rcan", "swin2mose"],
        default="swin2mose",
    )
    args = parser.parse_args()
    model_type = args.model_type

    launch_all(model_type=model_type)


if __name__ == "__main__":
    main()
