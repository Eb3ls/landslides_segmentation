# TODO: algoritmo di valutazione differenza tra input e output

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from Super_Resolution.config import ConfigRCAN, ConfigSwin2Mose, ConfigMyModel
from Super_Resolution.models_utils import launch_all


def main():
    # Prendiamo in input arg swin2mose o rcan
    parser = argparse.ArgumentParser(description="Launch Super Resolution Model")
    parser.add_argument(
        "--model",
        type=str,
        choices=["rcan", "swin2mose", "mymodel"],
    )
    args = parser.parse_args()
    model_type = args.model

    if model_type == "rcan":
        config = ConfigRCAN()
    elif model_type == "swin2mose":
        config = ConfigSwin2Mose()
    elif model_type == "mymodel":
        config = ConfigMyModel()
    else:
        raise ValueError(f"Model type {model_type} is not supported.")

    launch_all(config)


if __name__ == "__main__":
    main()
