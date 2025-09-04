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
    # Fine-tune flags
    parser.add_argument(
        "--finetune", action="store_true", help="Enable fine-tuning mode"
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        help="Path to checkpoint (.pth) to load for fine-tuning",
        required="--finetune" in sys.argv,
    )
    parser.add_argument(
        "--scope",
        type=str,
        default="head",
        choices=["head"],
        help="Fine-tuning scope (head-only supported)",
    )

    args = parser.parse_args()
    model_type = args.model

    try:
        config = load_config(model_type)
    except ValueError as e:
        print(f"Errore: {e}")
        return

    try:
        setattr(config.train, "finetune", bool(args.finetune))
        setattr(config.train, "finetune_from", str(args.ckpt))
        setattr(config.train, "finetune_scope", str(args.scope))
    except Exception:
        pass

    launch_all(config)


if __name__ == "__main__":
    main()
