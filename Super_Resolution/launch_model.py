import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from Super_Resolution.config import load_config


def main():
    parser = argparse.ArgumentParser(description="Launch Super Resolution Model")
    parser.add_argument(
        "--model",
        type=str,
        choices=["rcan", "swin2mose", "mymodel", "drct"],
        required=True,
        help="Model family selected by the configuration schema",
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to a model configuration file; defaults to the model example path",
    )
    # Fine-tune flags
    parser.add_argument(
        "--finetune", action="store_true", help="Enable fine-tuning mode"
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        help="Path to checkpoint (.pth) to load for fine-tuning",
    )
    parser.add_argument(
        "--scope",
        type=str,
        default="head",
        choices=["head"],
        help="Fine-tuning scope (head-only supported)",
    )

    args = parser.parse_args()
    if args.finetune and not args.ckpt:
        parser.error("--ckpt is required with --finetune")
    model_type = args.model

    try:
        config = load_config(model_type, args.config)
    except (OSError, ValueError, TypeError) as exc:
        parser.error(str(exc))

    try:
        setattr(config.train, "finetune", bool(args.finetune))
        setattr(config.train, "finetune_from", str(args.ckpt))
        setattr(config.train, "finetune_scope", str(args.scope))
    except Exception:
        pass

    # Training imports optional visualization dependencies, so keep --help usable
    # in a minimal environment.
    from Super_Resolution.models_utils import launch_all

    launch_all(config)


if __name__ == "__main__":
    main()
