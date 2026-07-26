from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .inference import infer_one_image


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run TableSeq inference on one table image.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to a TableSeq .pt checkpoint.")
    parser.add_argument("--image", type=Path, required=True, help="Path to a table image.")
    parser.add_argument("--tokenizer", type=Path, required=True, help="Path/name of the tokenizer.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/demo"),
        help="Directory for prediction artifacts.",
    )
    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    parser.add_argument("--device", default=default_device, help="Device to use, e.g. cuda or cpu.")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument(
        "--resize-factor",
        type=float,
        default=None,
        help=(
            "Image resize factor used by the checkpoint. New checkpoints store it automatically; "
            "pass it explicitly for older checkpoints."
        ),
    )
    parser.add_argument("--non-strict-load", action="store_true", help="Load checkpoint with strict=False.")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    outputs = infer_one_image(
        checkpoint=args.checkpoint,
        tokenizer=args.tokenizer,
        image=args.image,
        output_dir=args.output_dir,
        device=args.device,
        max_length=args.max_length,
        resize_factor=args.resize_factor,
        strict_load=not args.non_strict_load,
    )
    print(json.dumps(outputs, indent=2, ensure_ascii=False))
    print(f"\nOpen this file in a browser to inspect the prediction: {outputs['report']}")


if __name__ == "__main__":
    main()
