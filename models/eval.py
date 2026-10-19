import argparse
import json
from pathlib import Path

import pytorch_lightning as pl
import torch

from src.data import AM4DataModule
from src.models import load_model_from_checkpoint
from src.utils import get_base_dir, local_run


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained checkpoint on AM4.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model", choices=["unet", "attn_unet", "attention_unet"], required=True)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--base-dir", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--no-validate", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--accelerator", default="auto")
    parser.add_argument("--devices", default="auto")
    return parser.parse_args()


def main():
    args = parse_args()

    torch.set_float32_matmul_precision("high")

    base_dir = args.base_dir if args.base_dir is not None else get_base_dir()

    batch_size = args.batch_size
    if batch_size is None:
        batch_size = 4 if local_run() else 32

    num_workers = args.num_workers
    if num_workers is None:
        num_workers = 0 if local_run() else 4

    datamodule = AM4DataModule(
        base_dir=base_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        download=not args.no_download,
        validate=not args.no_validate,
    )

    model = load_model_from_checkpoint(
        args.model,
        args.checkpoint,
        num_inputs=4,
        num_outputs=1,
    )

    trainer = pl.Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        logger=False,
    )

    if args.split == "test":
        metrics = trainer.test(model, datamodule=datamodule)
    else:
        metrics = trainer.validate(model, datamodule=datamodule)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"Saved metrics to {args.output}")


if __name__ == "__main__":
    main()
