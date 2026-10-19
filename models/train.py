import argparse
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from src.data import AM4DataModule
from src.models import get_model
from src.utils import get_base_dir, local_run


class EpochMetricsPrinter(Callback):
    """Print the training and validation metrics once each epoch finishes."""

    metric_names = (
        "train_loss",
        "train_precision",
        "train_recall",
        "train_f1_score",
        "train_iou",
        "val_loss",
        "val_precision",
        "val_recall",
        "val_f1_score",
        "val_iou",
    )

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking or not trainer.is_global_zero:
            return

        metrics = trainer.callback_metrics
        formatted_metrics = []
        for name in self.metric_names:
            value = metrics.get(name)
            if value is None:
                continue
            if isinstance(value, torch.Tensor):
                value = value.detach().float().cpu().item()
            formatted_metrics.append(f"{name}={value:.4f}")

        learning_rate = trainer.optimizers[0].param_groups[0]["lr"]
        print(
            f"Epoch {trainer.current_epoch + 1}/{trainer.max_epochs} | "
            f"{' | '.join(formatted_metrics)} | lr={learning_rate:.2e}",
            flush=True,
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Train UNet or Attention UNet on AM4.")
    parser.add_argument("--model", choices=["unet", "attn_unet", "attention_unet"], default="unet")
    parser.add_argument("--base-dir", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--log-dir", type=Path, default=Path("tb_logs"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--validate-prob", type=float, default=0.5)
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--no-validate", action="store_true")
    parser.add_argument("--no-drive-cache", action="store_true")
    parser.add_argument("--no-pin-memory", action="store_true")
    parser.add_argument("--fast-dev-run", action="store_true")
    parser.add_argument("--accelerator", default="auto")
    parser.add_argument("--devices", default="auto")
    parser.add_argument("--log-every-n-steps", type=int, default=1)
    return parser.parse_args()


def default_epochs(model_name):
    if model_name == "unet":
        return 20
    return 60


def main():
    args = parse_args()

    torch.set_float32_matmul_precision("high")

    base_dir = args.base_dir if args.base_dir is not None else get_base_dir()
    checkpoint_dir = args.checkpoint_dir
    if not checkpoint_dir.is_absolute():
        checkpoint_dir = base_dir / checkpoint_dir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    log_dir = args.log_dir
    if not log_dir.is_absolute():
        log_dir = base_dir / log_dir
    log_dir.mkdir(parents=True, exist_ok=True)

    batch_size = args.batch_size
    if batch_size is None:
        batch_size = 4 if local_run() else 32

    num_workers = args.num_workers
    if num_workers is None:
        num_workers = 0 if local_run() else 4

    max_epochs = args.max_epochs if args.max_epochs is not None else default_epochs(args.model)

    for seed in args.seeds:
        logger_name = "attn_unet" if args.model in {"attn_unet", "attention_unet"} else "unet"
        logger = TensorBoardLogger(
            save_dir=str(log_dir),
            name=logger_name,
            version=f"seed_{seed}",
        )

        pl.seed_everything(seed, workers=True)

        datamodule = AM4DataModule(
            base_dir=base_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=not args.no_pin_memory,
            validate_prob=args.validate_prob,
            download=not args.no_download,
            validate=not args.no_validate,
            use_drive_cache=not args.no_drive_cache,
        )

        model = get_model(
            args.model,
            num_inputs=4,
            num_outputs=1,
            lr=args.lr,
        )

        checkpoint_callback = ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename=f"{logger_name}_seed_{seed}-epoch={{epoch:03d}}-val_loss={{val_loss:.4f}}",
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            save_last=True,
            auto_insert_metric_name=False,
        )

        trainer = pl.Trainer(
            default_root_dir=checkpoint_dir,
            accelerator=args.accelerator,
            devices=args.devices,
            max_epochs=max_epochs,
            log_every_n_steps=args.log_every_n_steps,
            fast_dev_run=args.fast_dev_run,
            logger=logger,
            callbacks=[EpochMetricsPrinter(), checkpoint_callback],
        )

        trainer.fit(model, datamodule=datamodule)
        best_checkpoint_path = checkpoint_callback.best_model_path if trainer.is_global_zero else ""
        best_checkpoint_path = trainer.strategy.broadcast(best_checkpoint_path, src=0)
        best_checkpoint = Path(best_checkpoint_path)
        if not best_checkpoint.is_file():
            raise RuntimeError("Training finished without producing a best-model checkpoint.")

        if trainer.is_global_zero:
            portable_weights_path = checkpoint_dir / f"{logger_name}_seed_{seed}_best.pt"
            checkpoint = torch.load(best_checkpoint, map_location="cpu", weights_only=False)
            torch.save(checkpoint["state_dict"], portable_weights_path)
            print(f"Best checkpoint: {best_checkpoint}", flush=True)
            print(f"Best model weights: {portable_weights_path}", flush=True)

        datamodule.setup(stage="test")
        trainer.test(model, datamodule=datamodule, ckpt_path=str(best_checkpoint))


if __name__ == "__main__":
    main()
