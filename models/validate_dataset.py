import argparse
from pathlib import Path

from src.data import AM4DataModule
from src.utils import get_base_dir
from src.validation import DatasetValidator


def parse_args():
    parser = argparse.ArgumentParser(description="Validate the AM4 dataset.")
    parser.add_argument("--base-dir", type=Path, default=None)
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--prob", type=float, default=0.5)
    parser.add_argument("--brightness-threshold", type=int, default=5000)
    parser.add_argument("--overlap-buffer", type=float, default=5)
    parser.add_argument("--plot-brightest", action="store_true")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--no-drive-cache", action="store_true")
    parser.add_argument("--no-move-invalid", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    base_dir = args.base_dir if args.base_dir is not None else get_base_dir()
    dataset_root = args.dataset_root

    if args.download:
        datamodule = AM4DataModule(
            base_dir=base_dir,
            download=True,
            validate=False,
            use_drive_cache=not args.no_drive_cache,
        )
        datamodule.prepare_data()

    if dataset_root is None:
        dataset_root = base_dir / "data" / "AM4" / "amazon"

    validator = DatasetValidator(
        dataset_root=dataset_root,
        prob=args.prob,
        brightness_threshold=args.brightness_threshold,
        overlap_buffer=args.overlap_buffer,
        move_invalid=not args.no_move_invalid,
    )

    validator.run_all(plot_brightest=args.plot_brightest)


if __name__ == "__main__":
    main()
