from pathlib import Path
import shutil
import subprocess

import numpy as np
import patoolib
import pytorch_lightning as pl
import rasterio as rio
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from .utils import get_base_dir, local_run
from .validation import DatasetValidator


class MinMaxScale:
    """Min-max scale one image array to the range [0, 1]."""

    def __call__(self, img):
        img_min = np.min(img)
        img_max = np.max(img)
        return (img - img_min) / (img_max - img_min + 1e-8)


class ToTensor:
    """Convert a NumPy array to a PyTorch tensor."""

    def __call__(self, img):
        return torch.from_numpy(img)


class DeforestationDataset(Dataset):
    """
    PyTorch Dataset for one AM4 split.

    root_directory should be one of:
    data/AM4/amazon/Training
    data/AM4/amazon/Test
    data/AM4/amazon/Validation

    image_name is the image folder name.
    label_name is the mask folder name.
    """

    def __init__(
        self,
        root_directory,
        image_name,
        label_name,
        indices=None,
        transform=None,
    ):
        self.root_directory = Path(root_directory)
        self.image_directory = self.root_directory / str(image_name)
        self.label_directory = self.root_directory / str(label_name)

        self.image_paths = sorted(self.image_directory.glob("*.tif"))
        self.label_paths = sorted(self.label_directory.glob("*.tif"))

        self.indices = indices
        self.transform = transform

    def __len__(self):
        if self.indices is not None:
            return len(self.indices)
        return len(self.image_paths)

    def __getitem__(self, index):
        if self.indices is not None:
            img_path = self.image_paths[self.indices[index]]
        else:
            img_path = self.image_paths[index]

        img_name = img_path.name
        lbl_path = self.label_directory / img_name

        with rio.open(img_path) as img:
            img = img.read().astype("float32")

        if self.transform is not None:
            img = self.transform(img)

        with rio.open(lbl_path) as lbl:
            lbl = lbl.read().astype("float32")

        lbl = torch.from_numpy(lbl)
        return img, lbl


class AM4DataModule(pl.LightningDataModule):
    """LightningDataModule for the AM4 deforestation dataset."""

    def __init__(
        self,
        base_dir=None,
        batch_size=None,
        num_workers=None,
        pin_memory=True,
        validate_prob=0.5,
        download=True,
        validate=True,
        use_drive_cache=True,
        move_invalid=True,
    ):
        super().__init__()

        self.base_dir = Path(base_dir) if base_dir is not None else get_base_dir()

        self.data_dir = self.base_dir / "data" / "AM4"
        self.dataset_root = self.data_dir / "amazon"

        self.url = "https://doi.org/10.5281/zenodo.4498086"
        self.archive_name = "AMAZON.rar"

        self.download = download
        self.validate = validate
        self.use_drive_cache = use_drive_cache
        self.validate_prob = validate_prob
        self.move_invalid = move_invalid

        if batch_size is None:
            self.batch_size = 4 if local_run() else 32
        else:
            self.batch_size = batch_size

        if num_workers is None:
            self.num_workers = 0 if local_run() else 4
        else:
            self.num_workers = num_workers

        self.pin_memory = pin_memory

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        self.transform = transforms.Compose([MinMaxScale(), ToTensor()])

    def _mount_drive_if_needed(self):
        """Mount Google Drive in Colab when used for archive caching."""
        if local_run() or not self.use_drive_cache:
            return None

        from google.colab import drive

        drive.mount("/content/drive")

        drive_dir = Path("/content/drive/MyDrive")
        drive_dir.mkdir(parents=True, exist_ok=True)
        return drive_dir

    def _download_and_extract_am4(self):
        """Download and extract AM4."""
        self.data_dir.mkdir(parents=True, exist_ok=True)

        local_archive_path = self.data_dir / self.archive_name
        drive_dir = self._mount_drive_if_needed()

        if drive_dir is not None:
            drive_archive_path = drive_dir / self.archive_name
        else:
            drive_archive_path = None

        archive_source_path = None

        if drive_archive_path is not None and drive_archive_path.exists():
            archive_source_path = drive_archive_path
            print(f"AM4 archive found in Google Drive: {archive_source_path}")

        elif local_archive_path.exists():
            archive_source_path = local_archive_path
            print(f"AM4 archive found locally: {archive_source_path}")

        else:
            print("Downloading AM4 dataset.")

            cmd = [
                "zenodo_get",
                "-o",
                str(self.data_dir),
                "-g",
                self.archive_name,
                self.url,
            ]

            subprocess.run(cmd, check=True)

            archive_source_path = local_archive_path
            print(f"AM4 downloaded to {self.data_dir}.")

            if drive_archive_path is not None and not drive_archive_path.exists():
                shutil.copy2(local_archive_path, drive_archive_path)
                print(f"Copied AM4 archive to Google Drive: {drive_archive_path}")

        if self.dataset_root.exists():
            print(f"AM4 already extracted at {self.dataset_root}.")
            return

        print("Extracting AM4.")

        patoolib.extract_archive(
            str(archive_source_path),
            outdir=str(self.data_dir),
        )

        extracted_dir_name = self.archive_name.rsplit(".", 1)[0]
        extracted_path = self.data_dir / extracted_dir_name

        if extracted_path.exists() and extracted_path != self.dataset_root:
            extracted_path.rename(self.dataset_root)

        print(f"AM4 extracted to {self.dataset_root}.")

    def _validate_am4(self):
        """Validate AM4 once and write a marker file after validation."""
        validation_marker = self.data_dir / "am4_validation_complete.txt"

        if validation_marker.exists():
            print("AM4 validation already completed. Skipping validation.")
            return

        print("Running AM4 dataset validation.")

        validator = DatasetValidator(
            dataset_root=self.dataset_root,
            prob=self.validate_prob,
            move_invalid=self.move_invalid,
        )

        validator.run_all(plot_brightest=False)
        validation_marker.write_text("AM4 validation completed.\n")

        print(f"Wrote validation marker to {validation_marker}.")

    def prepare_data(self):
        """Download, extract, and validate data."""
        if self.download:
            self._download_and_extract_am4()

        if self.validate:
            self._validate_am4()

    def setup(self, stage=None):
        """Create Dataset objects."""
        if stage == "fit" or stage is None:
            self.train_dataset = DeforestationDataset(
                root_directory=self.dataset_root / "Training",
                image_name="image",
                label_name="label",
                transform=self.transform,
            )

            self.val_dataset = DeforestationDataset(
                root_directory=self.dataset_root / "Validation",
                image_name="images",
                label_name="masks",
                transform=self.transform,
            )

        if stage == "test" or stage is None:
            self.test_dataset = DeforestationDataset(
                root_directory=self.dataset_root / "Test",
                image_name="image",
                label_name="mask",
                transform=self.transform,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
