from pathlib import Path
import shutil

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio as rio
import rasterio.plot as rplt


def centroid(bounds):
    """Return the centroid of raster bounds."""
    return (
        (bounds.left + bounds.right) / 2,
        (bounds.top + bounds.bottom) / 2,
    )


class DatasetValidator:
    """
    Validate AM4 image-mask pairs before training.

    This class checks metadata, corrupt files, empty images, empty masks,
    suspicious band values, image brightness, and geographical overlap
    between train, validation, and test splits.

    Invalid images and their corresponding masks are moved to an error folder
    by default, matching the notebook workflow.
    """

    def __init__(
        self,
        dataset_root,
        prob=0.5,
        brightness_threshold=5000,
        overlap_buffer=5,
        move_invalid=True,
    ):
        self.dataset_root = Path(dataset_root)
        self.prob = prob
        self.brightness_threshold = brightness_threshold
        self.overlap_buffer = overlap_buffer
        self.move_invalid = move_invalid

        self.dataset_dir_imgs = {
            "train": self.dataset_root / "Training" / "image",
            "test": self.dataset_root / "Test" / "image",
            "val": self.dataset_root / "Validation" / "images",
        }

        self.dataset_dir_masks = {
            "train": self.dataset_root / "Training" / "label",
            "test": self.dataset_root / "Test" / "mask",
            "val": self.dataset_root / "Validation" / "masks",
        }

        self.image_dict = {
            phase: {img.stem: img.name for img in img_dir.glob("*.tif")}
            for phase, img_dir in self.dataset_dir_imgs.items()
        }

        self.mask_dict = {
            phase: {mask.stem: mask.name for mask in mask_dir.glob("*.tif")}
            for phase, mask_dir in self.dataset_dir_masks.items()
        }

        self.brightness_scores = {}
        self.images_to_remove = set()

        self.centroids = {
            "train": {},
            "test": {},
            "val": {},
        }

    def validate_metadata(self, label, phase):
        """Validate metadata for one image-mask pair."""
        img_path = self.dataset_dir_imgs[phase] / self.image_dict[phase][label]

        if label not in self.mask_dict[phase]:
            print(f"{phase}: mask not found for image {label}.")
            self.images_to_remove.add(img_path)
            return False

        mask_path = self.dataset_dir_masks[phase] / self.mask_dict[phase][label]

        try:
            with rio.open(img_path) as image, rio.open(mask_path) as mask:
                try:
                    assert image.bounds == mask.bounds
                    assert image.crs == mask.crs
                    assert image.nodata is None and mask.nodata is None
                    assert image.res == mask.res
                    assert image.width == mask.width
                    assert image.height == mask.height
                    assert image.count == 4
                    assert mask.count == 1
                    assert image.dtypes[0] == "uint16"
                    assert mask.dtypes[0] == "uint8"

                except AssertionError:
                    print(f"Metadata validation failed for {phase} label {label}.")
                    print(f"Image bounds: {image.bounds}, Mask bounds: {mask.bounds}")
                    print(f"Image CRS: {image.crs}, Mask CRS: {mask.crs}")
                    print(f"Image nodata: {image.nodata}, Mask nodata: {mask.nodata}")
                    print(f"Image resolution: {image.res}, Mask resolution: {mask.res}")
                    print(
                        f"Image dimensions: ({image.width}, {image.height}), "
                        f"Mask dimensions: ({mask.width}, {mask.height})"
                    )
                    print(f"Image bands: {image.count}, Mask bands: {mask.count}")
                    print(f"Image dtypes: {image.dtypes}, Mask dtypes: {mask.dtypes}")

                    self.images_to_remove.add(img_path)
                    return False

                self.centroids[phase][label] = centroid(image.bounds)
                return True

        except Exception as e:
            print(f"Could not open image-mask pair for {phase} label {label}: {e}")
            self.images_to_remove.add(img_path)
            return False

    def validate_image_mask(self, label, phase):
        """
        Validate actual image and mask values.

        This reads raster data, so it can be slower than metadata validation.
        That is why it can be run only for a random subset using self.prob.
        """
        img_path = self.dataset_dir_imgs[phase] / self.image_dict[phase][label]
        mask_path = self.dataset_dir_masks[phase] / self.mask_dict[phase][label]

        try:
            with rio.open(img_path) as image, rio.open(mask_path) as mask:
                rgb = image.read([1, 2, 3]).astype(np.int32)

                bright_pixels = np.sum(
                    (rgb[0] + rgb[1] + rgb[2]) > self.brightness_threshold
                )

                self.brightness_scores[str(img_path)] = bright_pixels

                if not np.any(rgb > 0):
                    print(f"Image for {phase} label {label} contains only zeros.")
                    self.images_to_remove.add(img_path)
                    return False

                mask_data = mask.read(1)

                if not np.any(mask_data > 0):
                    print(f"Mask for {phase} label {label} contains only zeros.")
                    self.images_to_remove.add(img_path)
                    return False

                bands = image.read([1, 2, 3, 4]).astype(np.int32)
                band_names = ["red", "green", "blue", "nir"]

                for band_name, band in zip(band_names, bands):
                    if np.any(band > 10000):
                        print(
                            f"Image {img_path} has {band_name} band values above 10000."
                        )
                        self.images_to_remove.add(img_path)
                        return False

                return True

        except Exception as e:
            print(f"Value validation failed for {phase} label {label}: {e}")
            self.images_to_remove.add(img_path)
            return False

    def _make_centroid_gdf(self, phase):
        """Create a GeoDataFrame of buffered centroids for one split."""
        labels = list(self.centroids[phase].keys())

        if len(labels) == 0:
            return gpd.GeoDataFrame(columns=["label", "geometry"])

        coords = [self.centroids[phase][label] for label in labels]

        gdf = gpd.GeoDataFrame(
            {"label": labels},
            geometry=gpd.points_from_xy(
                [c[0] for c in coords],
                [c[1] for c in coords],
            ),
        )

        gdf["geometry"] = gdf.geometry.buffer(self.overlap_buffer)
        return gdf

    def _mark_overlapping_images(self, left_phase, right_phase):
        """Mark images from right_phase invalid if they overlap with left_phase."""
        left_gdf = self._make_centroid_gdf(left_phase)
        right_gdf = self._make_centroid_gdf(right_phase)

        if left_gdf.empty or right_gdf.empty:
            return

        spatial_index = left_gdf.sindex

        for right_idx, right_geom in enumerate(right_gdf.geometry):
            candidate_indices = spatial_index.intersection(right_geom.bounds)

            for left_idx in candidate_indices:
                left_geom = left_gdf.geometry.iloc[left_idx]

                if right_geom.intersects(left_geom):
                    right_label = right_gdf.label.iloc[right_idx]
                    left_label = left_gdf.label.iloc[left_idx]

                    print(
                        f"{right_phase} image {right_label} overlaps with "
                        f"{left_phase} image {left_label}."
                    )

                    image_path = (
                        self.dataset_dir_imgs[right_phase]
                        / self.image_dict[right_phase][right_label]
                    )

                    self.images_to_remove.add(image_path)

    def check_overlap(self):
        """Check possible geographical leakage between train, validation, and test."""
        self._mark_overlapping_images("train", "val")
        self._mark_overlapping_images("train", "test")
        self._mark_overlapping_images("test", "val")

    def plot_top_brightest(self, top_n=5):
        """Plot the brightest images found during value validation."""
        if len(self.brightness_scores) == 0:
            print("No brightness scores available.")
            return

        top_brightest = sorted(
            self.brightness_scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:top_n]

        n = len(top_brightest)
        fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))

        if n == 1:
            axes = [axes]

        for ax, (image_path, score) in zip(axes, top_brightest):
            print(f"Image: {image_path}, Brightness score: {score}")

            with rio.open(image_path) as src:
                rplt.show(src, ax=ax)

            ax.set_title(Path(image_path).name)

        plt.tight_layout()
        plt.show()

    def _get_phase_from_image_path(self, image_path):
        """Infer split name from an image path."""
        image_path = Path(image_path)

        for phase, img_dir in self.dataset_dir_imgs.items():
            if image_path.parent == img_dir:
                return phase

        return None

    def remove_invalid_images(self):
        """Move invalid images and corresponding masks to error folders."""
        if not self.move_invalid:
            if self.images_to_remove:
                print("Invalid images found but not moved because move_invalid=False.")
                for image_path in sorted(self.images_to_remove):
                    print(image_path)
            return

        for image_path in self.images_to_remove:
            image_path = Path(image_path)
            phase = self._get_phase_from_image_path(image_path)

            if phase is None:
                print(f"Could not infer phase for {image_path}. Skipping move.")
                continue

            label = image_path.stem
            mask_path = self.dataset_dir_masks[phase] / f"{label}.tif"

            image_error_dir = image_path.parent / "error"
            mask_error_dir = mask_path.parent / "error"

            image_error_dir.mkdir(exist_ok=True)
            mask_error_dir.mkdir(exist_ok=True)

            if image_path.exists():
                shutil.move(str(image_path), str(image_error_dir / image_path.name))
                print(f"Moved image {image_path} to {image_error_dir}.")

            if mask_path.exists():
                shutil.move(str(mask_path), str(mask_error_dir / mask_path.name))
                print(f"Moved mask {mask_path} to {mask_error_dir}.")

    def run_all(self, plot_brightest=False):
        """Run all validation checks."""
        for phase in ["train", "test", "val"]:
            print(f"Validating {phase} images and masks.")

            for label in list(self.image_dict[phase].keys()):
                metadata_valid = self.validate_metadata(label, phase)

                if metadata_valid and np.random.rand() < self.prob:
                    self.validate_image_mask(label, phase)

        self.check_overlap()

        if plot_brightest:
            self.plot_top_brightest(top_n=5)

        self.remove_invalid_images()
