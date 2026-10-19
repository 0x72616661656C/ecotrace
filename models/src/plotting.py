from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio as rio
import rasterio.plot as rplt
import torch

from .utils import get_device


def plot_nrg(src):
    """Plot an NRG false-color image from an opened rasterio source."""
    n = src.read(4)
    r = src.read(1)
    g = src.read(2)

    nrg = np.dstack((n, r, g))
    nrg = nrg.astype(float) / 6000
    nrg = np.clip(nrg, 0, 1)

    plt.figure(figsize=(7, 7))
    plt.imshow(nrg)
    plt.title("NRG Image")
    plt.xlabel("Column #")
    plt.ylabel("Row #")
    plt.show()


def make_img_plots(src):
    """Plot RGB composite, NIR band, NDVI, and NRG image."""
    red_band = src.read(1).astype("float32")
    green_band = src.read(2).astype("float32")
    nir_band = src.read(4).astype("float32")

    denominator = nir_band + red_band
    ndvi = np.where(denominator == 0, 0, (nir_band - red_band) / denominator)
    ndvi = np.clip(ndvi, -1, 1)

    nrg = np.dstack((nir_band, red_band, green_band))
    nrg = nrg.astype(float) / 6000
    nrg = np.clip(nrg, 0, 1)

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.flatten()

    rplt.show(src, ax=axes[0])
    axes[0].set_title("RGB Composite")

    im = axes[1].imshow(nir_band, cmap="Reds")
    axes[1].set_title("NIR Band")
    plt.colorbar(im, ax=axes[1], shrink=0.7)

    im = axes[2].imshow(ndvi, cmap="RdYlGn", vmin=-1, vmax=1)
    axes[2].set_title("NDVI")
    plt.colorbar(im, ax=axes[2], shrink=0.7)

    axes[3].imshow(nrg)
    axes[3].set_title("NRG Image")

    plt.tight_layout()
    plt.show()


def image_tensor_to_rgb(image):
    """Convert a [C,H,W] tensor or array to clipped RGB [H,W,3]."""
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()

    if image.ndim == 3 and image.shape[0] >= 3:
        rgb = np.transpose(image[:3], (1, 2, 0))
    elif image.ndim == 3 and image.shape[-1] >= 3:
        rgb = image[..., :3]
    else:
        return image.squeeze()

    return np.clip(rgb, 0, 1)


def plot_prediction_triplet(image, pred_mask, true_mask=None, save_path=None):
    """Plot input image, predicted mask, and optionally true mask."""
    ncols = 3 if true_mask is not None else 2
    plt.figure(figsize=(4 * ncols, 4))

    plt.subplot(1, ncols, 1)
    plt.imshow(image_tensor_to_rgb(image))
    plt.title("Input Image")
    plt.axis("off")

    plt.subplot(1, ncols, 2)
    if isinstance(pred_mask, torch.Tensor):
        pred_mask = pred_mask.detach().cpu().squeeze().numpy()
    plt.imshow(pred_mask, cmap="gray")
    plt.title("Predicted Mask")
    plt.axis("off")

    if true_mask is not None:
        plt.subplot(1, ncols, 3)
        if isinstance(true_mask, torch.Tensor):
            true_mask = true_mask.detach().cpu().squeeze().numpy()
        plt.imshow(true_mask, cmap="gray")
        plt.title("True Mask")
        plt.axis("off")

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")

    plt.show()


def plot_dataset_prediction(model, dataset, index=0, save_path=None):
    """Plot prediction for one item from a dataset."""
    device = get_device()
    model = model.to(device)
    model.eval()

    image, true_mask = dataset[index]
    image_batch = image.unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(image_batch)
        pred_mask = (logits > 0.0).float()

    plot_prediction_triplet(
        image=image,
        pred_mask=pred_mask.squeeze(0),
        true_mask=true_mask,
        save_path=save_path,
    )
