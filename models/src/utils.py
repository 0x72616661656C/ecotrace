from pathlib import Path
import os
import random

import numpy as np
import torch


def local_run():
    """Return True outside Google Colab."""
    return not os.path.exists("/content")


def get_base_dir():
    """
    Return the project base directory.

    In Colab, this assumes the project is cloned to:
    /content/deforestation-unet

    Locally, this supports normal Python scripts and interactive notebooks.
    """
    if not local_run():
        return Path("/content/deforestation-unet")

    try:
        return Path(__file__).resolve().parent.parent
    except NameError:
        return Path.cwd()


def set_seed(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    """Return CUDA device when available, otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
