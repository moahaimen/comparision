"""Global configuration for MOH-DETECTOR."""
from pathlib import Path
from dataclasses import dataclass


@dataclass
class Config:
    """Container for project-wide settings.

    Adjust these values according to your environment or override them via
    command-line arguments in the training and inference scripts.
    """

    # Paths
    data_root: Path = Path("data")
    alerts_root: Path = data_root / "alerts"
    dataset_root: Path = data_root / "ztf_yolo"

    # Image processing
    img_size: int = 512
    thr_sigma: float = 5.0
    max_blob_area: int = 2000

    # Training
    batch_size: int = 4
    num_workers: int = 2
    epochs: int = 10
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    train_split: float = 0.8

    # Patching
    patch_size: int = 256
    max_patches_per_image: int = 4


CFG = Config()
