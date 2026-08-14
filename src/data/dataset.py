"""Oxford 102 Flowers dataset wired through the preprocessing pipeline.

Uses torchvision.datasets.Flowers102, which downloads the dataset
automatically on first use and exposes the paper's official
train/val/test split.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import Flowers102

from src.preprocessing.pipeline import PreprocessingPipeline

IMAGE_SIZE = 224


class PreprocessedFlowers102(Dataset):
    """Wraps Flowers102, resizing then running every image through the
    CLAHE -> bilateral -> ROI -> texture -> color-norm pipeline before
    returning a (C, H, W) float tensor.
    """

    def __init__(self, root: str, split: str = "train", download: bool = True,
                 image_size: int = IMAGE_SIZE, augment: bool = False):
        self.base = Flowers102(root=root, split=split, download=download)
        self.image_size = image_size
        self.pipeline = PreprocessingPipeline()
        self.resize = transforms.Resize((image_size, image_size))
        self.augment = transforms.RandomHorizontalFlip(p=0.5) if augment else None

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        image, label = self.base[idx]
        image = self.resize(image)
        if self.augment is not None:
            image = self.augment(image)

        np_image = np.array(image.convert("RGB"), dtype=np.uint8)
        processed = self.pipeline(np_image)  # (H, W, 3) float32, normalized
        tensor = torch.from_numpy(processed).permute(2, 0, 1).contiguous().float()
        return tensor, label


def get_dataloaders(root: str = "./data/flowers102", batch_size: int = 32,
                     num_workers: int = 4, image_size: int = IMAGE_SIZE):
    train_ds = PreprocessedFlowers102(root, split="train", image_size=image_size, augment=True)
    val_ds = PreprocessedFlowers102(root, split="val", image_size=image_size, augment=False)
    test_ds = PreprocessedFlowers102(root, split="test", image_size=image_size, augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                               num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader


NUM_CLASSES = 102
