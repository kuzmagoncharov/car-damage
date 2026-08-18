from __future__ import annotations

from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.config import (
    BATCH_SIZE,
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    NUM_WORKERS,
    SPLIT_DIR,
)


def get_transforms(train: bool) -> transforms.Compose:
    if train:
        return transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.7, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
                transforms.RandomRotation(12),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def get_infer_transform() -> transforms.Compose:
    return get_transforms(train=False)


def create_loaders(batch_size: int = BATCH_SIZE) -> tuple[dict[str, DataLoader], list[str]]:
    loaders: dict[str, DataLoader] = {}
    class_names: list[str] | None = None

    for split, is_train in (("train", True), ("val", False), ("test", False)):
        folder = SPLIT_DIR / split
        dataset = datasets.ImageFolder(folder, transform=get_transforms(is_train))
        if class_names is None:
            class_names = list(dataset.classes)
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=is_train,
            num_workers=NUM_WORKERS,
            pin_memory=False,
        )

    if class_names is None:
        raise FileNotFoundError(f"Не найдены данные в {SPLIT_DIR}")
    return loaders, class_names


def image_paths_in(folder: Path) -> list[Path]:
    return sorted(path for path in folder.iterdir() if path.is_file())
