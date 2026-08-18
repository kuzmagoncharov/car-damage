"""Копирует фото из data/raw в data/split: train / val / test."""

from __future__ import annotations

import random
import shutil
from collections import Counter
from pathlib import Path

from sklearn.model_selection import train_test_split

from src.config import (
    CLASSES,
    IMAGE_EXTENSIONS,
    RAW_DIR,
    SEED,
    SPLIT_DIR,
    TEST_RATIO,
    TRAIN_RATIO,
    VAL_RATIO,
    configure_console,
)


def collect_images(class_name: str) -> list[Path]:
    folder = RAW_DIR / class_name
    if not folder.is_dir():
        raise FileNotFoundError(f"Нет папки класса: {folder}")
    images = [
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    if not images:
        raise FileNotFoundError(f"В {folder} нет изображений")
    return sorted(images)


def split_paths(paths: list[Path]) -> dict[str, list[Path]]:
    train_paths, rest_paths = train_test_split(
        paths,
        train_size=TRAIN_RATIO,
        random_state=SEED,
        shuffle=True,
    )
    relative_val = VAL_RATIO / (VAL_RATIO + TEST_RATIO)
    val_paths, test_paths = train_test_split(
        rest_paths,
        train_size=relative_val,
        random_state=SEED,
        shuffle=True,
    )
    return {"train": train_paths, "val": val_paths, "test": test_paths}


def reset_split_dir() -> None:
    if SPLIT_DIR.exists():
        shutil.rmtree(SPLIT_DIR)
    for split in ("train", "val", "test"):
        for class_name in CLASSES:
            (SPLIT_DIR / split / class_name).mkdir(parents=True, exist_ok=True)


def copy_split(class_name: str, split_map: dict[str, list[Path]]) -> None:
    for split, paths in split_map.items():
        for src in paths:
            dst = SPLIT_DIR / split / class_name / src.name
            shutil.copy2(src, dst)


def print_summary(counts: dict[str, Counter]) -> None:
    print("Разбиение готово:\n")
    header = f"{'split':<8} {'damaged':>10} {'normal':>10} {'всего':>10}"
    print(header)
    print("-" * len(header))
    for split in ("train", "val", "test"):
        damaged = counts[split]["damaged"]
        normal = counts[split]["normal"]
        print(f"{split:<8} {damaged:>10} {normal:>10} {damaged + normal:>10}")
    print(f"\nФайлы скопированы в: {SPLIT_DIR}")


def main() -> None:
    configure_console()
    random.seed(SEED)
    reset_split_dir()
    counts = {split: Counter() for split in ("train", "val", "test")}

    for class_name in CLASSES:
        images = collect_images(class_name)
        split_map = split_paths(images)
        copy_split(class_name, split_map)
        for split, paths in split_map.items():
            counts[split][class_name] = len(paths)

    print_summary(counts)


if __name__ == "__main__":
    main()
