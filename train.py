"""Обучает классификатор damaged / normal и считает метрики на тесте."""

from __future__ import annotations

import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader
from torchvision import datasets

from src.config import (
    ARCH,
    BATCH_SIZE,
    CHECKPOINT_PATH,
    EPOCHS,
    LEARNING_RATE,
    MODELS_DIR,
    REPORTS_DIR,
    SEED,
    SPLIT_DIR,
    WEIGHT_DECAY,
    configure_console,
)
from src.dataset import create_loaders, get_transforms
from src.model import create_model


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    dev: torch.device,
) -> tuple[float, float]:
    train_mode = optimizer is not None
    model.train(train_mode)
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(dev)
        labels = labels.to(dev)
        with torch.set_grad_enabled(train_mode):
            logits = model(images)
            loss = criterion(logits, labels)
            if train_mode:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    dev: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    y_prob: list[float] = []

    for images, labels in loader:
        images = images.to(dev)
        logits = model(images)
        probs = torch.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)
        conf = probs.max(dim=1).values
        y_true.extend(labels.tolist())
        y_pred.extend(preds.cpu().tolist())
        y_prob.extend(conf.cpu().tolist())

    return np.array(y_true), np.array(y_pred), np.array(y_prob)


def save_confusion_matrix(
    matrix: np.ndarray,
    class_names: list[str],
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Предсказание")
    ax.set_ylabel("Истина")
    ax.set_title("Матрица ошибок")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, int(matrix[i, j]), ha="center", va="center")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def evaluate(model: nn.Module, class_names: list[str], dev: torch.device) -> None:
    test_dir = SPLIT_DIR / "test"
    dataset = datasets.ImageFolder(test_dir, transform=get_transforms(False))
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    y_true, y_pred, y_prob = collect_predictions(model, loader, dev)

    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )
    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=3,
        zero_division=0,
    )
    matrix = confusion_matrix(y_true, y_pred)

    errors: list[str] = []
    for index, (true_idx, pred_idx, prob) in enumerate(zip(y_true, y_pred, y_prob)):
        if true_idx != pred_idx:
            rel = Path(dataset.samples[index][0]).relative_to(SPLIT_DIR)
            errors.append(
                f"{rel}  истина={class_names[true_idx]}  "
                f"предсказание={class_names[pred_idx]}  "
                f"уверенность={prob:.0%}"
            )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    save_confusion_matrix(matrix, class_names, REPORTS_DIR / "confusion_matrix.png")

    lines = [
        f"accuracy:  {accuracy:.3f}",
        f"precision: {precision:.3f}  (macro)",
        f"recall:    {recall:.3f}  (macro)",
        f"f1:        {f1:.3f}  (macro)",
        "",
        report,
        "Ошибки на тесте:",
        *(errors if errors else ["нет ошибок"]),
    ]
    report_path = REPORTS_DIR / "metrics.txt"
    report_path.write_text("\n".join(lines), encoding="utf-8")

    print("\n=== Тест ===")
    print(f"accuracy:  {accuracy:.3f}")
    print(f"precision: {precision:.3f}")
    print(f"recall:    {recall:.3f}")
    print()
    print(report)
    print("Ошибки:")
    if errors:
        for item in errors:
            print(" -", item)
    else:
        print(" нет ошибок")
    print(f"\nОтчёт: {report_path}")
    print(f"Матрица ошибок: {REPORTS_DIR / 'confusion_matrix.png'}")


def main() -> None:
    configure_console()
    if not (SPLIT_DIR / "train").exists():
        raise SystemExit("Сначала выполните: python prepare_data.py")

    set_seed()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    dev = device()
    print(f"Устройство: {dev}")
    print(f"Архитектура: {ARCH}")

    loaders, class_names = create_loaders()
    model = create_model(arch=ARCH, num_classes=len(class_names), freeze_backbone=True)
    model = model.to(dev)

    criterion = nn.CrossEntropyLoss()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    best_val_loss = float("inf")
    best_val_acc = -1.0
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = run_epoch(model, loaders["train"], criterion, optimizer, dev)
        val_loss, val_acc = run_epoch(model, loaders["val"], criterion, None, dev)
        marker = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            marker = "  <- лучшая"
            torch.save(
                {
                    "arch": ARCH,
                    "class_names": class_names,
                    "model_state": model.state_dict(),
                },
                CHECKPOINT_PATH,
            )
        print(
            f"эпоха {epoch:02d}/{EPOCHS}  "
            f"train loss={train_loss:.3f} acc={train_acc:.3f}  "
            f"val loss={val_loss:.3f} acc={val_acc:.3f}{marker}"
        )

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=dev, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    print(f"\nСохранена лучшая модель: {CHECKPOINT_PATH}")
    print(f"Лучшая val loss: {best_val_loss:.3f}, val accuracy: {best_val_acc:.3f}")

    meta_path = MODELS_DIR / "classes.json"
    meta_path.write_text(json.dumps(class_names, ensure_ascii=False, indent=2), encoding="utf-8")
    evaluate(model, class_names, dev)


if __name__ == "__main__":
    main()
