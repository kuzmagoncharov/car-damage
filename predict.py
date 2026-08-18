"""Предсказание по одному фото: класс и вероятность."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image

from src.config import ARCH, CHECKPOINT_PATH, configure_console
from src.dataset import get_infer_transform
from src.model import create_model


class CarDamageModel:
    """Загружает веса один раз и проверяет фото."""

    def __init__(self) -> None:
        if not CHECKPOINT_PATH.exists():
            raise FileNotFoundError("Модель ещё не обучена. Сначала выполните: python train.py")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=self.device, weights_only=False)
        self.class_names = list(checkpoint["class_names"])
        arch = checkpoint.get("arch", ARCH)

        model = create_model(arch=arch, num_classes=len(self.class_names), freeze_backbone=False)
        model.load_state_dict(checkpoint["model_state"])
        model.to(self.device)
        model.eval()
        self.model = model
        self._transform = get_infer_transform()

    def predict_image(self, image: Image.Image) -> tuple[str, float, dict[str, float]]:
        rgb = image.convert("RGB")
        batch = self._transform(rgb).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probs = torch.softmax(self.model(batch), dim=1)[0]

        scores = {
            name: float(probs[index].item())
            for index, name in enumerate(self.class_names)
        }
        label = max(scores, key=scores.get)
        return label, scores[label], scores

    def predict(self, image_path: Path) -> tuple[str, float, dict[str, float]]:
        path = Path(image_path)
        if not path.is_file():
            raise FileNotFoundError(f"Файл не найден: {path}")
        return self.predict_image(Image.open(path))


def predict(image_path: Path) -> tuple[str, float]:
    label, confidence, _scores = CarDamageModel().predict(image_path)
    return label, confidence


def main() -> None:
    configure_console()
    parser = argparse.ArgumentParser(description="Определяет, повреждён автомобиль или нет")
    parser.add_argument("image", type=Path, help="Путь к фото автомобиля")
    args = parser.parse_args()

    try:
        label, confidence = predict(args.image)
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from exc

    print(f"Класс: {label}")
    print(f"Вероятность: {confidence:.0%}")


if __name__ == "__main__":
    main()
