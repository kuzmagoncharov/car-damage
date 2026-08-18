from __future__ import annotations

import torch.nn as nn
from torchvision import models


def create_model(
    arch: str = "mobilenet_v3_small",
    num_classes: int = 2,
    freeze_backbone: bool = True,
) -> nn.Module:
    if arch == "mobilenet_v3_small":
        model = _mobilenet_v3_small()
        if freeze_backbone:
            for param in model.features.parameters():
                param.requires_grad = False
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        return model

    if arch == "resnet18":
        model = _resnet18()
        if freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model

    raise ValueError(f"Неизвестная архитектура: {arch}")


def _mobilenet_v3_small():
    try:
        weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
        return models.mobilenet_v3_small(weights=weights)
    except AttributeError:
        return models.mobilenet_v3_small(pretrained=True)


def _resnet18():
    try:
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        return models.resnet18(weights=weights)
    except AttributeError:
        return models.resnet18(pretrained=True)
