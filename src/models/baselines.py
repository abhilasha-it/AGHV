"""Baseline models used to benchmark AGHV-Net against in the comparison
tab of the controller (accuracy / precision / recall / F1 bar charts).

Each factory returns a plain nn.Module whose forward(x) -> logits, so the
same train/eval loop in src/train.py and src/evaluate.py works for all of
them as well as for AGHV-Net (which instead returns a dict; train.py
handles both cases, see `unwrap_logits`).
"""

from __future__ import annotations

import timm
import torch.nn as nn
import torchvision


def resnet50_baseline(num_classes: int = 102, pretrained: bool = True) -> nn.Module:
    weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    model = torchvision.models.resnet50(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def vgg16_baseline(num_classes: int = 102, pretrained: bool = True) -> nn.Module:
    weights = torchvision.models.VGG16_Weights.IMAGENET1K_V1 if pretrained else None
    model = torchvision.models.vgg16(weights=weights)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model


def vit_baseline(num_classes: int = 102, pretrained: bool = True,
                  model_name: str = "vit_small_patch16_224") -> nn.Module:
    return timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)


def plain_cnn_baseline(num_classes: int = 102) -> nn.Module:
    """A small from-scratch CNN, included as a lower-bound reference point."""

    def block(in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    return nn.Sequential(
        block(3, 32), block(32, 64), block(64, 128), block(128, 256),
        nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        nn.Linear(256, 128), nn.ReLU(inplace=True), nn.Dropout(0.3),
        nn.Linear(128, num_classes),
    )


BASELINE_REGISTRY = {
    "resnet50": resnet50_baseline,
    "vgg16": vgg16_baseline,
    "vit_small": vit_baseline,
    "plain_cnn": plain_cnn_baseline,
}


def build_baseline(name: str, num_classes: int = 102, pretrained: bool = True) -> nn.Module:
    if name not in BASELINE_REGISTRY:
        raise ValueError(f"Unknown baseline '{name}'. Choose from {list(BASELINE_REGISTRY)}")
    factory = BASELINE_REGISTRY[name]
    if name == "plain_cnn":
        return factory(num_classes)
    return factory(num_classes, pretrained)
