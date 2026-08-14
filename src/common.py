"""Shared helpers for building/running any model in the comparison set
(AGHV-Net returns a dict of logits; baselines return a plain tensor)."""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.aghv_net import AGHVNet, aghv_net_loss
from src.models.baselines import BASELINE_REGISTRY, build_baseline

MODEL_NAMES = ["aghv_net", *BASELINE_REGISTRY.keys()]

DISPLAY_NAMES = {
    "aghv_net": "AGHV-Net",
    "resnet50": "ResNet-50",
    "vgg16": "VGG-16",
    "vit_small": "ViT-Small",
    "plain_cnn": "Plain CNN",
}


def build_model(name: str, num_classes: int = 102, pretrained: bool = True) -> nn.Module:
    if name == "aghv_net":
        return AGHVNet(num_classes=num_classes, pretrained=pretrained)
    return build_baseline(name, num_classes=num_classes, pretrained=pretrained)


def forward_and_loss(model: nn.Module, model_name: str, x: torch.Tensor, y: torch.Tensor):
    """Returns (logits, loss) regardless of whether `model` is AGHV-Net (dict
    output, deep supervision loss) or a plain baseline (tensor output, CE loss).
    """
    if model_name == "aghv_net":
        outputs = model(x)
        loss = aghv_net_loss(outputs, y)
        return outputs["logits"], loss
    logits = model(x)
    loss = nn.functional.cross_entropy(logits, y)
    return logits, loss
