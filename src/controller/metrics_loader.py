"""Loads the JSON metrics/curves that src/train.py and src/evaluate.py write,
for display in the controller's metric cards and comparison charts."""

from __future__ import annotations

import json
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent.parent.parent / "results"
METRICS_DIR = RESULTS_DIR / "metrics"

MODEL_ORDER = ["aghv_net", "resnet50", "vgg16", "vit_small", "plain_cnn"]
DISPLAY_NAMES = {
    "aghv_net": "AGHV-Net",
    "resnet50": "ResNet-50",
    "vgg16": "VGG-16",
    "vit_small": "ViT-Small",
    "plain_cnn": "Plain CNN",
}


def load_metrics(model_name: str) -> dict | None:
    path = METRICS_DIR / f"{model_name}_metrics.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def load_curves(model_name: str) -> dict | None:
    path = METRICS_DIR / f"{model_name}_curves.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def available_models() -> list[str]:
    return [m for m in MODEL_ORDER if load_metrics(m) is not None]


def all_models_for_comparison() -> list[str]:
    """Models that have at least metrics or curves available."""
    return [m for m in MODEL_ORDER if load_metrics(m) is not None or load_curves(m) is not None]
