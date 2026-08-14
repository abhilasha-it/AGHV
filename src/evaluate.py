"""Evaluates a trained checkpoint on the Oxford 102 Flowers test split and
writes the metrics JSON the controller's metric cards and comparison
charts read from (results/metrics/{model}_metrics.json).

Usage:
    python -m src.evaluate --model aghv_net
    python -m src.evaluate --model resnet50
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from src.common import build_model
from src.data.dataset import NUM_CLASSES, get_dataloaders

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
CHECKPOINT_DIR = RESULTS_DIR / "checkpoints"
METRICS_DIR = RESULTS_DIR / "metrics"


@torch.no_grad()
def collect_predictions(model, model_name: str, loader, device):
    model.eval()
    all_preds, all_targets = [], []
    for x, y in loader:
        x = x.to(device)
        outputs = model(x)
        logits = outputs["logits"] if model_name == "aghv_net" else outputs
        preds = logits.argmax(dim=1).cpu()
        all_preds.append(preds)
        all_targets.append(y)
    return torch.cat(all_preds).numpy(), torch.cat(all_targets).numpy()


def evaluate(model_name: str, data_root: str, batch_size: int, num_workers: int, device: str | None = None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    _, _, test_loader = get_dataloaders(data_root, batch_size=batch_size, num_workers=num_workers)
    model = build_model(model_name, num_classes=NUM_CLASSES, pretrained=False).to(device)

    checkpoint_path = CHECKPOINT_DIR / f"{model_name}_best.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"No checkpoint at {checkpoint_path}; run src/train.py first.")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    preds, targets = collect_predictions(model, model_name, test_loader, device)

    accuracy = (preds == targets).mean()
    precision, recall, f1, _ = precision_recall_fscore_support(
        targets, preds, average="macro", zero_division=0)
    cm = confusion_matrix(targets, preds)

    metrics = {
        "model": model_name,
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "misclassification_rate": float(1 - accuracy),
        "num_test_samples": int(len(targets)),
        "confusion_matrix": cm.tolist(),
    }

    with open(METRICS_DIR / f"{model_name}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps({k: v for k, v in metrics.items() if k != "confusion_matrix"}, indent=2))
    return metrics


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=[
        "aghv_net", "resnet50", "vgg16", "vit_small", "plain_cnn"])
    parser.add_argument("--data-root", type=str, default="./data/flowers102")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(args.model, args.data_root, args.batch_size, args.num_workers)
