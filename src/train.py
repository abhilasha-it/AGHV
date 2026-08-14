"""Trains a single model (AGHV-Net or a baseline) on Oxford 102 Flowers and
saves per-epoch curves + the best checkpoint, ready for src/evaluate.py and
the controller's model-comparison tab.

Usage:
    python -m src.train --model aghv_net --epochs 30
    python -m src.train --model resnet50 --epochs 30
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.common import build_model, forward_and_loss
from src.data.dataset import NUM_CLASSES, get_dataloaders

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
CHECKPOINT_DIR = RESULTS_DIR / "checkpoints"
METRICS_DIR = RESULTS_DIR / "metrics"


def run_epoch(model, model_name, loader, device, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss, total_correct, total_count = 0.0, 0, 0
    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            if is_train:
                optimizer.zero_grad()
            logits, loss = forward_and_loss(model, model_name, x, y)
            if is_train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * x.size(0)
            total_correct += (logits.argmax(dim=1) == y).sum().item()
            total_count += x.size(0)

    return total_loss / total_count, total_correct / total_count


def train(model_name: str, epochs: int, batch_size: int, lr: float, data_root: str,
          pretrained: bool, num_workers: int, device: str | None = None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, _ = get_dataloaders(data_root, batch_size=batch_size, num_workers=num_workers)
    model = build_model(model_name, num_classes=NUM_CLASSES, pretrained=pretrained).to(device)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    curves = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "epoch_time_sec": []}
    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        start = time.time()
        train_loss, train_acc = run_epoch(model, model_name, train_loader, device, optimizer)
        val_loss, val_acc = run_epoch(model, model_name, val_loader, device, optimizer=None)
        scheduler.step()
        elapsed = time.time() - start

        curves["train_loss"].append(train_loss)
        curves["train_acc"].append(train_acc)
        curves["val_loss"].append(val_loss)
        curves["val_acc"].append(val_acc)
        curves["epoch_time_sec"].append(elapsed)

        print(f"[{model_name}] epoch {epoch}/{epochs} "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} ({elapsed:.1f}s)")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), CHECKPOINT_DIR / f"{model_name}_best.pt")

        with open(METRICS_DIR / f"{model_name}_curves.json", "w") as f:
            json.dump(curves, f, indent=2)

    return curves


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=[
        "aghv_net", "resnet50", "vgg16", "vit_small", "plain_cnn"])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--data-root", type=str, default="./data/flowers102")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args.model, args.epochs, args.batch_size, args.lr, args.data_root,
          pretrained=not args.no_pretrained, num_workers=args.num_workers)
