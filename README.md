# research

AGHV-Net (Attention-Guided Hybrid Vision Network) — a CNN + Vision
Transformer hybrid for flower classification (Oxford 102 Flowers),
fronted by an ANFIS (Adaptive Neuro-Fuzzy Inference System) fuzzy-logic
controller/simulator, with a desktop app to run inference and compare
AGHV-Net against baseline models.

## Structure

- `src/preprocessing/` — CLAHE, bilateral filtering, ROI segmentation, Gabor
  texture enhancement, color normalization pipeline
- `src/models/aghv_net.py` — AGHV-Net: ResNet-50 branch + ViT branch fused via
  bidirectional cross-attention, with CNN/ViT auxiliary heads for deep
  supervision (their confidences feed the ANFIS layer)
- `src/models/baselines.py` — ResNet-50, VGG-16, ViT-Small, plain CNN, used
  as comparison points
- `src/data/` — Oxford 102 Flowers dataset loader (auto-downloads via
  torchvision) and category-id -> name mapping
- `src/fuzzy/features.py` — derives petal-count, symmetry, color-intensity
  features from an image via classical CV
- `src/fuzzy/anfis.py` — the trainable neuro-fuzzy fusion network
- `src/train.py` / `src/evaluate.py` — train/evaluate any model in the
  comparison set, writing metrics + training curves to `results/metrics/`
- `src/controller/` — PyQt6 desktop app: metric cards, workflow pipeline
  panel, live ANFIS inference simulator, and a model-comparison tab with
  bar charts + training curves
- `results/` — checkpoints, metrics JSON, figures
- `notebooks/` — exploratory notebooks

## Setup

Python is required (not currently installed on this machine — install from
[python.org](https://www.python.org/downloads/), not the Windows Store
alias, then re-open a new terminal so `python` resolves correctly).

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Train and evaluate a model

```bash
python -m src.train --model aghv_net --epochs 30
python -m src.evaluate --model aghv_net

python -m src.train --model resnet50 --epochs 30
python -m src.evaluate --model resnet50
```

Repeat for `vgg16`, `vit_small`, `plain_cnn` to populate the comparison
charts. The Oxford 102 Flowers dataset downloads automatically on first run
into `./data/flowers102`.

## Run the controller / simulator

```bash
python -m src.controller.app
```

- **Simulator tab** — pick a model to show its accuracy/precision/recall/F1/
  misclassification metric cards (once evaluated); the Workflow Pipeline
  panel lets you load an image and run it through the full preprocessing +
  AGHV-Net pipeline (if a checkpoint exists) to auto-fill the ANFIS sliders;
  the Live ANFIS Inference Simulator lets you manually adjust petal count,
  symmetry, color intensity, and CNN/ViT confidence sliders and see the
  fused confidence and rule explanation.
- **Model Comparison tab** — grouped bar chart of accuracy/precision/recall/F1
  across every evaluated model, plus per-model training curves.
