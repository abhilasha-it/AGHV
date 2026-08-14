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
- `streamlit_app.py` — browser-based version of the same controller, for
  running locally at a `localhost` URL or deploying to a public URL
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

Both versions share the same panels: **Simulator** (metric cards for a
selected model, the Workflow Pipeline panel — load an image to auto-fill
petal count / symmetry / color intensity, and run it through a trained
AGHV-Net checkpoint if one exists — and the Live ANFIS Inference Simulator
sliders + rule explanation) and **Model Comparison** (accuracy/precision/
recall/F1 bars and per-model training curves across every evaluated model).

### Desktop app (PyQt6)

```bash
pip install -r requirements-desktop.txt
python -m src.controller.app
```

### Web app (Streamlit)

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

This opens at `http://localhost:8501`. To get a public URL instead of a
local one, deploy it for free on [Streamlit Community Cloud](https://share.streamlit.io):
sign in with GitHub, "New app", pick this repo, branch `master`, and set
"Main file path" to `streamlit_app.py`.
