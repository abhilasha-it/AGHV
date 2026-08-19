# research

AGHV-Net (Attention-Guided Hybrid Vision Network) — a CNN + Vision
Transformer hybrid for flower classification (Oxford 102 Flowers) — fronted
by an explainable fuzzy-logic controller/simulator that classifies a flower
from its botanical features via an inspectable IF-THEN rule base (in the
spirit of ANFIS/neuro-fuzzy systems: confirming, correcting, or expressing
uncertainty about the deep-learning branch's prediction), plus a
model-comparison view against baseline architectures.

## Structure

- `src/preprocessing/` — CLAHE, bilateral filtering, ROI segmentation, Gabor
  texture enhancement, color normalization pipeline
- `src/models/aghv_net.py` — AGHV-Net: ResNet-50 branch + ViT branch fused via
  bidirectional cross-attention, with CNN/ViT auxiliary heads for deep
  supervision (their confidences feed the fuzzy layer)
- `src/models/baselines.py` — ResNet-50, VGG-16, ViT-Small, plain CNN, used
  as comparison points
- `src/data/` — Oxford 102 Flowers dataset loader (auto-downloads via
  torchvision) and category-id -> name mapping
- `src/fuzzy/features.py` — derives petal count, radial symmetry, color
  intensity, petal-edge serration, and dominant hue from an image via
  classical CV
- `src/fuzzy/rule_base.py` — the expert IF-THEN fuzzy rule base (~15
  visually distinctive Oxford-102 species, including a deliberately
  ambiguous rose/camellia pair) with trapezoidal membership functions
- `src/fuzzy/inference.py` — the decision engine: runs the rule base against
  the current features and, if a trained AGHV-Net checkpoint is available,
  confirms or overrides its prediction with an explanation
- `src/fuzzy/anfis.py` — a standalone trainable Sugeno-style neuro-fuzzy
  network (not currently wired into the controller; kept for future
  data-driven calibration of the rule base's membership functions)
- `src/train.py` / `src/evaluate.py` — train/evaluate any model in the
  comparison set, writing metrics + training curves to `results/metrics/`
- `src/controller/` — PyQt6 desktop app (**legacy**: still uses the old
  single-candidate-class ANFIS flow, not yet updated to the rule-engine
  redesign in `streamlit_app.py` — ask if you want it brought up to date)
- `streamlit_app.py` — the up-to-date browser-based controller: metric
  cards, the workflow pipeline panel, and the live fuzzy inference
  simulator (predicted species updates automatically as any feature slider
  changes, with the fired rule, all rule match scores, and the full rule
  base shown for explainability)
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

The Streamlit app (below) is the current version. It has two tabs:
**Simulator** — metric cards for a selected model; the Workflow Pipeline
panel, where loading an image auto-fills petal count / symmetry / color
intensity / edge serration / hue, and runs it through a trained AGHV-Net
checkpoint if one exists; and the Live Fuzzy Inference Simulator, whose
predicted species, confidence, and explanation recompute the instant any
feature slider changes, with the fired IF-THEN rule, every rule's current
match score, and the full rule base all shown for explainability.
**Model Comparison** — accuracy/precision/recall/F1 bars and per-model
training curves across every evaluated model.

### Desktop app (PyQt6) — legacy, not yet updated

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

## Getting real (non-rule-only) results

Without a trained AGHV-Net checkpoint, the controller runs in **pure fuzzy
rule-engine mode** — the sliders drive the ~15-species hand-authored rule
base directly, which is illustrative/explainable but not calibrated
against real data, so it will misclassify anything outside those species
or with noisy feature extraction. To get genuine deep-learning-backed
results:

1. Open [`notebooks/colab_train_aghv_net.ipynb`](notebooks/colab_train_aghv_net.ipynb)
   in Google Colab (File -> Upload notebook, or File -> Open notebook ->
   GitHub -> this repo), enable a GPU runtime, and run it top to bottom.
   It clones this repo, trains + evaluates AGHV-Net (and a ResNet-50
   baseline), and walks through hosting the resulting checkpoint
   (`results/checkpoints/aghv_net_best.pt`, ~150-250 MB — too large to
   commit directly to git) on Hugging Face Hub or Google Drive.
2. In the Streamlit Cloud app's settings ("Manage app" -> Settings ->
   Secrets), add:
   ```toml
   AGHV_CHECKPOINT_URL = "https://huggingface.co/<you>/aghv-net-flowers102/resolve/main/aghv_net_best.pt"
   ```
   The app downloads it once on first run and caches it from then on. Running
   locally instead, just drop the file at `results/checkpoints/aghv_net_best.pt`
   and skip the secret.
3. Commit the notebook's `results/metrics/*.json` output back to the repo
   (the notebook's last cell does this) so the Model Comparison tab shows
   real accuracy/precision/recall/F1 numbers instead of "no evaluated
   models yet."

Once a checkpoint is deployed, the simulator switches from rule-only mode
to confirming/correcting the AGHV-Net prediction (see `src/fuzzy/inference.py`).
