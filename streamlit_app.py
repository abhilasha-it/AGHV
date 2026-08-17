"""Browser-based controller/simulator for AGHV-Net + ANFIS.

Run locally:
    streamlit run streamlit_app.py

Deploy for a public URL: push this repo to GitHub (already done) and
deploy on https://share.streamlit.io, pointing "Main file path" at
streamlit_app.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from src.controller import metrics_loader
from src.data.flower_names import CAT_TO_NAME
from src.fuzzy.anfis import ANFIS, run_anfis_simulation
from src.fuzzy.features import extract_botanical_features
from src.preprocessing.pipeline import PIPELINE_STAGE_LABELS, PreprocessingPipeline

st.set_page_config(page_title="AGHV-Net Fuzzy-Neuro Controller", layout="wide")

METRIC_KEYS = ["accuracy", "precision", "recall", "f1_score", "misclassification_rate"]
METRIC_LABELS = {
    "accuracy": "Accuracy", "precision": "Precision", "recall": "Recall",
    "f1_score": "F1 Score", "misclassification_rate": "Misclass. Rate",
}


@st.cache_resource
def get_anfis_model() -> ANFIS:
    return ANFIS()


@st.cache_resource
def get_aghv_net(checkpoint_path: str):
    import torch

    from src.models.aghv_net import AGHVNet

    model = AGHVNet(num_classes=102, pretrained=False)
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model.eval()
    return model


def render_metric_cards():
    st.subheader("Model metrics")
    models_with_metrics = metrics_loader.available_models()
    if not models_with_metrics:
        st.info("No evaluated models yet. Run `python -m src.train` then `python -m src.evaluate` to populate this.")
        return

    model_name = st.selectbox(
        "Model", metrics_loader.MODEL_ORDER,
        format_func=lambda m: metrics_loader.DISPLAY_NAMES.get(m, m),
        key="metrics_model_select",
    )
    metrics = metrics_loader.load_metrics(model_name)
    cols = st.columns(len(METRIC_KEYS))
    for col, key in zip(cols, METRIC_KEYS):
        value = metrics.get(key) if metrics else None
        col.metric(METRIC_LABELS[key], f"{value * 100:.2f}%" if value is not None else "--")


def render_pipeline_panel():
    st.subheader("Workflow Pipeline")
    stages = [*PIPELINE_STAGE_LABELS, ("Fusion & ANFIS", "CNN + ViT confidence fused via neuro-fuzzy rules")]
    for i, (title, desc) in enumerate(stages):
        st.markdown(f"**{i}. {title}**  \n<span style='color:#888;font-size:0.85em'>{desc}</span>",
                    unsafe_allow_html=True)

    uploaded = st.file_uploader("Load a flower image to run the pipeline", type=["png", "jpg", "jpeg"])
    if uploaded is None:
        return

    image = np.array(Image.open(uploaded).convert("RGB").resize((224, 224)))
    st.image(image, caption="Input (resized to 224x224)", width=200)

    features = extract_botanical_features(image)
    st.session_state["petal_count"] = float(features.petal_count)
    st.session_state["symmetry"] = float(features.symmetry)
    st.session_state["color_intensity"] = float(features.color_intensity)
    st.success(f"Extracted: petal_count={features.petal_count:.0f}, "
               f"symmetry={features.symmetry:.1f}, color_intensity={features.color_intensity:.1f}")

    checkpoint_path = ROOT_DIR / "results" / "checkpoints" / "aghv_net_best.pt"
    if checkpoint_path.exists():
        import torch

        model = get_aghv_net(str(checkpoint_path))
        processed = PreprocessingPipeline()(image)
        tensor = torch.from_numpy(processed).permute(2, 0, 1).unsqueeze(0).float()
        result = model.predict_with_confidences(tensor)
        st.session_state["cnn_conf"] = round(result["cnn_conf"].item() * 100)
        st.session_state["vit_conf"] = round(result["vit_conf"].item() * 100)
        pred_idx = int(result["pred_class"].item())
        st.session_state["candidate_class"] = CAT_TO_NAME.get(pred_idx + 1, f"class_{pred_idx}")
        st.session_state["checkpoint_used"] = True
        st.success("AGHV-Net checkpoint found — CNN/ViT confidence and candidate class auto-filled below.")
    else:
        st.session_state["checkpoint_used"] = False
        st.warning(
            "No trained AGHV-Net checkpoint is deployed, so this image's feature values were extracted "
            "but **the class was not classified**. Pick a candidate class yourself in the panel on the "
            "right and the simulator will show how confident the fuzzy rules are in *that* guess — it "
            "will not pick the class for you until a real checkpoint is trained and deployed."
        )


def render_anfis_panel():
    st.subheader("Live ANFIS Inference Simulator")
    checkpoint_used = st.session_state.get("checkpoint_used", False)
    if checkpoint_used:
        st.caption("Candidate class below was predicted by the AGHV-Net checkpoint; adjust sliders to see how confidence responds.")
    else:
        st.caption("⚠️ No checkpoint deployed — pick a candidate class yourself, then adjust sliders. "
                   "This scores your chosen class; it does not search across all 102 classes for you.")

    class_names = sorted(CAT_TO_NAME.values())
    default_class = st.session_state.get("candidate_class", "rose")
    candidate = st.selectbox(
        "Candidate class" + ("" if checkpoint_used else " (you choose — not model-predicted)"),
        class_names,
        index=class_names.index(default_class) if default_class in class_names else 0,
    )

    petal_count = st.slider("Petal count", 1, 30, int(st.session_state.get("petal_count", 10)))
    symmetry = st.slider("Symmetry", 0, 100, int(st.session_state.get("symmetry", 89)))
    color_intensity = st.slider("Color intens.", 0, 100, int(st.session_state.get("color_intensity", 52)))
    cnn_conf = st.slider("CNN conf.", 0, 100, int(st.session_state.get("cnn_conf", 91)))
    vit_conf = st.slider("ViT conf.", 0, 100, int(st.session_state.get("vit_conf", 92)))

    if st.button("Run ANFIS Inference ↗", type="primary"):
        result = run_anfis_simulation(
            get_anfis_model(), petal_count=petal_count, symmetry=symmetry,
            color_intensity=color_intensity, cnn_conf=cnn_conf, vit_conf=vit_conf,
            predicted_class_name=candidate.capitalize(),
        )
        with st.container(border=True):
            top_col, badge_col = st.columns([3, 1])
            top_col.markdown(f"### {candidate.capitalize()}")
            badge_col.markdown(f"**{result.confidence * 100:.1f}%**")
            st.write(result.explanation)
            st.progress(result.confidence)


def render_comparison_tab():
    models = metrics_loader.available_models()
    if models:
        st.subheader("Accuracy / Precision / Recall / F1 by model")
        rows = []
        for model_name in models:
            metrics = metrics_loader.load_metrics(model_name)
            rows.append({
                "model": metrics_loader.DISPLAY_NAMES.get(model_name, model_name),
                "accuracy": metrics["accuracy"], "precision": metrics["precision"],
                "recall": metrics["recall"], "f1_score": metrics["f1_score"],
            })
        df = pd.DataFrame(rows).set_index("model")
        st.bar_chart(df)
    else:
        st.info("No evaluated models yet. Run `python -m src.train` then `python -m src.evaluate`.")

    st.subheader("Training curves")
    curve_models = metrics_loader.all_models_for_comparison()
    if not curve_models:
        st.info("No training curves yet. Run `python -m src.train`.")
        return

    selected = st.selectbox("Model", curve_models,
                             format_func=lambda m: metrics_loader.DISPLAY_NAMES.get(m, m),
                             key="curves_model_select")
    curves = metrics_loader.load_curves(selected)
    if not curves:
        st.info(f"No training curves for {metrics_loader.DISPLAY_NAMES.get(selected, selected)} yet.")
        return

    curve_df = pd.DataFrame({
        "train_acc": curves["train_acc"], "val_acc": curves["val_acc"],
    }, index=range(1, len(curves["train_acc"]) + 1))
    curve_df.index.name = "epoch"
    st.line_chart(curve_df)


st.title("AGHV-Net Fuzzy-Neuro Controller — Flower Classification")

simulator_tab, comparison_tab = st.tabs(["Simulator", "Model Comparison"])

with simulator_tab:
    render_metric_cards()
    left, right = st.columns(2)
    with left:
        render_pipeline_panel()
    with right:
        render_anfis_panel()

with comparison_tab:
    render_comparison_tab()
