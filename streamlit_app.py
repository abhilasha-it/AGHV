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
from src.fuzzy.features import BotanicalFeatures, extract_botanical_features
from src.fuzzy.inference import classify
from src.fuzzy.rule_base import RULE_BASE
from src.preprocessing.pipeline import PIPELINE_STAGE_LABELS, PreprocessingPipeline

st.set_page_config(page_title="AGHV-Net Fuzzy-Neuro Controller", layout="wide")

METRIC_KEYS = ["accuracy", "precision", "recall", "f1_score", "misclassification_rate"]
METRIC_LABELS = {
    "accuracy": "Accuracy", "precision": "Precision", "recall": "Recall",
    "f1_score": "F1 Score", "misclassification_rate": "Misclass. Rate",
}

SOURCE_LABELS = {
    "rule_only": "Fuzzy rule engine (no DL checkpoint deployed)",
    "dl_confirmed": "DL branch prediction, confirmed by fuzzy rules",
    "dl_corrected": "DL branch prediction overridden by fuzzy rules",
    "dl_only": "DL branch prediction (species not in rule base — unverified)",
    "unknown": "Insufficient evidence",
}


def _hue_swatch(hue_0_179: float) -> str:
    hue_deg = hue_0_179 * 2
    return f"<div style='width:100%;height:14px;border-radius:4px;background:hsl({hue_deg:.0f},70%,50%);'></div>"


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
    stages = [*PIPELINE_STAGE_LABELS, ("Fusion & Fuzzy Inference", "Botanical features + DL branch fused via IF-THEN rules")]
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
    st.session_state["edge_serration"] = float(features.edge_serration)
    st.session_state["dominant_hue"] = float(features.dominant_hue)
    st.success(
        f"Extracted: petal_count={features.petal_count:.0f}, symmetry={features.symmetry:.1f}, "
        f"color_intensity={features.color_intensity:.1f}, edge_serration={features.edge_serration:.1f}, "
        f"hue={features.dominant_hue:.1f}"
    )

    checkpoint_path = ROOT_DIR / "results" / "checkpoints" / "aghv_net_best.pt"
    if checkpoint_path.exists():
        import torch

        model = get_aghv_net(str(checkpoint_path))
        processed = PreprocessingPipeline()(image)
        tensor = torch.from_numpy(processed).permute(2, 0, 1).unsqueeze(0).float()
        result = model.predict_with_confidences(tensor)
        st.session_state["dl_species"] = CAT_TO_NAME.get(int(result["pred_class"].item()) + 1, "unknown")
        st.session_state["dl_confidence"] = result["fused_conf"].item()
        st.success(f"AGHV-Net checkpoint found — DL branch predicts "
                   f"'{st.session_state['dl_species']}' ({st.session_state['dl_confidence']:.0%}). "
                   f"The fuzzy rules below will confirm or correct this.")
    else:
        st.session_state["dl_species"] = None
        st.session_state["dl_confidence"] = None
        st.warning(
            "No trained AGHV-Net checkpoint is deployed, so classification runs in **pure fuzzy "
            "rule-engine mode**: the sliders on the right feed straight into the IF-THEN rule base "
            "below, and the predicted species updates live as you change them."
        )


def render_anfis_panel():
    st.subheader("Live Fuzzy Inference Simulator")
    st.caption("The predicted species below is computed live from these feature values — change any "
               "slider and it updates automatically.")

    petal_count = st.slider("Petal count", 1, 90, int(st.session_state.get("petal_count", 20)))
    symmetry = st.slider("Symmetry", 0, 100, int(st.session_state.get("symmetry", 85)))
    color_intensity = st.slider("Color intensity", 0, 100, int(st.session_state.get("color_intensity", 60)))
    edge_serration = st.slider("Edge serration", 0, 100, int(st.session_state.get("edge_serration", 10)))
    dominant_hue = st.slider("Dominant hue (0=red, 30=yellow, 60=green, 120=blue)", 0, 179,
                              int(st.session_state.get("dominant_hue", 5)))
    st.markdown(_hue_swatch(dominant_hue), unsafe_allow_html=True)

    features = BotanicalFeatures(
        petal_count=petal_count, symmetry=symmetry, color_intensity=color_intensity,
        edge_serration=edge_serration, dominant_hue=dominant_hue,
    )
    result = classify(features, dl_species=st.session_state.get("dl_species"),
                       dl_confidence=st.session_state.get("dl_confidence"))

    with st.container(border=True):
        top_col, badge_col = st.columns([3, 1])
        top_col.markdown(f"### {result.species.capitalize()}")
        badge_col.markdown(f"**{result.confidence * 100:.1f}%**")
        st.caption(SOURCE_LABELS.get(result.source, result.source))
        if result.note:
            st.info(result.note)
        st.progress(min(max(result.confidence, 0.0), 1.0))
        st.write({
            "petal_shape": result.explanation.get("petal_shape"),
            "symmetry": result.explanation.get("symmetry"),
            "color_match": result.explanation.get("color_match"),
            "fuzzy_score": result.explanation.get("fuzzy_score"),
        })
        if result.fired_rule_text:
            st.code(result.fired_rule_text, language=None)

    with st.expander("All rule match scores (why not another species?)"):
        st.dataframe(
            pd.DataFrame([{"species": m.species, "fuzzy_score": round(m.fuzzy_score, 3)} for m in result.all_matches]),
            hide_index=True, use_container_width=True,
        )

    with st.expander("Fuzzy rule base (defined inference rules)"):
        for rule in RULE_BASE:
            st.code(rule.as_if_then(), language=None)


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
