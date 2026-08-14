"""Main controller/simulator window.

Tab 1 "Simulator": metric cards for a selected trained model, the
preprocessing Workflow Pipeline panel, and the Live ANFIS Inference
Simulator (sliders -> fuzzy fusion -> predicted class + confidence +
rule explanation). Optionally runs a real AGHV-Net checkpoint on a loaded
image to auto-populate the botanical features and CNN/ViT confidences.

Tab 2 "Model Comparison": grouped bar chart of accuracy/precision/recall/F1
across every model that has been evaluated, plus per-model training curves.
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QComboBox, QFileDialog, QFrame, QGroupBox, QHBoxLayout,
    QLabel, QMainWindow, QPushButton, QSlider, QTabWidget,
    QVBoxLayout, QWidget,
)

from src.controller import metrics_loader
from src.controller.widgets import ConfidenceBar, MetricCard, MplCanvas, PipelineStepCard
from src.data.flower_names import CAT_TO_NAME
from src.fuzzy.anfis import ANFIS, run_anfis_simulation
from src.fuzzy.features import extract_botanical_features
from src.preprocessing.pipeline import PIPELINE_STAGE_LABELS

METRIC_KEYS = ["accuracy", "precision", "recall", "f1_score", "misclassification_rate"]


class LabeledSlider(QWidget):
    def __init__(self, name: str, minimum: int, maximum: int, initial: int, parent=None):
        super().__init__(parent)
        self.name_label = QLabel(name)
        self.name_label.setFixedWidth(90)
        self.value_label = QLabel(str(initial))
        self.value_label.setStyleSheet("color: #4f5ff5; font-weight: 600;")
        self.value_label.setFixedWidth(32)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setMinimum(minimum)
        self.slider.setMaximum(maximum)
        self.slider.setValue(initial)
        self.slider.valueChanged.connect(lambda v: self.value_label.setText(str(v)))

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 2)
        layout.addWidget(self.name_label)
        layout.addWidget(self.slider)
        layout.addWidget(self.value_label)

    def value(self) -> int:
        return self.slider.value()

    def set_value(self, value: int):
        self.slider.setValue(int(value))


class SimulatorTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.anfis_model = ANFIS()
        self._aghv_net = None  # lazily loaded on first "Load Image" use

        root = QVBoxLayout(self)

        root.addLayout(self._build_metrics_row())

        panels = QHBoxLayout()
        panels.addWidget(self._build_pipeline_panel(), stretch=1)
        panels.addWidget(self._build_anfis_panel(), stretch=1)
        root.addLayout(panels)

        self._refresh_metric_cards()

    # ---- top metric cards -------------------------------------------------
    def _build_metrics_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        self.model_selector = QComboBox()
        self.model_selector.addItems(metrics_loader.MODEL_ORDER)
        self.model_selector.currentTextChanged.connect(lambda _: self._refresh_metric_cards())

        selector_box = QVBoxLayout()
        selector_box.addWidget(QLabel("Model:"))
        selector_box.addWidget(self.model_selector)
        row.addLayout(selector_box)

        self.metric_cards = {key: MetricCard(key) for key in METRIC_KEYS}
        for card in self.metric_cards.values():
            row.addWidget(card)
        return row

    def _refresh_metric_cards(self):
        metrics = metrics_loader.load_metrics(self.model_selector.currentText())
        for key, card in self.metric_cards.items():
            card.set_value(metrics.get(key) if metrics else None)

    # ---- workflow pipeline panel ------------------------------------------
    def _build_pipeline_panel(self) -> QGroupBox:
        box = QGroupBox("Workflow Pipeline")
        layout = QVBoxLayout(box)

        stages = [*PIPELINE_STAGE_LABELS, ("Fusion & ANFIS", "CNN + ViT confidence fused via neuro-fuzzy rules")]
        for i, (title, desc) in enumerate(stages):
            layout.addWidget(PipelineStepCard(str(i) if i > 0 else "◎", title, desc))

        load_button = QPushButton("Load Image & Run Pipeline")
        load_button.clicked.connect(self._on_load_image)
        layout.addWidget(load_button)

        self.pipeline_status = QLabel("")
        self.pipeline_status.setWordWrap(True)
        self.pipeline_status.setStyleSheet("color: #888888; font-size: 11px;")
        layout.addWidget(self.pipeline_status)
        layout.addStretch()
        return box

    def _on_load_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select flower image", "", "Images (*.png *.jpg *.jpeg)")
        if not path:
            return

        image = np.array(Image.open(path).convert("RGB").resize((224, 224)))
        features = extract_botanical_features(image)
        self.petal_slider.set_value(features.petal_count)
        self.symmetry_slider.set_value(features.symmetry)
        self.color_slider.set_value(features.color_intensity)

        status = (f"Extracted from image: petal_count={features.petal_count:.0f}, "
                  f"symmetry={features.symmetry:.1f}, color_intensity={features.color_intensity:.1f}.")

        checkpoint_available = self._try_run_aghv_net(image)
        if not checkpoint_available:
            status += " No trained AGHV-Net checkpoint found -- set CNN/ViT confidence sliders manually."
        self.pipeline_status.setText(status)

    def _try_run_aghv_net(self, image: np.ndarray) -> bool:
        from pathlib import Path
        checkpoint_path = Path(__file__).resolve().parent.parent.parent / "results" / "checkpoints" / "aghv_net_best.pt"
        if not checkpoint_path.exists():
            return False

        if self._aghv_net is None:
            from src.models.aghv_net import AGHVNet
            self._aghv_net = AGHVNet(num_classes=102, pretrained=False)
            self._aghv_net.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
            self._aghv_net.eval()

        from src.preprocessing.pipeline import PreprocessingPipeline
        processed = PreprocessingPipeline()(image)
        tensor = torch.from_numpy(processed).permute(2, 0, 1).unsqueeze(0).float()
        result = self._aghv_net.predict_with_confidences(tensor)

        self.cnn_conf_slider.set_value(round(result["cnn_conf"].item() * 100))
        self.vit_conf_slider.set_value(round(result["vit_conf"].item() * 100))
        pred_idx = int(result["pred_class"].item())
        name = CAT_TO_NAME.get(pred_idx + 1, f"class_{pred_idx}")
        self.class_selector.setCurrentText(name)
        return True

    # ---- ANFIS simulator panel --------------------------------------------
    def _build_anfis_panel(self) -> QGroupBox:
        box = QGroupBox("Live ANFIS Inference Simulator")
        layout = QVBoxLayout(box)
        layout.addWidget(QLabel("Adjust botanical features to classify"))

        self.class_selector = QComboBox()
        self.class_selector.addItems(sorted(CAT_TO_NAME.values()))
        self.class_selector.setCurrentText("rose")
        class_row = QHBoxLayout()
        class_row.addWidget(QLabel("Candidate class:"))
        class_row.addWidget(self.class_selector)
        layout.addLayout(class_row)

        self.petal_slider = LabeledSlider("Petal count", 1, 30, 10)
        self.symmetry_slider = LabeledSlider("Symmetry", 0, 100, 89)
        self.color_slider = LabeledSlider("Color intens.", 0, 100, 52)
        self.cnn_conf_slider = LabeledSlider("CNN conf.", 0, 100, 91)
        self.vit_conf_slider = LabeledSlider("ViT conf.", 0, 100, 92)
        for slider in (self.petal_slider, self.symmetry_slider, self.color_slider,
                       self.cnn_conf_slider, self.vit_conf_slider):
            layout.addWidget(slider)

        run_button = QPushButton("Run ANFIS Inference ↗")
        run_button.setStyleSheet(
            "background-color: #4f5ff5; color: white; font-weight: 600; padding: 8px; border-radius: 6px;"
        )
        run_button.clicked.connect(self._on_run_anfis)
        layout.addWidget(run_button)

        self.result_card = QFrame()
        self.result_card.setStyleSheet(
            "background-color: #eaf6ee; border-radius: 8px; padding: 10px;"
        )
        result_layout = QVBoxLayout(self.result_card)
        header_row = QHBoxLayout()
        self.result_class_label = QLabel("--")
        self.result_class_label.setStyleSheet("font-size: 18px; font-weight: 700; color: #1fa363;")
        self.result_conf_badge = QLabel("--")
        self.result_conf_badge.setStyleSheet(
            "background-color: #d3efdd; color: #1fa363; border-radius: 10px; padding: 2px 10px; font-weight: 600;"
        )
        header_row.addWidget(self.result_class_label)
        header_row.addStretch()
        header_row.addWidget(self.result_conf_badge)
        self.result_explanation = QLabel("")
        self.result_explanation.setWordWrap(True)
        self.result_bar = ConfidenceBar()

        result_layout.addLayout(header_row)
        result_layout.addWidget(self.result_explanation)
        result_layout.addWidget(self.result_bar)
        layout.addWidget(self.result_card)
        layout.addStretch()
        return box

    def _on_run_anfis(self):
        class_name = self.class_selector.currentText()
        result = run_anfis_simulation(
            self.anfis_model,
            petal_count=self.petal_slider.value(),
            symmetry=self.symmetry_slider.value(),
            color_intensity=self.color_slider.value(),
            cnn_conf=self.cnn_conf_slider.value(),
            vit_conf=self.vit_conf_slider.value(),
            predicted_class_name=class_name.capitalize(),
        )
        self.result_class_label.setText(class_name.capitalize())
        self.result_conf_badge.setText(f"{result.confidence * 100:.1f}%")
        self.result_explanation.setText(result.explanation)
        self.result_bar.setValue(int(result.confidence * 100))


class ComparisonTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)

        controls = QHBoxLayout()
        refresh_button = QPushButton("Refresh results")
        refresh_button.clicked.connect(self.refresh)
        controls.addWidget(refresh_button)
        controls.addStretch()
        layout.addLayout(controls)

        charts = QHBoxLayout()
        self.metrics_canvas = MplCanvas(width=6, height=4)
        self.curves_canvas = MplCanvas(width=6, height=4)
        charts.addWidget(self.metrics_canvas)
        charts.addWidget(self.curves_canvas)
        layout.addLayout(charts)

        curve_controls = QHBoxLayout()
        curve_controls.addWidget(QLabel("Training curves for:"))
        self.curve_model_selector = QComboBox()
        self.curve_model_selector.addItems(metrics_loader.MODEL_ORDER)
        self.curve_model_selector.currentTextChanged.connect(self._plot_curves)
        curve_controls.addWidget(self.curve_model_selector)
        curve_controls.addStretch()
        layout.addLayout(curve_controls)

        self.refresh()

    def refresh(self):
        self._plot_metric_bars()
        self._plot_curves()

    def _plot_metric_bars(self):
        ax = self.metrics_canvas.ax
        ax.clear()
        models = metrics_loader.available_models()
        if not models:
            ax.text(0.5, 0.5, "No evaluated models yet.\nRun src/train.py then src/evaluate.py.",
                    ha="center", va="center", transform=ax.transAxes)
            self.metrics_canvas.draw()
            return

        metric_keys = ["accuracy", "precision", "recall", "f1_score"]
        x = np.arange(len(metric_keys))
        width = 0.8 / len(models)

        for i, model_name in enumerate(models):
            metrics = metrics_loader.load_metrics(model_name)
            values = [metrics[k] for k in metric_keys]
            ax.bar(x + i * width, values, width, label=metrics_loader.DISPLAY_NAMES[model_name])

        ax.set_xticks(x + width * (len(models) - 1) / 2)
        ax.set_xticklabels(["Accuracy", "Precision", "Recall", "F1"])
        ax.set_ylim(0, 1)
        ax.set_title("Model comparison")
        ax.legend(fontsize=8)
        self.metrics_canvas.draw()

    def _plot_curves(self, *_):
        ax = self.curves_canvas.ax
        ax.clear()
        model_name = self.curve_model_selector.currentText()
        curves = metrics_loader.load_curves(model_name)
        if not curves:
            ax.text(0.5, 0.5, f"No training curves for {model_name} yet.",
                    ha="center", va="center", transform=ax.transAxes)
            self.curves_canvas.draw()
            return

        epochs = range(1, len(curves["train_acc"]) + 1)
        ax.plot(epochs, curves["train_acc"], label="Train acc")
        ax.plot(epochs, curves["val_acc"], label="Val acc")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"{metrics_loader.DISPLAY_NAMES.get(model_name, model_name)} training curves")
        ax.legend(fontsize=8)
        self.curves_canvas.draw()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AGHV-Net Fuzzy-Neuro Controller — Flower Classification")
        self.resize(1200, 800)

        tabs = QTabWidget()
        tabs.addTab(SimulatorTab(), "Simulator")
        tabs.addTab(ComparisonTab(), "Model Comparison")
        self.setCentralWidget(tabs)
