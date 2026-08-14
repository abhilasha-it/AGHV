"""Reusable Qt widgets for the controller: colored metric cards, pipeline
step cards, and an embedded matplotlib canvas for comparison charts."""

from __future__ import annotations

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QFrame, QLabel, QVBoxLayout, QHBoxLayout, QWidget, QProgressBar

CARD_COLORS = {
    "accuracy": "#4f5ff5",
    "precision": "#1fa363",
    "recall": "#12a37e",
    "f1_score": "#c98a12",
    "misclassification_rate": "#d1433f",
}

METRIC_LABELS = {
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "f1_score": "F1 Score",
    "misclassification_rate": "Misclass. Rate",
}


class MetricCard(QFrame):
    def __init__(self, key: str, value: float | None = None, parent=None):
        super().__init__(parent)
        self.key = key
        self.setObjectName("MetricCard")
        self.setFrameShape(QFrame.Shape.StyledPanel)

        color = CARD_COLORS.get(key, "#444444")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)

        self.value_label = QLabel(self._format(value))
        self.value_label.setStyleSheet(f"color: {color}; font-size: 22px; font-weight: 700;")
        self.value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        underline = QFrame()
        underline.setFixedHeight(3)
        underline.setStyleSheet(f"background-color: {color}; border-radius: 1px;")

        name_label = QLabel(METRIC_LABELS.get(key, key))
        name_label.setStyleSheet("color: #666666; font-size: 12px;")
        name_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        layout.addWidget(self.value_label)
        layout.addWidget(underline)
        layout.addWidget(name_label)
        self.setStyleSheet(
            "#MetricCard { background-color: white; border: 1px solid #e5e5e5; border-radius: 10px; }"
        )

    def _format(self, value: float | None) -> str:
        return f"{value * 100:.2f}%" if value is not None else "--"

    def set_value(self, value: float | None):
        self.value_label.setText(self._format(value))


class PipelineStepCard(QFrame):
    def __init__(self, index: str, title: str, description: str, parent=None):
        super().__init__(parent)
        self.setObjectName("PipelineStepCard")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)

        badge = QLabel(index)
        badge.setFixedSize(28, 28)
        badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        badge.setStyleSheet(
            "background-color: #eaf6ee; color: #1fa363; border-radius: 14px; font-weight: 600;"
        )

        text_layout = QVBoxLayout()
        title_label = QLabel(title)
        title_label.setStyleSheet("font-weight: 600; font-size: 13px;")
        desc_label = QLabel(description)
        desc_label.setStyleSheet("color: #888888; font-size: 11px;")
        text_layout.addWidget(title_label)
        text_layout.addWidget(desc_label)
        text_layout.setSpacing(2)

        layout.addWidget(badge)
        layout.addLayout(text_layout)
        layout.addStretch()
        self.setStyleSheet("#PipelineStepCard { background-color: #fafafa; border-radius: 8px; }")


class ConfidenceBar(QProgressBar):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setRange(0, 100)
        self.setTextVisible(False)
        self.setFixedHeight(8)
        self.setStyleSheet(
            "QProgressBar { background-color: #e2f3e8; border-radius: 4px; }"
            "QProgressBar::chunk { background-color: #1fa363; border-radius: 4px; }"
        )


class MplCanvas(FigureCanvasQTAgg):
    def __init__(self, width=5, height=4, dpi=100):
        self.figure = Figure(figsize=(width, height), dpi=dpi, tight_layout=True)
        super().__init__(self.figure)
        self.ax = self.figure.add_subplot(111)
