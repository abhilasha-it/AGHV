"""ANFIS: Adaptive Neuro-Fuzzy Inference System.

A 5-layer Takagi-Sugeno neuro-fuzzy network (Jang, 1993) that fuses the
botanical features (petal count, symmetry, color intensity) with the
AGHV-Net CNN-branch and ViT-branch confidences into one calibrated final
confidence, plus a human-readable rule explanation. This is the "fuzzy
logic controller" sitting on top of the deep-learning backend.

Layers:
  1. Fuzzification      - Gaussian membership degree of each input under
                           each linguistic term (trainable center/width).
  2. Rule / firing       - product T-norm over every input's selected term,
                           one firing strength per rule (all term
                           combinations across inputs).
  3. Normalization        - firing strengths normalized to sum to 1.
  4. Consequent           - per-rule first-order Sugeno output
                           f_i = p_i . [x, 1] (trainable linear params).
  5. Aggregation           - normalized-weighted sum of rule outputs,
                           squashed to [0, 1] as the final confidence.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import torch
import torch.nn as nn

INPUT_NAMES = ["petal_count", "symmetry", "color_intensity", "cnn_conf", "vit_conf"]

# (min, max) raw-value ranges used to scale each input onto a common 0-100 axis.
INPUT_RANGES = {
    "petal_count": (0.0, 30.0),
    "symmetry": (0.0, 100.0),
    "color_intensity": (0.0, 100.0),
    "cnn_conf": (0.0, 100.0),
    "vit_conf": (0.0, 100.0),
}

# Linguistic term labels used when n_mfs == 2, for the natural-language explanation.
LOW_HIGH_LABELS = {
    "petal_count": ("low petal count", "high petal count"),
    "symmetry": ("weak symmetry", "strong radial symmetry"),
    "color_intensity": ("muted color", "vivid color intensity"),
    "cnn_conf": ("low CNN confidence", "high CNN confidence"),
    "vit_conf": ("low ViT confidence", "high ViT confidence"),
}


class GaussianFuzzification(nn.Module):
    """Per-input Gaussian membership functions with trainable center/width."""

    def __init__(self, n_inputs: int, n_mfs: int = 2):
        super().__init__()
        self.n_inputs = n_inputs
        self.n_mfs = n_mfs
        # spread initial centers evenly across the normalized [0, 100] axis
        init_centers = torch.linspace(10, 90, n_mfs).unsqueeze(0).repeat(n_inputs, 1)
        self.centers = nn.Parameter(init_centers)
        self.widths = nn.Parameter(torch.full((n_inputs, n_mfs), 25.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, n_inputs) -> membership degrees (B, n_inputs, n_mfs)."""
        x = x.unsqueeze(-1)  # (B, n_inputs, 1)
        centers = self.centers.unsqueeze(0)  # (1, n_inputs, n_mfs)
        widths = self.widths.clamp(min=1e-2).unsqueeze(0)
        return torch.exp(-((x - centers) ** 2) / (2 * widths ** 2))


class ANFIS(nn.Module):
    def __init__(self, n_inputs: int = 5, n_mfs: int = 2):
        super().__init__()
        if n_inputs != len(INPUT_NAMES):
            raise ValueError(f"ANFIS is wired for inputs {INPUT_NAMES}; got n_inputs={n_inputs}")
        self.n_inputs = n_inputs
        self.n_mfs = n_mfs
        self.fuzzify = GaussianFuzzification(n_inputs, n_mfs)

        self.rule_indices = list(itertools.product(range(n_mfs), repeat=n_inputs))
        self.n_rules = len(self.rule_indices)
        # Sugeno first-order consequent params: [w_1..w_n, bias] per rule
        self.consequents = nn.Parameter(torch.randn(self.n_rules, n_inputs + 1) * 0.01)

    def _rule_firing_strengths(self, memberships: torch.Tensor) -> torch.Tensor:
        """memberships: (B, n_inputs, n_mfs) -> firing strengths (B, n_rules)."""
        batch_size = memberships.shape[0]
        strengths = torch.ones(batch_size, self.n_rules, device=memberships.device)
        for rule_idx, term_combo in enumerate(self.rule_indices):
            for input_idx, term in enumerate(term_combo):
                strengths[:, rule_idx] *= memberships[:, input_idx, term]
        return strengths

    def forward(self, x: torch.Tensor):
        """x: (B, n_inputs) raw values already scaled onto the shared 0-100 axis.
        Returns (confidence in [0,1] of shape (B,), normalized firing strengths (B, n_rules)).
        """
        memberships = self.fuzzify(x)  # (B, n_inputs, n_mfs)
        firing = self._rule_firing_strengths(memberships)  # (B, n_rules)
        norm_firing = firing / (firing.sum(dim=1, keepdim=True) + 1e-8)

        x_ext = torch.cat([x, torch.ones(x.shape[0], 1, device=x.device)], dim=1)  # (B, n_inputs+1)
        rule_outputs = x_ext @ self.consequents.T  # (B, n_rules)

        aggregated = (norm_firing * rule_outputs).sum(dim=1)  # (B,)
        confidence = torch.sigmoid(aggregated)
        return confidence, norm_firing, memberships


@dataclass
class ANFISResult:
    confidence: float
    explanation: str
    firing_strengths: torch.Tensor


def scale_raw_inputs(petal_count: float, symmetry: float, color_intensity: float,
                      cnn_conf: float, vit_conf: float) -> torch.Tensor:
    """Scales raw feature values (cnn_conf/vit_conf given as 0-1 probabilities or
    0-100 percentages) onto the shared 0-100 axis the network was built for.
    """
    if cnn_conf <= 1.0:
        cnn_conf *= 100
    if vit_conf <= 1.0:
        vit_conf *= 100
    raw = {"petal_count": petal_count, "symmetry": symmetry,
           "color_intensity": color_intensity, "cnn_conf": cnn_conf, "vit_conf": vit_conf}
    scaled = []
    for name in INPUT_NAMES:
        lo, hi = INPUT_RANGES[name]
        value = max(lo, min(hi, raw[name]))
        scaled.append((value - lo) / (hi - lo) * 100)
    return torch.tensor(scaled, dtype=torch.float32).unsqueeze(0)


def explain(model: ANFIS, memberships: torch.Tensor, predicted_class_name: str,
            confidence: float, top_k: int = 2) -> str:
    """Builds a rule-style explanation from the inputs with the strongest
    'high' membership degree, e.g. 'High petal count + strong radial symmetry
    -> Rose rule activated at high confidence.'
    """
    if model.n_mfs != 2:
        return f"{predicted_class_name} predicted with confidence {confidence:.1%}."

    high_term_degrees = memberships[0, :, 1]  # membership degree in the "high" term, per input
    top_indices = torch.topk(high_term_degrees, k=min(top_k, model.n_inputs)).indices.tolist()

    phrases = []
    for idx in top_indices:
        input_name = INPUT_NAMES[idx]
        degree = high_term_degrees[idx].item()
        low_label, high_label = LOW_HIGH_LABELS[input_name]
        phrases.append(high_label if degree >= 0.5 else low_label)

    level = "high" if confidence >= 0.7 else ("moderate" if confidence >= 0.4 else "low")
    joined = " + ".join(p.capitalize() if i == 0 else p for i, p in enumerate(phrases))
    return f"{joined} → {predicted_class_name} rule activated at {level} confidence."


def run_anfis_simulation(model: ANFIS, petal_count: float, symmetry: float, color_intensity: float,
                          cnn_conf: float, vit_conf: float, predicted_class_name: str) -> ANFISResult:
    model.eval()
    with torch.no_grad():
        x = scale_raw_inputs(petal_count, symmetry, color_intensity, cnn_conf, vit_conf)
        confidence, firing, memberships = model(x)
        explanation = explain(model, memberships, predicted_class_name, confidence.item())
    return ANFISResult(confidence=confidence.item(), explanation=explanation, firing_strengths=firing)
