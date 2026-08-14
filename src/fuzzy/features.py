"""Botanical feature extraction: derives the human-interpretable inputs
(petal count, radial symmetry, color intensity) that the ANFIS simulator
fuses alongside the AGHV-Net CNN/ViT confidences.

These are classical image-processing heuristics, not learned features --
that is the point: they give the fuzzy layer inputs a domain expert can
reason about ("high petal count + strong symmetry -> rose-like").
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from src.preprocessing.pipeline import segment_roi


@dataclass
class BotanicalFeatures:
    petal_count: float      # estimated count, roughly 3-30
    symmetry: float         # 0-100, higher = more radially symmetric
    color_intensity: float  # 0-100, mean saturation*value of the ROI


def _petal_count(mask: np.ndarray, min_defect_depth: float = 1000.0) -> int:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0
    contour = max(contours, key=cv2.contourArea)
    if len(contour) < 5:
        return 0

    hull_indices = cv2.convexHull(contour, returnPoints=False)
    hull_indices = np.sort(hull_indices, axis=0)
    try:
        defects = cv2.convexityDefects(contour, hull_indices)
    except cv2.error:
        return 0
    if defects is None:
        return 1

    deep_defects = sum(1 for d in defects[:, 0] if d[3] > min_defect_depth)
    # each petal typically contributes ~1 deep convexity defect at its base
    return max(1, deep_defects)


def _radial_symmetry(mask: np.ndarray, n_angles: int = 12) -> float:
    """Compares the mask against its own rotation by 180/n_angles-degree steps;
    higher average IoU across rotations means stronger radial symmetry.
    """
    moments = cv2.moments(mask, binaryImage=True)
    if moments["m00"] == 0:
        return 0.0
    cx, cy = moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]

    ious = []
    for i in range(1, n_angles):
        angle = 360.0 * i / n_angles
        rot_mat = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
        rotated = cv2.warpAffine(mask, rot_mat, (mask.shape[1], mask.shape[0]))
        intersection = np.logical_and(mask > 0, rotated > 0).sum()
        union = np.logical_or(mask > 0, rotated > 0).sum()
        if union > 0:
            ious.append(intersection / union)
    return float(np.mean(ious) * 100) if ious else 0.0


def _color_intensity(image: np.ndarray, mask: np.ndarray) -> float:
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
    sat, val = hsv[:, :, 1], hsv[:, :, 2]
    roi = mask > 0
    if roi.sum() == 0:
        return 0.0
    intensity = (sat[roi].mean() / 255.0) * (val[roi].mean() / 255.0) * 100
    return float(intensity)


def extract_botanical_features(image: np.ndarray) -> BotanicalFeatures:
    """image: RGB uint8 array (H, W, 3)."""
    _, mask = segment_roi(image)
    petal_count = _petal_count(mask)
    symmetry = _radial_symmetry(mask)
    color_intensity = _color_intensity(image, mask)
    return BotanicalFeatures(petal_count=petal_count, symmetry=symmetry, color_intensity=color_intensity)
