"""Image preprocessing pipeline for flower classification.

Stages: CLAHE -> bilateral filtering -> ROI segmentation ->
Gabor texture enhancement -> channel-wise color normalization.

Each stage is exposed as a standalone function (so the controller UI can
run/display them one at a time) plus a `PreprocessingPipeline` that chains
all of them and can be dropped into a torchvision transform.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def apply_clahe(image: np.ndarray, clip_limit: float = 2.0, tile_grid_size: int = 8) -> np.ndarray:
    """Contrast Limited Adaptive Histogram Equalization on the L channel of LAB space."""
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def apply_bilateral_filter(image: np.ndarray, d: int = 9, sigma_color: float = 75, sigma_space: float = 75) -> np.ndarray:
    """Edge-preserving noise reduction."""
    return cv2.bilateralFilter(image, d, sigma_color, sigma_space)


def segment_roi(image: np.ndarray, margin: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """Segment the flower region of interest using Otsu thresholding on saturation,
    returning (cropped_and_masked_image, binary_mask) both at the original size.
    Falls back to the full image if no contour is found.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    sat = hsv[:, :, 1]
    _, mask = cv2.threshold(sat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image, np.ones(image.shape[:2], dtype=np.uint8) * 255

    largest = max(contours, key=cv2.contourArea)
    roi_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.drawContours(roi_mask, [largest], -1, 255, thickness=cv2.FILLED)

    x, y, w, h = cv2.boundingRect(largest)
    x0, y0 = max(0, x - margin), max(0, y - margin)
    x1, y1 = min(image.shape[1], x + w + margin), min(image.shape[0], y + h + margin)

    masked = cv2.bitwise_and(image, image, mask=roi_mask)
    masked[roi_mask == 0] = image[roi_mask == 0] // 3  # dim background instead of blacking it out
    cropped = masked[y0:y1, x0:x1]
    cropped = cv2.resize(cropped, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
    return cropped, roi_mask


def enhance_texture(image: np.ndarray, ksize: int = 15, sigma: float = 4.0,
                     theta_steps: int = 4, lambd: float = 10.0, gamma: float = 0.5,
                     blend: float = 0.35) -> np.ndarray:
    """Gabor filter bank texture amplification, blended back onto the color image."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    accum = np.zeros_like(gray, dtype=np.float32)
    for i in range(theta_steps):
        theta = np.pi * i / theta_steps
        kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, 0, ktype=cv2.CV_32F)
        filtered = cv2.filter2D(gray, cv2.CV_32F, kernel)
        accum = np.maximum(accum, filtered)

    accum = cv2.normalize(accum, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    texture_rgb = cv2.cvtColor(accum, cv2.COLOR_GRAY2RGB)
    return cv2.addWeighted(image, 1 - blend, texture_rgb, blend, 0)


def normalize_color(image: np.ndarray) -> np.ndarray:
    """Channel-wise mean/std normalization to ImageNet statistics, returned as float32 in [0, 1]."""
    img = image.astype(np.float32) / 255.0
    return (img - IMAGENET_MEAN) / IMAGENET_STD


@dataclass
class PreprocessingPipeline:
    """Chains all preprocessing stages and records intermediate outputs for the
    'Workflow Pipeline' panel in the controller UI.
    """

    clip_limit: float = 2.0
    tile_grid_size: int = 8
    bilateral_d: int = 9
    bilateral_sigma_color: float = 75
    bilateral_sigma_space: float = 75
    texture_blend: float = 0.35
    stage_outputs: dict = field(default_factory=dict, repr=False)

    def run(self, image: np.ndarray, keep_intermediates: bool = False) -> np.ndarray:
        """image: RGB uint8 array (H, W, 3). Returns a normalized float32 (H, W, 3) array."""
        stages = {}

        stage1 = apply_clahe(image, self.clip_limit, self.tile_grid_size)
        stages["1_clahe"] = stage1

        stage2 = apply_bilateral_filter(stage1, self.bilateral_d, self.bilateral_sigma_color, self.bilateral_sigma_space)
        stages["2_bilateral_filter"] = stage2

        stage3, roi_mask = segment_roi(stage2)
        stages["3_roi_segmentation"] = stage3
        stages["3_roi_mask"] = roi_mask

        stage4 = enhance_texture(stage3, blend=self.texture_blend)
        stages["4_texture_enhancement"] = stage4

        stage5 = normalize_color(stage4)
        stages["5_color_normalization"] = stage5

        if keep_intermediates:
            self.stage_outputs = stages
        return stage5

    def __call__(self, image: np.ndarray) -> np.ndarray:
        return self.run(image, keep_intermediates=False)


PIPELINE_STAGE_LABELS = [
    ("Flower Image Input", "Raw RGB image from dataset"),
    ("CLAHE", "Contrast Limited Adaptive Histogram Equalization"),
    ("Bilateral Filtering", "Edge-preserving noise reduction"),
    ("ROI Segmentation", "Region of interest extraction"),
    ("Texture Enhancement", "Gabor filter-based texture amplification"),
    ("Color Normalization", "Channel-wise mean/std normalization"),
]
