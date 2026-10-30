"""CLAHE 강화 전처리 (v2 Phase 1, 그대로 이식)."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Preprocessed:
    gray: np.ndarray
    equalized: np.ndarray
    blurred: np.ndarray
    edges: np.ndarray
    visibility: float


def compute_visibility_score(gray: np.ndarray) -> float:
    brightness_score = 1.0 - abs(gray.mean() - 128) / 128.0
    contrast_score = min(gray.std() / 64.0, 1.0)
    return float(np.clip(0.5 * brightness_score + 0.5 * contrast_score, 0.0, 1.0))


def get_visibility_label(score: float) -> str:
    if score >= 0.6:
        return "Good"
    if score >= 0.35:
        return "Moderate"
    return "Poor"


def preprocess(img: np.ndarray, use_clahe: bool = True) -> Preprocessed:
    """시인성이 낮을 때만 CLAHE 적용 → Gaussian → Canny.

    use_clahe=False이면 v1처럼 전역 equalizeHist를 씁니다.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    vis = compute_visibility_score(gray)
    if vis < 0.5:
        if use_clahe:
            equalized = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
        else:
            equalized = cv2.equalizeHist(gray)
    else:
        equalized = gray.copy()
    blurred = cv2.GaussianBlur(equalized, (5, 5), 1.0)
    edges = cv2.Canny(blurred, 50, 150)
    return Preprocessed(gray, equalized, blurred, edges, vis)
