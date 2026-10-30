"""위험도 등급화.

- risk_v1: v1(Midterm) 공식. 면적 + 화면 중앙 거리 + 시인성.
- risk_v2: v2(Final) 공식 그대로. 계획서 원칙("위험도 v2 공식 유지")에 따라 E1~E5의 주 등급.
- risk_geometric: 추정한 (Z, X)에 GT와 같은 규칙을 적용. 검출·거리 품질만으로 얻을 수 있는
  등급 상한을 보기 위한 보조 지표.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from hfcw.types import CAUTION, DANGER, SAFE, Detection

# GT와 공유하는 기하 규칙 (README 초안)
DANGER_Z, DANGER_X = 10.0, 1.8
CAUTION_Z, CAUTION_X = 25.0, 3.5


def level_from_geometry(z: Optional[float], x: Optional[float]) -> int:
    if z is None or x is None or z <= 0:
        return SAFE
    if z < DANGER_Z and abs(x) < DANGER_X:
        return DANGER
    if z < CAUTION_Z and abs(x) < CAUTION_X:
        return CAUTION
    return SAFE


def _level(score: float) -> int:
    return DANGER if score >= 0.6 else (CAUTION if score >= 0.35 else SAFE)


def _center_score(det: Detection, img_w: int, img_h: int) -> float:
    dx = (det.cx - img_w / 2) / (img_w / 2)
    dy = (det.cy - img_h / 2) / (img_h / 2)
    return 1.0 - float(np.clip(np.sqrt(dx ** 2 + dy ** 2) / np.sqrt(2), 0.0, 1.0))


def risk_v1(det: Detection, img_w: int, img_h: int, visibility: float) -> Tuple[int, float]:
    size_score = min(det.area / (img_w * img_h * 0.08), 1.0)
    vis_weight = 1.0 + 0.3 * (1.0 - visibility)
    score = float(np.clip((0.5 * size_score + 0.5 * _center_score(det, img_w, img_h))
                          * vis_weight, 0.0, 1.0))
    return _level(score), score


def risk_v2(det: Detection, img_w: int, img_h: int, visibility: float,
            harris: int = 0, sift: int = 0, hog: float = 0.0, in_lane: bool = True,
            ttc_s: Optional[float] = None,
            distance_m: Optional[float] = None) -> Tuple[int, float]:
    """v2 risk.assess_risk와 같은 식. TTC 가중치는 v2 화면 표기 의도대로 초 단위로 해석합니다."""
    if distance_m is not None:
        size_score = float(np.clip(1.0 - (distance_m - 5.0) / 45.0, 0.0, 1.0))
    else:
        size_score = min(det.area / (img_w * img_h * 0.08), 1.0)
    feat = float(np.clip((harris + sift) / 500.0, 0.0, 1.0))
    feature_score = 0.5 * feat + 0.5 * float(np.clip(hog * 10.0, 0.0, 1.0))
    vis_weight = 1.0 + 0.3 * (1.0 - visibility)
    ttc_weight = 1.0
    if ttc_s is not None:
        ttc_weight = float(np.clip(1.0 + 0.4 * max(0.0, 1.0 - ttc_s / 3.0), 1.0, 1.4))
    lane_weight = 1.1 if in_lane else 0.85
    base = 0.45 * size_score + 0.45 * _center_score(det, img_w, img_h) + 0.10 * feature_score
    score = float(np.clip(base * vis_weight * ttc_weight * lane_weight, 0.0, 1.0))
    return _level(score), score
