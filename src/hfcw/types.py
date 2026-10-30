"""파이프라인 전체가 공유하는 자료형.

v2의 Detection은 정수 x, y, w, h를 썼습니다. v3에서는 YOLO 출력을 그대로 담기 위해
실수 좌표 (x1, y1, x2, y2)를 저장하고, v2 코드가 기대하는 x/y/w/h/cx/cy/area는
프로퍼티로 제공합니다.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# KITTI 평가 클래스 (Van은 Car로, Person_sitting은 Pedestrian으로 합산)
CAR = "Car"
PEDESTRIAN = "Pedestrian"
CYCLIST = "Cyclist"
TRUCK = "Truck"
EVAL_CLASSES = (CAR, PEDESTRIAN, CYCLIST)

# 위험 등급 (v2와 동일한 값)
SAFE, CAUTION, DANGER = 0, 1, 2
LEVEL_NAMES = {SAFE: "SAFE", CAUTION: "CAUTION", DANGER: "DANGER"}


@dataclass
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    score: float = 1.0
    cls: str = "unknown"
    contour: Optional[np.ndarray] = field(default=None, repr=False)
    # 컨투어 기반 검출기는 실제 컨투어 면적을 씀 (v1 크기 점수 재현용)
    contour_area: Optional[float] = None

    # ── v2 호환 프로퍼티 ────────────────────────────────────────────────
    @property
    def x(self) -> int:
        return int(round(self.x1))

    @property
    def y(self) -> int:
        return int(round(self.y1))

    @property
    def w(self) -> int:
        return max(int(round(self.x2 - self.x1)), 1)

    @property
    def h(self) -> int:
        return max(int(round(self.y2 - self.y1)), 1)

    @property
    def height(self) -> float:
        return float(self.y2 - self.y1)

    @property
    def cx(self) -> float:
        return 0.5 * (self.x1 + self.x2)

    @property
    def cy(self) -> float:
        return 0.5 * (self.y1 + self.y2)

    @property
    def area(self) -> float:
        if self.contour_area is not None:
            return float(self.contour_area)
        return float((self.x2 - self.x1) * (self.y2 - self.y1))

    @property
    def box(self) -> np.ndarray:
        return np.array([self.x1, self.y1, self.x2, self.y2], dtype=np.float64)


@dataclass
class ObjectEstimate:
    """한 검출에 대한 파이프라인 최종 출력."""

    det: Detection
    distance_m: Optional[float] = None   # 전방 거리 Z 추정
    lateral_m: Optional[float] = None    # 횡방향 X 추정 (카메라 기준, 오른쪽 +)
    in_lane: bool = True
    track_id: Optional[int] = None
    ttc_s: Optional[float] = None
    risk_v2: int = SAFE                 # v2 점수 공식 등급
    risk_v2_score: float = 0.0
    risk_geo: int = SAFE                # 추정 (Z, X)에 GT와 같은 규칙을 적용한 등급


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(N,4) × (M,4) → (N,M) IoU. 박스는 x1, y1, x2, y2."""
    a = np.asarray(a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)
