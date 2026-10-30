"""Hough 차선 검출 (v2 Phase 1, 그대로 이식)."""
from __future__ import annotations

import cv2
import numpy as np

Segment = tuple[int, int, int, int]


def detect_lanes(edges: np.ndarray, img_shape: tuple):
    """반환: (left_lines, right_lines, lane_mask)"""
    h, w = img_shape[:2]
    roi_edges = edges.copy()
    roi_edges[: int(h * 0.45), :] = 0  # 상단 45% 하늘·원경 제외

    lines = cv2.HoughLinesP(roi_edges, 1, np.pi / 180, threshold=50,
                            minLineLength=80, maxLineGap=60)
    left: list[Segment] = []
    right: list[Segment] = []
    if lines is not None:
        for x1, y1, x2, y2 in lines.reshape(-1, 4):
            if abs(x2 - x1) < 1:
                continue
            slope = (y2 - y1) / (x2 - x1)
            if abs(slope) < 0.3:  # 수평선 제외
                continue
            (left if slope < 0 else right).append((x1, y1, x2, y2))
    return left, right, _build_lane_mask(left, right, h, w)


def fit_lane_line(lines: list[Segment], h: int) -> Segment | None:
    if not lines:
        return None
    pts = np.array([p for x1, y1, x2, y2 in lines for p in ((x1, y1), (x2, y2))],
                   dtype=np.float32)
    vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
    if abs(vy) < 1e-6:
        return None
    y_bot, y_top = h, int(h * 0.55)
    return (int(x0 + (y_top - y0) / vy * vx), y_top, int(x0 + (y_bot - y0) / vy * vx), y_bot)


def _build_lane_mask(left, right, h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.uint8)
    lf, rf = fit_lane_line(left, h), fit_lane_line(right, h)
    if lf and rf:
        pts = np.array([[lf[0], lf[1]], [rf[0], rf[1]], [rf[2], rf[3]], [lf[2], lf[3]]],
                       dtype=np.int32)
        cv2.fillPoly(mask, [pts], 255)
    return mask


def is_in_lane(cx: float, cy: float, lane_mask: np.ndarray | None) -> bool:
    """객체 중심이 차선 내부인지. 차선 미검출 시 True (v2 동작 유지)."""
    if lane_mask is None or not lane_mask.any():
        return True
    h, w = lane_mask.shape
    return bool(lane_mask[min(max(int(cy), 0), h - 1), min(max(int(cx), 0), w - 1)] > 0)
