"""IoU 매칭 + LK 광학 흐름 추적, TTC (v2 Phase 3 이식 + TTC 보정).

v2의 TTC는 `area_prev / dArea` (프레임 단위)였습니다. 면적은 1/Z²에 비례하므로
이 식은 실제 TTC의 약 1/2을 줍니다. v3는 박스 높이 비율 s = h_t / h_{t-k}로
TTC = k·Δt / (s − 1) 을 쓰고 (Δt = 1/fps), v2 식은 비교용으로 남겨 둡니다.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from hfcw.types import Detection, box_iou


@dataclass
class Track:
    track_id: int
    det: Detection
    heights: List[float] = field(default_factory=list)
    areas: List[float] = field(default_factory=list)
    distances: List[Optional[float]] = field(default_factory=list)
    centroids: List[Tuple[float, float]] = field(default_factory=list)
    keypoints: Optional[np.ndarray] = None
    age: int = 0
    missed: int = 0

    @property
    def velocity(self) -> Tuple[float, float]:
        if len(self.centroids) < 2:
            return (0.0, 0.0)
        (x0, y0), (x1, y1) = self.centroids[-2], self.centroids[-1]
        return (x1 - x0, y1 - y0)

    def ttc_v2(self, fps: float) -> Optional[float]:
        """v2 원식 (면적 증가율), 초 단위로 환산."""
        if len(self.areas) < 2:
            return None
        prev, now = self.areas[-2], self.areas[-1]
        if now - prev <= 1e-3 or prev <= 0:
            return None
        return float(np.clip(prev / (now - prev), 0.5, 99.0)) / fps

    def ttc_scale(self, fps: float, k: int = 3) -> Optional[float]:
        """박스 높이 스케일 변화 기반 TTC (초). 접근 중이 아니면 None."""
        if len(self.heights) <= k:
            return None
        s = self.heights[-1] / max(self.heights[-1 - k], 1e-6)
        if s <= 1.0 + 1e-3:
            return None
        return float(min(k / fps / (s - 1.0), 99.0))

    def ttc_distance(self, fps: float, k: int = 3) -> Optional[float]:
        """추정 거리 차분 기반 TTC (초): Z / (−dZ/dt)."""
        if len(self.distances) <= k:
            return None
        z_now, z_prev = self.distances[-1], self.distances[-1 - k]
        if z_now is None or z_prev is None:
            return None
        closing = (z_prev - z_now) * fps / k
        if closing <= 1e-3:
            return None
        return float(min(z_now / closing, 99.0))


class ObjectTracker:
    LK_PARAMS = dict(winSize=(15, 15), maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    def __init__(self, max_missed: int = 5, iou_thresh: float = 0.25, use_lk: bool = True):
        self.tracks: Dict[int, Track] = {}
        self.next_id = 0
        self.max_missed = max_missed
        self.iou_thresh = iou_thresh
        self.use_lk = use_lk
        self.prev_gray: Optional[np.ndarray] = None

    def reset(self):
        self.tracks.clear()
        self.next_id = 0
        self.prev_gray = None

    def update(self, dets: List[Detection], gray: np.ndarray,
               distances: Optional[List[Optional[float]]] = None) -> List[Optional[Track]]:
        """검출 목록을 받아 트랙을 갱신하고, 각 검출에 대응하는 트랙을 돌려줍니다."""
        distances = distances if distances is not None else [None] * len(dets)
        if self.use_lk and self.prev_gray is not None and self.tracks:
            self._update_lk(gray)

        tracks = list(self.tracks.values())
        assigned: List[Optional[Track]] = [None] * len(dets)
        if tracks and dets:
            iou = box_iou(np.stack([t.det.box for t in tracks]), np.stack([d.box for d in dets]))
            # 그리디 매칭 (IoU 큰 순서)
            used_t, used_d = set(), set()
            for flat in np.argsort(-iou, axis=None):
                i, j = np.unravel_index(flat, iou.shape)
                if iou[i, j] < self.iou_thresh:
                    break
                if i in used_t or j in used_d:
                    continue
                used_t.add(i)
                used_d.add(j)
                assigned[j] = tracks[i]
        matched_ids = {t.track_id for t in assigned if t is not None}

        for t in tracks:
            if t.track_id not in matched_ids:
                t.missed += 1
        self.tracks = {tid: t for tid, t in self.tracks.items() if t.missed <= self.max_missed}

        for j, det in enumerate(dets):
            t = assigned[j]
            if t is None:
                t = Track(self.next_id, det)
                self.tracks[self.next_id] = t
                self.next_id += 1
                assigned[j] = t
            else:
                t.age += 1
                t.missed = 0
            t.det = det
            t.heights.append(det.height)
            t.areas.append(det.area)
            t.distances.append(distances[j])
            t.centroids.append((det.cx, det.cy))
            if self.use_lk:
                t.keypoints = self._init_keypoints(gray, det)

        self.prev_gray = gray
        return assigned

    def _init_keypoints(self, gray: np.ndarray, det: Detection) -> Optional[np.ndarray]:
        h, w = gray.shape
        x1, y1 = max(det.x, 0), max(det.y, 0)
        x2, y2 = min(det.x + det.w, w), min(det.y + det.h, h)
        roi = gray[y1:y2, x1:x2]
        if roi.size < 200:
            return None
        pts = cv2.goodFeaturesToTrack(roi, maxCorners=20, qualityLevel=0.3, minDistance=5)
        if pts is None:
            return None
        pts[:, 0, 0] += x1
        pts[:, 0, 1] += y1
        return pts

    def _update_lk(self, gray: np.ndarray):
        if self.prev_gray.shape != gray.shape:
            return
        for t in self.tracks.values():
            if t.keypoints is None or len(t.keypoints) == 0:
                continue
            try:
                nxt, status, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, t.keypoints,
                                                          None, **self.LK_PARAMS)
            except cv2.error:
                continue
            if nxt is not None and status is not None:
                good = nxt[status.flatten() == 1]
                if len(good):
                    t.keypoints = good.reshape(-1, 1, 2)
