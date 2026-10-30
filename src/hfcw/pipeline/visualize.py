"""결과 시각화: 위험 등급 색 박스 + 거리/TTC 라벨 + 차선 + (선택) GT 박스."""
from __future__ import annotations

import cv2
import numpy as np

from hfcw.data.kitti import Object3D
from hfcw.pipeline.lane import fit_lane_line
from hfcw.pipeline.risk import level_from_geometry
from hfcw.types import CAUTION, DANGER, LEVEL_NAMES, SAFE, ObjectEstimate

LEVEL_COLOR = {SAFE: (0, 200, 0), CAUTION: (0, 200, 255), DANGER: (0, 0, 255)}


def draw(img: np.ndarray, objects: list[ObjectEstimate], lanes=None,
         gt: list[Object3D] | None = None, use_geo: bool = False,
         title: str = "") -> np.ndarray:
    canvas = img.copy()
    if lanes is not None and lanes[2] is not None and lanes[2].any():
        overlay = np.zeros_like(canvas)
        overlay[lanes[2] > 0] = (0, 80, 0)
        canvas = cv2.addWeighted(canvas, 1.0, overlay, 0.35, 0)
        for seg in (fit_lane_line(lanes[0], canvas.shape[0]),
                    fit_lane_line(lanes[1], canvas.shape[0])):
            if seg:
                cv2.line(canvas, seg[:2], seg[2:], (0, 255, 255), 2)

    if gt:
        for o in gt:
            if o.cls is None or o.is_dontcare:
                continue
            x1, y1, x2, y2 = o.box.astype(int)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 255, 255), 1, cv2.LINE_AA)
            g = LEVEL_NAMES[level_from_geometry(o.z, o.x)][0]
            cv2.putText(canvas, f"GT {o.z:.0f}m {g}", (x1, y2 + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)

    for ob in objects:
        lvl = ob.risk_geo if use_geo else ob.risk_v2
        col = LEVEL_COLOR[lvl]
        d = ob.det
        p1, p2 = (int(d.x1), int(d.y1)), (int(d.x2), int(d.y2))
        cv2.rectangle(canvas, p1, p2, col, 2)
        label = f"{d.cls[:3]} {LEVEL_NAMES[lvl]}"
        if ob.distance_m is not None:
            label += f" {ob.distance_m:.1f}m"
        if ob.ttc_s is not None and ob.ttc_s < 10:
            label += f" TTC{ob.ttc_s:.1f}s"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        cv2.rectangle(canvas, (p1[0], p1[1] - th - 4), (p1[0] + tw + 2, p1[1]), col, -1)
        cv2.putText(canvas, label, (p1[0] + 1, p1[1] - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (0, 0, 0), 1, cv2.LINE_AA)
    if title:
        cv2.putText(canvas, title, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
                    cv2.LINE_AA)
    return canvas
