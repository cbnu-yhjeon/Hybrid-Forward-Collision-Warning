"""Ultralytics YOLO 검출기 + COCO → KITTI 클래스 매핑.

매핑 (계획서 리스크 대응: "매핑 표를 고정"):

| COCO        | KITTI       | 비고 |
|-------------|-------------|------|
| car         | Car         | |
| truck       | Car         | COCO truck에는 픽업·밴이 많음. KITTI Van도 Car로 합산하므로 일치 |
| bus         | Truck       | KITTI에 bus 클래스 없음. AP에서 무시, 위험도에는 포함 |
| person      | Pedestrian  | bicycle/motorcycle과 겹치면 Cyclist로 병합 |
| bicycle, motorcycle | (Cyclist 병합용) | 사람과 겹치지 않으면 버림 (주차된 자전거) |

KITTI로 미세조정한 가중치(클래스 이름이 이미 KITTI)도 그대로 받습니다.
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np

from hfcw.detectors.base import Detector
from hfcw.types import CAR, CYCLIST, PEDESTRIAN, TRUCK, Detection

COCO_TO_KITTI = {"car": CAR, "truck": CAR, "bus": TRUCK, "person": PEDESTRIAN}
COCO_RIDEABLE = {"bicycle", "motorcycle"}
KITTI_NAMES = {CAR, PEDESTRIAN, CYCLIST, TRUCK}


def merge_cyclists(people: List[Detection], rideables: List[Detection],
                   overlap: float = 0.3) -> List[Detection]:
    """사람 박스가 자전거 박스와 충분히 겹치면 두 박스를 합쳐 Cyclist 하나로 만든다."""
    out, used = [], set()
    for p in people:
        best, best_ov = None, overlap
        for k, r in enumerate(rideables):
            if k in used:
                continue
            ix = max(0.0, min(p.x2, r.x2) - max(p.x1, r.x1))
            iy = max(0.0, min(p.y2, r.y2) - max(p.y1, r.y1))
            ov = ix * iy / max((r.x2 - r.x1) * (r.y2 - r.y1), 1e-6)  # 자전거 박스 기준 겹침
            if ov > best_ov:
                best, best_ov = k, ov
        if best is None:
            out.append(p)
            continue
        used.add(best)
        r = rideables[best]
        out.append(Detection(min(p.x1, r.x1), min(p.y1, r.y1), max(p.x2, r.x2),
                             max(p.y2, r.y2), max(p.score, r.score), CYCLIST))
    return out


class YoloDetector(Detector):
    name = "yolo"

    def __init__(self, weights: str = "weights/yolo11m.pt", conf: float = 0.25,
                 iou: float = 0.5, imgsz: int = 1280, device: str | None = None,
                 half: bool = True):
        from ultralytics import YOLO

        self.model = YOLO(str(Path(weights)))
        self.conf, self.iou, self.imgsz = conf, iou, imgsz
        self.device = device
        self.half = half
        names = self.model.names
        self.names = names if isinstance(names, dict) else dict(enumerate(names))
        self.native_kitti = set(self.names.values()) <= KITTI_NAMES | {"Van", "DontCare"}

    def detect(self, img, pre=None, frame=None) -> List[Detection]:
        res = self.model.predict(img, conf=self.conf, iou=self.iou, imgsz=self.imgsz,
                                 device=self.device, half=self.half, verbose=False)[0]
        if res.boxes is None or len(res.boxes) == 0:
            return []
        xyxy = res.boxes.xyxy.cpu().numpy().astype(np.float64)
        scores = res.boxes.conf.cpu().numpy()
        labels = res.boxes.cls.cpu().numpy().astype(int)

        dets, people, rideables = [], [], []
        for box, s, c in zip(xyxy, scores, labels):
            name = self.names[int(c)]
            if self.native_kitti:
                cls = CAR if name == "Van" else name
                if cls in KITTI_NAMES:
                    dets.append(Detection(*box, float(s), cls))
                continue
            if name in COCO_RIDEABLE:
                rideables.append(Detection(*box, float(s), name))
            elif name in COCO_TO_KITTI:
                d = Detection(*box, float(s), COCO_TO_KITTI[name])
                (people if d.cls == PEDESTRIAN else dets).append(d)
        return dets + merge_cyclists(people, rideables)
