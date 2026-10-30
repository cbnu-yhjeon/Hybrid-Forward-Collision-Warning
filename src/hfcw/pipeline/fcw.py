"""FCW v3 파이프라인: 전처리 → 차선 → 검출(교체 가능) → 거리 → 추적·TTC → 위험도."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from hfcw.data.kitti import Calib, Frame
from hfcw.detectors import build_detector
from hfcw.distance import build_distance
from hfcw.pipeline.feature import extract_harris, extract_hog, extract_sift
from hfcw.pipeline.lane import detect_lanes, is_in_lane
from hfcw.pipeline.preprocessing import Preprocessed, preprocess
from hfcw.pipeline.risk import level_from_geometry, risk_v1, risk_v2
from hfcw.pipeline.tracking import ObjectTracker
from hfcw.types import ObjectEstimate


@dataclass
class FrameOutput:
    objects: list[ObjectEstimate]
    pre: Preprocessed
    lanes: tuple = (None, None, None)
    timings: dict = field(default_factory=dict)


class FCWPipeline:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.detector = build_detector(cfg.get("detector", {}))
        self.distance = build_distance(cfg.get("distance", {"type": "pinhole"}))
        self.risk_mode = cfg.get("risk", "v2")
        self.use_clahe = cfg.get("preprocess", {}).get("clahe", True)
        self.use_lanes = cfg.get("lanes", True)
        self.use_features = cfg.get("features", True)
        tr = cfg.get("tracking", {})
        self.tracking = bool(tr) if not isinstance(tr, dict) else tr.get("enabled", True)
        self.fps = float(cfg.get("fps", 10.0))
        self.ttc_mode = (tr.get("ttc", "scale") if isinstance(tr, dict) else "scale")
        self.tracker = ObjectTracker() if self.tracking else None

    def reset(self):
        if self.tracker:
            self.tracker.reset()

    def process(self, img: np.ndarray, calib: Calib, frame: Frame | None = None) -> FrameOutput:
        t = {}
        t0 = time.perf_counter()
        pre = preprocess(img, use_clahe=self.use_clahe)
        lanes = detect_lanes(pre.edges, img.shape) if self.use_lanes else (None, None, None)
        lane_mask = lanes[2]
        t["preprocess"] = time.perf_counter() - t0

        t1 = time.perf_counter()
        dets = self.detector.detect(img, pre=pre, frame=frame)
        t["detect"] = time.perf_counter() - t1

        t2 = time.perf_counter()
        if self.distance is not None:
            zx = self.distance.estimate(img, dets, calib)
        else:
            zx = [(None, None)] * len(dets)
        t["distance"] = time.perf_counter() - t2

        t3 = time.perf_counter()
        tracks = (self.tracker.update(dets, pre.gray, [z for z, _ in zx])
                  if self.tracker else [None] * len(dets))
        if self.use_features and dets and self.risk_mode == "v2":
            harris = extract_harris(pre.gray, dets)
            sift = extract_sift(pre.gray, dets)
            hog = extract_hog(pre.gray, dets)
        else:
            harris = sift = [0] * len(dets)
            hog = [0.0] * len(dets)

        h, w = img.shape[:2]
        objects = []
        for i, d in enumerate(dets):
            z, x = zx[i]
            tr = tracks[i]
            ttc = None
            if tr is not None:
                ttc = {"scale": tr.ttc_scale, "distance": tr.ttc_distance,
                       "v2": tr.ttc_v2}[self.ttc_mode](self.fps)
            in_lane = is_in_lane(d.cx, d.cy, lane_mask) if self.use_lanes else True
            if self.risk_mode == "v1":
                lvl, score = risk_v1(d, w, h, pre.visibility)
            else:
                lvl, score = risk_v2(d, w, h, pre.visibility, harris[i], sift[i], hog[i],
                                     in_lane, ttc, z)
            objects.append(ObjectEstimate(
                det=d, distance_m=z, lateral_m=x, in_lane=in_lane,
                track_id=tr.track_id if tr else None, ttc_s=ttc,
                risk_v2=lvl, risk_v2_score=score, risk_geo=level_from_geometry(z, x)))
        t["track_risk"] = time.perf_counter() - t3
        t["total"] = time.perf_counter() - t0
        return FrameOutput(objects, pre, lanes, t)
