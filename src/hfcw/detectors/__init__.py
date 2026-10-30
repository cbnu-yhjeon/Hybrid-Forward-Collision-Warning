"""검출기 팩토리. `--detector classical | yolo` 하나로 전환 (목표 G1)."""
from __future__ import annotations

from hfcw.detectors.base import Detector


def build_detector(cfg: dict) -> Detector:
    kind = cfg.get("type", "classical")
    if kind in ("classical", "classical_v2"):
        from hfcw.detectors.classical import ClassicalV2Detector
        return ClassicalV2Detector(use_pedestrian_hog=cfg.get("pedestrian_hog", True))
    if kind == "classical_v1":
        from hfcw.detectors.classical import ClassicalV1Detector
        return ClassicalV1Detector()
    if kind == "yolo":
        from hfcw.detectors.yolo import YoloDetector
        return YoloDetector(weights=cfg.get("weights", "weights/yolo11m.pt"),
                            conf=cfg.get("conf", 0.25), iou=cfg.get("iou", 0.5),
                            imgsz=cfg.get("imgsz", 1280), device=cfg.get("device"),
                            half=cfg.get("half", True))
    if kind == "gt":
        from hfcw.detectors.gt import GroundTruthDetector
        return GroundTruthDetector()
    raise ValueError(f"unknown detector: {kind}")


__all__ = ["Detector", "build_detector"]
