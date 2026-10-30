"""GT 박스 오라클 (E5): 박스 오차를 0으로 둔 거리 추정 상한선 실험용."""
from __future__ import annotations

from hfcw.detectors.base import Detector
from hfcw.types import Detection


class GroundTruthDetector(Detector):
    name = "gt"

    def detect(self, img, pre=None, frame=None):
        if frame is None:
            raise ValueError("GroundTruthDetector는 GT가 있는 Frame이 필요합니다")
        return [Detection(*o.box, 1.0, o.cls) for o in frame.objects
                if o.cls is not None and not o.is_dontcare]
