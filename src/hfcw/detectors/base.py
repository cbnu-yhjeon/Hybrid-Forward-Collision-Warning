"""공통 Detector 인터페이스 (목표 G1)."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np

from hfcw.data.kitti import Frame
from hfcw.pipeline.preprocessing import Preprocessed
from hfcw.types import Detection


class Detector(ABC):
    name = "base"

    @abstractmethod
    def detect(self, img: np.ndarray, pre: Optional[Preprocessed] = None,
               frame: Optional[Frame] = None) -> List[Detection]:
        """BGR 이미지 → 검출 목록.

        pre: 파이프라인이 이미 계산한 전처리 결과 (고전 검출기가 재사용)
        frame: GT 오라클 검출기 전용. 일반 검출기는 무시합니다.
        """
