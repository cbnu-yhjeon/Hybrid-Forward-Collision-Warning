"""핀홀 거리 추정 Z = H·fy / h (산업 컴퓨터비전 Lec 10).

v2는 실습 카메라 fy=613을 고정으로 썼습니다. v3는 프레임마다 KITTI P2에서 fy를 읽습니다.

높이 H 선택 (`height` 설정):
- mean:            v2 평균 높이 (차량 1.5 m, 사람 1.7 m)          → E1, E2, E5
- class_mean:      KITTI train 분할의 클래스별 3D 높이 평균          → E3a (ablation)
- class_effective: KITTI train 분할에서 median(h_box·z / fy)        → E3
  2D 박스는 3D 직육면체의 투영이라 실제 높이보다 크게 잡히므로(특히 가까운 차량)
  "2D 박스 높이에 대응하는 유효 높이"를 직접 학습하는 쪽이 핀홀 식과 맞습니다.
- class_effective_det: 위와 같지만 GT 박스 대신 검출기 박스로 train 분할에서 보정 → E3b
  (검출기마다 박스를 긋는 습관이 달라, 보정에 쓴 박스와 추론 박스가 같아야 편향이 없음)
"""
from __future__ import annotations

import json
from pathlib import Path

from hfcw.data.kitti import Calib
from hfcw.types import CAR, CYCLIST, PEDESTRIAN, TRUCK, Detection

V2_MEAN_HEIGHTS = {CAR: 1.5, TRUCK: 1.5, PEDESTRIAN: 1.7, CYCLIST: 1.7, "unknown": 1.5}
DEFAULT_STATS = Path(__file__).resolve().parents[3] / "configs" / "height_stats.json"


def load_height_table(mode: str, stats_path: Path = DEFAULT_STATS,
                      det_key: str = "yolo11m") -> dict[str, float]:
    if mode == "mean":
        return dict(V2_MEAN_HEIGHTS)
    stats = json.loads(Path(stats_path).read_text())
    if mode == "class_effective_det":
        table = dict(stats["detector_effective"][det_key])
    else:
        key = {"class_mean": "mean_height_m", "class_effective": "effective_height_m"}[mode]
        table = {cls: s[key] for cls, s in stats["classes"].items()}
    table.setdefault("unknown", table.get(CAR, 1.5))
    return table


class PinholeEstimator:
    def __init__(self, heights: dict[str, float]):
        self.heights = heights

    @classmethod
    def from_config(cls, cfg: dict) -> PinholeEstimator:
        return cls(load_height_table(cfg.get("height", "mean"),
                                     Path(cfg.get("stats", DEFAULT_STATS)),
                                     cfg.get("det_key", "yolo11m")))

    def height_of(self, det: Detection) -> float:
        return self.heights.get(det.cls, self.heights["unknown"])

    def distance(self, det: Detection, calib: Calib) -> float | None:
        if det.height <= 1:
            return None
        return float(min(max(self.height_of(det) * calib.fy / det.height, 0.5), 300.0))

    @staticmethod
    def lateral(det: Detection, z: float | None, calib: Calib) -> float | None:
        """X = (u − cx)·Z / fx, u는 박스 중심."""
        if z is None:
            return None
        return float((det.cx - calib.cx) * z / calib.fx)

    def estimate(self, img, dets: list[Detection],
                 calib: Calib) -> list[tuple[float | None, float | None]]:
        out = []
        for d in dets:
            z = self.distance(d, calib)
            out.append((z, self.lateral(d, z, calib)))
        return out
