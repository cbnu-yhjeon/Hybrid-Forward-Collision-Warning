"""KITTI 3D 라벨 → GT 위험 등급.

| 등급    | 조건 |
|---------|------|
| DANGER  | z < 10 m 이고 |x| < 1.8 m (자차 차로 안) |
| CAUTION | z < 25 m 이고 |x| < 3.5 m |
| SAFE    | 그 외 |

x는 image_2 카메라 기준 횡방향 거리(kitti.py에서 cam0→cam2 보정).
"""
from __future__ import annotations

from hfcw.data.kitti import Object3D
from hfcw.pipeline.risk import level_from_geometry
from hfcw.types import SAFE


def risk_objects(objects: list[Object3D]) -> list[Object3D]:
    """위험 등급 평가 대상: DontCare와 Tram/Misc를 뺀 모든 도로 이용자."""
    return [o for o in objects if o.cls is not None and not o.is_dontcare]


def object_level(o: Object3D) -> int:
    return level_from_geometry(o.z, o.x)


def frame_level(objects: list[Object3D]) -> int:
    return max((object_level(o) for o in risk_objects(objects)), default=SAFE)
