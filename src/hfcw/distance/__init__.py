"""거리 추정기 팩토리."""
from __future__ import annotations

from hfcw.distance.pinhole import PinholeEstimator


def build_distance(cfg: dict):
    kind = cfg.get("type", "pinhole")
    if kind == "none":
        return None
    pinhole = PinholeEstimator.from_config(cfg)
    if kind == "pinhole":
        return pinhole
    if kind == "mono_depth":
        from hfcw.distance.mono_depth import MonoDepthEstimator
        return MonoDepthEstimator(pinhole, model=cfg.get("model",
                                  "depth-anything/Depth-Anything-V2-Small-hf"),
                                  device=cfg.get("device"), metric=cfg.get("metric", False),
                                  scale=cfg.get("scale", "fit"), fusion=cfg.get("fusion", "none"))
    raise ValueError(f"unknown distance estimator: {kind}")
