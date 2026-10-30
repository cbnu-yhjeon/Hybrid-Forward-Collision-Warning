"""단안 깊이 + 핀홀 스케일 보정 (E4, RQ3).

Depth Anything V2(상대 깊이)는 affine-invariant 역깊이 d를 냅니다. 스케일이 없으므로
같은 프레임의 핀홀 거리들을 앵커로 1/Z ≈ a·d + b 를 가중 최소제곱으로 맞춥니다.

- 가중치: 박스 높이 (가까운 큰 박스일수록 핀홀 거리가 정확)
- 앵커 2개 미만이거나 a ≤ 0이면 b = 0으로 두고 a만 가중 중앙값으로 추정
- 앵커가 없으면 핀홀 거리를 그대로 사용

metric=True (E4m): Depth Anything V2 Metric-Outdoor처럼 미터 깊이 D를 내는 모델.
scale="fit"이면 z = s·D, s = 핀홀/D 비율의 가중 중앙값 (한 프레임에 스케일 하나),
scale="none"이면 모델 출력 그대로.

fusion="border" (E4h): 박스가 화면 경계에 닿은(잘렸을 가능성이 큰) 검출만 깊이를 쓰고
나머지는 핀홀을 쓴다. 잘린 박스는 높이가 작게 잡혀 핀홀이 크게 틀리지만 깊이는 영향이 적다.
"""
from __future__ import annotations

import numpy as np

from hfcw.data.kitti import Calib
from hfcw.distance.pinhole import PinholeEstimator
from hfcw.types import Detection

MIN_ANCHOR_HEIGHT = 25.0  # 이보다 작은 박스는 앵커로 쓰지 않음 (핀홀 거리 불안정)


def touches_border(det: Detection, shape, margin: float = 3.0) -> bool:
    h, w = shape[:2]
    return det.x1 <= margin or det.x2 >= w - 1 - margin or det.y2 >= h - 1 - margin


def box_disparity(disp: np.ndarray, det: Detection) -> float | None:
    """박스 안쪽(가로 중앙 50%, 세로 30~90%)의 역깊이 중앙값. 배경 섞임을 줄인다."""
    h, w = disp.shape
    bw, bh = det.x2 - det.x1, det.y2 - det.y1
    x1, x2 = int(max(det.x1 + 0.25 * bw, 0)), int(min(det.x2 - 0.25 * bw, w))
    y1, y2 = int(max(det.y1 + 0.30 * bh, 0)), int(min(det.y1 + 0.90 * bh, h))
    patch = disp[y1:max(y2, y1 + 1), x1:max(x2, x1 + 1)]
    if patch.size == 0:
        return None
    return float(np.median(patch))


def weighted_median(v: np.ndarray, w: np.ndarray) -> float:
    order = np.argsort(v)
    cw = np.cumsum(w[order])
    return float(v[order][np.searchsorted(cw, cw[-1] / 2)])


def fit_inverse_depth(d: np.ndarray, z: np.ndarray, w: np.ndarray) -> tuple[float, float]:
    """1/z ≈ a·d + b 가중 최소제곱. 실패하면 (a, 0)."""
    inv = 1.0 / z
    if len(d) >= 2 and np.ptp(d) > 1e-6:
        A = np.column_stack([d, np.ones_like(d)]) * np.sqrt(w)[:, None]
        (a, b), *_ = np.linalg.lstsq(A, inv * np.sqrt(w), rcond=None)
        if a > 0:
            return float(a), float(b)
    return weighted_median(inv / np.maximum(d, 1e-6), w), 0.0


class MonoDepthEstimator:
    def __init__(self, pinhole: PinholeEstimator,
                 model: str = "depth-anything/Depth-Anything-V2-Small-hf",
                 device: str | None = None, metric: bool = False, scale: str = "fit",
                 fusion: str = "none"):
        import torch
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        self.torch = torch
        self.pinhole = pinhole
        self.metric = metric
        self.scale = scale
        self.fusion = fusion
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoImageProcessor.from_pretrained(model)
        self.model = AutoModelForDepthEstimation.from_pretrained(model).to(self.device).eval()
        if self.device == "cuda":
            self.model = self.model.half()
        self.last_disparity: np.ndarray | None = None
        self.last_fit: tuple[float, float] = (0.0, 0.0)

    def predict_disparity(self, img_bgr: np.ndarray) -> np.ndarray:
        torch = self.torch
        rgb = np.ascontiguousarray(img_bgr[:, :, ::-1])
        inputs = self.processor(images=rgb, return_tensors="pt").to(self.device)
        if self.device == "cuda":
            inputs = {k: v.half() for k, v in inputs.items()}
        with torch.no_grad():
            pred = self.model(**inputs).predicted_depth  # 상대 역깊이 또는 미터 깊이
        pred = torch.nn.functional.interpolate(pred[:, None].float(), size=img_bgr.shape[:2],
                                               mode="bilinear", align_corners=False)
        return pred[0, 0].cpu().numpy()

    def estimate(self, img: np.ndarray, dets: list[Detection],
                 calib: Calib) -> list[tuple[float | None, float | None]]:
        pin = [self.pinhole.distance(d, calib) for d in dets]
        if not dets:
            return []
        if self.metric and self.fusion == "border" and not any(
                touches_border(d, img.shape) for d in dets):
            # 경계 융합에서는 경계 박스가 없으면 깊이를 쓰지 않으므로 추론을 건너뛴다
            return [(z, self.pinhole.lateral(d, z, calib)) for d, z in zip(dets, pin)]
        disp = self.predict_disparity(img)
        self.last_disparity = disp
        ds = [box_disparity(disp, d) for d in dets]
        if self.metric:
            return self._estimate_metric(dets, ds, pin, calib, img.shape)
        anchors = [(dv, z, d.height) for d, dv, z in zip(dets, ds, pin)
                   if dv is not None and z is not None and dv > 0
                   and d.height >= MIN_ANCHOR_HEIGHT]
        if not anchors:
            return [(z, self.pinhole.lateral(d, z, calib)) for d, z in zip(dets, pin)]
        a_d, a_z, a_w = (np.array(v, dtype=np.float64) for v in zip(*anchors))
        a, b = fit_inverse_depth(a_d, a_z, a_w)
        self.last_fit = (a, b)
        out = []
        for d, dv, zp in zip(dets, ds, pin):
            z = None
            if dv is not None:
                inv = a * dv + b
                z = float(min(max(1.0 / inv, 0.5), 300.0)) if inv > 1e-4 else None
            z = z if z is not None else zp
            out.append((z, self.pinhole.lateral(d, z, calib)))
        return out

    def _estimate_metric(self, dets, depths, pin, calib, shape):
        s = 1.0
        if self.scale == "fit":
            anchors = [(z / dv, d.height) for d, dv, z in zip(dets, depths, pin)
                       if dv is not None and z is not None and dv > 0
                       and d.height >= MIN_ANCHOR_HEIGHT]
            if anchors:
                r, w = (np.array(v, dtype=np.float64) for v in zip(*anchors))
                s = weighted_median(r, w)
        self.last_fit = (s, 0.0)
        out = []
        for d, dv, zp in zip(dets, depths, pin):
            if self.fusion == "border" and zp is not None and not touches_border(d, shape):
                z = zp
            else:
                z = float(min(max(s * dv, 0.5), 300.0)) if dv is not None and dv > 0 else zp
            out.append((z, self.pinhole.lateral(d, z, calib)))
        return out
