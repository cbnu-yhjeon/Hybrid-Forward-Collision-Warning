"""KITTI Tracking 시퀀스에서 TTC 평가.

GT TTC: 같은 GT 트랙의 z를 t−3 … t+3 (7프레임, 0.7 s) 창에서 직선 맞춤해 접근 속도
v = −dz/dt를 구하고 TTC_gt = z / v (v > 0일 때만). 자차 운동은 z가 이미 상대 좌표이므로
자동으로 반영됩니다.

평가 구간: 0 < TTC_gt < 5 s (계획서). 방법별로 MAE, 중앙 절대 오차, 커버리지(값을 낸 비율).
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from hfcw.data.kitti import Frame

HALF_WIN = 3
TTC_MAX = 5.0


def gt_ttc_table(frames: list[Frame], fps: float) -> dict[tuple, float]:
    """(frame_index, track_id) → GT TTC (초). 접근 중이 아니면 키 없음."""
    series = defaultdict(dict)
    for i, fr in enumerate(frames):
        for o in fr.objects:
            if o.track_id >= 0 and o.cls is not None and not o.is_dontcare:
                series[o.track_id][i] = o.z
    out = {}
    for tid, zs in series.items():
        for i, z in zs.items():
            idx = [j for j in range(i - HALF_WIN, i + HALF_WIN + 1) if j in zs]
            if len(idx) < HALF_WIN + 2:
                continue
            slope = np.polyfit(np.array(idx) / fps, [zs[j] for j in idx], 1)[0]
            if slope < -1e-3:
                out[(i, tid)] = z / -slope
    return out


def summarize(rows: list[dict], methods: list[str]) -> dict:
    sel = [r for r in rows if 0 < r["ttc_gt"] < TTC_MAX]
    out = {"n_pairs": len(sel)}
    for m in methods:
        vals = [(r["ttc_gt"], r[m]) for r in sel if r.get(m) is not None]
        if not vals:
            out[m] = {"coverage": 0.0, "mae_s": None, "median_ae_s": None}
            continue
        g, p = np.array(vals).T
        p = np.minimum(p, 99.0)
        ae = np.abs(p - g)
        out[m] = {"coverage": len(vals) / max(len(sel), 1), "mae_s": float(ae.mean()),
                  "median_ae_s": float(np.median(ae)),
                  "within_1s": float((ae < 1.0).mean())}
    return out


def ttc_or_none(v: float | None) -> float | None:
    return None if v is None or not np.isfinite(v) else float(v)
