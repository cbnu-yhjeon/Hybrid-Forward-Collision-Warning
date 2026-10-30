"""KITTI train 분할에서 클래스별 높이 통계를 계산해 configs/height_stats.json에 저장 (E3).

- mean_height_m:      3D 라벨 높이 h의 평균
- effective_height_m: median(h_box · z / fy) — 2D 박스 높이와 핀홀 식을 직접 맞춘 유효 높이
  (가려짐·잘림 영향을 줄이려고 occluded == 0, truncated ≤ 0.15, 박스 높이 ≥ 25 px만 사용)
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hfcw.data.kitti import KittiObject  # noqa: E402


def main():
    ds = KittiObject()
    ids = ds.split("train")
    H, Heff = defaultdict(list), defaultdict(list)
    for fid in ids:
        fr = ds.frame(fid)
        for o in fr.objects:
            if o.cls is None or o.is_dontcare:
                continue
            H[o.cls].append(o.height_m)
            if o.occluded == 0 and o.truncated <= 0.15 and o.box_height >= 25 and o.z > 0:
                Heff[o.cls].append(o.box_height * o.z / fr.calib.fy)
    stats = {"split": "train", "n_frames": len(ids), "classes": {}}
    for c in sorted(H):
        stats["classes"][c] = {
            "n": len(H[c]),
            "mean_height_m": round(float(np.mean(H[c])), 4),
            "std_height_m": round(float(np.std(H[c])), 4),
            "n_effective": len(Heff[c]),
            "effective_height_m": round(float(np.median(Heff[c])), 4),
        }
    out = ROOT / "configs" / "height_stats.json"
    out.write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
