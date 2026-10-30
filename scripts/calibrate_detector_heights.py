"""검출기 박스 기준 유효 높이 보정 (E3b). train 분할만 사용.

    uv run python scripts/calibrate_detector_heights.py --config configs/E2.yaml --key yolo11m

검출을 GT와 IoU ≥ 0.5로 매칭하고 (occluded == 0, truncated ≤ 0.15인 GT만),
검출 클래스별 median(h_det · z_gt / fy)를 configs/height_stats.json의
detector_effective[<key>]에 저장한다.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hfcw.data.kitti import KittiObject  # noqa: E402
from hfcw.detectors import build_detector  # noqa: E402
from hfcw.types import box_iou  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--key", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    det = build_detector(cfg["detector"])
    ds = KittiObject()
    samples = defaultdict(list)
    for fid in tqdm(ds.split("train")):
        fr = ds.frame(fid)
        dets = det.detect(cv2.imread(str(fr.image_path)))
        gts = [o for o in fr.objects if o.cls is not None and not o.is_dontcare]
        if not dets or not gts:
            continue
        iou = box_iou(np.stack([d.box for d in dets]), np.stack([o.box for o in gts]))
        for k, d in enumerate(dets):
            g = int(iou[k].argmax())
            o = gts[g]
            if iou[k, g] >= 0.5 and o.occluded == 0 and o.truncated <= 0.15 and o.z > 0:
                samples[d.cls].append(d.height * o.z / fr.calib.fy)
    path = ROOT / "configs" / "height_stats.json"
    stats = json.loads(path.read_text())
    stats.setdefault("detector_effective", {})[args.key] = {
        c: round(float(np.median(v)), 4) for c, v in sorted(samples.items())}
    stats.setdefault("detector_effective_n", {})[args.key] = {
        c: len(v) for c, v in sorted(samples.items())}
    path.write_text(json.dumps(stats, indent=2))
    print(stats["detector_effective"][args.key], stats["detector_effective_n"][args.key])


if __name__ == "__main__":
    main()
