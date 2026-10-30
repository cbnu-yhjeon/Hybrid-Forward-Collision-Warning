"""KITTI Tracking 시퀀스로 TTC 평가 (마일스톤 6).

    uv run python scripts/eval_ttc.py --configs configs/E1.yaml configs/E3.yaml configs/E4.yaml

시퀀스마다 트래커를 초기화하고 연속 프레임을 처리한 뒤, 예측 박스를 GT 트랙과
IoU ≥ 0.5로 매칭해 세 가지 TTC를 GT TTC와 비교한다.
- ttc_v2:       v2 원식 (면적 증가율)
- ttc_scale:    박스 높이 스케일 변화 (k=3)
- ttc_distance: 추정 거리 차분 (k=3)
결과: results/ttc/metrics.json, results/ttc/<name>_pairs.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hfcw.data.kitti import KittiTracking
from hfcw.eval.ttc_eval import gt_ttc_table, summarize, ttc_or_none
from hfcw.pipeline.fcw import FCWPipeline
from hfcw.types import box_iou

METHODS = ["ttc_v2", "ttc_scale", "ttc_distance"]


def run(cfg: dict, ds: KittiTracking, seqs, stride_frames: int) -> list:
    cfg = dict(cfg)
    cfg["tracking"] = {"enabled": True, "ttc": "scale"}
    pipe = FCWPipeline(cfg)
    rows = []
    for seq in seqs:
        frames = ds.frames(seq)[::stride_frames]
        fps = ds.FPS / stride_frames
        pipe.fps = fps
        gt = gt_ttc_table(frames, fps)
        pipe.reset()
        for i, fr in enumerate(tqdm(frames, desc=f"{cfg['name']} seq {seq}", leave=False)):
            img = cv2.imread(str(fr.image_path))
            out = pipe.process(img, fr.calib, frame=fr)
            gts = [o for o in fr.objects if o.track_id >= 0 and (i, o.track_id) in gt]
            if not gts or not out.objects:
                continue
            iou = box_iou(np.stack([o.det.box for o in out.objects]),
                          np.stack([o.box for o in gts]))
            used = set()
            for k in np.argsort(-iou.max(axis=1)):
                g = int(iou[k].argmax())
                if iou[k, g] < 0.5 or g in used:
                    continue
                used.add(g)
                ob, o = out.objects[k], gts[g]
                tr = pipe.tracker.tracks.get(ob.track_id)
                rows.append({
                    "seq": seq, "frame": i, "track_gt": o.track_id, "cls": o.cls,
                    "z_gt": o.z, "z": ob.distance_m, "ttc_gt": gt[(i, o.track_id)],
                    "ttc_v2": ttc_or_none(tr.ttc_v2(fps)) if tr else None,
                    "ttc_scale": ttc_or_none(tr.ttc_scale(fps)) if tr else None,
                    "ttc_distance": ttc_or_none(tr.ttc_distance(fps)) if tr else None,
                })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", required=True)
    ap.add_argument("--seqs", nargs="*", default=None, help="기본: 21개 전체")
    ap.add_argument("--stride", type=int, default=1, help="프레임 간격 (1 = 10 Hz)")
    args = ap.parse_args()

    ds = KittiTracking()
    seqs = args.seqs or ds.sequences
    out_dir = ROOT / "results" / "ttc"
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    for path in args.configs:
        cfg = yaml.safe_load(Path(path).read_text())
        rows = run(cfg, ds, seqs, args.stride)
        if rows:
            with open(out_dir / f"{cfg['name']}_pairs.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
        metrics[cfg["name"]] = {"sequences": seqs, "stride": args.stride,
                                **summarize(rows, METHODS)}
        print(cfg["name"], json.dumps(metrics[cfg["name"]], indent=1)[:800])
        metrics_path.write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
