"""실패 사례 갤러리 (마일스톤 7): 같은 val 프레임에서 E1(v2)과 비교 실험을 나란히 그린다.

    uv run python scripts/make_gallery.py --a E1 --b E3 --n 6

선정 기준 (predictions.jsonl 기반, 재실행 없이 그림만 그림):
- missed_danger: GT 프레임 등급이 DANGER인데 B가 DANGER를 내지 못함
- false_alarm:   GT 프레임 등급이 SAFE인데 B가 DANGER를 냄
- fixed:         A는 틀리고 B는 맞힌 DANGER 프레임
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hfcw.data.kitti import KittiObject  # noqa: E402
from hfcw.eval.gt_risk import frame_level  # noqa: E402
from hfcw.pipeline.visualize import draw  # noqa: E402
from hfcw.types import DANGER, SAFE, Detection, ObjectEstimate  # noqa: E402


def load_preds(name):
    path = ROOT / "results" / name / "predictions" / "predictions.jsonl"
    return {p["frame_id"]: p for p in map(json.loads, open(path))}


def to_objects(p):
    return [ObjectEstimate(det=Detection(*o["box"], o["score"], o["cls"]), distance_m=o["z"],
                           lateral_m=o["x"], risk_v2=o["risk_v2"], risk_geo=o["risk_geo"])
            for o in p["objects"]]


def frame_pred(p, key):
    return max((o[key] for o in p["objects"]), default=SAFE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="E1")
    ap.add_argument("--b", default="E3")
    ap.add_argument("--key", default="risk_v2")
    ap.add_argument("--n", type=int, default=4)
    args = ap.parse_args()

    ds = KittiObject()
    A, B = load_preds(args.a), load_preds(args.b)
    cases = {"missed_danger": [], "false_alarm": [], "fixed": []}
    for fid in sorted(set(A) & set(B)):
        fr = ds.frame(fid)
        g = frame_level(fr.objects)
        a, b = frame_pred(A[fid], args.key), frame_pred(B[fid], args.key)
        if g == DANGER and b != DANGER:
            cases["missed_danger"].append(fid)
        if g == SAFE and b == DANGER:
            cases["false_alarm"].append(fid)
        if g == DANGER and b == DANGER and a != DANGER:
            cases["fixed"].append(fid)

    out_dir = ROOT / "results" / "gallery"
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    for case, fids in cases.items():
        pick = rng.choice(fids, size=min(args.n, len(fids)), replace=False) if fids else []
        panels = []
        for fid in pick:
            fr = ds.frame(fid)
            img = cv2.imread(str(fr.image_path))
            use_geo = args.key == "risk_geo"
            pa = draw(img, to_objects(A[fid]), gt=fr.objects, use_geo=use_geo,
                      title=f"{args.a}  #{fid}")
            pb = draw(img, to_objects(B[fid]), gt=fr.objects, use_geo=use_geo,
                      title=f"{args.b}  #{fid}")
            panels.append(np.hstack([pa, pb]))
        if panels:
            w = max(p.shape[1] for p in panels)
            panels = [cv2.copyMakeBorder(p, 0, 4, 0, w - p.shape[1], cv2.BORDER_CONSTANT)
                      for p in panels]
            grid = np.vstack(panels)
            scale = 1600 / grid.shape[1]
            cv2.imwrite(str(out_dir / f"{case}_{args.a}_vs_{args.b}.jpg"),
                        cv2.resize(grid, None, fx=scale, fy=scale), [cv2.IMWRITE_JPEG_QUALITY, 85])
        print(f"{case}: {len(fids)} frames")


if __name__ == "__main__":
    main()
