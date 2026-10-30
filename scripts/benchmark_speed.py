"""경합 없는 환경에서 실험별 처리 속도 측정 → results/speed.json.

    uv run python scripts/benchmark_speed.py --configs configs/E1.yaml configs/E2.yaml --n 200
    uv run python scripts/benchmark_speed.py --configs configs/E2.yaml --device cpu

이미지는 미리 메모리에 올려 디코딩 시간을 빼고, 앞 10프레임은 워밍업으로 버린다.
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
sys.path.insert(0, str(ROOT / "scripts"))

from run_experiment import load_config

from hfcw.data.kitti import KittiObject
from hfcw.pipeline.fcw import FCWPipeline

WARMUP = 10


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    args = ap.parse_args()

    ds = KittiObject()
    frames = [ds.frame(f) for f in ds.split("val", limit=args.n + WARMUP)]
    imgs = [cv2.imread(str(f.image_path)) for f in frames]
    out_path = ROOT / "results" / "speed.json"
    speed = json.loads(out_path.read_text()) if out_path.exists() else {}
    for path in args.configs:
        cfg = load_config(Path(path))
        if args.device == "cpu":
            cfg["detector"]["device"] = "cpu"
            cfg["detector"]["half"] = False
            cfg["distance"]["device"] = "cpu"
        pipe = FCWPipeline(cfg)
        times = []
        for fr, img in zip(frames, imgs):
            pipe.reset()
            times.append(pipe.process(img, fr.calib, frame=fr).timings)
        times = times[WARMUP:]
        res = {k: float(np.mean([t[k] for t in times]) * 1000) for k in times[0]}
        res = {f"{k}_ms": round(v, 2) for k, v in res.items()}
        res["fps"] = round(1000.0 / res["total_ms"], 2)
        speed.setdefault(cfg["name"], {})[args.device] = res
        print(cfg["name"], args.device, res)
        out_path.write_text(json.dumps(speed, indent=2))


if __name__ == "__main__":
    main()
