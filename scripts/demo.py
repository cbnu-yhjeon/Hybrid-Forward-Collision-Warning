"""정성 확인용 데모: KITTI Tracking 시퀀스, 이미지 폴더, 동영상 → 결과 MP4/PNG.

    uv run python scripts/demo.py --config configs/E3.yaml --kitti-seq 0001 --out results/demo_0001.mp4
    uv run python scripts/demo.py --config configs/E2.yaml --input Data/open_dataset/drive.mp4 --fy 613
    uv run python scripts/demo.py --config configs/E1.yaml --detector yolo ...   # 검출기만 교체

KITTI가 아닌 입력은 --fy(기본 v2 실습 카메라 613)와 이미지 중심을 주점으로 쓴다.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_experiment import load_config

from hfcw.data.kitti import Calib, KittiTracking
from hfcw.pipeline.fcw import FCWPipeline
from hfcw.pipeline.visualize import draw


def iter_input(args):
    if args.kitti_seq:
        ds = KittiTracking()
        for fr in ds.frames(args.kitti_seq):
            yield cv2.imread(str(fr.image_path)), fr.calib, fr
        return
    p = Path(args.input)
    if p.is_dir():
        frames = (cv2.imread(str(f)) for f in sorted(p.iterdir())
                  if f.suffix.lower() in {".png", ".jpg", ".jpeg"})
    else:
        cap = cv2.VideoCapture(str(p))

        def gen():
            while True:
                ok, f = cap.read()
                if not ok:
                    break
                yield f
        frames = gen()
    calib = None
    for img in frames:
        if calib is None:
            h, w = img.shape[:2]
            calib = Calib(np.array([[args.fy, 0, w / 2, 0], [0, args.fy, h / 2, 0], [0, 0, 1, 0]]))
        yield img, calib, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--detector", choices=["classical", "classical_v1", "yolo"])
    ap.add_argument("--kitti-seq")
    ap.add_argument("--input")
    ap.add_argument("--fy", type=float, default=613.0)
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--max", type=int, default=0)
    ap.add_argument("--gt", action="store_true", help="KITTI GT 박스 함께 그리기")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = load_config(Path(args.config), args.detector)
    cfg["fps"] = args.fps
    pipe = FCWPipeline(cfg)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    for k, (img, calib, fr) in enumerate(tqdm(iter_input(args))):
        if args.max and k >= args.max:
            break
        res = pipe.process(img, calib, frame=fr)
        canvas = draw(img, res.objects, res.lanes, gt=fr.objects if (fr and args.gt) else None,
                      title=f"{cfg['name']} | {1 / res.timings['total']:.1f} FPS")
        if out.suffix == ".mp4":
            if writer is None:
                h, w = canvas.shape[:2]
                writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w, h))
            writer.write(canvas)
        else:
            out.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out / f"{k:06d}.png"), canvas)
    if writer:
        writer.release()
    print("saved", out)


if __name__ == "__main__":
    main()
