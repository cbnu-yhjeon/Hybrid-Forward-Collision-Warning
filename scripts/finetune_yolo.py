"""KITTI train 분할로 YOLO 미세조정 (E2ft, 선택 실험).

    uv run python scripts/finetune_yolo.py --epochs 20 --imgsz 960

- 데이터: KITTI Object train 분할(3,740장)만 사용. val 분할은 평가 전용이라 학습에 쓰지 않음.
  Ultralytics의 학습 중 검증에는 train 분할의 10%를 떼어 사용.
- 클래스: Car(+Van), Pedestrian(+Person_sitting), Cyclist, Truck
- 결과 가중치: weights/yolo11m_kitti.pt
"""
from __future__ import annotations

import argparse
import os
import random
import shutil
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hfcw.data.kitti import KittiObject  # noqa: E402
from hfcw.types import CAR, CYCLIST, PEDESTRIAN, TRUCK  # noqa: E402

NAMES = [CAR, PEDESTRIAN, CYCLIST, TRUCK]


def build_dataset(out: Path) -> Path:
    ds = KittiObject()
    ids = ds.split("train")
    random.Random(1).shuffle(ids)
    n_hold = len(ids) // 10
    parts = {"val": ids[:n_hold], "train": ids[n_hold:]}
    for part, pids in parts.items():
        (out / "images" / part).mkdir(parents=True, exist_ok=True)
        (out / "labels" / part).mkdir(parents=True, exist_ok=True)
        for fid in pids:
            fr = ds.frame(fid)
            link = out / "images" / part / f"{fid}.png"
            if not link.exists():
                os.symlink(fr.image_path, link)
            h, w = cv2.imread(str(fr.image_path), cv2.IMREAD_UNCHANGED).shape[:2]
            lines = []
            for o in fr.objects:
                if o.cls not in NAMES:
                    continue
                x1, y1, x2, y2 = o.box
                lines.append(f"{NAMES.index(o.cls)} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f}"
                             f" {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
            (out / "labels" / part / f"{fid}.txt").write_text("\n".join(lines))
    yaml_path = out / "kitti.yaml"
    yaml_path.write_text(f"path: {out}\ntrain: images/train\nval: images/val\nnames:\n"
                         + "".join(f"  {i}: {n}\n" for i, n in enumerate(NAMES)))
    return yaml_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=str(ROOT / "weights" / "yolo11m.pt"))
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--data-dir", default=str(ROOT / "data" / "kitti_yolo"))
    args = ap.parse_args()

    from ultralytics import YOLO

    data_yaml = build_dataset(Path(args.data_dir))
    model = YOLO(args.base)
    model.train(data=str(data_yaml), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                project=str(ROOT / "runs"), name="kitti_ft", exist_ok=True, seed=0,
                workers=8, cos_lr=True, close_mosaic=5, fliplr=0.5, plots=True)
    best = ROOT / "runs" / "kitti_ft" / "weights" / "best.pt"
    shutil.copy(best, ROOT / "weights" / "yolo11m_kitti.pt")
    print("saved", ROOT / "weights" / "yolo11m_kitti.pt")


if __name__ == "__main__":
    main()
