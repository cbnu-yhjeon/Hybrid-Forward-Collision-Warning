"""KITTI Object val 분할에서 실험 하나를 실행하고 평가한다.

    uv run python scripts/run_experiment.py --config configs/E2.yaml
    uv run python scripts/run_experiment.py --config configs/E1.yaml --limit 200
    uv run python scripts/run_experiment.py --config configs/E2.yaml --detector classical  # 검출기만 교체

결과: results/<name>/metrics.json, distance_pairs.csv, predictions/predictions.jsonl
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
from pathlib import Path

import cv2
import yaml
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hfcw.data.kitti import KittiObject
from hfcw.eval.kitti_eval import evaluate
from hfcw.pipeline.fcw import FCWPipeline


def load_config(path: Path, detector: str | None = None) -> dict:
    cfg = yaml.safe_load(Path(path).read_text())
    if detector:
        cfg = copy.deepcopy(cfg)
        cfg["detector"] = {"type": detector, **({"weights": "weights/yolo11m.pt"}
                                                if detector == "yolo" else {})}
        cfg["name"] = f"{cfg['name']}_{detector}"
    return cfg


def serialize(frame_id: str, fy: float, out) -> dict:
    return {
        "frame_id": frame_id,
        "fy": fy,
        "objects": [{
            "box": [round(float(v), 2) for v in o.det.box],
            "score": round(float(o.det.score), 4),
            "cls": o.det.cls,
            "z": None if o.distance_m is None else round(o.distance_m, 3),
            "x": None if o.lateral_m is None else round(o.lateral_m, 3),
            "in_lane": o.in_lane,
            "risk_v2": int(o.risk_v2),
            "risk_v2_score": round(o.risk_v2_score, 4),
            "risk_geo": int(o.risk_geo),
        } for o in out.objects],
        "timings": {k: round(v, 5) for k, v in out.timings.items()},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--detector", choices=["classical", "classical_v1", "yolo", "gt"],
                    help="설정 파일의 검출기를 덮어쓴다")
    ap.add_argument("--split", default="val")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "results"))
    ap.add_argument("--reuse", action="store_true", help="기존 predictions.jsonl로 평가만 다시")
    args = ap.parse_args()

    cfg = load_config(Path(args.config), args.detector)
    out_dir = Path(args.out) / cfg["name"]
    (out_dir / "predictions").mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "predictions" / "predictions.jsonl"

    ds = KittiObject()
    ids = ds.split(args.split, limit=args.limit)
    frames = {fid: ds.frame(fid) for fid in ids}

    if not (args.reuse and pred_path.exists()):
        pipe = FCWPipeline(cfg)
        with open(pred_path, "w") as f:
            for fid in tqdm(ids, desc=cfg["name"]):
                fr = frames[fid]
                img = cv2.imread(str(fr.image_path))
                if img is None:
                    raise FileNotFoundError(fr.image_path)
                pipe.reset()  # Object 데이터는 프레임 간 연속성이 없음
                out = pipe.process(img, fr.calib, frame=fr)
                f.write(json.dumps(serialize(fid, fr.calib.fy, out)) + "\n")

    preds = {}
    for line in open(pred_path):
        p = json.loads(line)
        preds[p["frame_id"]] = p
    frames = {k: v for k, v in frames.items() if k in preds}
    metrics, rows = evaluate(frames, preds)
    metrics["config"] = cfg
    metrics["split"] = args.split
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    if rows:
        with open(out_dir / "distance_pairs.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    d = metrics["detection"]
    print(f"[{cfg['name']}] frames={metrics['n_frames']}")
    print("  AP50 " + " ".join(f"{k}={v['AP50_R40']:.3f}" for k, v in d.items())
          + f" | FP/frame={d['All']['fp_per_frame']:.2f}")
    if metrics["distance"]:
        o = metrics["distance"]["overall"]
        print(f"  dist MAE={o['mae_m']:.2f}m rel={o['rel_err']:.3f} n={o['n']}")
    for key in ("risk_v2", "risk_geo"):
        r = metrics.get(key)
        if r:
            print(f"  {key}: frame macroF1={r['frame']['macro_f1']:.3f} "
                  f"DANGER recall={r['frame']['danger_recall']:.3f} | object macroF1="
                  f"{r['object']['macro_f1']:.3f} DANGER recall={r['object']['danger_recall']:.3f}")
    print(f"  speed: {metrics['speed'].get('fps', 0):.1f} FPS")


if __name__ == "__main__":
    main()
