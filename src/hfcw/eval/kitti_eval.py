"""KITTI Object 평가: 검출 AP · 거리 오차 · 위험 등급.

예측 형식 (프레임당 한 줄 JSON):
    {"frame_id": "000123", "fy": 721.5, "objects": [
        {"box": [x1,y1,x2,y2], "score": .9, "cls": "Car", "z": 12.3, "x": -0.4,
         "risk_v2": 1, "risk_geo": 1}, ...], "timings": {...}}

검출 평가 규칙 (KITTI moderate 기준을 단순화):
- 양성 GT: 해당 클래스, 박스 높이 ≥ 25 px, occluded ≤ 1, truncated ≤ 0.3
- 무시 GT: 해당 클래스지만 위 조건 미달, Car 평가에서는 Truck도 무시 (COCO truck→Car 매핑)
- 무시 검출: 높이 < 25 px, 또는 DontCare 영역과 50% 이상 겹침
- AP: IoU 0.5, 40점 보간 (KITTI R40)
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from hfcw.data.kitti import Frame, Object3D
from hfcw.eval.gt_risk import object_level, risk_objects
from hfcw.types import CAR, EVAL_CLASSES, LEVEL_NAMES, SAFE, TRUCK, box_iou

IOU_THR = 0.5
MIN_HEIGHT = 25.0
MAX_OCC = 1
MAX_TRUNC = 0.3
DIST_BINS = [(0, 10), (10, 20), (20, 40), (40, 1e9)]
BIN_NAMES = ["0-10", "10-20", "20-40", "40+"]


def _is_moderate(o: Object3D) -> bool:
    return o.box_height >= MIN_HEIGHT and o.occluded <= MAX_OCC and o.truncated <= MAX_TRUNC


def _dontcare_overlap(boxes: np.ndarray, dc: np.ndarray) -> np.ndarray:
    """검출 박스 면적 대비 DontCare 영역과의 교집합 비율 (최댓값)."""
    if len(boxes) == 0 or len(dc) == 0:
        return np.zeros(len(boxes))
    ix = np.clip(np.minimum(boxes[:, None, 2], dc[None, :, 2])
                 - np.maximum(boxes[:, None, 0], dc[None, :, 0]), 0, None)
    iy = np.clip(np.minimum(boxes[:, None, 3], dc[None, :, 3])
                 - np.maximum(boxes[:, None, 1], dc[None, :, 1]), 0, None)
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return (ix * iy / np.maximum(area[:, None], 1e-6)).max(axis=1)


def _ap_r40(scores: np.ndarray, tps: np.ndarray, n_pos: int) -> Tuple[float, list]:
    if n_pos == 0:
        return float("nan"), []
    order = np.argsort(-scores, kind="stable")
    tp = tps[order].astype(np.float64)
    ctp, cfp = np.cumsum(tp), np.cumsum(1 - tp)
    recall = ctp / n_pos
    precision = ctp / np.maximum(ctp + cfp, 1e-9)
    # 단조 감소 정밀도 포락선
    prec_env = np.maximum.accumulate(precision[::-1])[::-1] if len(precision) else precision
    ap = 0.0
    for r in np.linspace(1 / 40, 1, 40):
        idx = np.searchsorted(recall, r, side="left")
        ap += prec_env[idx] if idx < len(prec_env) else 0.0
    curve = [[float(r), float(p)] for r, p in zip(recall[::max(len(recall) // 200, 1)],
                                                    precision[::max(len(recall) // 200, 1)])]
    return ap / 40, curve


def _match_detection(frame: Frame, preds: List[dict], cls: Optional[str]):
    """한 프레임, 한 클래스(None=클래스 무관)에 대해 (score, tp) 리스트와 양성 수, FP 수를 반환."""
    objs = [o for o in frame.objects if o.cls is not None and not o.is_dontcare]
    dc = np.array([o.box for o in frame.objects if o.is_dontcare]).reshape(-1, 4)
    if cls is None:
        pos = [o for o in objs if o.cls in EVAL_CLASSES and _is_moderate(o)]
        ign = [o for o in objs if not (o.cls in EVAL_CLASSES and _is_moderate(o))]
        dets = [p for p in preds if p["cls"] in EVAL_CLASSES or p["cls"] == TRUCK]
    else:
        pos = [o for o in objs if o.cls == cls and _is_moderate(o)]
        ign = [o for o in objs if (o.cls == cls and not _is_moderate(o))
               or (cls == CAR and o.cls == TRUCK)]
        dets = [p for p in preds if p["cls"] == cls]
    dets = sorted(dets, key=lambda p: -p["score"])
    boxes = np.array([p["box"] for p in dets]).reshape(-1, 4)
    iou_pos = box_iou(boxes, np.array([o.box for o in pos]).reshape(-1, 4))
    iou_ign = box_iou(boxes, np.array([o.box for o in ign]).reshape(-1, 4))
    dc_ov = _dontcare_overlap(boxes, dc)

    used, out, n_fp = set(), [], 0
    for k, p in enumerate(dets):
        best, best_iou = -1, IOU_THR
        for g in range(iou_pos.shape[1]):
            if g not in used and iou_pos[k, g] >= best_iou:
                best, best_iou = g, iou_pos[k, g]
        if best >= 0:
            used.add(best)
            out.append((p["score"], 1))
            continue
        h = boxes[k, 3] - boxes[k, 1]
        if (iou_ign.shape[1] and iou_ign[k].max() >= IOU_THR) or h < MIN_HEIGHT or dc_ov[k] > 0.5:
            continue
        out.append((p["score"], 0))
        n_fp += 1
    return out, len(pos), n_fp


def detection_metrics(frames: Dict[str, Frame], preds: Dict[str, dict]) -> dict:
    res = {}
    for cls in list(EVAL_CLASSES) + [None]:
        name = cls or "All"
        rows, n_pos, n_fp = [], 0, 0
        for fid, fr in frames.items():
            r, npos, nfp = _match_detection(fr, preds[fid]["objects"], cls)
            rows += r
            n_pos += npos
            n_fp += nfp
        scores = np.array([s for s, _ in rows])
        tps = np.array([t for _, t in rows])
        ap, curve = _ap_r40(scores, tps, n_pos)
        n_tp = int(tps.sum()) if len(tps) else 0
        res[name] = {
            "AP50_R40": ap,
            "n_gt": n_pos,
            "precision": n_tp / max(n_tp + n_fp, 1),
            "recall": n_tp / max(n_pos, 1),
            "fp_per_frame": n_fp / max(len(frames), 1),
            "pr_curve": curve,
        }
    return res


def match_any(frame: Frame, preds: List[dict]) -> Tuple[List[Tuple[dict, Object3D]],
                                                         List[Object3D], List[dict]]:
    """클래스 무관 그리디 매칭 (거리·위험 등급 평가용). DontCare 위 검출은 버린다."""
    gts = risk_objects(frame.objects)
    dc = np.array([o.box for o in frame.objects if o.is_dontcare]).reshape(-1, 4)
    dets = sorted(preds, key=lambda p: -p["score"])
    boxes = np.array([p["box"] for p in dets]).reshape(-1, 4)
    iou = box_iou(boxes, np.array([o.box for o in gts]).reshape(-1, 4))
    dc_ov = _dontcare_overlap(boxes, dc)
    pairs, used, fps = [], set(), []
    for k, p in enumerate(dets):
        best, best_iou = -1, IOU_THR
        for g in range(iou.shape[1]):
            if g not in used and iou[k, g] >= best_iou:
                best, best_iou = g, iou[k, g]
        if best >= 0:
            used.add(best)
            pairs.append((p, gts[best]))
        elif dc_ov[k] <= 0.5:
            fps.append(p)
    missed = [o for g, o in enumerate(gts) if g not in used]
    return pairs, missed, fps


def distance_pairs(frames: Dict[str, Frame], preds: Dict[str, dict]) -> List[dict]:
    rows = []
    for fid, fr in frames.items():
        pairs, _, _ = match_any(fr, preds[fid]["objects"])
        for p, o in pairs:
            if p.get("z") is None:
                continue
            rows.append({
                "frame_id": fid, "cls": o.cls, "z_gt": o.z, "x_gt": o.x, "z": p["z"],
                "x": p.get("x"), "h_pred": p["box"][3] - p["box"][1], "h_gt": o.box_height,
                "H_gt": o.height_m, "fy": fr.calib.fy, "occluded": o.occluded,
                "truncated": o.truncated,
            })
    return rows


def _err_stats(z_gt: np.ndarray, z: np.ndarray) -> dict:
    if len(z_gt) == 0:
        return {"n": 0, "mae_m": None, "rel_err": None, "median_rel_err": None}
    ae = np.abs(z - z_gt)
    return {"n": int(len(z_gt)), "mae_m": float(ae.mean()),
            "rel_err": float((ae / z_gt).mean()), "median_rel_err": float(np.median(ae / z_gt))}


def distance_metrics(rows: List[dict]) -> dict:
    if not rows:
        return {}
    z_gt = np.array([r["z_gt"] for r in rows])
    z = np.array([r["z"] for r in rows])
    out = {"overall": _err_stats(z_gt, z), "bins": {}, "occlusion": {}, "truncation": {},
           "class": {}}
    for (lo, hi), name in zip(DIST_BINS, BIN_NAMES):
        m = (z_gt >= lo) & (z_gt < hi)
        out["bins"][name] = _err_stats(z_gt[m], z[m])
    occ = np.array([r["occluded"] for r in rows])
    for lvl, name in [(0, "visible"), (1, "partly"), (2, "largely"), (3, "unknown")]:
        out["occlusion"][name] = _err_stats(z_gt[occ == lvl], z[occ == lvl])
    tr = np.array([r["truncated"] for r in rows])
    out["truncation"]["<=0.15"] = _err_stats(z_gt[tr <= 0.15], z[tr <= 0.15])
    out["truncation"][">0.15"] = _err_stats(z_gt[tr > 0.15], z[tr > 0.15])
    cls = np.array([r["cls"] for r in rows])
    for c in sorted(set(cls)):
        out["class"][c] = _err_stats(z_gt[cls == c], z[cls == c])
    return out


def _confusion_stats(pairs: Iterable[Tuple[int, int]]) -> dict:
    cm = np.zeros((3, 3), dtype=int)  # [gt, pred]
    for g, p in pairs:
        cm[g, p] += 1
    f1s, per = [], {}
    for k in range(3):
        tp = cm[k, k]
        prec = tp / max(cm[:, k].sum(), 1)
        rec = tp / max(cm[k, :].sum(), 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        f1s.append(f1)
        per[LEVEL_NAMES[k]] = {"precision": float(prec), "recall": float(rec), "f1": float(f1),
                               "support": int(cm[k, :].sum())}
    return {"confusion": cm.tolist(), "macro_f1": float(np.mean(f1s)),
            "danger_recall": per["DANGER"]["recall"],
            "danger_precision": per["DANGER"]["precision"], "per_class": per}


def risk_metrics(frames: Dict[str, Frame], preds: Dict[str, dict], key: str) -> dict:
    """key: 'risk_v2' 또는 'risk_geo'. 객체 단위·프레임 단위 혼동행렬."""
    obj_pairs, frame_pairs = [], []
    for fid, fr in frames.items():
        objs = preds[fid]["objects"]
        pairs, missed, fps = match_any(fr, objs)
        obj_pairs += [(object_level(o), p[key]) for p, o in pairs]
        obj_pairs += [(object_level(o), SAFE) for o in missed]
        obj_pairs += [(SAFE, p[key]) for p in fps]
        gt_f = max((object_level(o) for o in risk_objects(fr.objects)), default=SAFE)
        pr_f = max((p[key] for p in objs), default=SAFE)
        frame_pairs.append((gt_f, pr_f))
    return {"object": _confusion_stats(obj_pairs), "frame": _confusion_stats(frame_pairs)}


def speed_metrics(preds: Dict[str, dict]) -> dict:
    t = defaultdict(list)
    for p in preds.values():
        for k, v in p.get("timings", {}).items():
            t[k].append(v)
    out = {f"{k}_ms": float(np.mean(v) * 1000) for k, v in t.items()}
    if "total" in t:
        # 첫 10프레임은 워밍업으로 제외
        tot = np.array(t["total"][10:] or t["total"])
        out["fps"] = float(1.0 / tot.mean())
    return out


def evaluate(frames: Dict[str, Frame], preds: Dict[str, dict]) -> Tuple[dict, List[dict]]:
    rows = distance_pairs(frames, preds)
    has_dist = any(o.get("z") is not None for p in preds.values() for o in p["objects"])
    return {
        "n_frames": len(frames),
        "detection": detection_metrics(frames, preds),
        "distance": distance_metrics(rows) if has_dist else None,
        "risk_v2": risk_metrics(frames, preds, "risk_v2"),
        "risk_geo": risk_metrics(frames, preds, "risk_geo") if has_dist else None,
        "speed": speed_metrics(preds),
    }, rows
