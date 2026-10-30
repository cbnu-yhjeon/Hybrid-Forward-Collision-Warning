"""Harris + SIFT + HoG 특징 (v2 Phase 2). 위험도 feature_score(10%)에 쓰입니다."""
from __future__ import annotations

import cv2
import numpy as np

from hfcw.types import Detection

_sift = None
_hog = None


def _roi(gray: np.ndarray, det: Detection) -> np.ndarray:
    h, w = gray.shape[:2]
    x1, y1 = max(det.x, 0), max(det.y, 0)
    x2, y2 = min(det.x + det.w, w), min(det.y + det.h, h)
    return gray[y1:y2, x1:x2]


def extract_harris(gray: np.ndarray, dets: list[Detection]) -> list[int]:
    counts = []
    for det in dets:
        roi = _roi(gray, det)
        if roi.size < 9:
            counts.append(0)
            continue
        r = cv2.cornerHarris(np.float32(roi), blockSize=2, ksize=3, k=0.04)
        counts.append(int(np.sum(r > 0.01 * r.max())) if r.max() > 0 else 0)
    return counts


def extract_sift(gray: np.ndarray, dets: list[Detection]) -> list[int]:
    global _sift
    if _sift is None:
        _sift = cv2.SIFT_create()
    counts = []
    for det in dets:
        roi = _roi(gray, det)
        counts.append(len(_sift.detect(roi, None)) if roi.size else 0)
    return counts


def extract_hog(gray: np.ndarray, dets: list[Detection]) -> list[float]:
    global _hog
    if _hog is None:
        _hog = cv2.HOGDescriptor((64, 64), (16, 16), (8, 8), (8, 8), 9)
    scores = []
    for det in dets:
        roi = _roi(gray, det)
        if roi.size < 64 * 64:
            scores.append(0.0)
            continue
        desc = _hog.compute(cv2.resize(roi, (64, 64)))
        scores.append(float(np.mean(np.abs(desc))) if desc is not None else 0.0)
    return scores
