"""고전 검출기: v1 (Contour) / v2 (Contour + Watershed + 이상치 제거 + HOG 보행자).

v2 detection.py · classifier.py 로직을 그대로 옮기고 Detector 인터페이스로 감쌌습니다.
고전 검출기는 신뢰도가 없으므로 score=1.0을 붙입니다 (AP는 단일 PR 점이 됩니다).
"""
from __future__ import annotations

import cv2
import numpy as np

from hfcw.detectors.base import Detector
from hfcw.pipeline.feature import extract_harris, extract_sift
from hfcw.pipeline.preprocessing import Preprocessed, preprocess
from hfcw.types import CAR, PEDESTRIAN, Detection, box_iou

MAX_DETS = 10


def _is_vehicle_like(x, y, w, h, img_w, img_h) -> bool:
    area, img_area = w * h, img_w * img_h
    if area < img_area * 0.001 or area > img_area * 0.15:
        return False
    if w > img_w * 0.7 or h < 15:
        return False
    if y + h < img_h * 0.30:  # 상단 30% 하늘
        return False
    return True


def _det(x, y, w, h, area, contour=None, cls=CAR) -> Detection:
    return Detection(float(x), float(y), float(x + w), float(y + h), 1.0, cls,
                     contour=contour, contour_area=float(area))


def detect_contours(edges: np.ndarray, img_shape: tuple) -> list[Detection]:
    """Canny → Morphology → Contour → 차량 크기 필터"""
    img_h, img_w = img_shape[:2]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    dilated = cv2.dilate(closed, kernel, iterations=1)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    dets = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if _is_vehicle_like(x, y, w, h, img_w, img_h):
            dets.append(_det(x, y, w, h, cv2.contourArea(cnt), cnt))
    dets.sort(key=lambda d: d.area, reverse=True)
    return dets[:MAX_DETS]


def segment_road_objects(gray: np.ndarray, img_shape: tuple) -> list[Detection]:
    """Otsu + ConnectedComponents 보조 검출"""
    img_h, img_w = img_shape[:2]
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    binary[: int(img_h * 0.30), :] = 0
    n, _, stats, _ = cv2.connectedComponentsWithStats(binary)
    dets = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if _is_vehicle_like(x, y, w, h, img_w, img_h):
            dets.append(_det(x, y, w, h, area))
    dets.sort(key=lambda d: d.area, reverse=True)
    return dets[:MAX_DETS]


def filter_statistical_outliers(dets: list[Detection]) -> list[Detection]:
    """면적 중앙값 대비 20배 초과 / 5% 미만 박스 제거"""
    if len(dets) < 3:
        return dets
    med = float(np.median([d.area for d in dets]))
    kept = [d for d in dets if med * 0.05 <= d.area <= med * 20]
    return kept or dets


def apply_watershed(gray: np.ndarray, dets: list[Detection], img_shape: tuple) -> list[Detection]:
    """IoU > 0.4로 겹친 박스 쌍을 Watershed로 분리. 실패 시 원본 유지."""
    if len(dets) < 2:
        return dets
    img_h, img_w = img_shape[:2]

    def iou(a: Detection, b: Detection) -> float:
        # v2와 같이 컨투어 면적을 분모로 사용
        x1, y1 = max(a.x, b.x), max(a.y, b.y)
        x2, y2 = min(a.x + a.w, b.x + b.w), min(a.y + a.h, b.y + b.h)
        if x2 <= x1 or y2 <= y1:
            return 0.0
        inter = (x2 - x1) * (y2 - y1)
        return inter / (a.area + b.area - inter + 1e-6)

    processed, new_dets = set(), []
    kernel = np.ones((3, 3), np.uint8)
    for i in range(len(dets)):
        for j in range(i + 1, len(dets)):
            if iou(dets[i], dets[j]) <= 0.4:
                continue
            d1, d2 = dets[i], dets[j]
            xmin, ymin = max(min(d1.x, d2.x), 0), max(min(d1.y, d2.y), 0)
            xmax = min(max(d1.x + d1.w, d2.x + d2.w), img_w)
            ymax = min(max(d1.y + d1.h, d2.y + d2.h), img_h)
            roi = gray[ymin:ymax, xmin:xmax]
            if roi.size < 200:
                continue
            try:
                _, fg = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                sure_bg = cv2.dilate(fg, kernel, iterations=3)
                dist = cv2.distanceTransform(fg, cv2.DIST_L2, 5)
                _, sure_fg = cv2.threshold(dist, 0.5 * dist.max(), 255, 0)
                sure_fg = np.uint8(sure_fg)
                unknown = cv2.subtract(sure_bg, sure_fg)
                _, markers = cv2.connectedComponents(sure_fg)
                markers += 1
                markers[unknown == 255] = 0
                cv2.watershed(cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR), markers)
                for lbl in range(2, markers.max() + 1):
                    mask = np.uint8(markers == lbl) * 255
                    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    for cnt in cnts:
                        a = cv2.contourArea(cnt)
                        if a < 200:
                            continue
                        nx, ny, nw, nh = cv2.boundingRect(cnt)
                        nx, ny = nx + xmin, ny + ymin
                        if _is_vehicle_like(nx, ny, nw, nh, img_w, img_h):
                            new_dets.append(_det(nx, ny, nw, nh, a, cnt))
                processed.update((i, j))
            except cv2.error:
                pass
    final = [d for k, d in enumerate(dets) if k not in processed] + new_dets
    final.sort(key=lambda d: d.area, reverse=True)
    return final[:MAX_DETS] if final else dets


# ── HOG 보행자 (v2 classifier.py) ───────────────────────────────────────────
_hog_people = None


def _nms(rects: np.ndarray, weights: np.ndarray, thresh: float = 0.4) -> list[int]:
    boxes = np.column_stack([rects[:, 0], rects[:, 1],
                             rects[:, 0] + rects[:, 2], rects[:, 1] + rects[:, 3]])
    order, keep = list(np.argsort(-weights)), []
    while order:
        i = order.pop(0)
        keep.append(i)
        if order:
            ious = box_iou(boxes[i:i + 1], boxes[order])[0]
            order = [o for o, v in zip(order, ious) if v <= thresh]
    return keep


def detect_pedestrians(img: np.ndarray) -> list[Detection]:
    global _hog_people
    if _hog_people is None:
        _hog_people = cv2.HOGDescriptor()
        _hog_people.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    img_h, img_w = img.shape[:2]
    scale = 0.5
    small = cv2.resize(img, None, fx=scale, fy=scale)
    try:
        rects, weights = _hog_people.detectMultiScale(small, winStride=(8, 8), padding=(4, 4),
                                                      scale=1.05, finalThreshold=2.0)
    except cv2.error:
        return []
    if len(rects) == 0:
        return []
    rects = np.asarray(rects, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64).flatten()
    if len(weights) != len(rects):
        weights = np.ones(len(rects))
    dets = []
    for k in _nms(rects, weights):
        x, y, w, h = (rects[k] / scale).astype(int)
        if y + h < img_h * 0.30 or w * h < img_h * img_w * 0.001:
            continue
        dets.append(Detection(float(x), float(y), float(x + w), float(y + h), 1.0, PEDESTRIAN))
    return dets[:5]


def classify_object(det: Detection, harris: int, sift: int) -> str:
    """종횡비 + 특징점 휴리스틱 (v2). 세로로 길고 특징점 적으면 보행자."""
    if det.cls == PEDESTRIAN:
        return PEDESTRIAN
    if det.h / max(det.w, 1) > 1.4 and harris + sift < 150:
        return PEDESTRIAN
    return CAR


class ClassicalV1Detector(Detector):
    """v1 (Midterm): Contour + Otsu 보조 검출. 분류 없음 → 전부 Car."""

    name = "classical_v1"

    def detect(self, img, pre=None, frame=None):
        pre = pre or preprocess(img, use_clahe=False)
        dets = detect_contours(pre.edges, img.shape)
        return dets or segment_road_objects(pre.gray, img.shape)


class ClassicalV2Detector(Detector):
    """v2 (Final): Contour → Watershed → 이상치 제거 → HOG 보행자 병합 → 휴리스틱 분류."""

    name = "classical"

    def __init__(self, use_pedestrian_hog: bool = True):
        self.use_ped = use_pedestrian_hog

    def detect(self, img, pre: Preprocessed | None = None, frame=None):
        pre = pre or preprocess(img)
        dets = detect_contours(pre.edges, img.shape)
        if not dets:
            dets = segment_road_objects(pre.gray, img.shape)
        dets = apply_watershed(pre.gray, dets, img.shape)
        dets = filter_statistical_outliers(dets)
        if self.use_ped:
            for pd in detect_pedestrians(img):
                if not dets or box_iou(pd.box, np.stack([d.box for d in dets])).max() <= 0.3:
                    dets.append(pd)
            dets = sorted(dets, key=lambda d: d.area, reverse=True)[:MAX_DETS]
        if dets:
            harris = extract_harris(pre.gray, dets)
            sift = extract_sift(pre.gray, dets)
            for d, hc, sc in zip(dets, harris, sift):
                d.cls = classify_object(d, hc, sc)
        return dets
