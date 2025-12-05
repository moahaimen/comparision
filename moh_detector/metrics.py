"""Metrics and post-processing utilities for MOH-DETECTOR."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw
from skimage import measure

BBox = Tuple[float, float, float, float]


def heatmap_to_boxes(heatmap: np.ndarray, threshold: float = 0.4, min_area: int = 4) -> List[BBox]:
    """Convert a predicted heatmap to bounding boxes via thresholding."""
    mask = heatmap > threshold
    labeled = measure.label(mask, connectivity=2)
    props = measure.regionprops(labeled)
    boxes: List[BBox] = []
    h, w = heatmap.shape
    for prop in props:
        minr, minc, maxr, maxc = prop.bbox
        area = (maxr - minr) * (maxc - minc)
        if area < min_area:
            continue
        boxes.append((float(minc), float(minr), float(maxc), float(maxr)))
    return boxes


def iou(box_a: BBox, box_b: BBox) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area + 1e-6
    return inter_area / union


@dataclass
class DetectionResult:
    boxes: List[BBox]
    scores: List[float]


def match_detections(pred_boxes: Sequence[BBox], gt_boxes: Sequence[BBox], iou_thr: float = 0.5) -> Tuple[int, int, int]:
    """Return true positives, false positives, false negatives based on IoU matching."""
    matched_gt = set()
    tp = 0
    for p_box in pred_boxes:
        best_iou = 0.0
        best_idx = -1
        for i, g_box in enumerate(gt_boxes):
            if i in matched_gt:
                continue
            score = iou(p_box, g_box)
            if score > best_iou:
                best_iou = score
                best_idx = i
        if best_iou >= iou_thr and best_idx >= 0:
            matched_gt.add(best_idx)
            tp += 1
    fp = max(0, len(pred_boxes) - tp)
    fn = max(0, len(gt_boxes) - tp)
    return tp, fp, fn


def precision_recall_ap(pred_boxes: Sequence[Sequence[BBox]], gt_boxes: Sequence[Sequence[BBox]], iou_thr: float = 0.5) -> Tuple[float, float, float]:
    """Compute precision, recall, and a simple AP@0.5 metric across a dataset."""
    all_scores = []
    all_matches = []
    for preds, gts in zip(pred_boxes, gt_boxes):
        tp, fp, fn = match_detections(preds, gts, iou_thr=iou_thr)
        score = 1.0 if preds else 0.0
        all_scores.append(score)
        all_matches.append((tp, fp, fn))
    tp_sum = sum(m[0] for m in all_matches)
    fp_sum = sum(m[1] for m in all_matches)
    fn_sum = sum(m[2] for m in all_matches)
    precision = tp_sum / (tp_sum + fp_sum + 1e-6)
    recall = tp_sum / (tp_sum + fn_sum + 1e-6)
    ap = precision * recall  # placeholder simplified AP
    return precision, recall, ap


def draw_boxes(img: Image.Image, boxes: Sequence[BBox]) -> Image.Image:
    draw = ImageDraw.Draw(img)
    for x1, y1, x2, y2 in boxes:
        draw.rectangle((x1, y1, x2, y2), outline="lime", width=2)
    return img


__all__ = [
    "heatmap_to_boxes",
    "iou",
    "DetectionResult",
    "match_detections",
    "precision_recall_ap",
    "draw_boxes",
]
