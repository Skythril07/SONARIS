"""Confidence filtering + per-tile NMS (Stage 5 MVP; plan gate P1.7; layout section 5).

Cheap, world-agnostic dedup inside a tile: drop low-confidence detections, then greedily suppress
overlapping boxes. Cross-tile deduplication is deliberately NOT here -- doing it in pixel space
breaks at tile seams; it is a world-space job for P2.1. This module never re-reads raw files.
"""
from __future__ import annotations

from ..models.detector import RawDetection


def filter_confidence(dets: list[RawDetection], conf_min: float) -> list[RawDetection]:
    """Keep detections at or above ``conf_min``."""
    return [d for d in dets if d.conf >= conf_min]


def _iou(a: RawDetection, b: RawDetection) -> float:
    ix1, iy1 = max(a.x1, b.x1), max(a.y1, b.y1)
    ix2, iy2 = min(a.x2, b.x2), min(a.y2, b.y2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = (a.x2 - a.x1) * (a.y2 - a.y1)
    area_b = (b.x2 - b.x1) * (b.y2 - b.y1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def nms(dets: list[RawDetection], iou_thresh: float) -> list[RawDetection]:
    """Greedy non-maximum suppression per tile: highest confidence wins, overlaps above
    ``iou_thresh`` (same tile, same class) are suppressed."""
    kept: list[RawDetection] = []
    for d in sorted(dets, key=lambda r: r.conf, reverse=True):
        if all(
            not (k.tile_id == d.tile_id and k.cls == d.cls and _iou(k, d) > iou_thresh)
            for k in kept
        ):
            kept.append(d)
    return kept
