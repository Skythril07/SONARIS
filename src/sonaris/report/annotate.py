"""Annotated waterfall PNG (plan gate P1.8; layout section 5).

Draws detection boxes on the ground-range waterfall so an operator can eyeball a pin against the
sonar return -- the manual-confirmation step in the Phase 1 definition of done. Dumb and total:
it takes an 8-bit image and boxes already in image coordinates and draws; it computes nothing.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def annotate_waterfall(image_u8: np.ndarray, boxes: list[tuple[int, int, int, int, str]], path) -> Path:
    """Draw ``(x1, y1, x2, y2, label)`` boxes on a single-channel 8-bit image; write a PNG."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas = cv2.cvtColor(np.ascontiguousarray(image_u8, dtype=np.uint8), cv2.COLOR_GRAY2BGR)
    for x1, y1, x2, y2, label in boxes:
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 165, 255), 1)   # amber, BGR
        if label:
            cv2.putText(canvas, label, (x1, max(0, y1 - 2)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 165, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(path), canvas)
    return path
