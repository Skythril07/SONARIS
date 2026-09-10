"""Shadow-consistency check (plan gate P2.2; layout section 5.3).

On a ground-range-corrected waterfall a real raised object returns a bright highlight and casts a
dark acoustic shadow on its FAR side from the sonar -- i.e. at greater ground range, which is a
HIGHER column index (column 0 is nadir; see ``geometry/correct.py``). A bright blob with no shadow
beyond it is more likely speckle or a flat reflective patch than a true raised target, so a
darker-than-highlight region immediately past the detection is corroborating evidence.

Deliberately a boolean: shadow-derived *height* (plan override #5) stacks fragile assumptions and
is a separate later gate (P2.4). The ratio test here is pure and unit-tested.
"""
from __future__ import annotations

import numpy as np


def shadow_consistency(
    image: np.ndarray,
    box: tuple[int, int, int, int],
    *,
    shadow_len_frac: float = 1.0,
    min_drop_frac: float = 0.5,
    min_shadow_px: int = 4,
) -> tuple[bool, dict]:
    """Is there an acoustic shadow just beyond ``box`` (towards greater ground range = higher col)?

    ``box`` is ``(col_start, row_start, col_end, row_end)`` in corrected-image pixels. The mean
    intensity of the detection highlight is compared with a window of comparable width immediately
    at higher columns (the shadow zone). A relative drop of at least ``min_drop_frac`` reads as a
    consistent shadow. Returns ``(consistent, evidence)``.
    """
    c0, r0, c1, r1 = (int(v) for v in box)
    h, w = image.shape[:2]
    r0 = max(0, min(r0, h - 1))
    r1 = max(r0 + 1, min(r1, h))
    c0 = max(0, min(c0, w - 1))
    c1 = max(c0 + 1, min(c1, w))

    box_w = c1 - c0
    shadow_w = max(min_shadow_px, round(box_w * shadow_len_frac))
    sc0, sc1 = c1, min(w, c1 + shadow_w)
    if sc1 - sc0 < min_shadow_px:
        return False, {"reason": "no room for shadow beyond object"}

    obj = image[r0:r1, c0:c1].astype(float)
    shadow = image[r0:r1, sc0:sc1].astype(float)
    obj_mean = float(obj.mean()) if obj.size else 0.0
    shadow_mean = float(shadow.mean()) if shadow.size else 0.0
    drop = 1.0 - (shadow_mean / obj_mean) if obj_mean > 0 else 0.0
    consistent = drop >= min_drop_frac
    return consistent, {
        "obj_mean": round(obj_mean, 2),
        "shadow_mean": round(shadow_mean, 2),
        "shadow_drop_frac": round(drop, 3),
    }
