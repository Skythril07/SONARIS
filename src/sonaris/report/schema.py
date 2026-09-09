"""The deliverable record (layout §2.5).

``report/writers.py`` turns a ``list[Detection]`` into GeoJSON / CSV / GPX. Keep the writers
dumb and total. Phase-2 signals have safe defaults so MVP writers never hit a missing field.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Detection:
    detection_id: str
    line_id: str
    cls: str
    confidence: float                    # calibrated (== confidence_raw until P2.5)
    confidence_raw: float
    lat: float
    lon: float
    position_uncertainty_m: float
    length_m: float
    width_m: float
    ground_range_m: float
    channel: str
    ping_index: int

    # --- populated by Phase 2 upgrades; defaults keep MVP writers total ---
    height_est_m: float | None = None       # from shadow length (P2.4)
    height_source: str | None = None        # 'shadow' | None
    persistence_count: int = 1              # world-space cluster size (P2.1)
    shadow_consistent: bool | None = None   # P2.2
    metric_plausible: bool | None = None    # P2.3
    evidence: dict = field(default_factory=dict)   # per-signal scores for the UI panel
    priority: str = "unranked"
    review_status: str = "unreviewed"
