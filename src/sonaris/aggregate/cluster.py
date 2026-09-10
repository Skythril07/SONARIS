"""World-space aggregation of georeferenced detections (plan gate P2.1; layout section 5.1).

Georeference first, then cluster: detections whose WGS84 positions fall within
``aggregate.cluster_radius_m`` of one another are the SAME physical object seen across overlapping
tiles, adjacent pings, or both channels. Collapsing them removes tile-seam duplicates, and the size
of each cluster becomes the survivor's ``persistence_count`` -- an object seen many times is more
trustworthy than a one-tile flash (this is where dedup *and* persistence both come from, plan
override #7).

Clustering is single-linkage over geodesic distance (union-find): exact and cheap at the tens-to-
hundreds of detections a survey produces. The distance/cluster math is pure and unit-tested; only
``aggregate_detections`` touches the Detection contract, and it never invents a new coordinate --
the survivor keeps its own ping-traceable lat/lon.
"""
from __future__ import annotations

from pyproj import Geod

from ..report.schema import Detection

# fwd/inv take lon-before-lat; inv returns (fwd_az, back_az, distance_m).
_GEOD = Geod(ellps="WGS84")


def geodesic_distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Geodesic distance in metres between two WGS84 points."""
    return float(_GEOD.inv(lon1, lat1, lon2, lat2)[2])


def cluster_indices(points: list[tuple[float, float]], radius_m: float) -> list[list[int]]:
    """Single-linkage clusters of ``(lat, lon)`` points within ``radius_m`` (geodesic).

    Returns one list of indices per cluster (singletons included); every index appears exactly
    once, and indices within a cluster are ascending.
    """
    n = len(points)
    parent = list(range(n))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]  # path halving
            a = parent[a]
        return a

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(n):
        lat_i, lon_i = points[i]
        for j in range(i + 1, n):
            lat_j, lon_j = points[j]
            if geodesic_distance_m(lat_i, lon_i, lat_j, lon_j) <= radius_m:
                union(i, j)

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return [sorted(g) for g in groups.values()]


def cluster_spread_m(points: list[tuple[float, float]]) -> float:
    """Maximum pairwise geodesic distance within a set of points (0.0 for fewer than two)."""
    m = 0.0
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            m = max(m, geodesic_distance_m(*points[i], *points[j]))
    return m


def aggregate_detections(
    dets: list[Detection], radius_m: float, alpha: float = 0.9
) -> list[Detection]:
    """Collapse near-coincident detections into one survivor per physical object.

    The survivor is the highest-confidence cluster member (its exact lat/lon is kept -- clustering
    never fabricates a coordinate). ``persistence_count`` becomes the cluster size; ``evidence``
    records the cluster spread and the merged ids; and a ranking ``salience`` blends confidence
    with persistence via ``alpha`` (config ``aggregate.alpha``). Output is sorted salience-first so
    the operator triages the strongest contacts at the top.
    """
    if not dets:
        return []
    points = [(d.lat, d.lon) for d in dets]
    out: list[Detection] = []
    for group in cluster_indices(points, radius_m):
        members = [dets[i] for i in group]
        rep = max(members, key=lambda d: d.confidence)
        rep.persistence_count = len(members)
        spread = cluster_spread_m([(m.lat, m.lon) for m in members])
        persistence_factor = 1.0 - 1.0 / len(members)  # 0 for a singleton, -> 1 as it grows
        rep.evidence = {
            **rep.evidence,
            "cluster_size": len(members),
            "cluster_spread_m": round(spread, 3),
            "merged_ids": [m.detection_id for m in members if m.detection_id != rep.detection_id],
            "salience": round(alpha * rep.confidence + (1.0 - alpha) * persistence_factor, 4),
        }
        out.append(rep)
    out.sort(key=lambda d: d.evidence.get("salience", d.confidence), reverse=True)
    return out
