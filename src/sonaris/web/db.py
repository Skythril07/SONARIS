"""SQLite persistence for the operator console (plan gate P3.1): ``jobs`` + ``detections``.

Async (aiosqlite) so it never blocks the event loop. Rows mirror the Detection contract plus the
operator ``review_status`` written back via ``PATCH /api/detections/{id}`` (P3.8). The DB is the
system of record the UI reads; the pipeline's report files remain the ground truth on disk.
"""
from __future__ import annotations

import json
from pathlib import Path

import aiosqlite

from ..report.schema import Detection

DB_PATH = Path("data/outputs/sonaris.db")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id          TEXT PRIMARY KEY,
    filename    TEXT NOT NULL,
    status      TEXT NOT NULL,          -- queued | running | done | error
    stage       TEXT,                   -- last stage seen from progress.jsonl
    created_at  REAL NOT NULL,
    out_dir     TEXT NOT NULL,
    detections  INTEGER NOT NULL DEFAULT 0,
    error       TEXT
);
CREATE TABLE IF NOT EXISTS detections (
    detection_id           TEXT PRIMARY KEY,
    job_id                 TEXT NOT NULL,
    cls                    TEXT,
    confidence             REAL,
    confidence_raw         REAL,
    lat                    REAL,
    lon                    REAL,
    position_uncertainty_m REAL,
    length_m               REAL,
    width_m                REAL,
    ground_range_m         REAL,
    channel                TEXT,
    ping_index             INTEGER,
    persistence_count      INTEGER,
    shadow_consistent      INTEGER,     -- 0/1/NULL
    metric_plausible       INTEGER,     -- 0/1/NULL
    review_status          TEXT,
    evidence               TEXT,        -- JSON blob
    FOREIGN KEY (job_id) REFERENCES jobs(id)
);
CREATE INDEX IF NOT EXISTS idx_det_job ON detections(job_id);
"""


async def init_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    async with aiosqlite.connect(DB_PATH) as db:
        await db.executescript(_SCHEMA)
        await db.commit()


async def create_job(job_id: str, filename: str, out_dir: str, created_at: float) -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT INTO jobs (id, filename, status, stage, created_at, out_dir) "
            "VALUES (?, ?, 'queued', NULL, ?, ?)",
            (job_id, filename, created_at, out_dir),
        )
        await db.commit()


async def set_job(job_id: str, **fields) -> None:
    if not fields:
        return
    cols = ", ".join(f"{k} = ?" for k in fields)
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(f"UPDATE jobs SET {cols} WHERE id = ?", (*fields.values(), job_id))
        await db.commit()


def _row_to_dict(cursor, row) -> dict:
    return {d[0]: row[i] for i, d in enumerate(cursor.description)}


async def get_job(job_id: str) -> dict | None:
    async with aiosqlite.connect(DB_PATH) as db:
        cur = await db.execute("SELECT * FROM jobs WHERE id = ?", (job_id,))
        row = await cur.fetchone()
        return _row_to_dict(cur, row) if row else None


async def list_jobs(limit: int = 50) -> list[dict]:
    async with aiosqlite.connect(DB_PATH) as db:
        cur = await db.execute("SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,))
        rows = await cur.fetchall()
        return [_row_to_dict(cur, r) for r in rows]


def _bool_to_int(v) -> int | None:
    return None if v is None else int(bool(v))


async def insert_detections(job_id: str, dets: list[Detection]) -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.executemany(
            "INSERT OR REPLACE INTO detections (detection_id, job_id, cls, confidence, "
            "confidence_raw, lat, lon, position_uncertainty_m, length_m, width_m, ground_range_m, "
            "channel, ping_index, persistence_count, shadow_consistent, metric_plausible, "
            "review_status, evidence) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            [
                (
                    d.detection_id, job_id, d.cls, d.confidence, d.confidence_raw, d.lat, d.lon,
                    d.position_uncertainty_m, d.length_m, d.width_m, d.ground_range_m, d.channel,
                    d.ping_index, d.persistence_count, _bool_to_int(d.shadow_consistent),
                    _bool_to_int(d.metric_plausible), d.review_status, json.dumps(d.evidence),
                )
                for d in dets
            ],
        )
        await db.commit()


async def list_detections(job_id: str) -> list[dict]:
    async with aiosqlite.connect(DB_PATH) as db:
        cur = await db.execute(
            "SELECT * FROM detections WHERE job_id = ? ORDER BY confidence DESC", (job_id,)
        )
        rows = await cur.fetchall()
        out = []
        for r in rows:
            rec = _row_to_dict(cur, r)
            rec["evidence"] = json.loads(rec["evidence"]) if rec["evidence"] else {}
            for k in ("shadow_consistent", "metric_plausible"):
                rec[k] = None if rec[k] is None else bool(rec[k])
            out.append(rec)
        return out


async def update_review(detection_id: str, review_status: str) -> bool:
    async with aiosqlite.connect(DB_PATH) as db:
        cur = await db.execute(
            "UPDATE detections SET review_status = ? WHERE detection_id = ?",
            (review_status, detection_id),
        )
        await db.commit()
        return cur.rowcount > 0
