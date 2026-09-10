"""FastAPI operator console backend (plan gates P3.1, P3.2, P3.4, P3.5, P3.8, P3.9).

A thin front-end onto ``sonaris run``: an upload creates a job, the pipeline runs as a subprocess
writing ``progress.jsonl``, and an SSE stream tails that file to drive the 9-stage rail. When the
run finishes, detections are loaded from ``detections.geojson`` into SQLite and served to the map /
table / evidence panel. Operator verdicts are written back via ``PATCH /api/detections/{id}``.

fastapi/uvicorn are only imported here (the ``app`` extra), never from the core package.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ..report.schema import Detection
from . import db

REPO_ROOT = Path(__file__).resolve().parents[3]   # src/sonaris/web/app.py -> repo root
JOBS_DIR = REPO_ROOT / "data" / "outputs" / "jobs"
FRONTEND_DIST = REPO_ROOT / "sonaris-demo-frontend" / "dist"

REVIEW_VALUES = {"CONFIRMED", "REJECTED", "NEEDS_REVIEW", "UNREVIEWED"}


@asynccontextmanager
async def lifespan(_: FastAPI):
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    await db.init_db()
    yield


app = FastAPI(title="SONARIS operator console", version="0.1", lifespan=lifespan)
# Dev convenience: the Vite dev server (5173) calls this API (8000) cross-origin.
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)


def _load_detections(geojson_path: Path) -> list[Detection]:
    """Rebuild Detection objects from a written detections.geojson (properties + geometry)."""
    data = json.loads(geojson_path.read_text(encoding="utf-8"))
    out: list[Detection] = []
    for feat in data.get("features", []):
        props = dict(feat.get("properties", {}))
        lon, lat = feat["geometry"]["coordinates"]
        props["lat"], props["lon"] = lat, lon
        out.append(Detection(**props))
    return out


async def _run_job(job_id: str, input_path: Path, jobdir: Path) -> None:
    """Run ``sonaris run`` as a subprocess, then load its detections into the DB."""
    progress = jobdir / "progress.jsonl"
    await db.set_job(job_id, status="running")
    cmd = [
        sys.executable, "-m", "sonaris", "run",
        "--input", str(input_path), "--out", str(jobdir),
        "--progress", str(progress), "--verify",
    ]
    # Ensure the subprocess can import sonaris even if the editable install is unavailable.
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")}
    proc = await asyncio.create_subprocess_exec(
        *cmd, cwd=str(REPO_ROOT), env=env,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
    )
    out, _ = await proc.communicate()

    if proc.returncode != 0:
        tail = out.decode(errors="ignore")[-2000:] if out else "sonaris run failed"
        await db.set_job(job_id, status="error", error=tail)
        with progress.open("a", encoding="utf-8") as f:      # unblock any SSE listener
            f.write(json.dumps({"stage": "ERROR", "index": -1}) + "\n")
        return

    geojson = jobdir / "detections.geojson"
    dets = _load_detections(geojson) if geojson.exists() else []
    await db.insert_detections(job_id, dets)
    await db.set_job(job_id, status="done", stage="DONE", detections=len(dets))


@app.post("/api/jobs")
async def create_job(file: UploadFile):
    """Accept a sonar file, start the pipeline, return the new job id (P3.1)."""
    job_id = uuid.uuid4().hex[:12]
    jobdir = JOBS_DIR / job_id
    jobdir.mkdir(parents=True, exist_ok=True)
    saved = jobdir / (file.filename or "upload.xtf")
    saved.write_bytes(await file.read())
    await db.create_job(job_id, saved.name, str(jobdir), time.time())
    asyncio.create_task(_run_job(job_id, saved, jobdir))
    return {"id": job_id, "filename": saved.name, "status": "queued"}


@app.get("/api/jobs")
async def get_jobs():
    return await db.list_jobs()


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str):
    job = await db.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    return job


@app.get("/api/jobs/{job_id}/events")
async def job_events(job_id: str, request: Request):
    """SSE: stream each progress.jsonl stage, then a final 'complete' event (P3.2/P3.4)."""
    job = await db.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    progress = Path(job["out_dir"]) / "progress.jsonl"

    async def gen():
        sent = 0
        while True:
            if await request.is_disconnected():
                return
            if progress.exists():
                lines = progress.read_text(encoding="utf-8", errors="ignore").splitlines()
                for ln in lines[sent:]:
                    if ln.strip():
                        yield f"event: stage\ndata: {ln}\n\n"
                sent = len(lines)
            current = await db.get_job(job_id)
            if current and current["status"] in ("done", "error"):
                payload = json.dumps({
                    "status": current["status"],
                    "detections": current["detections"],
                    "error": current["error"],
                })
                yield f"event: complete\ndata: {payload}\n\n"
                return
            await asyncio.sleep(0.4)

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.get("/api/jobs/{job_id}/detections")
async def get_detections(job_id: str):
    return await db.list_detections(job_id)


@app.get("/api/jobs/{job_id}/track")
async def get_track(job_id: str):
    job = await db.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    track = Path(job["out_dir"]) / "track.geojson"
    if not track.exists():
        raise HTTPException(status_code=404, detail="track not available yet")
    return FileResponse(track, media_type="application/geo+json")


@app.get("/api/jobs/{job_id}/report/{fmt}")
async def get_report(job_id: str, fmt: str):
    if fmt not in ("geojson", "csv", "gpx"):
        raise HTTPException(status_code=400, detail="format must be geojson, csv, or gpx")
    job = await db.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    path = Path(job["out_dir"]) / f"detections.{fmt}"
    if not path.exists():
        raise HTTPException(status_code=404, detail="report not available yet")
    return FileResponse(path, filename=f"sonaris-{job_id}.{fmt}")


class ReviewUpdate(BaseModel):
    review_status: str


@app.patch("/api/detections/{detection_id}")
async def patch_detection(detection_id: str, body: ReviewUpdate):
    """Operator verdict: Confirm / Reject / Needs Review (P3.8)."""
    status = body.review_status.upper()
    if status not in REVIEW_VALUES:
        raise HTTPException(status_code=400, detail=f"review_status must be one of {REVIEW_VALUES}")
    if not await db.update_review(detection_id, status):
        raise HTTPException(status_code=404, detail="detection not found")
    return {"detection_id": detection_id, "review_status": status}


@app.get("/api/health")
async def health():
    return {"ok": True, "frontend_built": FRONTEND_DIST.exists()}


# Serve the built React app at / when present (after API routes, so /api/* still wins).
if FRONTEND_DIST.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIST), html=True), name="frontend")
