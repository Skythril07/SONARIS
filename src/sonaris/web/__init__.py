"""Operator-console web backend (Phase 3): a thin FastAPI front-end onto ``sonaris run``.

Imports fastapi/uvicorn lazily via :mod:`sonaris.web.app`, so the core package never depends on the
``app`` extra. Launch with ``sonaris serve`` (or ``uvicorn sonaris.web.app:app``).
"""
