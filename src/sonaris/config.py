"""Config loader.

One file, ``configs/default.yaml``. If a number appears in code, it is a bug (layout §4).
Configs are resolved relative to the repo root in dev/editable installs, overridable with
the ``SONARIS_CONFIG_DIR`` environment variable.
"""
from __future__ import annotations

import os
from functools import cache
from pathlib import Path

import yaml


def _repo_root() -> Path:
    # src/sonaris/config.py -> parents[2] == repo root (dev / editable install).
    here = Path(__file__).resolve()
    for base in (here.parents[2], Path.cwd()):
        if (base / "configs" / "default.yaml").exists():
            return base
    return here.parents[2]


def config_dir() -> Path:
    override = os.environ.get("SONARIS_CONFIG_DIR")
    return Path(override) if override else _repo_root() / "configs"


@cache
def load_config(name: str = "default") -> dict:
    """Load and cache ``configs/{name}.yaml`` as a plain dict."""
    path = config_dir() / f"{name}.yaml"
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_classes() -> dict:
    return load_config("classes")


def load_datasets() -> dict:
    return load_config("datasets")
