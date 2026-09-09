#!/usr/bin/env python3
"""G0 / Phase 0.3 — environment check.

Verifies Phase-1 dependencies import and that pyxtf exposes the structs we rely on.
    Run:  python scripts/verify_env.py
Exit code 0 = core environment ready; 1 = a required core dep or pyxtf is missing.
ML and app deps are reported but not required this early.
"""
from __future__ import annotations

import importlib
import sys

CORE = ["numpy", "pandas", "pyarrow", "scipy", "cv2",
        "pyxtf", "pyproj", "shapely", "gpxpy", "yaml"]
ML = ["torch", "ultralytics", "onnx", "onnxruntime"]
APP = ["fastapi", "uvicorn", "aiosqlite", "PIL"]


def check(mods: list[str], required: bool) -> bool:
    ok = True
    for m in mods:
        try:
            mod = importlib.import_module(m)
            ver = getattr(mod, "__version__", "")
            print(f"  [ OK ] {m:16s} {ver}")
        except Exception as e:  # noqa: BLE001 - report any import failure
            if required:
                ok = False
            print(f"  [{'MISS' if required else 'skip'}] {m:16s} ({e.__class__.__name__})")
    return ok


def check_pyxtf_structs() -> bool:
    try:
        import pyxtf
    except Exception as e:  # noqa: BLE001
        print(f"  [MISS] pyxtf structs ({e.__class__.__name__})")
        return False
    needed = ["XTFPingHeader", "XTFFileHeader", "XTFHeaderType"]
    missing = [n for n in needed if not hasattr(pyxtf, n)]
    if missing:
        print(f"  [WARN] pyxtf present but missing: {missing}")
        return False
    print("  [ OK ] pyxtf structs present")
    return True


def main() -> int:
    print(f"Python {sys.version.split()[0]}\n")
    print("Core (Phase 1, required):")
    core_ok = check(CORE, required=True)
    print("\npyxtf structs:")
    structs_ok = check_pyxtf_structs()
    print("\nML (Phase 1 detector - needed for train/infer):")
    check(ML, required=False)
    print("\nApp (Phase 3 - optional now):")
    check(APP, required=False)

    print()
    if core_ok and structs_ok:
        print("G0 PASS: core environment ready.")
        return 0
    print("G0 FAIL: install missing core deps  ->  pip install -e .")
    return 1


if __name__ == "__main__":
    sys.exit(main())
