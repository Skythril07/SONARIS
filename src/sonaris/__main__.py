"""Enable ``python -m sonaris ...`` (used by the web backend to spawn ``run`` as a subprocess)."""
from __future__ import annotations

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
