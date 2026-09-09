"""Make ``src/`` and the repo root importable so tests run without an editable install
(``import sonaris...`` and ``import tests.synthetic``)."""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
for p in (_ROOT / "src", _ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
