"""Put the module's backend/python on sys.path for the tests."""

import sys
from pathlib import Path

_BACKEND_PY = Path(__file__).resolve().parents[1] / "backend" / "python"
if str(_BACKEND_PY) not in sys.path:
    sys.path.insert(0, str(_BACKEND_PY))
