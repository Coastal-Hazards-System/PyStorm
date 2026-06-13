"""Put the module backend/python and the shared common/python on sys.path for tests."""

import sys
from pathlib import Path

_here = Path(__file__).resolve()
for _p in (_here.parents[1] / "backend" / "python",     # module orchestration package
           _here.parents[3] / "common" / "python"):     # shared CyHAN common library (§5.2)
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
