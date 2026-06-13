    # pystorm_common

Shared, cross-module presentation helpers for PyStorm: the canonical Wave Maker
design palette, the clean-axes styling helper (`style_ax`), and the standard
figure writer (`save_figure`, which fixes the PyStorm figure DPI standard).

This is the CyHAN v2.2 shared **common library** (CyHAN-Standard-v2.2 §5.2 /
§16.10). It holds presentation and pure-utility code only: no module domain
logic, numerical kernels, or orchestration. A module may depend on it and still
run in isolation through its launcher, because `common/` is an integration-tier
dependency, not a sibling-module source dependency.

```python
from pystorm_common import WAVE_MAKER, INK, GRID, C, style_ax, save_figure
style_ax(ax)                       # despined, light grid, muted ticks
save_figure(fig, out_path)         # parent dirs created, standard DPI
```

Modules find this package by adding `common/python/` to `sys.path` in their
launcher (alongside `backend/python/`), or via an editable install
(`pip install -e common/`).

## Layout

```
common/
├── pyproject.toml
├── README.md
└── python/
    └── pystorm_common/
        ├── __init__.py     re-exports palette tokens + style_ax + save_figure
        ├── palette.py      Wave Maker design palette (tokens, ramp, cycle, cmaps)
        └── figure.py       style_ax + save_figure + DEFAULT_DPI
```
