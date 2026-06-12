"""pystorm_common - shared, cross-module helpers for PyStorm (CyHAN 5.2 common library).

Presentation and pure-utility helpers only (the design palette, axes styling,
and the figure writer). No module domain logic, numerical kernels, or
orchestration live here. A module may depend on this library and still run in
isolation through its launcher; it is an integration-tier dependency, not a
sibling-module source dependency.
"""

from .palette import *          # noqa: F401,F403  (design tokens + apply/band)
from .palette import __all__ as _palette_all
from .figure import DEFAULT_DPI, style_ax, save_figure

__all__ = list(_palette_all) + ["DEFAULT_DPI", "style_ax", "save_figure"]
