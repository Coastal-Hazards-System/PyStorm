"""pystorm_common.figure - shared matplotlib axes styling and figure saving.

The clean-axes styling (``style_ax``) and the figure writer (``save_figure``)
were previously duplicated per module; they live here as the single source of
truth (CyHAN 5.2). ``save_figure`` also fixes the PyStorm figure DPI standard so
diagnostic figures are consistent across modules.
"""
from __future__ import annotations

from pathlib import Path

from .palette import INK, GRID

# PyStorm figure DPI standard for on-disk diagnostic figures. One value across
# modules (previously 300 in POT, 150 in PST). 150 dpi is sharp for screen and
# print at the sizes these figures are used, without oversized files.
DEFAULT_DPI = 150


def style_ax(ax, *, spine: str = INK, grid: str = GRID,
             tick_color: str = "#333333", tick_labelsize: float = 10.0) -> None:
    """Apply the shared clean axes styling: despined top/right, colored
    left/bottom spines, a light grid behind the data, and muted ticks."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(spine)
    ax.grid(True, color=grid, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=tick_color, labelsize=tick_labelsize)


def save_figure(fig, path, *, dpi: int = DEFAULT_DPI,
                bbox_inches: str = "tight", close: bool = False, **savefig_kw) -> Path:
    """Write ``fig`` to ``path`` at the standard DPI, creating parent dirs.

    Returns the path. Set ``close=True`` to release the figure afterwards
    (equivalent to a following ``plt.close(fig)``).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches=bbox_inches, **savefig_kw)
    if close:
        import matplotlib.pyplot as plt
        plt.close(fig)
    return path


__all__ = ["DEFAULT_DPI", "style_ax", "save_figure"]
