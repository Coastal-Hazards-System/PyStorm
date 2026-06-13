"""plots - NaN-aware time-series plotter for POT diagnostics.

Author : Norberto C. Nadal-Caraballo, PhD  <norberto.c.nadal-caraballo@usace.army.mil>

Draws the original time series, the selected peaks, and the chosen threshold.
Long gaps in the input (default > 48 h) are split into separate line segments
so matplotlib does not draw horizontal "bridges" across them.

Colors come from the canonical Wave Maker design palette (``pystorm_common``),
shared with the PST module for a consistent, publication-grade look.
"""

from typing import Optional

import pandas as pd

from pystorm_common import WAVE_MAKER, EMPHASIS, EMPH_DARK, INK, MUTED, GRID, C, style_ax

# Semantic roles for POT figures, mapped to the canonical palette tokens per
# the design handoff's figure-type conventions.
PALETTE = {
    "series":    WAVE_MAKER,       # primary signal / detrended / NTR series
    "peaks":     EMPHASIS,         # selected peaks (coral)
    "threshold": INK,             # POT threshold dashed line (REF_DASH)
    "measured":  INK,             # measured backdrop series (behind detrended)
    "trend":     C["deep_ocean"], # linear-trend line
    "ref":       MUTED,           # reference (0) line
    "midpoint":  EMPH_DARK,       # NTDE midpoint pivot (emphasis dark)
    "grid":      GRID,
    "spine":     INK,
}


class TimeSeriesPlotter:
    """Render a time series + peaks + threshold onto a matplotlib Axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Destination axes; the caller owns the figure lifecycle.
    datetime_col, value_col : str
        Column names in the input DataFrame.
    ylabel : str
        Y-axis label (units appended if non-empty).
    units : str
        Optional unit string. Appears parenthesized after ``ylabel``.
    title : str
        Plot title.
    gap_hours : float
        Threshold for segmenting the line on data gaps.
    """

    def __init__(
        self,
        ax,
        datetime_col: str   = "datetime",
        value_col:    str   = "value",
        ylabel:       str   = "Response",
        units:        str   = "",
        title:        str   = "Time Series",
        gap_hours:    float = 48.0,
    ) -> None:
        self.ax           = ax
        self.datetime_col = datetime_col
        self.value_col    = value_col
        self.ylabel       = f"{ylabel} ({units})" if units else ylabel
        self.title        = title
        self.gap_hours    = float(gap_hours)

    # ──────────────────────────────────────────────────────────────────────
    def plot(self, df: pd.DataFrame, label: Optional[str] = None,
             color: Optional[str] = None, linestyle: str = "-") -> None:
        x = df[self.datetime_col]
        y = df[self.value_col]

        mask = x.notna() & y.notna()
        x = x[mask].reset_index(drop=True)
        y = y[mask].reset_index(drop=True)
        if x.empty:
            return

        segments  = []
        prev      = 0
        gap_secs  = self.gap_hours * 3600.0
        for i in range(1, len(x)):
            dt = (x[i] - x[i - 1]).total_seconds()
            if dt > gap_secs:
                segments.append((x[prev:i], y[prev:i]))
                prev = i
        segments.append((x[prev:], y[prev:]))

        for xi, yi in segments:
            self.ax.plot(xi, yi, label=label if label else None,
                         color=color, linestyle=linestyle)
            label = None  # legend only once

    # ──────────────────────────────────────────────────────────────────────
    def finalize(self, xlabel: str = "Date") -> None:
        self.ax.set_xlabel(xlabel, fontsize=12)
        self.ax.set_ylabel(self.ylabel, fontsize=12)
        self.ax.set_title(self.title, fontsize=13, fontweight="bold")
        style_ax(self.ax)
        leg = self.ax.legend(loc="upper left", frameon=True, framealpha=0.95,
                             edgecolor=PALETTE["grid"], fontsize=9)
        leg.get_frame().set_linewidth(0.8)
        if self.ax.lines:
            xdata = self.ax.lines[0].get_xdata()
            if pd.api.types.is_datetime64_any_dtype(xdata):
                self.ax.figure.autofmt_xdate()
        self.ax.figure.tight_layout()
