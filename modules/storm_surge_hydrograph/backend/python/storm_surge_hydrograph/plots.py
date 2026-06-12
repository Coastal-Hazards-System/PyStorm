"""Per-save-point diagnostic plots for the SSH module.

Author / POC : Norberto C. Nadal-Caraballo, PhD  <norberto.c.nadal-caraballo@usace.army.mil>

Per save point:
  * SSH_SP#####.png  - three panels: the canonical (double-normalized) unit
    hydrograph with the normalized ensemble and the rising/falling parametric fit;
    a peak-scaling family (median duration, several peaks); and a duration envelope
    (median peak, P25/P50/P75 equivalent widths).
  * SSH_ensemble_SP#####.png - every storm's unnormalized hydrograph (m NAVD88)
    aligned at its peak, colored by peak elevation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from storm_surge_hydrograph.hydrograph import UnitHydrograph, scale_to_peak, width_stats


def _pyplot():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except Exception as exc:                                   # noqa: BLE001
        raise RuntimeError("matplotlib is required for SSH plots "
                           "(pip install matplotlib).") from exc


def plot_aligned_ensemble(uh: UnitHydrograph, out_path, *, lat=None, lon=None,
                          dpi: int = 110) -> Path:
    """Every storm's UNNORMALIZED hydrograph (m NAVD88), aligned at its peak.

    Each storm's elevation is reconstructed from the peak-aligned normalized stack;
    for the double-normalized model the dimensionless grid is mapped back to physical
    time via each storm's own duration. Lines are colored by peak elevation.
    """
    plt = _pyplot()
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if uh.stack is None:
        raise RuntimeError("unit hydrograph has no stored storm stack to plot")

    peaks = np.asarray(uh.peaks, dtype=float)
    elev = uh.ground_elev + uh.stack * (peaks[:, None] - uh.ground_elev)  # (n, n_grid)
    norm = Normalize(vmin=float(peaks.min()), vmax=float(peaks.max()))
    cmap = plt.get_cmap("viridis")

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.subplots_adjust(left=0.10, right=0.99, top=0.92, bottom=0.11)
    loc = f"  ({lat:.4f}, {lon:.4f})" if lat is not None and lon is not None else ""
    ax.set_title(f"CHS — SSH SP{uh.sp_id:05d} peak-aligned hydrographs (n={uh.n_storms})"
                 f"{loc}", fontweight="bold")
    order = np.argsort(peaks)
    for i in order:
        tau_i = uh.grid * uh.equiv_widths[i] if uh.dimensionless else uh.grid
        ax.plot(tau_i, elev[i], color=cmap(norm(peaks[i])), linewidth=0.8, alpha=0.7)
    ax.axhline(uh.ground_elev, color="0.4", linewidth=1.0, linestyle="--",
               label=f"ground = {uh.ground_elev:.2f} m", zorder=5)
    ax.axvline(0, color="0.4", linewidth=0.8, linestyle=":", zorder=5)
    span = float(np.max(uh.equiv_widths)) * uh.window if uh.dimensionless else uh.window
    ax.set_xlim(-span, span)
    ax.set_xlabel("Time relative to peak (h)")
    ax.set_ylabel("Water-surface elevation (m NAVD88)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=8)
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label="peak elevation (m NAVD88)", pad=0.01)
    fig.savefig(out_path, dpi=dpi, pil_kwargs={"compress_level": 1})
    plt.close(fig)
    return out_path


def plot_save_point(uh: UnitHydrograph, out_path, *, lat=None, lon=None,
                    scale_peaks: Optional[Sequence[float]] = None, dpi: int = 110) -> Path:
    """Three-panel figure: canonical shape, peak scaling, and duration envelope."""
    plt = _pyplot()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ws = width_stats(uh)
    Wmed = ws["p50"]

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 11))
    fig.subplots_adjust(left=0.11, right=0.97, top=0.95, bottom=0.06, hspace=0.30)
    loc = f"  ({lat:.4f}, {lon:.4f})" if lat is not None and lon is not None else ""
    fig.suptitle(f"CHS — SSH save point SP{uh.sp_id:05d}{loc}", fontweight="bold")

    # Panel 1: canonical (double-normalized) unit hydrograph.
    if uh.stack is not None:
        for row in uh.stack:
            ax1.plot(uh.grid, row, color="0.78", linewidth=0.5, zorder=1)
        ax1.plot([], [], color="0.78", linewidth=0.5,
                 label=f"normalized storms (n={uh.n_storms})")
    ax1.plot(uh.grid, uh.u, color="#0050FF", linewidth=2.0, zorder=3,
             label=f"unit hydrograph ({uh.aggregate})")
    if uh.fit is not None:
        ax1.plot(uh.grid, uh.fit.u_param, color="#FF2020", linewidth=1.6, linestyle="--",
                 zorder=4, label="parametric (rise/fall fit)")
        txt = (f"rise: sigma={uh.fit.sigma_rise:.2f}, p={uh.fit.p_rise:.2f}\n"
               f"fall: sigma={uh.fit.sigma_fall:.2f}, p={uh.fit.p_fall:.2f}\n"
               f"RMSE={uh.fit.rmse:.3f}")
        ax1.text(0.985, 0.97, txt, transform=ax1.transAxes, ha="right", va="top",
                 fontsize=7, family="monospace",
                 bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="0.6"))
    ax1.axvline(0, color="0.4", linewidth=0.8, linestyle=":")
    ax1.set_xlim(uh.grid[0], uh.grid[-1]); ax1.set_ylim(0, 1.05)
    xlab = "dimensionless time  s = tau / D" if uh.dimensionless else "Time relative to peak (h)"
    ax1.set_xlabel(xlab); ax1.set_ylabel("Normalized surge (peak = 1)")
    ax1.grid(True, alpha=0.3); ax1.legend(loc="upper left", fontsize=8)

    # Panel 2: peak scaling at the median equivalent width.
    peaks = list(scale_peaks) if scale_peaks else [float(np.median(uh.peaks)),
                                                   float(np.max(uh.peaks))]
    colors = ["#1F9E1F", "#E69500", "#B0179E", "#0050FF"]
    for i, P in enumerate(peaks):
        tau, elev = scale_to_peak(uh, P, equiv_width=Wmed)
        ax2.plot(tau, elev, color=colors[i % len(colors)], linewidth=1.8,
                 label=f"peak = {P:.2f} m")
    ax2.axhline(uh.ground_elev, color="0.4", linewidth=1.0, linestyle="--",
                label=f"ground = {uh.ground_elev:.2f} m")
    ax2.axvline(0, color="0.4", linewidth=0.8, linestyle=":")
    ax2.set_xlabel("Time relative to peak (h)")
    ax2.set_ylabel("Elevation (m NAVD88)")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper left", fontsize=8,
               title=f"Peak scaling (W_eq = {Wmed:.1f} h)")

    # Panel 3: equivalent-width envelope at the median peak.
    Pmed = float(np.median(uh.peaks))
    for (lab, W), col in zip([("P25", ws["p25"]), ("P50", ws["p50"]), ("P75", ws["p75"])],
                             ["#3aa0ff", "#0050FF", "#002a8f"]):
        tau, elev = scale_to_peak(uh, Pmed, equiv_width=W)
        ax3.plot(tau, elev, color=col, linewidth=1.8, label=f"{lab} W_eq = {W:.1f} h")
    ax3.axhline(uh.ground_elev, color="0.4", linewidth=1.0, linestyle="--",
                label=f"ground = {uh.ground_elev:.2f} m")
    ax3.axvline(0, color="0.4", linewidth=0.8, linestyle=":")
    ax3.set_xlabel("Time relative to peak (h)")
    ax3.set_ylabel("Elevation (m NAVD88)")
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc="upper left", fontsize=8,
               title=f"Equivalent-width envelope (peak = {Pmed:.2f} m)")

    fig.savefig(out_path, dpi=dpi, pil_kwargs={"compress_level": 1})
    plt.close(fig)
    return out_path
