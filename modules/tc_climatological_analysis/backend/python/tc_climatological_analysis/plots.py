"""Per-CRL selected-TC maps (annual and monthly).

Author / POC : Norberto C. Nadal-Caraballo, PhD  <norberto.c.nadal-caraballo@usace.army.mil>

Port of the map figure in ``CHS_Atlantic_StormSelection.m``: one PNG per CRL
showing every selected tropical cyclone (the representative point) colored by
intensity - High (red), Med (yellow), Low (green) by central-pressure deficit -
over a Natural Earth basemap, with the CRL marked in blue. Each map also prints
the storm recurrence rate in the order All, High, Med, Low (storms/km/year).

Three products:
  * annual  - one map per CRL (the selected storms over all months).
  * monthly - one map per CRL and calendar month (Jan-Dec), with that month's
    storms and SRR; sequenced CRL1 -> months, CRL2 -> months, ...
  * daily   - one line plot per CRL of the continuous daily SRR over day-of-year
    1..365, with the All/High/Med/Low intensity curves on the same axes. This is an
    XY curve, not a map (no basemap), so it needs only matplotlib.

Speed: the basemap is the costly part, so one figure is built per worker with the
basemap drawn ONCE; each map only updates the storm scatter, the CRL point, the
title, and the SRR box. ``n_jobs`` spreads the work over worker processes.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from tc_climatological_analysis import basemap as _basemap
from tc_climatological_analysis.gkf import MONTHS

# SRR is always reported in this order.
_SRR_ORDER = ("all", "high", "med", "low")
_SRR_LABELS = ("All", "High", "Med", "Low")

# Vivid traffic-light intensity colors, brighter than the dull r/y/g matplotlib
# defaults so the storm markers pop like the CHS MATLAB figures.
_INTENSITY_FACE = {"high": "#FF2020", "med": "#FFC400", "low": "#21C521"}
# Thin, slightly transparent dark edge: enough to define each marker without the
# black outlines piling up and muddying dense clusters.
_MARKER_EDGE = "#303030"
_MARKER_EDGEWIDTH = 0.4
_MARKER_SIZE = 4

# Curve colors for the daily-SRR line plots. All is black; High/Low use the vivid
# palette; Med uses a deeper gold than the marker so the line reads on white.
_DAILY_COLORS = {"all": "k", "high": _INTENSITY_FACE["high"],
                 "med": "#E6A100", "low": _INTENSITY_FACE["low"]}
# Day-of-year of the first of each month on the fixed 365-day (non-leap) calendar,
# used for the daily-plot x-axis month ticks.
_MONTH_START_DOY = (1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335)

# Per-basin map extent and ticks. The Pacific basin (Guam 147E ... Hawaii/Samoa
# ... 120W) straddles the antimeridian, so it is drawn in a 0-360 longitude frame
# (lon360=True); the Atlantic uses the standard -180..180 frame.
_EXTENT = {
    "atlantic": {"xlim": (-110, -50), "ylim": (10, 50),
                 "xticks": range(-110, -40, 10), "yticks": range(10, 55, 5),
                 "lon360": False},
    "pacific":  {"xlim": (130, 230), "ylim": (-20, 35),
                 "xticks": range(130, 250, 20), "yticks": range(-20, 40, 10),
                 "lon360": True},
}


def _lat_label(v) -> str:
    return f"{abs(int(round(v)))}°{'S' if v < 0 else 'N'}"


def _lon_label(v) -> str:
    x = ((v + 180) % 360) - 180                      # fold to -180..180
    if abs(abs(x) - 180) < 1e-6:
        return "180°"
    return f"{abs(int(round(x)))}°{'W' if x < 0 else 'E'}"


def _sci(v) -> str:
    """Format a value as mathtext ``M.MM x 10^E`` (e.g. 1.84e-3 -> 1.84x10^-3)."""
    if not np.isfinite(v) or v <= 0:
        return "0"
    exp = int(np.floor(np.log10(v)))
    mant = v / 10.0 ** exp
    if mant >= 9.995:                                # rounding guard
        mant /= 10.0
        exp += 1
    return rf"${mant:.2f}\times10^{{{exp}}}$"


def _srr_text(srr4, scale=1.0, label="SRR (TC/km/yr)") -> str:
    """SRR text box content, in the All/High/Med/Low order (``scale`` for SRR_<R>km)."""
    lines = [label]
    for lab, v in zip(_SRR_LABELS, srr4):
        lines.append(f"{lab:<4} {_sci(v * scale)}")
    return "\n".join(lines)


def _pyplot():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except Exception as exc:                                   # noqa: BLE001
        raise RuntimeError(
            "matplotlib is required for the CRL maps; install it with "
            "`pip install -e .[plots]`.") from exc


def _resolve_jobs(n_jobs, n: int) -> int:
    if n_jobs is None or n_jobs <= 0:
        n_jobs = max(1, min((os.cpu_count() or 2) - 1, 8))
    if n < 50:
        return 1
    return max(1, min(n_jobs, n))


def _id_index(crls: pd.DataFrame) -> dict:
    """CRL id -> row position (the rates arrays are indexed by CRL order)."""
    return {int(i): k for k, i in enumerate(crls["id"].to_numpy())}


def _srr4_annual(rates: dict, idx: int):
    return tuple(float(rates[b]["srr"][idx]) for b in _SRR_ORDER)


def _srr4_monthly(rates: dict, idx: int, m0: int):
    return tuple(float(rates[b]["srr_monthly"][idx, m0]) for b in _SRR_ORDER)


# A render spec is: (fname, title, clat, clon, lon, lat, dp, srr4).
def _annual_specs(selection, crls, rates, basin, max_crls):
    title_basin = basin.capitalize()
    pos = {int(r.id): (float(r.lat), float(r.lon)) for r in crls.itertuples()}
    idx = _id_index(crls)
    specs = []
    for cid, g in selection.groupby("crl_id", sort=True):
        cid = int(cid)
        clat, clon = pos.get(cid, (np.nan, np.nan))
        srr4 = _srr4_annual(rates, idx[cid])
        specs.append((f"CHS_{title_basin}_CRL_{cid:04d}.png",
                      f"CHS — {title_basin} CRL {cid:04d}",
                      clat, clon, g["lon"].to_numpy(float), g["lat"].to_numpy(float),
                      g["dp"].to_numpy(float), srr4))
        if max_crls is not None and len(specs) >= max_crls:
            break
    return specs


def _monthly_specs(selection, crls, rates, basin, max_crls):
    title_basin = basin.capitalize()
    pos = {int(r.id): (float(r.lat), float(r.lon)) for r in crls.itertuples()}
    idx = _id_index(crls)
    specs = []
    ncrl = 0
    for cid, g in selection.groupby("crl_id", sort=True):
        cid = int(cid)
        clat, clon = pos.get(cid, (np.nan, np.nan))
        ci = idx[cid]
        for m in range(1, 13):                       # CRL -> all 12 months in order
            gm = g[g["month"] == m]
            srr4 = _srr4_monthly(rates, ci, m - 1)
            specs.append((f"CHS_{title_basin}_CRL_{cid:04d}_{m:02d}_{MONTHS[m - 1]}.png",
                          f"CHS — {title_basin} CRL {cid:04d} — {MONTHS[m - 1]}",
                          clat, clon, gm["lon"].to_numpy(float), gm["lat"].to_numpy(float),
                          gm["dp"].to_numpy(float), srr4))
        ncrl += 1
        if max_crls is not None and ncrl >= max_crls:
            break
    return specs


def _render_specs(specs, *, basin, out_dir, dp_low, dp_med, resolution,
                  cache_dir, have_basemap, srr_scale=1.0,
                  srr_label="SRR (TC/km/yr)", dpi=110) -> int:
    """Render one PNG per spec, reusing the figure and BLITTING a cached background.

    The Natural Earth basemap is vector (thousands of line segments), and a plain
    ``savefig`` re-rasterizes all of it on every map (~400 ms each, the dominant cost).
    Instead the static scene (basemap + legend + axes) is drawn and rasterized ONCE,
    captured with ``copy_from_bbox``, and per map only the dynamic storm scatter, the
    CRL point, the title, and the SRR box are blitted on top; the canvas RGBA buffer is
    written straight to PNG. This is ~20x faster per map with identical output.
    """
    plt = _pyplot()
    from matplotlib.lines import Line2D
    from PIL import Image

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ext = _EXTENT.get(basin, _EXTENT["atlantic"])
    lon360 = ext.get("lon360", False)

    layers = None
    if have_basemap:
        try:
            layers = _basemap.basemap_lines(resolution, cache_dir)
        except Exception:                                     # noqa: BLE001
            layers = None

    fig, ax = plt.subplots(figsize=(8, 6), dpi=dpi)
    fig.subplots_adjust(left=0.10, right=0.97, top=0.92, bottom=0.10)
    ax.set_xlim(*ext["xlim"])
    ax.set_ylim(*ext["ylim"])
    ax.set_xticks(list(ext["xticks"]))
    ax.set_yticks(list(ext["yticks"]))
    ax.set_xticklabels([_lon_label(v) for v in ext["xticks"]])
    ax.set_yticklabels([_lat_label(v) for v in ext["yticks"]])
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True, alpha=0.3)
    if layers is not None:
        _basemap.draw_basemap(ax, layers, lon360=lon360)

    proxies = [
        Line2D([], [], marker="o", linestyle="none", markeredgecolor=_MARKER_EDGE,
               markeredgewidth=_MARKER_EDGEWIDTH, markerfacecolor=_INTENSITY_FACE["high"],
               markersize=6, label="High"),
        Line2D([], [], marker="o", linestyle="none", markeredgecolor=_MARKER_EDGE,
               markeredgewidth=_MARKER_EDGEWIDTH, markerfacecolor=_INTENSITY_FACE["med"],
               markersize=6, label="Med"),
        Line2D([], [], marker="o", linestyle="none", markeredgecolor=_MARKER_EDGE,
               markeredgewidth=_MARKER_EDGEWIDTH, markerfacecolor=_INTENSITY_FACE["low"],
               markersize=6, label="Low"),
    ]
    leg = ax.legend(handles=proxies, loc="upper left", fontsize=8, title="TC Intensity")
    leg.set_zorder(5)
    title = ax.set_title("", fontweight="bold")
    srr_box = ax.text(0.985, 0.985, "", transform=ax.transAxes, ha="right",
                      va="top", fontsize=7, family="monospace", zorder=6,
                      bbox=dict(boxstyle="round", facecolor="white", alpha=0.85,
                                edgecolor="0.6"))

    # Rasterize the static scene once, then cache it for per-map blitting.
    fig.canvas.draw()
    background = fig.canvas.copy_from_bbox(fig.bbox)

    written = 0
    for fname, ttl, clat, clon, lon, lat, dp, srr4 in specs:
        fig.canvas.restore_region(background)
        dynamic = []
        lon = np.asarray(lon, dtype=float)
        lat = np.asarray(lat, dtype=float)
        dp = np.asarray(dp, dtype=float)
        clon_p = clon
        if lon360:
            lon = lon % 360.0
            clon_p = clon % 360.0 if np.isfinite(clon) else clon
        if dp.size:
            low = dp < dp_low
            med = (dp >= dp_low) & (dp < dp_med)
            high = dp >= dp_med
            # Plot low -> med -> high so the most intense storms sit on top.
            for mask, key, z in ((low, "low", 3), (med, "med", 4), (high, "high", 5)):
                if mask.any():
                    dynamic += ax.plot(lon[mask], lat[mask], marker="o", linestyle="none",
                                       markersize=_MARKER_SIZE, markeredgecolor=_MARKER_EDGE,
                                       markeredgewidth=_MARKER_EDGEWIDTH,
                                       markerfacecolor=_INTENSITY_FACE[key], zorder=z)
        if np.isfinite(clat) and np.isfinite(clon_p):
            dynamic += ax.plot([clon_p], [clat], marker="o", linestyle="none",
                               markersize=7, markeredgecolor="k", markeredgewidth=0.6,
                               markerfacecolor="#0050FF", zorder=6)
        title.set_text(ttl)
        srr_box.set_text(_srr_text(srr4, srr_scale, srr_label))
        for art in dynamic:
            ax.draw_artist(art)
        ax.draw_artist(title)
        ax.draw_artist(srr_box)
        fig.canvas.blit(fig.bbox)
        Image.fromarray(np.asarray(fig.canvas.buffer_rgba())).save(
            out_dir / fname, compress_level=1)
        for art in dynamic:
            art.remove()
        written += 1

    plt.close(fig)
    return written


def _render_chunk(payload) -> int:
    specs, kw = payload
    return _render_specs(specs, **kw)


def _ensure_basemap(resolution, cache_dir, basin, verbose) -> bool:
    try:
        _basemap.basemap_lines(resolution, cache_dir)         # download + cache once
        return True
    except Exception as exc:                                   # noqa: BLE001
        if verbose:
            print(f"[tca] {basin}: basemap unavailable ({exc}); maps omit coastlines")
        return False


def _dispatch(specs, *, basin, out_dir, dp_low, dp_med, resolution, cache_dir,
              have_basemap, n_jobs, verbose, label, srr_scale=1.0,
              srr_label="SRR (TC/km/yr)") -> int:
    if not specs:
        return 0
    kw = dict(basin=basin, out_dir=str(out_dir), dp_low=dp_low, dp_med=dp_med,
              resolution=resolution, cache_dir=cache_dir, have_basemap=have_basemap,
              srr_scale=srr_scale, srr_label=srr_label)
    jobs = _resolve_jobs(n_jobs, len(specs))
    if jobs > 1:
        from concurrent.futures import ProcessPoolExecutor
        if verbose:
            print(f"[tca] {basin}: rendering {len(specs):,} {label} on {jobs} processes")
        payloads = [(specs[i::jobs], kw) for i in range(jobs)]
        total = 0
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            for c in ex.map(_render_chunk, payloads):
                total += c
        return total
    if verbose:
        print(f"[tca] {basin}: rendering {len(specs):,} {label} (serial)")
    return _render_specs(specs, **kw)


def plot_selected_storms(
    selection: pd.DataFrame,
    crls: pd.DataFrame,
    rates: dict,
    *,
    basin: str,
    out_dir,
    dp_low: float = 28.0,
    dp_med: float = 48.0,
    resolution: str = "50m",
    cache_dir=None,
    n_jobs: Optional[int] = None,
    max_crls: Optional[int] = None,
    srr_scale: float = 1.0,
    srr_label: str = "SRR (TC/km/yr)",
    verbose: bool = True,
) -> int:
    """One annual selected-TC map per CRL (with the All/High/Med/Low SRR box)."""
    _pyplot()
    if cache_dir is None:
        cache_dir = Path(out_dir) / "_basemap_cache"
    cache_dir = str(cache_dir)
    have = _ensure_basemap(resolution, cache_dir, basin, verbose)
    specs = _annual_specs(selection, crls, rates, basin, max_crls)
    return _dispatch(specs, basin=basin, out_dir=out_dir, dp_low=dp_low,
                     dp_med=dp_med, resolution=resolution, cache_dir=cache_dir,
                     have_basemap=have, n_jobs=n_jobs, verbose=verbose,
                     label="CRL maps", srr_scale=srr_scale, srr_label=srr_label)


def plot_selected_storms_monthly(
    selection: pd.DataFrame,
    crls: pd.DataFrame,
    rates: dict,
    *,
    basin: str,
    out_dir,
    dp_low: float = 28.0,
    dp_med: float = 48.0,
    resolution: str = "50m",
    cache_dir=None,
    n_jobs: Optional[int] = None,
    max_crls: Optional[int] = None,
    srr_scale: float = 1.0,
    srr_label: str = "SRR (TC/km/yr)",
    verbose: bool = True,
) -> int:
    """One map per CRL and calendar month (Jan-Dec) with that month's storms + SRR.

    Sequenced CRL1 -> months, CRL2 -> months, ... (filenames embed CRL then month).
    """
    _pyplot()
    if cache_dir is None:
        cache_dir = Path(out_dir) / "_basemap_cache"
    cache_dir = str(cache_dir)
    have = _ensure_basemap(resolution, cache_dir, basin, verbose)
    specs = _monthly_specs(selection, crls, rates, basin, max_crls)
    return _dispatch(specs, basin=basin, out_dir=out_dir, dp_low=dp_low,
                     dp_med=dp_med, resolution=resolution, cache_dir=cache_dir,
                     have_basemap=have, n_jobs=n_jobs, verbose=verbose,
                     label="monthly CRL maps", srr_scale=srr_scale, srr_label=srr_label)


# ── Daily-SRR curve plots (one line plot per CRL; All/High/Med/Low vs day-of-year) ──

def _daily_specs(selection, crls, rates, basin, max_crls):
    """Render specs for the daily plots: (fname, title, (all,high,med,low) curves)."""
    title_basin = basin.capitalize()
    idx = _id_index(crls)
    specs = []
    for cid, _g in selection.groupby("crl_id", sort=True):
        cid = int(cid)
        ci = idx[cid]
        curves = tuple(np.asarray(rates[b]["srr_daily"][ci], dtype=float)
                       for b in _SRR_ORDER)
        specs.append((f"CHS_{title_basin}_CRL_{cid:04d}.png",
                      f"CHS — {title_basin} CRL {cid:04d}", curves))
        if max_crls is not None and len(specs) >= max_crls:
            break
    return specs


def _render_daily_specs(specs, *, out_dir, doys, srr_scale=1.0,
                        srr_label="Daily SRR (TC/km/yr, per day)",
                        srr_note="The daily curve is the annual SRR spread over the\n"
                                 "calendar: the 365 values sum to the annual SRR.",
                        dpi=110) -> int:
    """Draw the day-of-year SRR curves; one reused figure, restyled per CRL.

    ``srr_note`` is a static annotation clarifying the units: the daily value is the
    annual rate (TC per year) per day-of-year, so summing the 365 days gives the
    annual SRR. ``per day`` means per day-of-year, not a second time axis.
    """
    plt = _pyplot()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    doys = np.asarray(doys, dtype=float)

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.subplots_adjust(left=0.12, right=0.97, top=0.92, bottom=0.12)
    ax.set_xlim(1, 365)
    ax.set_xticks(list(_MONTH_START_DOY))
    ax.set_xticklabels(list(MONTHS))
    ax.set_xlabel("Day of year")
    ax.set_ylabel(srr_label)
    ax.grid(True, alpha=0.3)
    title = ax.set_title("", fontweight="bold")
    lines = {}
    for b, lab in zip(_SRR_ORDER, _SRR_LABELS):
        (ln,) = ax.plot([], [], color=_DAILY_COLORS[b], linewidth=1.6, label=lab)
        lines[b] = ln
    ax.legend(loc="upper left", fontsize=8, title="Intensity")
    if srr_note:                                     # units clarifier (upper-right)
        ax.text(0.985, 0.985, srr_note, transform=ax.transAxes, ha="right", va="top",
                fontsize=7, style="italic", zorder=6,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.85,
                          edgecolor="0.6"))

    written = 0
    for fname, ttl, curves in specs:
        ymax = 0.0
        for b, arr in zip(_SRR_ORDER, curves):
            y = np.asarray(arr, dtype=float) * srr_scale
            lines[b].set_data(doys, y)
            if y.size:
                ymax = max(ymax, float(np.nanmax(y)))
        ax.set_ylim(0.0, ymax * 1.08 if ymax > 0 else 1.0)
        title.set_text(ttl)
        fig.savefig(out_dir / fname, dpi=dpi, pil_kwargs={"compress_level": 1})
        written += 1

    plt.close(fig)
    return written


def _render_daily_chunk(payload) -> int:
    specs, kw = payload
    return _render_daily_specs(specs, **kw)


def _dispatch_daily(specs, *, out_dir, doys, n_jobs, verbose, basin, label,
                    srr_scale=1.0, srr_label="Daily SRR (TC/km/yr, per day)",
                    srr_note=None) -> int:
    if not specs:
        return 0
    kw = dict(out_dir=str(out_dir), doys=np.asarray(doys, dtype=float),
              srr_scale=srr_scale, srr_label=srr_label)
    if srr_note is not None:
        kw["srr_note"] = srr_note
    jobs = _resolve_jobs(n_jobs, len(specs))
    if jobs > 1:
        from concurrent.futures import ProcessPoolExecutor
        if verbose:
            print(f"[tca] {basin}: rendering {len(specs):,} {label} on {jobs} processes")
        payloads = [(specs[i::jobs], kw) for i in range(jobs)]
        total = 0
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            for c in ex.map(_render_daily_chunk, payloads):
                total += c
        return total
    if verbose:
        print(f"[tca] {basin}: rendering {len(specs):,} {label} (serial)")
    return _render_daily_specs(specs, **kw)


def plot_daily_srr(
    selection: pd.DataFrame,
    crls: pd.DataFrame,
    rates: dict,
    *,
    basin: str,
    out_dir,
    n_jobs: Optional[int] = None,
    max_crls: Optional[int] = None,
    srr_scale: float = 1.0,
    srr_label: str = "Daily SRR (TC/km/yr, per day)",
    srr_note: Optional[str] = None,
    verbose: bool = True,
) -> int:
    """One daily-SRR line plot per CRL: All/High/Med/Low curves over day-of-year 1..365.

    ``srr_scale`` / ``srr_label`` produce the SRR_<R>km variant (the daily rate times
    the 2R-km diameter; expected TC/yr within R, per day-of-year). ``srr_note`` is the
    static units clarifier (the 365 daily values sum to the annual SRR). No basemap.
    """
    _pyplot()
    specs = _daily_specs(selection, crls, rates, basin, max_crls)
    doys = np.asarray(rates["_meta"]["doys"], dtype=float)
    return _dispatch_daily(specs, out_dir=out_dir, doys=doys, n_jobs=n_jobs,
                           verbose=verbose, basin=basin, label="daily SRR plots",
                           srr_scale=srr_scale, srr_label=srr_label, srr_note=srr_note)
