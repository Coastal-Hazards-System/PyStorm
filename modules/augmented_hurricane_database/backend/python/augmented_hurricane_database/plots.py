"""Per-TC imputation diagnostic plots.

Author / POC : Norberto C. Nadal-Caraballo, PhD  <norberto.c.nadal-caraballo@usace.army.mil>

Renders one PNG per tropical cyclone so the imputed values can be inspected along
each storm's time history (a Python port of the CHS MATLAB DataImputation plots).
Each figure shows the GP-metamodel-completed series as a line with markers (``GPM``)
and the originally observed values as red dots (``Obs``):

  * Central pressure (``cp``)  - y = central-pressure deficit, ``1013 - pmin`` (hPa)
  * Radius of max wind (``rmax``) - y = ``rmax_km`` (km)

``Obs`` are the rows whose value was present BEFORE GP-metamodel imputation
(HURDAT2, plus any EBTRK backfill for Rmax); every other point on the line is an
imputed value, so observed dots lie on the completed line.

Speed. A full basin is ~1300-2000 storms, so the renderer is built to be fast:
matplotlib's Agg backend, ONE figure reused across storms (only the line data,
title, and axis limits change per storm - figure/artist construction is the
dominant cost), pre-converted numeric dates, a fixed layout (no per-figure
``tight_layout``), and low-compression PNG writes. ``n_jobs`` spreads the storms
over worker processes for a further near-linear speedup.

matplotlib is imported lazily (an optional dependency); if it is unavailable the
caller is told to install it and plotting is skipped.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from pystorm_common import save_figure

# Sea-level reference for the central-pressure deficit (matches the GP metamodel).
_PREF_HPA = 1013.0

# Per-target plot spec: filename tag, y-label, the value series, and the y-axis
# floor used to mimic the MATLAB ranges without ever clipping an extreme storm.
_SPEC = {
    "cp": {
        "file": "Cp",
        "ylabel": r"Central Pressure Deficit, $\Delta p$ (hPa)",
        "ylim_floor": (-20.0, 140.0),
        "value": lambda d: _PREF_HPA - d["pmin_hpa"].to_numpy(dtype=float),
    },
    "rmax": {
        "file": "Rm",
        "ylabel": r"Radius of Maximum Winds, $R_{max}$ (km)",
        "ylim_floor": (0.0, 300.0),
        "value": lambda d: d["rmax_km"].to_numpy(dtype=float),
    },
}

TARGETS = tuple(_SPEC)


def _pyplot():
    """Import matplotlib with the non-interactive Agg backend, or explain why not."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except Exception as exc:                                   # noqa: BLE001
        raise RuntimeError(
            "matplotlib is required for imputation plots; install it with "
            "`pip install matplotlib` (or `pip install -e .[plots]`).") from exc


def _id_label(nhc_id: str) -> str:
    """Cyclone id 'AL021880' -> 'AL 1880 02' (prefix, year, storm number)."""
    if len(nhc_id) >= 8:
        return f"{nhc_id[:2]} {nhc_id[4:8]} {nhc_id[2:4]}"
    return nhc_id


def _file_id(nhc_id: str) -> str:
    """Cyclone id 'AL021880' -> 'AL1880_02' for filenames: NHC basin, year, number.

    The basin prefix (AL/EP/CP) keeps Pacific EP and CP storms from colliding, and
    putting the year before the storm number sorts the files chronologically.
    """
    if len(nhc_id) >= 8:
        return f"{nhc_id[:2]}{nhc_id[4:8]}_{nhc_id[2:4]}"
    return nhc_id


def _storm_specs(df, obs, value, tnum, max_storms=None):
    """Lightweight, picklable per-storm payloads (chronological), skipping empties."""
    specs = []
    for tc_no, pos in df.groupby("tc_no", sort=True).indices.items():
        pos = np.sort(np.asarray(pos))
        pos = pos[np.argsort(tnum[pos], kind="stable")]    # chronological within storm
        v = value[pos]
        if not np.isfinite(v).any():                       # nothing to show
            continue
        row0 = df.iloc[pos[0]]
        specs.append((int(tc_no), str(row0["name"]), str(row0["nhc_id"]),
                      int(row0["year"]), tnum[pos], v, obs[pos]))
        if max_storms is not None and len(specs) >= max_storms:
            break
    return specs


def _render_specs(specs, *, basin, target, out_dir, dpi=150) -> int:
    """Render a list of storm specs reusing ONE figure. Returns the count written."""
    plt = _pyplot()
    import matplotlib.dates as mdates

    spec = _SPEC[target]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    floor_lo, floor_hi = spec["ylim_floor"]

    # Build the figure and its artists ONCE; per storm only the data and a few
    # texts/limits change. This is the main speed lever.
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.subplots_adjust(left=0.12, right=0.96, top=0.86, bottom=0.20)
    ax.set_xlabel("Date (UTC)")
    ax.set_ylabel(spec["ylabel"])
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b-%d %H"))
    ax.tick_params(axis="x", rotation=30)
    (gpm_line,) = ax.plot([], [], "-o", color="#0072BD", markersize=4,
                          linewidth=1.5, markerfacecolor="#0072BD", label="GPM")
    (obs_pts,) = ax.plot([], [], linestyle="None", marker="o", markersize=6,
                         color="red", label="Obs")
    ax.legend(loc="best")
    title = ax.set_title("", fontweight="bold")
    year_txt = fig.text(0.98, 0.02, "", ha="right", va="bottom",
                        fontsize=9, color="0.3")

    written = 0
    for _tc_no, name, nhc_id, year, t, v, o in specs:
        gpm_line.set_data(t, v)
        obs_pts.set_data(t[o], v[o])
        title.set_text(
            f"HURDAT Data Imputation\n(TC: {name}, ID: {_id_label(nhc_id)})")
        year_txt.set_text(str(int(year)))
        lo, hi = float(t.min()), float(t.max())
        if hi <= lo:                                   # single fix / one timestamp
            lo, hi = lo - 0.25, hi + 0.25              # ±6 h so xlim is not singular
        ax.set_xlim(lo, hi)
        ax.set_ylim(floor_lo, max(floor_hi, float(np.nanmax(v)) * 1.1))
        fname = f"DataImputation_{spec['file']}_HURDAT_{basin}_{_file_id(nhc_id)}.png"
        save_figure(fig, out_dir / fname, dpi=dpi, bbox_inches=None,
                    pil_kwargs={"compress_level": 1})
        written += 1

    plt.close(fig)
    return written


def _render_chunk(payload) -> int:
    """Process-pool entry point: render one chunk of storm specs."""
    specs, basin, target, out_dir, dpi = payload
    return _render_specs(specs, basin=basin, target=target, out_dir=out_dir, dpi=dpi)


def _resolve_jobs(n_jobs, n_specs: int) -> int:
    """Worker count: None/0 -> auto (cores-1, capped). Serial for small batches."""
    if n_jobs is None or n_jobs <= 0:
        n_jobs = max(1, min((os.cpu_count() or 2) - 1, 8))
    if n_specs < 200:        # process startup/import is not worth it for a few plots
        return 1
    return max(1, min(n_jobs, n_specs))


def plot_basin_imputation(
    df: pd.DataFrame,
    *,
    basin: str,
    target: str,
    obs_mask,
    out_dir,
    dpi: int = 150,
    max_storms: int | None = None,
    n_jobs: int | None = None,
    verbose: bool = True,
) -> int:
    """Write one imputation PNG per TC for ``target`` (``cp`` or ``rmax``).

    ``df`` is the completed per-fix table (after GP-metamodel imputation);
    ``obs_mask`` is a boolean array, aligned to ``df`` by position, marking the
    rows whose value was observed before imputation. ``n_jobs`` spreads the
    storms over worker processes (None/0 = auto). Returns the number of plots
    written; ``max_storms`` caps it.
    """
    if target not in _SPEC:
        raise ValueError(f"Unknown plot target '{target}'; expected one of {TARGETS}.")
    _pyplot()                                              # fail fast if matplotlib is missing
    import matplotlib.dates as mdates

    df = df.reset_index(drop=True)
    obs = np.asarray(obs_mask, dtype=bool)
    if obs.shape[0] != len(df):
        raise ValueError(f"obs_mask length {obs.shape[0]} != number of rows {len(df)}.")

    value = _SPEC[target]["value"](df)
    tnum = mdates.date2num(pd.to_datetime(df["time_utc"]).to_numpy())
    specs = _storm_specs(df, obs, value, tnum, max_storms=max_storms)
    if not specs:
        return 0

    jobs = _resolve_jobs(n_jobs, len(specs))
    out_dir = str(out_dir)
    if jobs > 1:
        from concurrent.futures import ProcessPoolExecutor
        if verbose:
            print(f"[ahd]   rendering {len(specs):,} {basin} {target} plots "
                  f"on {jobs} processes")
        payloads = [(specs[i::jobs], basin, target, out_dir, dpi) for i in range(jobs)]
        total = 0
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            for cnt in ex.map(_render_chunk, payloads):
                total += cnt
        return total

    if verbose:
        print(f"[ahd]   rendering {len(specs):,} {basin} {target} plots (serial)")
    return _render_specs(specs, basin=basin, target=target, out_dir=out_dir, dpi=dpi)
