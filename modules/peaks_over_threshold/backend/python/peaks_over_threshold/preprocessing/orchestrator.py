"""orchestrator — run the selected NTR-pipeline stages with I/O and plots.

Author / POC : Norberto C. Nadal-Caraballo, PhD  <norberto.c.nadal-caraballo@usace.army.mil>

Wraps the pure preprocessing engines (download / detrend / ntr) with file
reads, writes, and diagnostic plots, executing only the stages requested in
``PreprocessConfig.stages``. The POT stage itself is run by the module
orchestrator (``main_peaks_over_threshold``); this class stops at the NTR CSV
and reports the path so POT can consume it.

Public API
----------
  PreprocessResult       paths produced by the chain
  PreprocessOrchestrator accepts PreprocessConfig, exposes ``.run()``
"""

from dataclasses import dataclass
from pathlib     import Path
from typing      import Optional

import matplotlib.pyplot as plt
import pandas as pd

from ..config   import PreprocessConfig
from ..io       import read_time_series_csv, write_series_csv
from ..postproc import TimeSeriesPlotter, PALETTE
from pystorm_common import save_figure
from .detrend   import detrend_time_series, ntde_midpoint_timestamp
from .ntr       import estimate_ntr
from .noaa_download import download_noaa_wl_data


def _fmt_year(y: float) -> str:
    """Compact year label: ``1983`` for whole years, ``2012.42`` otherwise."""
    return f"{int(y)}" if float(y).is_integer() else f"{y:g}"


@dataclass
class PreprocessResult:
    """Paths produced by the preprocessing chain (None where not run)."""
    raw_wl_csv:    Optional[Path] = None
    raw_tide_csv:  Optional[Path] = None
    detrended_csv: Optional[Path] = None
    trend_csv:     Optional[Path] = None
    ntr_csv:       Optional[Path] = None


class PreprocessOrchestrator:
    """Run download -> detrend -> ntr for the requested stages."""

    def __init__(self, config: PreprocessConfig) -> None:
        self.cfg = config

    # ──────────────────────────────────────────────────────────────────────
    def run(self) -> PreprocessResult:
        cfg = self.cfg
        sid = cfg.station_id
        raw, proc, plots = Path(cfg.raw_dir), Path(cfg.processed_dir), Path(cfg.plots_dir)

        res = PreprocessResult(
            raw_wl_csv    = raw  / f"water_level_{sid}.csv",
            raw_tide_csv  = raw  / f"tide_prediction_{sid}.csv",
            detrended_csv = proc / f"dwl_{sid}.csv",      # detrended water level
            trend_csv     = proc / f"trend_{sid}.csv",
            ntr_csv       = proc / f"ntr_{sid}.csv",
        )
        full_units = cfg.units + (f", {cfg.vdatum}" if cfg.vdatum else "")

        # ── Stage: download ────────────────────────────────────────────────
        if "download" in cfg.stages:
            print(f"[preprocess] download — station {sid}, {cfg.start_year}–{cfg.end_year}")
            years = range(cfg.start_year, cfg.end_year + 1)
            res.raw_wl_csv = download_noaa_wl_data(
                sid, years, "hourly_height", raw,
                datum=cfg.datum, time_zone=cfg.time_zone, units=cfg.download_units,
            )
            res.raw_tide_csv = download_noaa_wl_data(
                sid, years, "predictions", raw,
                datum=cfg.datum, time_zone=cfg.time_zone, units=cfg.download_units,
                interval=cfg.tide_interval,   # "h" → hourly tide, matches the WL grid
            )

        wl_detrended_df: Optional[pd.DataFrame] = None

        # ── Stage: detrend ─────────────────────────────────────────────────
        if "detrend" in cfg.stages:
            print(f"[preprocess] detrend — method={cfg.detrend_method}")
            wl_df = read_time_series_csv(res.raw_wl_csv, cfg.datetime_col, cfg.wl_value_col)
            wl_detrended_df, trend_df, slope_yr = detrend_time_series(
                wl_df, method=cfg.detrend_method,
                ntde_range=(cfg.ntde_start, cfg.ntde_end),
                slope_per_year=cfg.detrend_slope,
            )
            write_series_csv(wl_detrended_df, res.detrended_csv,
                             datetime_header=cfg.datetime_col, value_header="Water Level")
            write_series_csv(trend_df, res.trend_csv,
                             datetime_header=cfg.datetime_col, value_header="Water Level")
            slope_src = "user override" if cfg.detrend_slope is not None else "fitted"
            print(f"[preprocess] detrend — slope {slope_yr:+.4f} {cfg.units}/yr "
                  f"({slope_src}); wrote {res.detrended_csv.name}")
            self._plot_detrend(wl_df, wl_detrended_df, trend_df, full_units, plots, sid,
                               method=cfg.detrend_method,
                               ntde_range=(cfg.ntde_start, cfg.ntde_end))

        # ── Stage: ntr ─────────────────────────────────────────────────────
        if "ntr" in cfg.stages:
            print("[preprocess] ntr — NTR = detrended WL - interpolated tide")
            if wl_detrended_df is None:
                wl_detrended_df = read_time_series_csv(
                    res.detrended_csv, cfg.datetime_col, "Water Level")
            tide_df = read_time_series_csv(res.raw_tide_csv, cfg.datetime_col, cfg.tide_value_col)
            ntr_full = estimate_ntr(wl_detrended_df, tide_df)
            write_series_csv(ntr_full, res.ntr_csv, value_col="ntr",
                             datetime_header=cfg.datetime_col, value_header="NTR")
            print(f"[preprocess] ntr — wrote {res.ntr_csv.name}")
            self._plot_ntr(ntr_full, full_units, plots, sid)

        return res

    # ──────────────────────────────────────────────────────────────────────
    def _plot_detrend(self, measured, detrended, trend, units, plots_dir, sid,
                      method, ntde_range):
        plots_dir = Path(plots_dir); plots_dir.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(10, 5))
        p = TimeSeriesPlotter(ax, "datetime", "value", ylabel="Water Level",
                              units=units, title="PyStorm: Time Series Detrending")
        p.plot(measured,  label="Measured",     color=PALETTE["measured"])
        p.plot(detrended, label="Detrended",    color=PALETTE["series"])
        p.plot(trend,     label="Linear Trend", color=PALETTE["trend"], linestyle="--")
        ax.axhline(0, color=PALETTE["ref"], linestyle="--", linewidth=1,
                   label="Reference (0)")

        # NTDE midpoint — the pivot the midpoint-method trend rotates about
        # (the linear trend passes through zero here). Matches detrend.py's
        # centering: midpoint of [NTDE_start-01-01, (NTDE_end+1)-01-01).
        if method == "midpoint":
            midpoint = ntde_midpoint_timestamp(ntde_range)
            ax.axvline(midpoint, color=PALETTE["midpoint"], linestyle=":", linewidth=2,
                       label=f"NTDE midpoint "
                             f"({_fmt_year(ntde_range[0])}–{_fmt_year(ntde_range[1])})")

        p.finalize()
        out = plots_dir / f"dwl_{sid}.png"
        save_figure(fig, out, close=True)
        print(f"[preprocess] plot saved: {out}")

    def _plot_ntr(self, ntr_full, units, plots_dir, sid):
        plots_dir = Path(plots_dir); plots_dir.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(10, 5))
        p = TimeSeriesPlotter(ax, "datetime", "ntr", ylabel="Non-Tidal Residual",
                              units=units, title="PyStorm: Non-Tidal Residual (NTR)")
        p.plot(ntr_full, label="NTR", color=PALETTE["series"])
        p.finalize()
        out = plots_dir / f"ntr_{sid}.png"
        save_figure(fig, out, close=True)
        print(f"[preprocess] plot saved: {out}")
