"""orchestrator - end-to-end POT extraction workflow runner.

Author : Norberto C. Nadal-Caraballo, PhD  <norberto.c.nadal-caraballo@usace.army.mil>

Composes the time-series reader, the iterative threshold-search kernel
(``_pot`` when built, pure Python otherwise), the peaks writer, and the
diagnostic plotter into a single deterministic run.

Public API
----------
  POTResult              dataclass with the run's in-memory outputs
  POTOrchestrator        accepts POTConfig and exposes ``.run() -> POTResult``

Algorithm
---------
Step 1 - Read the input CSV; sort ascending; canonicalize column names.
Step 2 - Convert datetimes to Unix epoch seconds (float64).
Step 3 - Iteratively search for a percentile threshold that yields the
         target event rate after the configured segmentation method.
Step 4 - Materialize the peaks DataFrame from the chosen indices.
Step 5 - Save the peaks CSV and render the diagnostic plot.
"""

from dataclasses import dataclass
from pathlib     import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config                import POTConfig
from .io                    import read_time_series_csv, write_pot_peaks
from .postproc              import TimeSeriesPlotter, PALETTE
from pystorm_common          import save_figure
from .sampling              import IterativeThresholdSearch, ThresholdSearchResult


@dataclass
class POTResult:
    """In-memory result bundle from one POT extraction."""
    threshold:        float
    peaks_df:         pd.DataFrame      # columns: datetime, value
    converged:        bool
    iterations:       int
    events_per_year:  float
    final_percentile: float
    used_cpp_kernel:  bool
    effective_duration_years: float = 0.0   # valid-hours / (365.25 * 24)


class POTOrchestrator:
    """End-to-end POT runner.

    Parameters
    ----------
    config : POTConfig
        Validated job request (immutable).
    """

    def __init__(self, config: POTConfig) -> None:
        self.config = config

    # ──────────────────────────────────────────────────────────────────────
    def run(self) -> POTResult:
        cfg = self.config
        print(f"[POT] Loading time series: {cfg.input_csv}")
        df = read_time_series_csv(cfg.input_csv, cfg.datetime_col, cfg.value_col)
        # Drop NaN-value rows: missing observations are excluded from both the
        # search and the effective-duration count.
        df = df.dropna(subset=["value"]).reset_index(drop=True)
        n  = len(df)
        if n < 2:
            raise RuntimeError(
                f"input has only {n} valid row(s); need at least 2 for POT"
            )

        # Effective duration = valid hourly steps / hours-per-year. POT targets
        # the rate against THIS (not the calendar span), and trims to exactly
        # round(target * eff_dur) peaks so PST can recover eff_dur = n_pot / rate.
        eff_dur = n / (365.25 * 24.0)
        n_keep  = int(round(cfg.target_events_per_year * eff_dur))
        print(f"[POT] effective duration = {eff_dur:.2f} yr "
              f"({n:,} non-NaN hourly steps); target {cfg.target_events_per_year} "
              f"ev/yr -> keep {n_keep} peaks")

        # Step 2 - convert to numpy float arrays for the kernel. Use a
        # resolution-independent epoch-seconds cast: pandas datetime64 may be
        # ns or us, so a fixed //1e9 on the raw int64 would misscale times.
        times_sec = df["datetime"].to_numpy("datetime64[s]").astype(np.int64).astype(np.float64)
        values    = df["value"].to_numpy(dtype=np.float64)

        # Step 3 - threshold search (one-sided, rate measured against eff_dur).
        searcher = IterativeThresholdSearch(
            interevent_sec         = cfg.interevent_hours * 3600.0,
            method                 = cfg.method,
            target_events_per_year = cfg.target_events_per_year,
            tolerance              = cfg.tolerance,
            start_percentile       = cfg.start_percentile,
            step_size              = cfg.step_size,
            max_iter               = cfg.max_iter,
            record_length_years    = eff_dur,
        )
        used_cpp = searcher.use_cpp
        print(f"[POT] Threshold-search backend: "
              f"{'C++ (_pot)' if used_cpp else 'pure Python'}")

        r: ThresholdSearchResult = searcher.run(values, times_sec)

        if not r.converged:
            print(
                f"[POT] WARNING: search did not land in [{cfg.target_events_per_year}, "
                f"{cfg.target_events_per_year + cfg.tolerance}] ev/yr; tightest "
                f"state is {r.events_per_year:.4f} ev/yr at percentile "
                f"{r.final_percentile:.4f}"
            )
        else:
            print(
                f"[POT] Threshold {r.threshold:.4f} {cfg.units} at percentile "
                f"{r.final_percentile:.4f}: {r.events_per_year:.4f} ev/yr "
                f"(>= target {cfg.target_events_per_year}, within +{cfg.tolerance})"
            )

        # Step 4 - rank-trim the (one-sided) peaks to exactly n_keep largest, so
        # the written count is deterministic: n_pot = round(target * eff_dur).
        peak_idx = np.asarray(r.peak_indices, dtype=np.int64)
        if n_keep >= 1 and peak_idx.size > n_keep:
            top = np.argsort(values[peak_idx], kind="stable")[::-1][:n_keep]
            peak_idx = np.sort(peak_idx[top])     # restore chronological order
        print(f"[POT] Retained {peak_idx.size} peaks "
              f"(effective rate = {peak_idx.size / eff_dur:.4f} ev/yr)")

        # Materialize peaks DataFrame in time order.
        peaks_df = df.iloc[peak_idx].reset_index(drop=True).copy()

        # Step 5 - save + plot.
        base_filename = Path(cfg.input_csv).stem
        out_csv = write_pot_peaks(cfg.output_dir, base_filename, peaks_df)
        print(f"[POT] Peaks written to: {out_csv}")

        self._render_plot(
            df            = df,
            peaks_df      = peaks_df,
            threshold     = r.threshold,
            base_filename = base_filename,
            value_col_label = cfg.value_col,
            units           = cfg.units,
            vdatum          = cfg.vdatum,
            plots_dir       = cfg.plots_dir,
        )

        return POTResult(
            threshold        = r.threshold,
            peaks_df         = peaks_df,
            converged        = r.converged,
            iterations       = r.iterations,
            events_per_year  = r.events_per_year,
            final_percentile = r.final_percentile,
            used_cpp_kernel  = used_cpp,
            effective_duration_years = eff_dur,
        )

    # ──────────────────────────────────────────────────────────────────────
    @staticmethod
    def _render_plot(
        df:              pd.DataFrame,
        peaks_df:        pd.DataFrame,
        threshold:       float,
        base_filename:   str,
        value_col_label: str,
        units:           str,
        vdatum:          str,
        plots_dir:       Path,
    ) -> None:
        plots_dir = Path(plots_dir)
        plots_dir.mkdir(parents=True, exist_ok=True)
        out_path = plots_dir / f"{base_filename}_pot.png"

        full_units = units + (f", {vdatum}" if vdatum else "")

        fig, ax = plt.subplots(figsize=(10, 5))
        plotter = TimeSeriesPlotter(
            ax, datetime_col="datetime", value_col="value",
            ylabel=value_col_label, units=full_units,
            title=f"PyStorm-POT {value_col_label} — Peaks Over Threshold",
        )
        df_valid = df.dropna(subset=["datetime", "value"])
        plotter.plot(df_valid, label=value_col_label, color=PALETTE["series"])

        ax.plot(peaks_df["datetime"], peaks_df["value"], "o",
                color=PALETTE["peaks"], markersize=4, markeredgecolor="white",
                markeredgewidth=0.4, zorder=5, label="Peaks")
        ax.axhline(threshold, color=PALETTE["threshold"], linestyle="--", linewidth=1.4,
                   label=f"Threshold = {threshold:.2f} {units}")
        plotter.finalize()

        save_figure(fig, out_path, close=True)
        print(f"[POT] Plot saved: {out_path}")
