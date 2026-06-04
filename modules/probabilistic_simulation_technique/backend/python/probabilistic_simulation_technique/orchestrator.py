"""orchestrator — end-to-end PST workflow runner.

Author / POC : Norberto C. Nadal-Caraballo, PhD  <norberto.c.nadal-caraballo@usace.army.mil>

Composes the bootstrap kernel, GPD-threshold selector, hazard-curve assembly,
I/O, and plotting layers into a single deterministic run. CyHAN v2 §1 places
all side-effect orchestration here; engines and helper modules are pure.

Public API
----------
  PSTResult              dataclass holding the run outputs in memory
  PSTOrchestrator        accepts a PSTConfig and exposes `.run() -> PSTResult`

Algorithm
---------
Step 1 — Load the POT column and compute the population intensity (lambda).
Step 2 — Sort descending; build Weibull-plotting-position empirical AERs.
Step 3 — Choose the GPD location μ via QDO-WMSE.
Step 4 — Split values into the GPD tail and the empirical bulk.
Step 5 — Bootstrap the GPD-tail exceedances (C++ kernel when available).
Step 6 — Fit GPD per realization and evaluate ICDF on the plot AER grid.
Step 7 — Splice GPD tail and empirical bulk into one hazard curve;
         interpolate to the 22-AER reporting grid.
Step 8 — Save outputs and render the hazard-curve plot.
"""

from dataclasses import dataclass
from pathlib     import Path
from typing      import Optional

import matplotlib.pyplot as plt
import numpy as np

from .config                 import PSTConfig
from .hazard                 import (
    make_aer_grids,
    fit_gpd_ensemble,
    assemble_hazard_curve,
    interpolate_to_table,
)
from .io                     import read_pot_csv, write_pst_outputs
from .postproc               import HazardCurvePlotter, plot_qdo_diagnostics
from .sampling               import BootstrapGenerator, select_gpd_threshold_qdo
from .solver                 import CPP_KERNEL_AVAILABLE


@dataclass
class PSTResult:
    """In-memory result bundle from one PST run."""
    gpd_threshold: float
    lambda_val:    float
    lambda_mu:     float
    aer_plot:      np.ndarray
    aer_table:     np.ndarray
    hc_plot_aer:   np.ndarray
    hc_plot_be:    np.ndarray
    hc_plot_cb10:  np.ndarray
    hc_plot_cb90:  np.ndarray
    hc_table_be:   np.ndarray
    hc_table_cb10: np.ndarray
    hc_table_cb90: np.ndarray
    ensemble:      np.ndarray
    used_cpp_kernel: bool


class PSTOrchestrator:
    """End-to-end PST runner.

    Parameters
    ----------
    config : PSTConfig
        Validated job request (immutable).
    """

    def __init__(self, config: PSTConfig) -> None:
        self.config = config

    # ──────────────────────────────────────────────────────────────────────
    def run(self) -> PSTResult:
        cfg = self.config
        print(f"[PST] Loading POT input: {cfg.input_csv}")
        values = read_pot_csv(cfg.input_csv, cfg.storm_column)
        if values.size == 0:
            raise ValueError(f"POT column '{cfg.storm_column}' has no values")

        # Record length: explicit, else auto = n_pot / events_per_year. POT
        # trims to exactly events_per_year × effective_duration peaks, so this
        # recovers the effective duration (e.g. 250 peaks / 10 = 25 yr).
        n_pot = values.size
        record_length = cfg.record_length_years
        if record_length is None:
            record_length = n_pot / cfg.events_per_year
            print(f"[PST] record length (auto) = n_pot / events_per_year = "
                  f"{n_pot} / {cfg.events_per_year:g} = {record_length:.2f} yr")

        # POT rate λ_u: all peaks extracted above the POT threshold u.
        lambda_val = n_pot / record_length
        print(f"[PST] n_pot = {n_pot}; record_length = "
              f"{record_length:.4f} yr; lambda_u = {lambda_val:.4f} /yr")

        # Step 2 — descending sort, empirical AERs (Weibull plotting positions).
        values_pot  = np.sort(values)[::-1]
        n           = values_pot.size
        weibull_aer = (np.arange(1, n + 1) / (n + 1)) * lambda_val

        # Step 3 — GPD location μ via QDO (a threshold re-optimized for the
        # distribution fit, distinct from the POT threshold u). Default method
        # gates on WMSE; "stability" (opt-in) gates on the ξ plateau.
        qdo = select_gpd_threshold_qdo(
            values_pot     = values_pot,
            weibull_aer    = weibull_aer,
            lambda_val     = lambda_val,
            record_length  = record_length,
            min_percentile = cfg.threshold_min_percentile,
            max_percentile = cfg.threshold_max_percentile,
            n_candidates   = cfg.n_threshold_candidates,
            min_exceedances = cfg.min_exceedances,
            shape_clip_low  = cfg.shape_clip_low,
            shape_clip_high = cfg.shape_clip_high,
            selection       = cfg.gpd_selection,
            tiebreak        = cfg.gpd_tiebreak,
            stability_window = cfg.stability_window,
            stab_tol        = cfg.stability_tol,
            tol             = cfg.wmse_tolerance,
            fit_method      = cfg.gpd_fit_method,
            gof_statistic   = cfg.gof_statistic,
            gof_significance = cfg.gof_significance,
        )
        threshold = qdo.best_threshold
        if qdo.selection_method == "mrl":
            print(f"[PST] GPD location mu = {threshold:.4f}  (selection: mrl "
                  f"[mean-residual-life]; recovered xi = "
                  f"{qdo.mrl_slope/(1+qdo.mrl_slope):+.3f})")
        elif qdo.selection_method == "gof":
            print(f"[PST] GPD location mu = {threshold:.4f}  (selection: gof "
                  f"[{qdo.gof_statistic.upper()} failure-to-reject @ "
                  f"{qdo.gof_significance:g}]; {qdo.selected_set_idx.size} "
                  f"non-rejected candidate(s))")
        else:
            set_name = ("stability plateau" if qdo.selection_method == "stability"
                        else "WMSE-tolerance set")
            print(f"[PST] GPD location mu = {threshold:.4f}  (selection: "
                  f"{qdo.selection_method}; {set_name}: {qdo.selected_set_idx.size} "
                  f"candidate(s); tie-break: {qdo.tiebreak})")
        if qdo.selection_warning:
            print(f"[PST] WARNING: {qdo.selection_warning}")

        # Step 4 — split exceedances (> μ) / bulk (<= μ).
        exceed_mask  = values_pot > threshold
        pot_above_th = values_pot[exceed_mask]
        aer_above_th = weibull_aer[exceed_mask]   # empirical WPP above μ
        pot_below_th = values_pot[~exceed_mask]
        aer_below_th = weibull_aer[~exceed_mask]

        if pot_above_th.size < 2:
            raise RuntimeError(
                "Fewer than 2 POT values exceed the GPD location μ; cannot fit GPD."
            )

        # Exceedance rate at μ, λ_μ (drives the GPD-tail AER grid).
        lambda_mu = pot_above_th.size / record_length

        # Step 5 — bootstrap (C++ when available, Python otherwise).
        bootstrap = BootstrapGenerator(
            distribution = cfg.bootstrap.distribution,
            truncation   = cfg.bootstrap.truncation,
            seed         = cfg.random_seed,
        )
        used_cpp = bootstrap.use_cpp
        print(f"[PST] Bootstrap backend: {'C++ (_pst)' if used_cpp else 'pure Python'}")
        print(f"[PST] Generating {cfg.num_simulations} realizations from "
              f"{pot_above_th.size} exceedances ...")
        boot_matrix = bootstrap.generate(pot_above_th, cfg.num_simulations)

        # Step 6 — fit GPD per realization, evaluate on the plot AER grid.
        aer_table, aer_plot = make_aer_grids()
        ensemble, gpd_be, gpd_cb10, gpd_cb90, aer_gpd_mask = fit_gpd_ensemble(
            boot_matrix     = boot_matrix,
            threshold       = threshold,
            aer_plot        = aer_plot,
            lambda_mu       = lambda_mu,
            shape_clip_low  = cfg.shape_clip_low,
            shape_clip_high = cfg.shape_clip_high,
            fit_method      = cfg.gpd_fit_method,
        )
        valid = int(np.sum(~np.isnan(ensemble).all(axis=1)))
        print(f"[PST] Valid GPD realizations: {valid} / {cfg.num_simulations}")

        # Step 7 — splice GPD tail + empirical bulk; interpolate to table grid.
        aer_gpd = aer_plot[aer_gpd_mask]
        hc_plot_aer, hc_plot_be, hc_plot_cb10, hc_plot_cb90 = assemble_hazard_curve(
            aer_gpd      = aer_gpd,
            gpd_be       = gpd_be,
            gpd_cb10     = gpd_cb10,
            gpd_cb90     = gpd_cb90,
            aer_below_th = aer_below_th,
            pot_below_th = pot_below_th,
        )
        hc_table_be, hc_table_cb10, hc_table_cb90 = interpolate_to_table(
            aer_table = aer_table,
            hc_aer    = hc_plot_aer,
            hc_be     = hc_plot_be,
            hc_cb10   = hc_plot_cb10,
            hc_cb90   = hc_plot_cb90,
        )

        # Step 8 — save and plot.
        base_filename = self._derive_base_filename(cfg.input_csv)
        write_pst_outputs(
            output_dir    = cfg.output_dir,
            base_filename = base_filename,
            ensemble      = ensemble,
            aer_plot      = aer_plot,
            aer_table     = aer_table,
            hc_table_be   = hc_table_be,
            hc_table_cb10 = hc_table_cb10,
            hc_table_cb90 = hc_table_cb90,
            hc_plot_aer   = hc_plot_aer,
            hc_plot_be    = hc_plot_be,
            hc_plot_cb10  = hc_plot_cb10,
            hc_plot_cb90  = hc_plot_cb90,
        )
        print(f"[PST] Outputs written to {cfg.output_dir}")

        self._render_plot(
            base_filename = base_filename,
            lambda_val    = lambda_val,
            threshold     = threshold,
            lambda_mu     = lambda_mu,
            aer_gpd       = aer_gpd,
            gpd_be        = gpd_be,
            gpd_cb10      = gpd_cb10,
            gpd_cb90      = gpd_cb90,
            aer_below_th  = aer_below_th,
            pot_below_th  = pot_below_th,
            aer_above_th  = aer_above_th,
            pot_above_th  = pot_above_th,
            plots_dir     = cfg.plots_dir,
            ylabel        = cfg.y_axis_label,
            series        = cfg.plot_series,
        )

        # QDO GPD-location selection diagnostics (visual QA of Step 3).
        if cfg.plot_threshold_diagnostics:
            diag_path = Path(cfg.plots_dir) / f"{base_filename}_qdo_threshold.png"
            plot_qdo_diagnostics(qdo, diag_path, ylabel=cfg.y_axis_label)
            print(f"[PST] QDO diagnostics saved: {diag_path}")

        return PSTResult(
            gpd_threshold   = threshold,
            lambda_val      = lambda_val,
            lambda_mu       = lambda_mu,
            aer_plot        = aer_plot,
            aer_table       = aer_table,
            hc_plot_aer     = hc_plot_aer,
            hc_plot_be      = hc_plot_be,
            hc_plot_cb10    = hc_plot_cb10,
            hc_plot_cb90    = hc_plot_cb90,
            hc_table_be     = hc_table_be,
            hc_table_cb10   = hc_table_cb10,
            hc_table_cb90   = hc_table_cb90,
            ensemble        = ensemble,
            used_cpp_kernel = used_cpp,
        )

    # ──────────────────────────────────────────────────────────────────────
    @staticmethod
    def _derive_base_filename(input_csv: Path) -> str:
        # Mirror v1 behavior: strip the trailing "_<token>" if present.
        stem = Path(input_csv).stem
        return stem.rsplit("_", 1)[0] if "_" in stem else stem

    @staticmethod
    def _render_plot(
        base_filename: str,
        lambda_val:    float,
        threshold:     float,
        lambda_mu:     float,
        aer_gpd:       np.ndarray,
        gpd_be:        np.ndarray,
        gpd_cb10:      np.ndarray,
        gpd_cb90:      np.ndarray,
        aer_below_th:  np.ndarray,
        pot_below_th:  np.ndarray,
        aer_above_th:  np.ndarray,
        pot_above_th:  np.ndarray,
        plots_dir:     Path,
        ylabel:        str,
        series,
    ) -> None:
        plots_dir = Path(plots_dir)
        plots_dir.mkdir(parents=True, exist_ok=True)
        out_path = plots_dir / f"{base_filename}_pst_hc.png"

        fig, ax  = plt.subplots(figsize=(10, 6))
        plotter  = HazardCurvePlotter(ax=ax, lambda_val=lambda_val)
        plotter.plot_hazard_curve(
            empirical_below = (aer_below_th, pot_below_th),
            empirical_above = (aer_above_th, pot_above_th),
            gpd_curve       = (aer_gpd, gpd_be),
            gpd_cb          = (gpd_cb10, gpd_cb90),
            gpd_threshold   = threshold,
            threshold_aer   = lambda_mu,
            series          = series,
            ylabel          = ylabel,
            output_path     = out_path,
        )
        plt.close(fig)
        print(f"[PST] Plot saved: {out_path}")
