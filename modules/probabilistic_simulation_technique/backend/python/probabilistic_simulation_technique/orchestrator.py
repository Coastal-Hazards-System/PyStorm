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
Step 2 — Sort descending; build Weibull-plotting-position empirical AEFs.
Step 3 — Choose the GPD threshold via QDM-WMSE.
Step 4 — Split values into the GPD tail and the empirical bulk.
Step 5 — Bootstrap the GPD-tail exceedances (C++ kernel when available).
Step 6 — Fit GPD per realization and evaluate ICDF on the plot AEF grid.
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
    make_aef_grids,
    fit_gpd_ensemble,
    assemble_hazard_curve,
    interpolate_to_table,
)
from .io                     import read_pot_csv, write_pst_outputs
from .postproc               import HazardCurvePlotter
from .sampling               import BootstrapGenerator, select_gpd_threshold_qdm
from .solver                 import CPP_KERNEL_AVAILABLE


@dataclass
class PSTResult:
    """In-memory result bundle from one PST run."""
    gpd_threshold: float
    lambda_val:    float
    lambda_th:     float
    aef_plot:      np.ndarray
    aef_table:     np.ndarray
    hc_plot_aef:   np.ndarray
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

        lambda_val = values.size / cfg.record_length_years
        print(f"[PST] n_pot = {values.size}; record_length = "
              f"{cfg.record_length_years} yr; lambda = {lambda_val:.4f} /yr")

        # Step 2 — descending sort, empirical AEFs (Weibull plotting positions).
        values_pot  = np.sort(values)[::-1]
        n           = values_pot.size
        weibull_aef = (np.arange(1, n + 1) / (n + 1)) * lambda_val

        # Step 3 — GPD threshold via QDM-WMSE.
        threshold, _wmse, _candidates = select_gpd_threshold_qdm(
            values_pot     = values_pot,
            weibull_aef    = weibull_aef,
            lambda_val     = lambda_val,
            min_percentile = cfg.threshold_min_percentile,
            max_percentile = cfg.threshold_max_percentile,
            n_candidates   = cfg.n_threshold_candidates,
        )
        print(f"[PST] GPD threshold = {threshold:.4f}")

        # Step 4 — split exceedances / bulk.
        exceed_mask  = values_pot > threshold
        pot_above_th = values_pot[exceed_mask]
        aef_above_th = weibull_aef[exceed_mask]   # noqa: F841 (retained for symmetry)
        pot_below_th = values_pot[~exceed_mask]
        aef_below_th = weibull_aef[~exceed_mask]

        if pot_above_th.size < 2:
            raise RuntimeError(
                "Fewer than 2 POT values exceed the selected threshold; cannot fit GPD."
            )

        lambda_th = pot_above_th.size / cfg.record_length_years

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

        # Step 6 — fit GPD per realization, evaluate on the plot AEF grid.
        aef_table, aef_plot = make_aef_grids()
        ensemble, gpd_be, gpd_cb10, gpd_cb90, aef_gpd_mask = fit_gpd_ensemble(
            boot_matrix     = boot_matrix,
            threshold       = threshold,
            aef_plot        = aef_plot,
            lambda_th       = lambda_th,
            shape_clip_low  = cfg.shape_clip_low,
            shape_clip_high = cfg.shape_clip_high,
        )
        valid = int(np.sum(~np.isnan(ensemble).all(axis=1)))
        print(f"[PST] Valid GPD realizations: {valid} / {cfg.num_simulations}")

        # Step 7 — splice GPD tail + empirical bulk; interpolate to table grid.
        aef_gpd = aef_plot[aef_gpd_mask]
        hc_plot_aef, hc_plot_be, hc_plot_cb10, hc_plot_cb90 = assemble_hazard_curve(
            aef_gpd      = aef_gpd,
            gpd_be       = gpd_be,
            gpd_cb10     = gpd_cb10,
            gpd_cb90     = gpd_cb90,
            aef_below_th = aef_below_th,
            pot_below_th = pot_below_th,
        )
        hc_table_be, hc_table_cb10, hc_table_cb90 = interpolate_to_table(
            aef_table = aef_table,
            hc_aef    = hc_plot_aef,
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
            aef_plot      = aef_plot,
            aef_table     = aef_table,
            hc_table_be   = hc_table_be,
            hc_table_cb10 = hc_table_cb10,
            hc_table_cb90 = hc_table_cb90,
            hc_plot_aef   = hc_plot_aef,
            hc_plot_be    = hc_plot_be,
            hc_plot_cb10  = hc_plot_cb10,
            hc_plot_cb90  = hc_plot_cb90,
        )
        print(f"[PST] Outputs written to {cfg.output_dir}")

        self._render_plot(
            base_filename = base_filename,
            lambda_val    = lambda_val,
            threshold     = threshold,
            aef_gpd       = aef_gpd,
            gpd_be        = gpd_be,
            gpd_cb10      = gpd_cb10,
            gpd_cb90      = gpd_cb90,
            aef_below_th  = aef_below_th,
            pot_below_th  = pot_below_th,
            plots_dir     = cfg.plots_dir,
            ylabel        = cfg.y_axis_label,
        )

        return PSTResult(
            gpd_threshold   = threshold,
            lambda_val      = lambda_val,
            lambda_th       = lambda_th,
            aef_plot        = aef_plot,
            aef_table       = aef_table,
            hc_plot_aef     = hc_plot_aef,
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
        aef_gpd:       np.ndarray,
        gpd_be:        np.ndarray,
        gpd_cb10:      np.ndarray,
        gpd_cb90:      np.ndarray,
        aef_below_th:  np.ndarray,
        pot_below_th:  np.ndarray,
        plots_dir:     Path,
        ylabel:        str,
    ) -> None:
        plots_dir = Path(plots_dir)
        plots_dir.mkdir(parents=True, exist_ok=True)
        out_path = plots_dir / f"{base_filename}_PST_HC.png"

        fig, ax  = plt.subplots(figsize=(10, 6))
        plotter  = HazardCurvePlotter(ax=ax, lambda_val=lambda_val)
        plotter.plot_hazard_curve(
            empirical_cdf = (aef_below_th, pot_below_th),
            gpd_curve     = (aef_gpd, gpd_be),
            gpd_cb        = (gpd_cb10, gpd_cb90),
            gpd_threshold = threshold,
            ylabel        = ylabel,
            output_path   = out_path,
        )
        plt.close(fig)
        print(f"[PST] Plot saved: {out_path}")
