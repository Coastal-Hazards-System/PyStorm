"""Orchestrator - per CRL: load the SRR drivers, run the Monte-Carlo, write the
synthetic TC catalog + summary, and optionally plot a QC figure.

Author : Norberto C. Nadal-Caraballo, PhD  <norberto.c.nadal-caraballo@usace.army.mil>
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from life_cycle_simulation.config import LCSConfig
from life_cycle_simulation import srr_source, simulator, writer, calibration


@dataclass
class CRLResult:
    crl_id: int
    lat: float
    lon: float
    lam: float                       # Poisson rate lambda (TC/yr)
    p_low: float
    p_med: float
    p_high: float
    n_events: int                    # total synthetic TCs across all realizations
    daily_used: bool                 # day-of-year drawn from the smooth daily SRR
    catalog_path: Path
    summary_path: Path
    fano: float = 1.0                # realized variance/mean of annual counts
    acf1: float = 0.0                # realized lag-1 autocorrelation of annual counts
    plot_paths: List[Path] = field(default_factory=list)


@dataclass
class LCSResult:
    results: Dict[int, CRLResult] = field(default_factory=dict)
    sim_years: int = 0
    n_realizations: int = 0


class LCSOrchestrator:
    """Runs the life-cycle Monte-Carlo for the configured CRLs off one SRR source."""

    def __init__(self, config: LCSConfig) -> None:
        self.cfg = config

    def _child_rng(self, crl_id: int):
        """Independent, reproducible RNG sub-stream per CRL (None seed -> fresh)."""
        if self.cfg.seed is None:
            return np.random.default_rng()
        return np.random.default_rng(np.random.SeedSequence([int(self.cfg.seed), crl_id]))

    def _file_tag(self, crl_id: int) -> str:
        """Compact, self-describing output tag, e.g. crl0001_R200km_100yr_1000real."""
        r = int(round(self.cfg.radius_km))
        return (f"crl{crl_id:04d}_R{r}km_{self.cfg.sim_years}yr_"
                f"{self.cfg.n_realizations}real")

    def _pool_ids(self, srr_df, crl_id, pool_km):
        """(ids, dist_km) for CRLs within ``pool_km`` great-circle km of ``crl_id``."""
        if crl_id not in srr_df.index:
            return [crl_id], np.zeros(1)
        lat0 = float(srr_df.loc[crl_id, "lat"]); lon0 = float(srr_df.loc[crl_id, "lon"])
        lat = np.radians(srr_df["lat"].to_numpy(float))
        lon = np.radians(srr_df["lon"].to_numpy(float))
        dlat = lat - np.radians(lat0); dlon = lon - np.radians(lon0)
        a = (np.sin(dlat / 2) ** 2
             + np.cos(np.radians(lat0)) * np.cos(lat) * np.sin(dlon / 2) ** 2)
        dist_km = 2.0 * 6371.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
        mask = dist_km <= pool_km
        return [int(c) for c in srr_df.index.to_numpy()[mask]], dist_km[mask]

    def _resolve_correlation(self, crl_id, srr_df, selection_df, cal_start):
        """Resolve (ar_phi, ar_beta, overdispersion): calibrate the None ones from
        the historical annual counts (regionally pooled when configured), keep
        operator overrides."""
        cfg = self.cfg
        if selection_df is None:
            # No selection table: use explicit values, else neutral defaults.
            return (cfg.ar_phi if cfg.ar_phi is not None else 0.5,
                    cfg.ar_beta if cfg.ar_beta is not None else 0.0,
                    cfg.overdispersion if cfg.overdispersion is not None else 0.0, None)

        kw = dict(ar_phi=cfg.ar_phi, ar_beta=cfg.ar_beta,
                  overdispersion=cfg.overdispersion)
        if cfg.regional_pool_km:
            pool, dist_km = self._pool_ids(srr_df, crl_id, cfg.regional_pool_km)
            series = [calibration.crl_annual_counts(
                selection_df, c, radius_km=cfg.radius_km, start_year=cal_start)
                for c in pool]
            weights = None
            if cfg.regional_pool_sigma_km:                # Gaussian distance taper
                sig = cfg.regional_pool_sigma_km
                weights = np.exp(-(dist_km ** 2) / (2.0 * sig * sig))
            cal = calibration.calibrate_correlation_regional(series, weights, **kw)
        else:
            counts = calibration.crl_annual_counts(
                selection_df, crl_id, radius_km=cfg.radius_km, start_year=cal_start)
            cal = calibration.calibrate_correlation(counts, **kw)
        return cal["ar_phi"], cal["ar_beta"], cal["overdispersion"], cal

    def _process_crl(self, crl_id, srr_df, daily_df, selection_df=None,
                     cal_start=1938) -> CRLResult:
        cfg = self.cfg
        srr = srr_source.build_crl_srr(srr_df, daily_df, crl_id,
                                       day_method=cfg.day_method)
        rng = self._child_rng(crl_id)

        ar_phi, ar_beta, overdispersion, cal = 0.0, 0.0, 0.0, None
        if cfg.correlation:
            ar_phi, ar_beta, overdispersion, cal = self._resolve_correlation(
                crl_id, srr_df, selection_df, cal_start)

        out = simulator.simulate(
            srr, radius_km=cfg.radius_km, sim_years=cfg.sim_years,
            n_realizations=cfg.n_realizations, rng=rng,
            correlation=cfg.correlation, ar_phi=ar_phi, ar_beta=ar_beta,
            overdispersion=overdispersion, sequencing=cfg.sequencing)

        tag = self._file_tag(crl_id)
        odir = Path(cfg.output_dir)
        catalog_path = writer.write_catalog(out.catalog, odir / f"lcs_catalog_{tag}.csv")
        summary = writer.build_summary(out.catalog, cfg.n_realizations)
        summary_path = writer.write_summary(summary, odir / f"lcs_summary_{tag}.csv")

        mean_rate = out.n_events / (cfg.n_realizations * cfg.sim_years)
        corr = ""
        if cfg.correlation:
            pool = (f" pool={cal['n_pooled']}" if cal and cal.get("n_pooled", 1) > 1
                    else "")
            corr = (f"  corr: phi={ar_phi:.2f} beta={ar_beta:.2f} "
                    f"od={overdispersion:.3f}{pool}  Fano={out.fano:.2f} ACF1={out.acf1:+.2f}")
        print(f"[lcs] CRL {crl_id}: lambda={out.lam:.4f} TC/yr  "
              f"split low/med/high={out.p[0]:.2f}/{out.p[1]:.2f}/{out.p[2]:.2f}  "
              f"{out.n_events:,} TCs (mean {mean_rate:.3f}/yr)  "
              f"day={'daily' if srr.daily_used else 'monthly'}{corr}  -> {catalog_path.name}")

        plot_paths: List[Path] = []
        if cfg.make_plots and cfg.plots:
            plot_paths = self._render_plots(out, summary, srr, crl_id, tag)

        return CRLResult(
            crl_id=int(crl_id), lat=srr.lat, lon=srr.lon, lam=out.lam,
            p_low=float(out.p[0]), p_med=float(out.p[1]), p_high=float(out.p[2]),
            n_events=out.n_events, daily_used=srr.daily_used,
            catalog_path=catalog_path, summary_path=summary_path,
            fano=out.fano, acf1=out.acf1, plot_paths=plot_paths)

    def _render_plots(self, out, summary, srr, crl_id, tag) -> List[Path]:
        cfg = self.cfg
        from life_cycle_simulation import plots
        base = Path(cfg.plot_dir) if cfg.plot_dir else (Path(cfg.output_dir) / "plots")
        crl_dir = base / f"crl{int(crl_id):04d}"       # one folder per CRL
        try:
            paths = plots.render_suite(
                out.catalog, summary, srr, lam=out.lam, p=out.p,
                sim_years=cfg.sim_years, n_realizations=cfg.n_realizations,
                plots=cfg.plots, out_dir=crl_dir, tag=tag)
            print(f"[lcs] CRL {crl_id}: wrote {len(paths)} figure(s) -> {crl_dir}")
            return paths
        except RuntimeError as exc:                    # matplotlib missing
            print(f"[lcs] CRL {crl_id}: plots skipped ({exc})")
            return []

    def run(self) -> LCSResult:
        cfg = self.cfg
        if cfg.storm_type == "etc":
            raise NotImplementedError(
                "storm_type='etc' (extratropical-cyclone life cycle) is a placeholder "
                "and not yet implemented; use storm_type='tc'. The same Poisson / "
                "stratum / day-of-year machinery would run on an ETC SRR source.")
        if cfg.input_csv is None:
            raise ValueError("input_csv is required (the SCA srr_<basin>_<v>.csv table).")

        srr_df = srr_source.load_srr_table(cfg.input_csv)

        daily_df = None
        if cfg.day_method == "daily":
            daily_csv = cfg.daily_csv or srr_source.locate_daily_companion(cfg.input_csv)
            if daily_csv is None:
                print("[lcs] day_method='daily' but no daily SRR table found next to "
                      "input_csv; falling back to the monthly seasonal shape.")
            else:
                daily_df = srr_source.load_daily_table(daily_csv, cfg.crl_ids)
                print(f"[lcs] daily SRR seasonal shape from {Path(daily_csv).name}")

        # Correlation calibration source: the SCA selection table and the rate-period
        # start year (parsed from the SRR filename, e.g. ..._1938-2025_...).
        selection_df = None
        cal_start = 1938
        if cfg.correlation:
            m = re.search(r"_(\d{4})-(\d{4})_", Path(cfg.input_csv).name)
            cal_start = int(m.group(1)) if m else 1938
            sel_csv = cfg.selection_csv or srr_source.locate_selection_companion(cfg.input_csv)
            if sel_csv is None:
                print("[lcs] correlation on but no selection table found next to "
                      "input_csv; using explicit/neutral params (no calibration).")
            else:
                # Regional pooling needs the neighbours too, so load the whole table;
                # otherwise just the requested CRLs (cheaper).
                sel_filter = None if cfg.regional_pool_km else cfg.crl_ids
                selection_df = srr_source.load_selection_table(sel_csv, sel_filter)
                pooled = " (regional pooling)" if cfg.regional_pool_km else ""
                print(f"[lcs] correlation calibration from {Path(sel_csv).name} "
                      f"(history from {cal_start}){pooled}")

        result = LCSResult(sim_years=cfg.sim_years, n_realizations=cfg.n_realizations)
        for crl_id in cfg.crl_ids:
            result.results[int(crl_id)] = self._process_crl(
                crl_id, srr_df, daily_df, selection_df, cal_start)
        return result
