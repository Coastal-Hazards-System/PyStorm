"""Orchestrator - per basin: load CRLs + augmented HURDAT, select storms,
compute SRR/DSRR (annual + monthly), write tables, and optionally map each CRL.

Author / POC : Norberto C. Nadal-Caraballo, PhD  <norberto.c.nadal-caraballo@usace.army.mil>
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from tc_climatological_analysis.config import TCAConfig
from tc_climatological_analysis.crls import load_crls
from tc_climatological_analysis.hurdat_source import (
    locate_augmented_hurdat, load_augmented_hurdat, created_date,
)
from tc_climatological_analysis.selection import select_storms
from tc_climatological_analysis.gkf import compute_rates
from tc_climatological_analysis import writer


@dataclass
class BasinResult:
    basin: str
    crl_file: Path
    hurdat_file: Path
    n_crls: int
    n_selected: int
    nyrs: int
    selection_path: Path
    srr_path: Path
    srr_daily_path: Path
    dsrr_summary_path: Path
    dsrr_arrays_path: Path
    srr_radius_path: Optional[Path] = None
    n_maps: int = 0
    n_monthly_maps: int = 0
    n_daily_plots: int = 0


@dataclass
class TCAResult:
    results: Dict[str, BasinResult] = field(default_factory=dict)


class TCAOrchestrator:
    """Runs the climatological analysis for the configured basins."""

    def __init__(self, config: TCAConfig) -> None:
        self.cfg = config

    def _resolve_in(self, p) -> Path:
        """Resolve a source file: absolute, or relative to the raw inputs dir."""
        p = Path(p)
        return p if p.is_absolute() else self.cfg.raw_dir / p

    def _process_basin(self, basin: str) -> Optional[BasinResult]:
        cfg = self.cfg
        crl_file = cfg.crl_file_for(basin)
        if crl_file is None:
            print(f"[tca] {basin}: no CRL set configured; skipped.")
            return None

        crl_path = self._resolve_in(crl_file)
        crls = load_crls(crl_path)
        print(f"[tca] {basin}: {len(crls):,} CRLs from {crl_path.name}")

        hurdat_path = locate_augmented_hurdat(
            basin, explicit_file=cfg.hurdat_file_for(basin),
            input_dir=cfg.input_dir,
            ahd_outputs_dir=cfg.ahd_outputs_dir)
        hurdat = load_augmented_hurdat(hurdat_path)
        # Effective rate period. START_YEAR = None uses the entire HURDAT record;
        # otherwise it is clamped up to the basin's first season (Atlantic ~1851 ->
        # 1938 floor; the Pacific record starts 1949).
        data_min = int(hurdat["year"].min())
        start_year = (data_min if cfg.start_year is None
                      else max(int(cfg.start_year), data_min))
        end_year = int(cfg.end_year) if cfg.end_year else int(hurdat["year"].max())
        nyrs = end_year - start_year + 1
        print(f"[tca] {basin}: augmented HURDAT {hurdat_path.name} "
              f"({len(hurdat):,} fixes); rate over {start_year}-{end_year} "
              f"(Nyrs={nyrs})")

        selection = select_storms(
            hurdat, crls, k_size=cfg.k_size, max_dist=cfg.max_dist,
            max_cp=cfg.max_cp, ref_pressure=cfg.ref_pressure)
        print(f"[tca] {basin}: selected {len(selection):,} CRL-TC pairs "
              f"(<= {cfg.max_dist:.0f} km)")

        rates = compute_rates(
            selection, crls, k_size=cfg.k_size, dir_kernel=cfg.dir_kernel,
            day_kernel=cfg.day_kernel, start_year=start_year, end_year=end_year,
            min_dp=cfg.min_dp, dp_low=cfg.dp_low, dp_med=cfg.dp_med)

        # Tag every non-plot output with the rate period and the NHC HURDAT file
        # date, e.g. ..._1938-2025_20260227.csv (start year = the rate start).
        created = created_date(hurdat_path)
        span = f"{start_year}-{end_year}"
        suf = f"_{span}_{created}" if created else f"_{span}"

        out = cfg.output_dir
        sel_path = writer.write_selection(selection, out / f"selection_{basin}{suf}.csv")
        srr_path = writer.write_srr_table(rates, crls, out / f"srr_{basin}{suf}.csv")
        srr_daily_path = writer.write_srr_daily_table(
            rates, crls, out / f"srr_daily_{basin}{suf}.csv")
        dsrr_sum = writer.write_dsrr_summary(rates, crls, out / f"dsrr_{basin}{suf}.csv")
        dsrr_arr = writer.write_dsrr_arrays(rates, crls, out / f"dsrr_{basin}{suf}.npz")
        print(f"[tca] {basin}: wrote SRR/DSRR (annual + monthly + daily) -> {out}")

        srr_radius_path = None
        if cfg.srr_radial:
            r = int(round(cfg.srr_radius_km))
            rad_dir = out / f"srr_{r}km"
            srr_radius_path = writer.write_srr_radius_table(
                rates, crls, cfg.srr_radius_km, rad_dir / f"srr_{r}km_{basin}{suf}.csv")
            print(f"[tca] {basin}: wrote SRR_{r}km (within {r} km; TC/yr) -> {rad_dir}")

        n_maps = n_monthly = n_daily = 0
        if cfg.plot_selection or cfg.plot_monthly or cfg.plot_daily:
            n_maps, n_monthly, n_daily = self._render_maps(selection, crls, rates, basin)

        return BasinResult(
            basin=basin, crl_file=crl_path, hurdat_file=hurdat_path,
            n_crls=len(crls), n_selected=len(selection), nyrs=nyrs,
            selection_path=sel_path, srr_path=srr_path,
            srr_daily_path=srr_daily_path,
            dsrr_summary_path=dsrr_sum, dsrr_arrays_path=dsrr_arr,
            srr_radius_path=srr_radius_path,
            n_maps=n_maps, n_monthly_maps=n_monthly, n_daily_plots=n_daily)

    def _render_maps(self, selection, crls, rates, basin: str):
        cfg = self.cfg
        from tc_climatological_analysis import plots
        base = Path(cfg.plot_dir) if cfg.plot_dir else (cfg.output_dir / "plots")
        cache_dir = cfg.raw_dir / "naturalearth"
        common = dict(dp_low=cfg.dp_low, dp_med=cfg.dp_med,
                      resolution=cfg.basemap_resolution, cache_dir=cache_dir,
                      n_jobs=cfg.plot_jobs)
        r = int(round(cfg.srr_radius_km))
        # (folder tag, SRR scale, map SRR-box label, daily y-axis label, daily note).
        # The radius variant is opt-in. The daily y-axis is a rate density over
        # day-of-year (the 365 values sum to the annual SRR); the note spells that out.
        daily_note = ("Daily rate density over day-of-year:\n"
                      "the 365 values sum to the annual SRR.\n"
                      "(\"per day\" = per day-of-year, not a 2nd time axis)")
        variants = [(None, 1.0, "SRR (TC/km/yr)",
                     "Daily SRR (TC/km/yr, per day-of-year)", daily_note)]
        if cfg.srr_radial:
            variants.append((f"{r}km", 2.0 * cfg.srr_radius_km, f"SRR_{r}km (TC/yr)",
                             f"Daily SRR_{r}km (TC/yr within {r}km, per day-of-year)",
                             daily_note))

        n_annual = n_monthly = n_daily = 0
        try:
            for tag, scale, label, daily_label, daily_note_v in variants:
                vkw = dict(srr_scale=scale, srr_label=label, **common)
                pre = f"SRR_{tag} " if tag else ""
                if cfg.plot_selection:
                    od = base / (f"selection_{tag}_{basin}" if tag else f"selection_{basin}")
                    na = plots.plot_selected_storms(
                        selection, crls, rates, basin=basin, out_dir=od, **vkw)
                    print(f"[tca] {basin}: wrote {na:,} {pre}CRL maps -> {od}")
                    n_annual += na
                if cfg.plot_monthly:
                    od = base / (f"selection_monthly_{tag}_{basin}" if tag
                                 else f"selection_monthly_{basin}")
                    nm = plots.plot_selected_storms_monthly(
                        selection, crls, rates, basin=basin, out_dir=od, **vkw)
                    print(f"[tca] {basin}: wrote {nm:,} {pre}monthly CRL maps -> {od}")
                    n_monthly += nm
                if cfg.plot_daily:
                    od = base / (f"daily_{tag}_{basin}" if tag else f"daily_{basin}")
                    nd = plots.plot_daily_srr(
                        selection, crls, rates, basin=basin, out_dir=od,
                        srr_scale=scale, srr_label=daily_label, srr_note=daily_note_v,
                        n_jobs=cfg.plot_jobs)
                    print(f"[tca] {basin}: wrote {nd:,} {pre}daily SRR plots -> {od}")
                    n_daily += nd
        except RuntimeError as exc:                            # e.g. matplotlib missing
            print(f"[tca] {basin}: CRL plots skipped ({exc})")
        return n_annual, n_monthly, n_daily

    def run(self) -> TCAResult:
        result = TCAResult()
        for basin in self.cfg.basins:
            r = self._process_basin(basin)
            if r is not None:
                result.results[basin] = r
        return result
