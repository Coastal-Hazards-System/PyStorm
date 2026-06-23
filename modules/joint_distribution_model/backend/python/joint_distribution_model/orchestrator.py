"""Orchestrator - per basin: load the SCA inputs, fit per-CRL marginals + copula,
write the joint-distribution outputs, and optionally plot each CRL.

Author : Norberto C. Nadal-Caraballo, PhD  <norberto.c.nadal-caraballo@usace.army.mil>
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from joint_distribution_model.config import (
    JDMConfig, INTENSITY_BINS, INTENSITY_LABEL, PARAM_NAMES,
)
from joint_distribution_model import sca_source, adjust, marginals, copula, writer
from joint_distribution_model.solver import CPP_KERNEL_AVAILABLE

# Columns of the selection table that the per-CRL fit actually reads (the rest of
# the wide table is skipped on load). cp_source is added at load time.
_SELECTION_COLS = ["crl_id", "year", "heading_deg", "trans_kmh", "rmax_km", "gaussW"]

_NPARAM = len(PARAM_NAMES)
_STRATUM = ("high", "med", "low")     # the partition labels (by descending deficit)


@dataclass
class BasinResult:
    basin: str
    selection_file: Path
    dsrr_file: Path
    n_crls: int
    n_records: int
    marginals_path: Path
    copula_path: Path
    adjusted_path: Path
    n_plots: int = 0


@dataclass
class JDMResult:
    results: Dict[str, BasinResult] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Per-CRL worker (top-level so it is picklable for ProcessPoolExecutor)
# ---------------------------------------------------------------------------

def _process_crl(bundle: dict, params: dict) -> dict:
    """Fit one CRL: distance-weighted adjust + bin, marginals, and copula per bin."""
    crl_id = int(bundle["crl_id"])
    # Independent, reproducible per-CRL seed (an int, so the C++ kernel can use it).
    seed = params["seed"]
    seed_crl = None if seed is None else int(seed) * 1_000_003 + crl_id
    rng = np.random.default_rng(seed_crl) if seed_crl is not None else np.random.default_rng()

    bins = adjust.adjust_crl(
        heading=bundle["heading"], cp=bundle["cp"], rmax=bundle["rmax"],
        vt=bundle["vt"], gaussW=bundle["gaussW"], year=bundle["year"],
        dsrr_mean_all=bundle["mean"]["all"], dsrr_stdv_all=bundle["stdv"]["all"],
        ref_pressure=params["ref_pressure"], start_year=params["start_year"],
        min_dp=params["min_dp"], dp_low=params["dp_low"], dp_med=params["dp_med"],
        vt_clip=params["vt_clip"], rmax_clip=params["rmax_clip"])

    records, boot_extra = marginals.fit_crl_marginals(
        bins, bundle["mean"], bundle["stdv"], n_boot=params["n_boot"], rng=rng,
        seed=seed_crl, min_dp=params["min_dp"], dp_low=params["dp_low"],
        dp_med=params["dp_med"])

    cop = {}
    for b in INTENSITY_BINS:
        tau, rho = copula.fit_copula(bins[b])          # handles small/empty bins
        cop[b] = {"tau": tau, "rho": rho}

    return {"crl_id": crl_id, "lat": bundle["lat"], "lon": bundle["lon"],
            "records": records, "copula": cop, "bins": bins, "boot_extra": boot_extra}


def _plot_one(result: dict, basin: str, base_dir: str) -> int:
    """Render one CRL's marginal + copula figures (top-level, for the plot pool)."""
    from joint_distribution_model import plots
    base = Path(base_dir)
    n = plots.plot_crl_marginals(
        result["crl_id"], result["bins"], result["records"], result["boot_extra"],
        basin=basin, out_dir=str(base / f"marginals_{basin}"))
    n += plots.plot_crl_copula(
        result["crl_id"], result["bins"], result["copula"],
        basin=basin, out_dir=str(base / f"copula_{basin}"))
    return n


class JDMOrchestrator:
    """Fits the joint distribution model for the configured basins off the SCA outputs."""

    def __init__(self, config: JDMConfig) -> None:
        self.cfg = config

    def _bundles(self, selection, dsrr, crl_ids, cp_col):
        """Build a lightweight per-CRL input bundle (picklable, no DataFrame)."""
        g = selection.groupby("crl_id")
        for cid in crl_ids:
            rows = g.get_group(cid)
            lat, lon = dsrr.coord(cid)
            mean = {b: dsrr.heading_stats(cid, b)[0] for b in INTENSITY_BINS}
            stdv = {b: dsrr.heading_stats(cid, b)[1] for b in INTENSITY_BINS}
            yield {
                "crl_id": int(cid), "lat": lat, "lon": lon,
                "heading": rows["heading_deg"].to_numpy(float),
                "cp": rows[cp_col].to_numpy(float),
                "rmax": rows["rmax_km"].to_numpy(float),
                "vt": rows["trans_kmh"].to_numpy(float),
                "gaussW": rows["gaussW"].to_numpy(float),
                "year": rows["year"].to_numpy(float),
                "mean": mean, "stdv": stdv,
            }

    def _process_basin(self, basin: str) -> Optional[BasinResult]:
        cfg = self.cfg
        sel_path = sca_source.locate_selection(
            basin, cfg.selection_file_for(basin), cfg.input_dir, cfg.sca_outputs_dir)
        dsrr_path = sca_source.locate_dsrr(
            basin, cfg.dsrr_file_for(basin), cfg.input_dir, cfg.sca_outputs_dir)
        usecols = _SELECTION_COLS + [cfg.cp_source]
        selection = sca_source.load_selection(sel_path, usecols=usecols)
        dsrr = sca_source.load_dsrr(dsrr_path, bins=INTENSITY_BINS)
        tag = sca_source.version_tag(sel_path, basin)

        # CRLs present in both the selection and the DSRR arrays.
        crl_ids = [c for c in dsrr.crl_ids.astype(int)
                   if c in set(selection["crl_id"].astype(int))]
        print(f"[jdm] {basin}: {len(crl_ids):,} CRLs  "
              f"(selection {sel_path.name}, dsrr {dsrr_path.name}); "
              f"cp_source={cfg.cp_source}, n_boot={cfg.n_boot}")

        params = dict(
            seed=cfg.seed, ref_pressure=cfg.ref_pressure, start_year=cfg.start_year,
            min_dp=cfg.min_dp, dp_low=cfg.dp_low, dp_med=cfg.dp_med,
            vt_clip=tuple(cfg.vt_clip), rmax_clip=tuple(cfg.rmax_clip),
            n_boot=cfg.n_boot)
        bundles = list(self._bundles(selection, dsrr, crl_ids, cfg.cp_source))

        n_jobs = self._resolve_jobs()
        worker = partial(_process_crl, params=params)
        if n_jobs == 1:
            results = [worker(b) for b in bundles]
        else:
            # With the C++ kernel the heavy bootstrap releases the GIL, so threads
            # parallelize it without the Windows process-spawn / pickling overhead;
            # without it, the fit is pure Python, so use processes.
            pool = ThreadPoolExecutor if CPP_KERNEL_AVAILABLE else ProcessPoolExecutor
            with pool(max_workers=n_jobs) as ex:
                results = list(ex.map(worker, bundles))
        kern = "C++ kernel + threads" if CPP_KERNEL_AVAILABLE else "NumPy + processes"
        print(f"[jdm] {basin}: fit {len(results):,} CRLs (n_jobs={n_jobs}, {kern})")

        return self._write_basin(basin, tag, sel_path, dsrr_path, results)

    def _write_basin(self, basin, tag, sel_path, dsrr_path, results) -> BasinResult:
        cfg = self.cfg
        out = cfg.output_dir
        ncrl = len(results)

        marg_rows: List[dict] = []
        adj_rows: List[dict] = []
        cop = {b: {"tau": np.full((ncrl, _NPARAM, _NPARAM), np.nan),
                   "rho": np.full((ncrl, _NPARAM, _NPARAM), np.nan)}
               for b in INTENSITY_BINS}
        crl_ids, lats, lons = [], [], []

        for i, r in enumerate(results):
            crl_ids.append(r["crl_id"]); lats.append(r["lat"]); lons.append(r["lon"])
            for rec in r["records"]:
                marg_rows.append({"crl_id": r["crl_id"], "lat": r["lat"],
                                  "lon": r["lon"], **rec})
            for b in INTENSITY_BINS:
                cop[b]["tau"][i] = r["copula"][b]["tau"]
                cop[b]["rho"][i] = r["copula"][b]["rho"]
            # Adjusted per-storm rows (All bin) with a stratum label.
            adj_rows.extend(self._adjusted_rows(r["crl_id"], r["bins"]))

        marg_path = writer.write_marginals(marg_rows, out / f"jdm_marginals_{tag}.csv")
        cop_path = writer.write_copula(cop, crl_ids, lats, lons,
                                       out / f"jdm_copula_{tag}.npz")
        adj_path = writer.write_adjusted(adj_rows, out / f"jdm_adjusted_{tag}.csv")
        print(f"[jdm] {basin}: wrote marginals ({len(marg_rows):,} rows), copula, "
              f"adjusted -> {out}")

        n_plots = 0
        if cfg.make_plots:
            n_plots = self._render_plots(basin, results)

        return BasinResult(
            basin=basin, selection_file=sel_path, dsrr_file=dsrr_path,
            n_crls=ncrl, n_records=len(marg_rows), marginals_path=marg_path,
            copula_path=cop_path, adjusted_path=adj_path, n_plots=n_plots)

    def _adjusted_rows(self, crl_id, bins):
        """One row per All-bin storm, tagged with its stratum (high/med/low)."""
        cfg = self.cfg
        all_d = bins["all"]
        for hd, dp, rm, vt in all_d:
            stratum = ("high" if dp >= cfg.dp_med
                       else "med" if dp >= cfg.dp_low else "low")
            yield {"crl_id": crl_id, "stratum": stratum,
                   "Hd": hd, "Dp": dp, "Rmax": rm, "Vt": vt}

    def _render_plots(self, basin, results) -> int:
        cfg = self.cfg
        base = Path(cfg.plot_dir) if cfg.plot_dir else (Path(cfg.output_dir) / "plots")
        (base / f"marginals_{basin}").mkdir(parents=True, exist_ok=True)
        (base / f"copula_{basin}").mkdir(parents=True, exist_ok=True)
        worker = partial(_plot_one, basin=basin, base_dir=str(base))
        n_jobs = self._resolve_jobs()
        try:
            if n_jobs == 1 or len(results) < 8:
                n = sum(worker(r) for r in results)
            else:
                # matplotlib holds the GIL and C-level state, so render in processes.
                with ProcessPoolExecutor(max_workers=n_jobs) as ex:
                    n = sum(ex.map(worker, results))
            print(f"[jdm] {basin}: wrote {n:,} figures (marginals + copula) -> {base}")
        except RuntimeError as exc:                    # matplotlib missing
            print(f"[jdm] {basin}: plots skipped ({exc})")
            return 0
        return n

    def _resolve_jobs(self) -> int:
        j = self.cfg.n_jobs
        j = max(1, (os.cpu_count() or 2) - 1) if (j is None or j == 0) else max(1, int(j))
        # Windows ProcessPoolExecutor caps the worker count at 61.
        if os.name == "nt":
            j = min(j, 60)
        return j

    def run(self) -> JDMResult:
        if self.cfg.storm_type == "etc":
            raise NotImplementedError(
                "storm_type='etc' is a placeholder and not yet implemented; use 'tc'.")
        result = JDMResult()
        for basin in self.cfg.basins:
            r = self._process_basin(basin)
            if r is not None:
                result.results[basin] = r
        return result
