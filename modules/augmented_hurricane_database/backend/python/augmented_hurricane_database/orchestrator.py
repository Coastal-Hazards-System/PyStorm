"""Orchestrator - resolve sources, parse, and write one CSV per basin.

Author / POC : Norberto C. Nadal-Caraballo, PhD  <norberto.c.nadal-caraballo@usace.army.mil>
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import pandas as pd

from augmented_hurricane_database import ebtrk as _ebtrk
from augmented_hurricane_database.config import AHDConfig
from augmented_hurricane_database.parser import HURDAT2
from augmented_hurricane_database.sources import resolve_source
from augmented_hurricane_database.writer import write_csv, write_metrics, write_parquet

# Recover the record's end year from an NHC filename (…-<startyr>-<endyr>-<stamp>.txt).
_END_YEAR = re.compile(r"-(\d{4})-(\d{4})-\d{6,8}\.txt$")


@dataclass
class BasinResult:
    basin: str
    source_file: Path
    csv_path: Path
    parquet_path: Path | None
    n_storms: int
    n_rows: int
    ebtrk_source: Path | None = None
    n_rmax_filled: int = 0
    n_pmin_gpm: int = 0
    n_rmax_gpm: int = 0


@dataclass
class AHDResult:
    results: Dict[str, BasinResult] = field(default_factory=dict)

    @property
    def csv_paths(self) -> List[Path]:
        return [r.csv_path for r in self.results.values()]


class AHDOrchestrator:
    """Builds the HURDAT-like CSV(s) for the configured basins."""

    def __init__(self, config: AHDConfig) -> None:
        self.cfg = config

    def _end_year(self, source: Path) -> str:
        m = _END_YEAR.search(source.name)
        return m.group(2) if m else "latest"

    def _process_basin(self, basin: str) -> BasinResult:
        cfg = self.cfg
        source = resolve_source(
            basin,
            download_latest=cfg.download,
            input_dir=cfg.input_dir,
            explicit_file=cfg.file_for(basin),
            # Also look under data/ and beside the module (legacy raw file).
            extra_search_dirs=(cfg.input_dir.parent, cfg.input_dir.parent.parent),
            overwrite=cfg.overwrite,
        )
        print(f"[ahd] {basin}: parsing {source.name}")

        df = HURDAT2(source, basin=basin).to_dataframe()
        n_storms = int(df["tc_no"].nunique()) if not df.empty else 0

        ebtrk_source: Path | None = None
        n_filled = 0
        if cfg.append_ebtrk_rmax:
            ebtrk_source, n_filled = self._backfill_rmax(df, basin)

        n_pmin_gpm = n_rmax_gpm = 0
        gpm_reports = None
        if cfg.impute_gpm:
            df, n_pmin_gpm, n_rmax_gpm, gpm_reports = self._impute_gpm(df, basin)

        stem = cfg.output_stem.format(basin=basin, end_year=self._end_year(source))
        csv_path = write_csv(df, cfg.output_dir / f"{stem}.csv")
        print(f"[ahd] {basin}: wrote {len(df):,} rows / {n_storms:,} storms -> {csv_path}")

        if gpm_reports is not None:
            metrics = self._gpm_metrics(basin, source.name, len(df), n_storms, gpm_reports)
            metrics_path = write_metrics(metrics, cfg.output_dir / f"{stem}_gpm_metrics.json")
            print(f"[ahd] {basin}: wrote GP-metamodel LOOCV metrics -> {metrics_path}")

        parquet_path = None
        if cfg.write_parquet:
            parquet_path = write_parquet(df, cfg.output_dir / f"{stem}.parquet")
            print(f"[ahd] {basin}: wrote Parquet -> {parquet_path}")

        return BasinResult(
            basin=basin,
            source_file=source,
            csv_path=csv_path,
            parquet_path=parquet_path,
            n_storms=n_storms,
            n_rows=len(df),
            ebtrk_source=ebtrk_source,
            n_rmax_filled=n_filled,
            n_pmin_gpm=n_pmin_gpm,
            n_rmax_gpm=n_rmax_gpm,
        )

    def _backfill_rmax(self, df: pd.DataFrame, basin: str) -> tuple[Path | None, int]:
        """Fill missing rmax_km in ``df`` (in place) from EBTRK.

        Atlantic uses the AL file; Pacific (HURDAT nepac = EP + CP storms) uses
        both the EP and CP files.
        """
        cfg = self.cfg
        if basin not in _ebtrk.EBTRK_BASINS:
            print(f"[ahd] {basin}: EBTRK Rmax backfill skipped (no EBTRK source).")
            return None, 0

        n_missing = int(df["rmax_km"].isna().sum())
        sources = _ebtrk.resolve_ebtrk_sources(
            basin,
            download=cfg.download,
            input_dir=cfg.input_dir,
            explicit_files=cfg.ebtrk_file,
            # data/, module root, and the legacy ebtrk/ reference folder.
            extra_search_dirs=(cfg.input_dir.parent,
                               cfg.input_dir.parent.parent,
                               cfg.input_dir.parent.parent / "ebtrk"),
            overwrite=cfg.overwrite,
        )
        names = ", ".join(p.name for p in sources)
        print(f"[ahd] {basin}: EBTRK Rmax from {names} "
              f"({n_missing:,} HURDAT rows missing Rmax)")
        ebtrk_df = _ebtrk.parse_ebtrk_files(sources)
        filled, n_filled = _ebtrk.fill_missing_rmax(df, ebtrk_df)
        df["rmax_km"] = filled["rmax_km"].to_numpy()
        print(f"[ahd] {basin}: filled {n_filled:,} Rmax values from EBTRK")
        return sources[0] if sources else None, n_filled

    def _gpm_metrics(self, basin, source_name, n_rows, n_storms, reports) -> dict:
        """Build the per-run GP-metamodel metrics record (LOOCV scores + settings)."""
        cfg = self.cfg

        def target_block(rep, full, small):
            def model(name, n_train):
                lc = rep.loocv.get(name, {})
                return {"n_train": int(n_train),
                        "loocv_r2": lc.get("r2"), "loocv_rmse": lc.get("rmse")}
            return {"n_missing": int(rep.n_missing), "n_filled": int(rep.n_filled),
                    "models": {full: model(full, rep.n_train_full),
                               small: model(small, rep.n_train_small)}}

        return {
            "basin": basin,
            "source_file": source_name,
            "n_rows": int(n_rows),
            "n_storms": int(n_storms),
            "central_pressure": target_block(reports["pmin"], "Cp6", "Cp3"),
            "radius_max_wind": target_block(reports["rmax"], "Rm7", "Rm4"),
            "settings": {
                "physical_mean": cfg.gpm_physical_mean, "log_rmax": cfg.gpm_log_rmax,
                "cp6_vecchia": cfg.gpm_cp6_vecchia, "cp3_vecchia": cfg.gpm_cp3_vecchia,
                "rm7_vecchia": cfg.gpm_rm7_vecchia, "rm4_vecchia": cfg.gpm_rm4_vecchia,
                "cp_max_support": cfg.gpm_cp_max_support, "cp_neighbors": cfg.gpm_cp_neighbors,
                "cp_n_cal": cfg.gpm_cp_n_cal, "cp_n_lhs": cfg.gpm_cp_n_lhs,
                "rmax_max_support": cfg.gpm_rmax_max_support, "rmax_neighbors": cfg.gpm_rmax_neighbors,
                "rmax_n_cal": cfg.gpm_rmax_n_cal, "rmax_n_lhs": cfg.gpm_rmax_n_lhs,
                "retrain": cfg.gpm_retrain,
            },
            "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }

    def _impute_gpm(self, df: pd.DataFrame, basin: str):
        """Fill remaining missing pmin_hpa and rmax_km with the GP metamodels.

        Returns (df, n_pmin_filled, n_rmax_filled, reports). ``reports`` carries the
        per-model LOOCV scores, written next to the output as a metrics JSON.
        """
        from .gp_metamodel import impute_all
        cfg = self.cfg
        print(f"[ahd] {basin}: GP-metamodel imputation "
              f"(pmin missing={int(df['pmin_hpa'].isna().sum()):,}, "
              f"rmax missing={int(df['rmax_km'].isna().sum()):,})")
        df, reports = impute_all(
            df, physical_mean=cfg.gpm_physical_mean,
            log_rmax=cfg.gpm_log_rmax, parallel=cfg.gpm_parallel,
            cp6_vecchia=cfg.gpm_cp6_vecchia, cp3_vecchia=cfg.gpm_cp3_vecchia,
            rm7_vecchia=cfg.gpm_rm7_vecchia, rm4_vecchia=cfg.gpm_rm4_vecchia,
            cp_max_support=cfg.gpm_cp_max_support, cp_neighbors=cfg.gpm_cp_neighbors,
            cp_n_cal=cfg.gpm_cp_n_cal, cp_n_lhs=cfg.gpm_cp_n_lhs,
            rmax_max_support=cfg.gpm_rmax_max_support,
            rmax_neighbors=cfg.gpm_rmax_neighbors,
            rmax_n_cal=cfg.gpm_rmax_n_cal, rmax_n_lhs=cfg.gpm_rmax_n_lhs,
            basin=basin, model_dir=cfg.gpm_model_dir, retrain=cfg.gpm_retrain)
        return df, int(reports["pmin"].n_filled), int(reports["rmax"].n_filled), reports

    def run(self) -> AHDResult:
        result = AHDResult()
        for basin in self.cfg.basins:
            result.results[basin] = self._process_basin(basin)
        return result
