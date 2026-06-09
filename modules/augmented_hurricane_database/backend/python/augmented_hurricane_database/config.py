"""Configuration model for the augmented hurricane database module.

Author / POC : Norberto C. Nadal-Caraballo, PhD  <norberto.c.nadal-caraballo@usace.army.mil>

A single validated ``AHDConfig`` carries the operator options from the launcher
to the orchestrator. Paths are coerced to ``Path``; ``basins`` is normalized to
the canonical lower-case basin list.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Union

from pydantic import BaseModel, field_validator

from augmented_hurricane_database.sources import BASINS


class AHDConfig(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    # Which basins to build: any of "atlantic", "pacific". "both" expands to all.
    basins: List[str] = ["atlantic", "pacific"]

    # Fetch the newest file from NHC (True) or use a local file (False).
    download: bool = True
    # Re-download even if the dated file already exists locally.
    overwrite: bool = False

    # Operator-pinned input files (skip discovery). Absolute, or relative to
    # input_dir. One per basin is keyed by name; None -> auto-resolve.
    atlantic_file: Optional[Union[str, Path]] = None
    pacific_file: Optional[Union[str, Path]] = None

    # I/O layout.
    input_dir: Path = Path("data/inputs")
    output_dir: Path = Path("data/outputs")

    # Output: write CSV (always) and optionally Parquet alongside it.
    write_parquet: bool = False
    # Output filename stem per basin; {basin} and {end_year} are substituted.
    output_stem: str = "hurdat2_{basin}_{end_year}"

    # EBTRK backfill: fill missing rmax_km from the Extended Best Track dataset.
    # Atlantic uses the AL file; Pacific uses both the EP and CP files.
    append_ebtrk_rmax: bool = False
    # Optional explicit EBTRK file(s) to use instead of automatic resolution: a
    # single path or a list of paths (absolute, or relative to input_dir). None
    # resolves the correct file(s) per basin automatically.
    ebtrk_file: Optional[Union[str, Path, list]] = None

    # GP-metamodel imputation: fill still-missing pmin_hpa and rmax_km with the
    # Gaussian-process metamodels (self-trained on the observed rows). Runs after
    # any EBTRK backfill.
    impute_gpm: bool = False
    # GP-metamodel quality/speed upgrades (all on by default):
    gpm_vecchia: bool = True          # NNGP: predict from all data, not just the support
    gpm_physical_mean: bool = True    # wind–pressure / lat·deficit kriging trend
    gpm_log_rmax: bool = True         # fit the Rmax models in log space (lognormal target)
    gpm_parallel: bool = True         # train the two models per target concurrently
    # Per-target method settings (empirically tuned — Cp is smooth/long-range and
    # wants more calibration support; Rmax is short-range/noisy and wants a small
    # conditioning set). The physical-mean trend leaves a short-range residual, so
    # neither target benefits from a large NNGP neighbour set.
    gpm_cp_max_support: int = 6000    # Cp: support points for θ / trend-β calibration
    gpm_cp_neighbors: int = 30        # Cp: NNGP conditioning-set size
    gpm_rmax_max_support: int = 3000  # Rmax: support points
    gpm_rmax_neighbors: int = 10      # Rmax: NNGP conditioning-set size

    @field_validator("basins", mode="before")
    @classmethod
    def _expand_basins(cls, v):
        if isinstance(v, str):
            v = [v]
        out: List[str] = []
        for b in v:
            b = str(b).strip().lower()
            if b == "both":
                out.extend(BASINS)
            else:
                out.append(b)
        # De-duplicate, preserve canonical order.
        seen = set()
        ordered = [b for b in BASINS if b in out and not (b in seen or seen.add(b))]
        unknown = sorted(set(out) - set(BASINS))
        if unknown:
            raise ValueError(f"Unknown basin(s) {unknown}; expected {BASINS} or 'both'.")
        if not ordered:
            raise ValueError("No basins selected.")
        return ordered

    @field_validator("input_dir", "output_dir", mode="before")
    @classmethod
    def _as_path(cls, v):
        return Path(v)

    def file_for(self, basin: str) -> Optional[Union[str, Path]]:
        """Operator-pinned file for a basin, if any."""
        return {"atlantic": self.atlantic_file, "pacific": self.pacific_file}.get(basin)
