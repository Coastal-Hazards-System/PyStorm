"""config - configuration model for the augmented_hurricane_database module.

Author : Norberto C. Nadal-Caraballo, PhD  <norberto.c.nadal-caraballo@usace.army.mil>

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

    # Per-basin HURDAT2 download URL override. When set (and download=True), the
    # exact URL is fetched instead of auto-discovering the newest NHC file; None
    # -> discover. Ignored when download=False (no network). A local
    # atlantic_file/pacific_file pin still wins over the URL.
    atlantic_url: Optional[str] = None
    pacific_url: Optional[str] = None

    # I/O layout.
    input_dir: Path = Path("data/inputs")
    output_dir: Path = Path("data/outputs")

    # Output: write CSV (always) and optionally Parquet alongside it.
    write_parquet: bool = False
    # Output filename stem per basin. Substituted fields: {basin}, {start_year},
    # {end_year}, and {created} (the NHC source-file date as YYYYMMDD). Default
    # marks the file as augmented HURDAT2 and carries the source record span and
    # NHC vintage, e.g. augmented_hurdat2_atlantic_1851-2025_20260227.csv.
    output_stem: str = "augmented_hurdat2_{basin}_{start_year}-{end_year}_{created}"

    # EBTRK backfill: fill missing rmax_km from the Extended Best Track dataset.
    # Atlantic uses the AL file; Pacific uses both the EP and CP files.
    append_ebtrk_rmax: bool = False
    # Optional explicit EBTRK file(s) to use instead of automatic resolution: a
    # single path or a list of paths (absolute, or relative to input_dir). None
    # resolves the correct file(s) per basin automatically.
    ebtrk_file: Optional[Union[str, Path, list]] = None
    # Per-file EBTRK download URL overrides, one per cyclone-id code. When set
    # (and download=True), the exact URL is fetched instead of discovering the
    # newest file from the CIRA listing; None -> discover. Ignored when
    # download=False, and superseded by an ebtrk_file local pin. The Atlantic
    # uses AL; the Pacific uses EP and CP.
    ebtrk_al_url: Optional[str] = None
    ebtrk_ep_url: Optional[str] = None
    ebtrk_cp_url: Optional[str] = None

    # Per-TC imputation diagnostic plots (off by default). One PNG per storm: the
    # GP-metamodel-completed series as a line (GPM) with the observed values as red
    # dots (Obs), for central pressure (cp) and radius of max wind (rmax). Needs
    # impute_gpm=True and matplotlib. A master switch can flip all four at once:
    # plot_imputation=True -> all on, False -> all off, None -> use the four
    # per-(basin, target) flags below.
    plot_imputation: Optional[bool] = None
    plot_atlantic_cp: bool = False
    plot_atlantic_rmax: bool = False
    plot_pacific_cp: bool = False
    plot_pacific_rmax: bool = False
    # Where plots are written; None -> output_dir / "plots".
    plot_dir: Optional[Union[str, Path]] = None
    # Worker processes for plotting; None/0 -> auto (cores-1, capped), 1 -> serial.
    plot_jobs: Optional[int] = None

    # GP-metamodel imputation: fill still-missing pmin_hpa and rmax_km with the
    # Gaussian-process metamodels (self-trained on the observed rows). Runs after
    # any EBTRK backfill.
    impute_gpm: bool = False
    # GP-metamodel quality/speed upgrades (all on by default):
    gpm_physical_mean: bool = True    # wind-pressure / lat·deficit kriging trend
    gpm_log_rmax: bool = True         # fit the Rmax models in log space (lognormal target)
    gpm_parallel: bool = True         # train the two models per target concurrently
    # Per-MODEL solver: True = nearest-neighbor GP (NNGP, predict from all data);
    # False = exact full GP over the support (denser/slower, needs deeper
    # calibration). Set per model so, e.g., Cp6 can use the full GP while Cp3 uses
    # the NNGP. The NNGP default is recommended for cost and generalization.
    gpm_cp6_vecchia: bool = True      # central pressure, full-feature model
    gpm_cp3_vecchia: bool = True      # central pressure, reduced model
    gpm_rm7_vecchia: bool = True      # radius of max wind, full-feature model
    gpm_rm4_vecchia: bool = True      # radius of max wind, reduced model
    # Per-target method settings (empirically tuned - Cp is smooth/long-range and
    # wants more calibration support; Rmax is short-range/noisy and wants a small
    # conditioning set). The physical-mean trend leaves a short-range residual, so
    # neither target benefits from a large NNGP neighbor set. n_cal is the
    # hyperparameter-calibration subset size; n_lhs the Latin-hypercube budget for
    # that search. The exact full GP needs a deeper n_cal (about 4000) and a larger
    # n_lhs to reach its best accuracy; the NNGP is fine at the defaults.
    gpm_cp_max_support: int = 6000    # Cp: support points for θ / trend-β calibration
    gpm_cp_neighbors: int = 30        # Cp: NNGP conditioning-set size
    gpm_cp_n_cal: int = 4000          # Cp: calibration-subset size (deep: lifts Cp6 above the MATLAB)
    gpm_cp_n_lhs: int = 250           # Cp: Latin-hypercube budget (>=250 converges Cp6 calibration)
    gpm_rmax_max_support: int = 8000  # Rmax: support points (large; the EBTRK-augmented set is ~15k)
    gpm_rmax_neighbors: int = 30      # Rmax: NNGP conditioning-set size
    gpm_rmax_n_cal: int = 4000        # Rmax: calibration-subset size (deep: lifts Rm7/Rm4 above the MATLAB)
    gpm_rmax_n_lhs: int = 250         # Rmax: Latin-hypercube budget for calibration
    # Trained-model cache. When gpm_model_dir is set, each fitted model is saved
    # there as a compressed .npz keyed by basin, model, and a signature of the
    # settings and training data. gpm_retrain=False reuses a matching cached model
    # and skips training; gpm_retrain=True retrains regardless and overwrites.
    gpm_model_dir: Optional[Union[str, Path]] = None
    gpm_retrain: bool = False

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

    def url_for(self, basin: str) -> Optional[str]:
        """Operator-supplied HURDAT2 download URL override for a basin, if any."""
        return {"atlantic": self.atlantic_url, "pacific": self.pacific_url}.get(basin)

    def ebtrk_urls(self) -> dict:
        """Per-code EBTRK URL overrides keyed by cyclone-id code (AL/EP/CP)."""
        return {"AL": self.ebtrk_al_url, "EP": self.ebtrk_ep_url, "CP": self.ebtrk_cp_url}

    def plot_targets_for(self, basin: str) -> List[str]:
        """Which targets ('cp'/'rmax') to plot for a basin.

        The master ``plot_imputation`` switch, when not None, overrides the four
        per-(basin, target) flags (True -> both targets, False -> none).
        """
        flags = {
            ("atlantic", "cp"): self.plot_atlantic_cp,
            ("atlantic", "rmax"): self.plot_atlantic_rmax,
            ("pacific", "cp"): self.plot_pacific_cp,
            ("pacific", "rmax"): self.plot_pacific_rmax,
        }
        out: List[str] = []
        for target in ("cp", "rmax"):
            on = (self.plot_imputation if self.plot_imputation is not None
                  else flags.get((basin, target), False))
            if on:
                out.append(target)
        return out
