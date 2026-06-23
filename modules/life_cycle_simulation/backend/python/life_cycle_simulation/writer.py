"""writer - output writers for the life-cycle simulation (catalog + summary).

Author : Norberto C. Nadal-Caraballo, PhD  <norberto.c.nadal-caraballo@usace.army.mil>

Two tables per CRL: the full synthetic TC catalog (one row per event) and a
per-realization summary (TC counts overall and by stratum). Post-processing only;
no simulation logic here.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from life_cycle_simulation.srr_source import STRATA


def write_catalog(catalog: pd.DataFrame, path) -> Path:
    """Write the synthetic TC catalog (one row per event)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    catalog.to_csv(path, index=False)
    return path


def build_summary(catalog: pd.DataFrame, n_realizations: int) -> pd.DataFrame:
    """Per-realization TC counts overall and per stratum (zero-filled for quiet years).

    Every realization 1..R appears, including those that produced no TCs, so the
    table is a complete sample of the annualized count distribution.
    """
    base = pd.DataFrame({"realization": np.arange(1, n_realizations + 1, dtype=np.int32)})
    if len(catalog) == 0:
        for col in ["n_tc", *(f"n_{s}" for s in STRATA)]:
            base[col] = 0
        return base
    total = (catalog.groupby("realization").size()
             .rename("n_tc").reset_index())
    out = base.merge(total, on="realization", how="left")
    for s in STRATA:
        cnt = (catalog[catalog["intensity"] == s].groupby("realization").size()
               .rename(f"n_{s}").reset_index())
        out = out.merge(cnt, on="realization", how="left")
    count_cols = ["n_tc", *(f"n_{s}" for s in STRATA)]
    out[count_cols] = out[count_cols].fillna(0).astype(np.int64)
    return out


def write_summary(summary: pd.DataFrame, path) -> Path:
    """Write the per-realization summary table."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(path, index=False)
    return path
