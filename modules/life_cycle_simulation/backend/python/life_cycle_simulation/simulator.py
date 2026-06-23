"""simulator - the vectorized life-cycle Monte-Carlo for one CRL.

Author : Norberto C. Nadal-Caraballo, PhD  <norberto.c.nadal-caraballo@usace.army.mil>

Given one CRL's driving SRR (Poisson rate, stratum split, per-stratum day-of-year
pmf), draws a synthetic catalog of tropical cyclones over ``n_realizations`` x
``sim_years`` of life cycle. Everything is drawn in one pass over the whole
(realization x year) grid, so cost scales with the number of TCs produced, not
with loops over years or realizations.

Pipeline (all arrays, no Python loops over events):
  1. counts ~ Poisson(lambda) on the (R, Y) grid; lambda = SRR(all) * 2 * radius.
  2. expand the counts to one row per TC (its realization and year).
  3. stratum ~ Categorical(p_low, p_med, p_high) from the annual stratum SRRs.
  4. day-of-year ~ the chosen stratum's day pmf; map doy -> (month, day).
No engine: the work is light and fully NumPy-vectorized (CyHAN: pure-Python module).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from life_cycle_simulation.calendar365 import doy_to_day, doy_to_month
from life_cycle_simulation.srr_source import STRATA, CRLSrr


@dataclass
class SimOutput:
    """The synthetic catalog plus the rate/split actually used."""
    catalog: pd.DataFrame          # one row per synthetic TC
    lam: float                     # Poisson rate lambda (TC/yr) used
    p: np.ndarray                  # [3] stratum probabilities (low/med/high)
    n_events: int                  # total TCs across all realizations


def poisson_rate(srr_all: float, radius_km: float) -> float:
    """Poisson rate lambda (TC/yr) = SRR(all) [TC/km/yr] x the 2R-km band."""
    return float(srr_all) * 2.0 * float(radius_km)


def stratum_probs(annual: dict) -> np.ndarray:
    """Normalized [p_low, p_med, p_high] from the annual stratum SRRs.

    Falls back to the marginal ``all`` rate if low+med+high is zero (degenerate),
    and to an even split if every stratum is zero so a draw never divides by zero.
    """
    s = np.array([annual[k] for k in STRATA], dtype=float)
    total = s.sum()
    if total > 0:
        return s / total
    return np.full(len(STRATA), 1.0 / len(STRATA))


def _expand_counts(counts: np.ndarray, n_years: int):
    """Expand an (R, Y) count grid to per-TC realization, year, and in-year index."""
    flat = counts.reshape(-1)                     # (R*Y,) cell counts, row-major
    n = int(flat.sum())
    cell = np.repeat(np.arange(flat.size), flat)  # (N,) source cell per TC
    realization = (cell // n_years).astype(np.int32) + 1
    year = (cell % n_years).astype(np.int32) + 1
    # In-year event ordinal 1..count: position within the run of each cell.
    starts = np.cumsum(flat) - flat               # first global TC index per cell
    event = (np.arange(n) - np.repeat(starts, flat)).astype(np.int32) + 1
    return realization, year, event, n


def _draw_doy(stratum: np.ndarray, doy_pmf: np.ndarray, rng) -> np.ndarray:
    """Inverse-CDF day-of-year (1..365) per TC, from its stratum's day pmf."""
    doy = np.ones(stratum.size, dtype=np.int32)
    for s in range(doy_pmf.shape[0]):
        mask = stratum == s
        ns = int(mask.sum())
        if ns == 0:
            continue
        cdf = np.cumsum(doy_pmf[s])
        if cdf[-1] <= 0:                          # stratum has no seasonal mass
            doy[mask] = rng.integers(1, 366, size=ns).astype(np.int32)
            continue
        cdf = cdf / cdf[-1]
        # searchsorted on a right-continuous CDF maps U(0,1) -> day index 0..364.
        idx = np.searchsorted(cdf, rng.random(ns), side="right")
        doy[mask] = np.clip(idx, 0, 364).astype(np.int32) + 1
    return doy


def simulate(srr: CRLSrr, *, radius_km: float, sim_years: int,
             n_realizations: int, rng) -> SimOutput:
    """Run the life-cycle Monte-Carlo for one CRL and return its synthetic catalog.

    Returns
    -------
    SimOutput with ``catalog`` columns:
        realization (1..R), year (1..Y), event (in-year ordinal), intensity
        (low/med/high), doy (1..365), month (1..12), day (1..31).
    """
    lam = poisson_rate(srr.annual["all"], radius_km)
    p = stratum_probs(srr.annual)

    counts = rng.poisson(lam, size=(n_realizations, sim_years))
    realization, year, event, n = _expand_counts(counts, sim_years)

    if n == 0:                                    # no activity (e.g. inland CRL)
        empty = pd.DataFrame({
            "realization": np.empty(0, np.int32), "year": np.empty(0, np.int32),
            "event": np.empty(0, np.int32), "intensity": np.empty(0, object),
            "doy": np.empty(0, np.int32), "month": np.empty(0, np.int32),
            "day": np.empty(0, np.int32)})
        return SimOutput(catalog=empty, lam=lam, p=p, n_events=0)

    # Stratum by inverse-CDF on the categorical: U(0,1) -> {0,1,2}.
    cum = np.cumsum(p)
    stratum = np.searchsorted(cum, rng.random(n), side="right")
    stratum = np.clip(stratum, 0, len(STRATA) - 1).astype(np.int32)

    doy = _draw_doy(stratum, srr.doy_pmf, rng)

    catalog = pd.DataFrame({
        "realization": realization,
        "year": year,
        "event": event,
        "intensity": np.asarray(STRATA, dtype=object)[stratum],
        "doy": doy,
        "month": doy_to_month(doy).astype(np.int32),
        "day": doy_to_day(doy).astype(np.int32),
    })
    return SimOutput(catalog=catalog, lam=lam, p=p, n_events=int(n))
