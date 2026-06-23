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
    fano: float = 1.0              # realized variance/mean of annual counts
    acf1: float = 0.0              # realized lag-1 autocorrelation of annual counts


def poisson_rate(srr_all: float, radius_km: float) -> float:
    """Poisson rate lambda (TC/yr) = SRR(all) [TC/km/yr] x the 2R-km band."""
    return float(srr_all) * 2.0 * float(radius_km)


def draw_counts(lam: float, n_real: int, n_years: int, *, correlation: bool = False,
                ar_phi: float = 0.0, ar_beta: float = 0.0,
                overdispersion: float = 0.0, rng) -> np.ndarray:
    """Annual TC counts on the (R, Y) grid.

    correlation=False is the independent Poisson(lambda) baseline. When True, the
    rate is modulated by a per-realization AR(1) latent climate state (mean-
    preserving) and/or an i.i.d. Gamma overdispersion, so active and quiet years
    cluster. The annual mean stays lambda, so the overall rate is unchanged.
    """
    if not correlation or (ar_beta == 0.0 and overdispersion == 0.0):
        return rng.poisson(lam, size=(n_real, n_years))

    rate = np.full((n_real, n_years), float(lam))
    if ar_beta != 0.0:
        # Stationary AR(1) state with unit marginal variance, per realization.
        s = np.empty((n_real, n_years))
        s[:, 0] = rng.standard_normal(n_real)
        innov = np.sqrt(1.0 - ar_phi * ar_phi)
        for y in range(1, n_years):
            s[:, y] = ar_phi * s[:, y - 1] + innov * rng.standard_normal(n_real)
        rate *= np.exp(ar_beta * s - 0.5 * ar_beta * ar_beta)   # mean-preserving
    if overdispersion > 0.0:
        # Gamma multiplier with mean 1 and variance = overdispersion (-> NegBin).
        k = 1.0 / overdispersion
        rate *= rng.gamma(shape=k, scale=1.0 / k, size=(n_real, n_years))
    return rng.poisson(rate)


def _count_diagnostics(counts: np.ndarray):
    """Realized Fano factor and lag-1 autocorrelation of the annual counts."""
    flat = counts.reshape(-1).astype(float)
    mean = flat.mean()
    var = flat.var()
    fano = float(var / mean) if mean > 0 else 1.0
    # Lag-1 autocorrelation pooled within realizations (year y vs y-1).
    a, b = counts[:, :-1].reshape(-1).astype(float), counts[:, 1:].reshape(-1).astype(float)
    if a.size and a.std() > 0 and b.std() > 0:
        acf1 = float(np.corrcoef(a, b)[0, 1])
    else:
        acf1 = 0.0
    return fano, acf1


def add_sequencing(catalog: pd.DataFrame, n_years: int) -> pd.DataFrame:
    """Add the chronological event timeline to a catalog.

    Adds ``event_time`` (continuous years = (year-1) + (doy-1)/365), sorts each
    realization chronologically, and adds ``seq`` (per-realization chronological
    ordinal) and ``wait_yr`` (inter-arrival from the previous event; NaN for the
    first event of a realization). Returns the reordered catalog.
    """
    if len(catalog) == 0:
        for col in ("event_time", "seq", "wait_yr"):
            catalog[col] = np.empty(0, float if col != "seq" else np.int32)
        return catalog
    cat = catalog.copy()
    cat["event_time"] = (cat["year"] - 1).astype(float) + (cat["doy"] - 1) / 365.0
    cat = cat.sort_values(["realization", "event_time"], kind="stable").reset_index(drop=True)
    cat["seq"] = cat.groupby("realization").cumcount().astype(np.int32) + 1
    wait = cat.groupby("realization")["event_time"].diff()
    cat["wait_yr"] = wait.to_numpy()
    return cat


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


def simulate(srr: CRLSrr, *, radius_km: float, sim_years: int, n_realizations: int,
             rng, correlation: bool = False, ar_phi: float = 0.0, ar_beta: float = 0.0,
             overdispersion: float = 0.0, sequencing: bool = True) -> SimOutput:
    """Run the life-cycle Monte-Carlo for one CRL and return its synthetic catalog.

    ``correlation`` adds serial correlation / overdispersion to the annual counts;
    ``sequencing`` adds the chronological event timeline. Both default to the
    independent-Poisson, with sequencing on.

    Returns
    -------
    SimOutput with ``catalog`` columns: realization, year, event (in-year ordinal),
    intensity (low/med/high), doy, month, day, and (when sequencing) event_time,
    seq (chronological order), wait_yr (inter-arrival).
    """
    lam = poisson_rate(srr.annual["all"], radius_km)
    p = stratum_probs(srr.annual)

    counts = draw_counts(lam, n_realizations, sim_years, correlation=correlation,
                         ar_phi=ar_phi, ar_beta=ar_beta, overdispersion=overdispersion,
                         rng=rng)
    fano, acf1 = _count_diagnostics(counts)
    realization, year, event, n = _expand_counts(counts, sim_years)

    if n == 0:                                    # no activity (e.g. inland CRL)
        empty = pd.DataFrame({
            "realization": np.empty(0, np.int32), "year": np.empty(0, np.int32),
            "event": np.empty(0, np.int32), "intensity": np.empty(0, object),
            "doy": np.empty(0, np.int32), "month": np.empty(0, np.int32),
            "day": np.empty(0, np.int32)})
        if sequencing:
            empty = add_sequencing(empty, sim_years)
        return SimOutput(catalog=empty, lam=lam, p=p, n_events=0, fano=fano, acf1=acf1)

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
    if sequencing:
        catalog = add_sequencing(catalog, sim_years)
    return SimOutput(catalog=catalog, lam=lam, p=p, n_events=int(n),
                     fano=fano, acf1=acf1)
