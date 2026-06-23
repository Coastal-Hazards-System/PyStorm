"""calibration - estimate the correlation parameters from historical annual counts.

Author : Norberto C. Nadal-Caraballo, PhD  <norberto.c.nadal-caraballo@usace.army.mil>

When the serial-correlation layer is enabled, its parameters default to values
calibrated from each CRL's historical annual TC counts (the SCA selection within the
radius of influence), rather than hand-set: the overdispersion from the count Fano
factor and the AR(1) terms from the lag-1/lag-2 count autocorrelation. Any parameter
the operator supplies explicitly overrides its estimate. A sparse, low-rate CRL
typically calibrates to ~0 (the Poisson baseline), which is the appropriate result.
No simulation logic here.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd


def crl_annual_counts(selection: pd.DataFrame, crl_id: int, *, radius_km: float,
                      start_year: int) -> np.ndarray:
    """Historical annual TC counts within ``radius_km`` of one CRL (zero-filled years).

    Counts the selected TCs whose closest approach is within the radius, per season
    from ``start_year`` to the last season in the table, filling quiet years with 0.
    """
    sub = selection[(selection["crl_id"] == crl_id)
                    & (selection["dist"] <= radius_km)
                    & (selection["year"] >= start_year)]
    if sub.empty:
        return np.zeros(0)
    years = np.arange(int(start_year), int(selection["year"].max()) + 1)
    return (sub.groupby("year").size().reindex(years, fill_value=0)
            .to_numpy().astype(float))


def calibrate_correlation(counts: np.ndarray, *, ar_phi: Optional[float] = None,
                          ar_beta: Optional[float] = None,
                          overdispersion: Optional[float] = None,
                          default_phi: float = 0.5, beta_max: float = 2.0) -> Dict:
    """Resolve (ar_phi, ar_beta, overdispersion), estimating any left as None.

    Estimators (mean = lambda, Fano = var/mean, r_k = lag-k count autocorrelation):
      * The total relative rate-variance is disp = max(0, (Fano - 1)/mean), which the
        Cox model splits as disp = ar_beta^2 + overdispersion (serial + i.i.d.). The
        AR(1) count autocorrelation r1 = mean * ar_beta^2 * ar_phi / Fano gives the
        serial part ar_beta^2 = r1 * Fano / (mean * ar_phi); it is CAPPED by disp, and
        overdispersion takes the remainder disp - ar_beta^2. Because beta^2 cannot
        exceed the total dispersion, a Poisson-like series (Fano ~ 1, so disp ~ 0)
        forces ar_beta ~ 0 even if its sparse-count lag-1 ACF is spuriously nonzero.
      * ar_phi = clip(r2 / r1) (the geometric decay) when both lags are clearly
        positive, else ``default_phi``.
    Explicit (non-None) arguments override their estimate. Returns the resolved
    parameters plus the realized mean/Fano/acf1 used.
    """
    counts = np.asarray(counts, float)
    n = counts.size
    mean = float(counts.mean()) if n else 0.0
    fano = float(counts.var() / mean) if mean > 0 else 1.0

    def acf(k: int) -> float:
        if n <= k + 1:
            return 0.0
        a, b = counts[:-k], counts[k:]
        if a.std() == 0 or b.std() == 0:
            return 0.0
        return float(np.corrcoef(a, b)[0, 1])

    r1, r2 = acf(1), acf(2)
    # Total relative variance of the annual rate implied by the overdispersion,
    # disp = (Fano - 1)/mean = beta^2 + nu. The serial (beta^2) and i.i.d. (nu)
    # components must sum to it, so the AR(1) variance is bounded by the total:
    # a Cox process cannot have serial correlation without overdispersion, so a
    # Poisson-like series (Fano ~ 1) forces beta ~ 0 regardless of a noisy lag-1 ACF.
    disp = max(0.0, (fano - 1.0) / mean) if mean > 0 else 0.0
    phi_auto = float(np.clip(r2 / r1, 0.0, 0.95)) if (r1 > 0.05 and r2 > 0) else default_phi
    if r1 > 0 and mean > 0:
        beta2_serial = r1 * fano / (mean * max(phi_auto, 1e-3))     # from lag-1 ACF
    else:
        beta2_serial = 0.0
    beta2 = min(beta2_serial, disp, beta_max ** 2)                  # capped by the total
    beta_auto = float(np.sqrt(max(beta2, 0.0)))
    nu_auto = max(0.0, disp - beta2)                               # i.i.d. remainder

    return {
        "ar_phi": phi_auto if ar_phi is None else float(ar_phi),
        "ar_beta": beta_auto if ar_beta is None else float(ar_beta),
        "overdispersion": nu_auto if overdispersion is None else float(overdispersion),
        "mean": mean, "fano": fano, "acf1": r1, "n_years": n,
    }
