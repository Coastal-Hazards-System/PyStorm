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
        "mean": mean, "fano": fano, "acf1": r1, "n_years": n, "n_pooled": 1,
    }


def _series_moments(counts: np.ndarray):
    """(mean, var, gamma1, gamma2): mean, variance, lag-1 and lag-2 autocovariances."""
    c = np.asarray(counts, float)
    n = c.size
    if n == 0:
        return 0.0, 0.0, 0.0, 0.0
    mean = float(c.mean())

    def autocov(k: int) -> float:
        if n <= k + 1:
            return 0.0
        return float(np.mean((c[:-k] - mean) * (c[k:] - mean)))

    return mean, float(c.var()), autocov(1), autocov(2)


def calibrate_correlation_regional(series, weights=None, *, ar_phi: Optional[float] = None,
                                   ar_beta: Optional[float] = None,
                                   overdispersion: Optional[float] = None,
                                   default_phi: float = 0.5, beta_max: float = 2.0) -> Dict:
    """Calibrate the correlation parameters from a POOL of CRL annual-count series.

    The clustering signal (AMO/ENSO-like) is basin- to regional-scale and usually too
    weak to detect in one sparse, low-rate CRL. Pooling neighbouring CRLs sharpens it.
    For a shared multiplicative rate factor of relative variance v, each CRL has
    Var_i - mean_i = mean_i^2 * v and lag-k autocovariance gamma_k,i = mean_i^2 *
    v_serial * phi^k, so the pooled, mean^2-weighted moments give

        v       = sum_i w_i (Var_i - mean_i) / sum_i w_i mean_i^2  (total dispersion)
        phi     = sum_i w_i gamma_2,i / sum_i w_i gamma_1,i        (geometric decay)
        beta^2  = (sum_i w_i gamma_1,i / sum_i w_i mean_i^2) / phi, capped by v
        nu      = v - beta^2

    ``weights`` (aligned with ``series``, default all 1) multiply each CRL's
    contribution: pass a Gaussian distance taper exp(-d^2/2 sigma^2) to make
    neighbours nearer the target count more. This is mean-invariant (no bias from CRLs
    of different rate) and avoids storm double-counting (it sums each CRL's own
    moments, never the overlapping counts). Explicit (non-None) arguments override.
    """
    if weights is None:
        weights = np.ones(len(series))
    rows = [(float(w),) + _series_moments(c)
            for w, c in zip(weights, series) if np.asarray(c).size]
    m2 = sum(w * m * m for w, m, _, _, _ in rows)
    g0 = sum(w * (v - m) for w, m, v, _, _ in rows)  # pooled excess variance
    g1 = sum(w * g for w, _, _, g, _ in rows)
    g2 = sum(w * g for w, _, _, _, g in rows)
    wsum = sum(w for w, _, _, _, _ in rows)
    mean_pool = sum(w * m for w, m, _, _, _ in rows) / wsum if wsum > 0 else 0.0

    disp = max(0.0, g0 / m2) if m2 > 0 else 0.0
    # Attribute variance to the serial term only with geometric-decay evidence
    # (positive lag-1 AND lag-2 autocovariance, gamma_k = mean^2 v_serial phi^k);
    # otherwise it is i.i.d. overdispersion (phi ~ 0 carries no multi-year memory).
    if g1 > 0 and g2 > 0 and m2 > 0:
        phi_auto = float(np.clip(g2 / g1, 0.0, 0.95))
        beta2_serial = (g1 / m2) / max(phi_auto, 1e-3)
    else:
        phi_auto, beta2_serial = default_phi, 0.0
    beta2 = min(max(beta2_serial, 0.0), disp, beta_max ** 2)
    beta_auto = float(np.sqrt(beta2))
    nu_auto = max(0.0, disp - beta2)

    return {
        "ar_phi": phi_auto if ar_phi is None else float(ar_phi),
        "ar_beta": beta_auto if ar_beta is None else float(ar_beta),
        "overdispersion": nu_auto if overdispersion is None else float(overdispersion),
        "mean": mean_pool, "fano": 1.0 + disp * mean_pool if mean_pool > 0 else 1.0,
        "acf1": float(g1 / g0) if g0 > 0 else 0.0, "n_pooled": len(rows),
    }


def _norm_ppf(p: np.ndarray) -> np.ndarray:
    """Inverse standard normal CDF (probit), vectorized (Acklam's algorithm, ~1e-9)."""
    p = np.asarray(p, float)
    a = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
    b = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00)
    plow, phigh = 0.02425, 1.0 - 0.02425
    x = np.empty_like(p)
    lo, hi = p < plow, p > phigh
    mid = ~(lo | hi)
    q = np.sqrt(-2.0 * np.log(np.clip(p[lo], 1e-300, None)))
    x[lo] = (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0)
    q = np.sqrt(-2.0 * np.log(np.clip(1.0 - p[hi], 1e-300, None)))
    x[hi] = -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0)
    q = p[mid] - 0.5
    r = q * q
    x[mid] = (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1.0)
    return x


def within_year_latent(doy, doy_cdf) -> np.ndarray:
    """Latent standard-normal z for each storm: z = Phi^-1(F(doy)), standardized.

    ``doy_cdf`` is the cumulative seasonal probability over days 1..365 (the stratum-
    weighted daily SRR). The probability-integral transform maps each storm day to a
    seasonal quantile; the probit lifts it to the copula's latent normal scale.
    Re-standardizing makes the estimate robust to small mismatch between the empirical
    days and the climatological seasonal shape.
    """
    cdf = np.asarray(doy_cdf, float)
    u = np.clip(cdf[np.clip(np.asarray(doy, int) - 1, 0, cdf.size - 1)], 1e-6, 1 - 1e-6)
    z = _norm_ppf(u)
    return (z - z.mean()) / (z.std() + 1e-12) if z.size else z


def within_year_rho_estimate(z, group, weight=None):
    """Exchangeable copula rho from the within-group correlation of latent normals z.

    Storms grouped by (CRL, year) share a latent season-phase factor, so same-group
    pairs have correlation rho. Estimated as the pooled mean cross-product over all
    within-group pairs (sum of products / number of pairs), using only groups with at
    least two storms. ``weight`` (per storm, default 1) multiplies each group's
    contribution; pass a Gaussian distance taper to pool neighbours regionally with
    nearer CRLs counting more. The signed estimate is shrunk toward zero by one
    standard error (~1/sqrt(number of multi-storm groups)) so a sparse pool is not
    assigned spurious structure. A positive rho is within-season clustering, a negative
    rho is within-season regularity (storms spaced more than random). Returns (rho in
    (-0.95, 0.99), n groups).
    """
    df = pd.DataFrame({"g": np.asarray(group), "z": np.asarray(z, float),
                       "w": 1.0 if weight is None else np.asarray(weight, float)})
    gb = df.groupby("g")
    s = gb["z"].sum().to_numpy()
    q = gb["z"].apply(lambda v: float(np.dot(v, v))).to_numpy()
    n = gb["z"].count().to_numpy().astype(float)
    w = gb["w"].mean().to_numpy()                            # one weight per CRL-year
    m = n >= 2
    n_groups = int(m.sum())
    pairs = (w[m] * n[m] * (n[m] - 1.0) / 2.0).sum()
    if pairs <= 0:
        return 0.0, 0
    rho_raw = (w[m] * (s[m] ** 2 - q[m]) / 2.0).sum() / pairs  # weighted within-yr corr
    shrink = 1.0 / np.sqrt(max(n_groups, 1))                 # ~1 SE in the group count
    rho = np.sign(rho_raw) * max(0.0, abs(rho_raw) - shrink)  # shrink toward 0, signed
    return float(np.clip(rho, -0.95, 0.99)), n_groups
