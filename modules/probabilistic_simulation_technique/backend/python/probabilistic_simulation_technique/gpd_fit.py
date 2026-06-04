"""gpd_fit — the single GPD fitting primitive shared across PST.

Author / POC : Norberto C. Nadal-Caraballo, PhD  <norberto.c.nadal-caraballo@usace.army.mil>

Both the QDO location search (``sampling/gpd_threshold.py``) and the hazard
ensemble (``hazard/curve.py``) fit a Generalized Pareto distribution to
threshold exceedances. Centralizing that fit here guarantees they score and
predict with the *same* model.

The fit holds the location at the threshold (``floc``) and clips the shape ξ to
the admissible band. Two estimators are offered:

  "mle" (default) — maximum likelihood. When clipping bites, the scale σ is
      refit with ξ (and the location) held fixed, so the returned (ξ, σ) pair is
      a genuine constrained MLE rather than a clipped ξ paired with the stale σ.
  "mom"           — method of moments. From the excess mean m and (unbiased)
      variance v above ``floc``:  ξ = ½(1 − m²/v),  σ = m(1 − ξ); ξ is then
      clipped and σ recomputed at the clipped ξ. More robust for small samples /
      heavy quantization (no optimizer), at some efficiency cost.

Public API
----------
  fit_gpd_clipped(data, floc, shape_clip_low, shape_clip_high, method="mle")
      -> (shape, loc, scale)
"""

import warnings
from typing import Tuple

import numpy as np
from scipy.stats import genpareto


def fit_gpd_clipped(
    data,
    floc:            float,
    shape_clip_low:  float,
    shape_clip_high: float,
    method:          str = "mle",
) -> Tuple[float, float, float]:
    """Fit a GPD with fixed location and clipped shape; "mle" (default) or "mom".

    Parameters
    ----------
    data : array_like
        Exceedances (all strictly greater than ``floc``).
    floc : float
        Fixed GPD location (the threshold / μ).
    shape_clip_low, shape_clip_high : float
        Admissible bounds for the GPD shape ξ.
    method : {"mle", "mom"}
        Estimator. "mle" = maximum likelihood with constrained σ-refit when ξ is
        clipped; "mom" = method of moments (closed form from the excess
        mean/variance), ξ clipped and σ recomputed at the clipped ξ.

    Returns
    -------
    (shape, loc, scale) : tuple of float
        ``shape`` is the clipped ξ; ``loc == floc``; ``scale`` is σ.

    Raises
    ------
    Exception
        "mle": propagates a failure of the initial (unconstrained-shape) fit so
        the caller can skip the sample (a failed *constrained* σ-refit is
        non-fatal — the unconstrained-fit σ is kept). "mom": raises ValueError if
        the excess variance is non-positive (degenerate sample).
    """
    if method not in ("mle", "mom"):
        raise ValueError(f"method must be 'mle' or 'mom'; got {method!r}")

    if method == "mom":
        excess = np.asarray(data, dtype=np.float64) - float(floc)
        m = float(np.mean(excess))
        v = float(np.var(excess, ddof=1)) if excess.size > 1 else 0.0
        if not (v > 0.0 and m > 0.0):
            raise ValueError("MoM GPD fit needs positive excess mean and variance")
        c = 0.5 * (1.0 - m * m / v)                       # ξ = ½(1 − m²/v)
        c = max(min(c, shape_clip_high), shape_clip_low)  # clip
        scale = m * (1.0 - c)                             # σ = m(1 − ξ)
        return c, float(floc), max(scale, 1e-12)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        c, loc, scale = genpareto.fit(data, floc=floc)
        c_clip = max(min(c, shape_clip_high), shape_clip_low)
        if c_clip != c:
            # ξ hit a bound — refit σ with ξ and the location held fixed so the
            # returned pair is the constrained MLE, not clipped-ξ / stale-σ.
            try:
                _c, _loc, scale = genpareto.fit(data, fc=c_clip, floc=floc)
            except Exception:
                pass   # keep the unconstrained-fit σ as a fallback
    return c_clip, floc, scale
