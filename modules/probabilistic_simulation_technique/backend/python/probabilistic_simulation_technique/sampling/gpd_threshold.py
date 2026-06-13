"""gpd_threshold - Quantile Delta Optimization (QDO) GPD-location search.

Author : Norberto C. Nadal-Caraballo, PhD  <norberto.c.nadal-caraballo@usace.army.mil>

Picks the Generalized-Pareto location μ by minimizing a frequency-weighted
mean-square error (WMSE) between Weibull-plotting-position empirical AERs and
the GPD ICDF predictions, scanned across a candidate grid in the configured
percentile band (Quantile Delta Optimization, QDO).

Objective consistency
---------------------
The empirical annual exceedance rate (AER) of each POT value is unconditional,
``rank/(n+1) · λ_u`` (λ_u = the POT base rate). The GPD models the conditional
tail ``X | X > μ``, so its unconditional AER is ``AER(x) = λ_μ · (1 − F(x))``
where ``λ_μ = n_exc/record_length`` is the exceedance rate ABOVE the candidate
location μ. Predicting a magnitude at an empirical AER therefore inverts with
λ_μ - ``x_pred = ppf(1 − AER/λ_μ)`` - identical to the hazard-curve convention
in ``hazard/curve.py``. (λ_u builds the AER; λ_μ converts an AER to a GPD
quantile - two distinct rates, each in its own place.) The fitted GPD shape is
clipped to the same admissible bounds the ensemble uses, so the objective
scores the model the pipeline actually fits.

Four selection methods (``selection=``); the GPD fit uses ``fit_method`` ("mle"
default, or "mom" method-of-moments):

  "wmse" (DEFAULT) - a relative tolerance ``tol`` (default 5%) on the in-band
      minimum WMSE defines the WMSE-tolerance set of statistically-
      indistinguishable candidates; a ``tiebreak`` chooses within it. This is
      the established behaviour. CAVEAT: the absolute-magnitude WMSE structurally
      shrinks into the over-fit sparse tail, so on some stations its minimum is a
      degenerate high-μ fit (ξ pinned at the lower clip); ``selection_warning``
      flags such a pick and points to "stability".

  "stability" (OPT-IN) - stability-primary, lower-clip-guarded: among eligible
      candidates (in-band, above the exceedance floor, ξ NOT pinned at the lower
      clip) the stability plateau is those within ``stab_tol`` of the minimum
      robust ξ-dispersion (the flat-ξ threshold-stability shelf); a ``tiebreak``
      chooses within it. WMSE is not used to gate. Robust to the degenerate
      sparse tail with no per-station tuning.

  "mrl" (OPT-IN) - automated mean-residual-life (Langousis et al. 2016, eqs 4-6):
      the lowest in-band order statistic at which the mean-excess curve becomes
      linear (a local minimum of the weighted-least-squares fit error). Non-
      parametric (fits a line to e(u), not the GPD itself).

  "gof" (OPT-IN) - Choulakian-Stephens "failure-to-reject" (Langousis §2.3): the
      lowest in-band threshold at which the GPD fit is NOT rejected by an EDF
      goodness-of-fit test (``gof_statistic`` "ad"=Anderson-Darling A² or
      "cvm"=Cramér-von Mises W²) at significance ``gof_significance``, using the
      C&S continuous-data asymptotic critical values (interpolated by ξ).

The per-candidate scan diagnostics (WMSE, fitted GPD shape/scale, robust
ξ-dispersion, exceedance count, λ_μ) are returned in ``QDOResult`` so the
selection can be plotted and visually assessed.

Public API
----------
  QDOResult                     per-candidate scan diagnostics + selected μ
  select_gpd_threshold_qdo(
      values_pot, weibull_aer, lambda_val, record_length,
      min_percentile=50, max_percentile=95, n_candidates=50,
      min_exceedances=30, shape_clip_low=-0.5, shape_clip_high=0.33,
      selection="wmse", tiebreak="stability", stability_window=3,
      stab_tol=0.02, tol=0.05,
  ) -> QDOResult
"""

import warnings
from dataclasses import dataclass, field

import numpy as np
from scipy.stats import genpareto

from ..gpd_fit import fit_gpd_clipped


@dataclass
class QDOResult:
    """Quantile-Delta-Optimization GPD-location scan diagnostics.

    Attributes
    ----------
    best_threshold : float    selected GPD location μ
    best_idx       : int      index of μ in ``candidates``
    candidates     : ndarray  μ candidate grid (response units); spans the full
                              evaluated range, from the data minimum to the top
                              of the selection band
    wmse           : ndarray  frequency-weighted MSE per candidate (NaN if no fit)
    n_exceed       : ndarray  number of exceedances above each candidate
    shape          : ndarray  fitted GPD shape ξ per candidate, shape-clipped
                              (NaN if no fit)
    scale          : ndarray  fitted GPD scale per candidate (NaN if no fit)
    shape_stability: ndarray  ROBUST local ξ-dispersion per candidate (scaled
                              MAD over a neighbour window; lower = flatter/more
                              stable; inf where unassessable) - the PRIMARY
                              selection signal (GPD threshold-stability)
    lambda_mu      : ndarray  exceedance rate above each candidate μ,
                              ``n_exceed / record_length`` (events/yr)
    band_lo        : float    lower edge of the selection band (min_percentile)
    band_hi        : float    upper edge of the selection band (max_percentile)
    record_length  : float    record length (yr) used to form λ_μ
    min_exceedances: int      candidates below this exceedance count are not
                              selectable (avoids over-fitting the sparse tail)
    shape_clip_low : float    lower ξ clip; the "stability" method excludes
                              candidates pinned here (degenerate over-fit)
    stab_tol       : float    ("stability") ξ-dispersion tolerance defining the
                              plateau (eligible candidates within stab_tol of min)
    tol            : float    ("wmse") WMSE-spread fraction defining the accept
                              ceiling = best + tol·(upper − best), where upper is
                              the Tukey-robust max in-band WMSE
    wmse_ceiling   : float    ("wmse") the WMSE accept ceiling actually used (NaN
                              for other methods); in-band candidates at or below
                              it form the WMSE-tolerance set
    selection_method : str    "wmse" (default) or "stability" - which method ran
    tiebreak       : str      arbiter within the chosen set ("stability" = min
                              dispersion, ties → lowest μ; "lowest_mu" = lowest μ)
    selected_set_idx : ndarray  indices of the candidate set the selection chose
                              among (WMSE-tolerance set or stability plateau)
    selection_warning : str   non-empty when the pick looks degenerate/unstable
                              ("wmse") or no flat plateau exists ("stability")
    """
    best_threshold: float
    best_idx:       int
    candidates:     np.ndarray
    wmse:           np.ndarray
    n_exceed:       np.ndarray
    shape:          np.ndarray
    scale:          np.ndarray
    shape_stability: np.ndarray
    lambda_mu:      np.ndarray
    band_lo:        float
    band_hi:        float
    record_length:  float = 0.0
    min_exceedances: int = 0
    shape_clip_low: float = -0.5
    stab_tol:       float = 0.02
    tol:            float = 0.05
    wmse_ceiling:   float = float("nan")
    selection_method: str = "wmse"
    tiebreak:       str = "stability"
    selected_set_idx: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.int64))
    selection_warning: str = ""
    # MRL ("mrl" method) diagnostics - empty for other methods.
    mrl_u:          np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float64))
    mrl_excess:     np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float64))
    mrl_wmse:       np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float64))
    mrl_slope:      float = 0.0
    mrl_intercept:  float = 0.0
    # GoF ("gof" method) diagnostics - NaN for other methods.
    gof_stat:       np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float64))
    gof_crit:       np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float64))
    gof_statistic:  str = "ad"
    gof_significance: float = 0.05


def select_gpd_threshold_qdo(
    values_pot:     np.ndarray,
    weibull_aer:    np.ndarray,
    lambda_val:     float,
    record_length:  float,
    min_percentile: float = 50.0,
    max_percentile: float = 95.0,
    n_candidates:   int   = 50,
    min_exceedances: int  = 30,
    shape_clip_low:  float = -0.5,
    shape_clip_high: float =  0.33,
    selection:       str   = "wmse",
    tiebreak:        str   = "stability",
    stability_window: int  = 3,
    stab_tol:        float = 0.02,
    tol:             float = 0.05,
    fit_method:      str   = "mle",
    gof_statistic:   str   = "ad",
    gof_significance: float = 0.05,
) -> QDOResult:
    """Scan candidate GPD locations μ and return the QDO selection + diagnostics.

    Parameters
    ----------
    values_pot : (n,) float64
        Descending-sorted POT magnitudes.
    weibull_aer : (n,) float64
        Empirical (unconditional) AERs at each POT value - Weibull plotting
        positions scaled by the POT base rate λ_u.
    lambda_val : float
        POT base rate λ_u = ``len(values_pot) / record_length``. Retained for
        context/reporting; the objective converts AERs with λ_μ (below).
    record_length : float
        Record length in years. Forms the per-candidate exceedance rate above μ,
        ``λ_μ = n_exc / record_length`` - the rate used to map an empirical AER
        to a GPD non-exceedance probability (hazard-curve convention).
    min_percentile, max_percentile : float
        Bounds of the candidate-μ scan, expressed as empirical PERCENTILES of
        the POT values (data quantiles). Must satisfy 0 <= min < max <= 100.
    n_candidates : int
        Number of μ candidates uniformly spaced in the band.
    min_exceedances : int
        A candidate μ must retain at least this many exceedances to be
        selectable (guards against over-fitting the sparse tail).
    shape_clip_low, shape_clip_high : float
        Admissible bounds for the fitted GPD shape ξ - applied here exactly as
        in the hazard ensemble, so the objective scores the model actually used.
    selection : {"wmse", "stability", "mrl", "gof"}
        Which selection method to use (see below). DEFAULT "wmse".
    fit_method : {"mle", "mom"}
        GPD estimator for every per-candidate / per-realization fit. DEFAULT
        "mle"; "mom" is the closed-form method of moments (more robust for
        small / quantized samples). Passed through to ``gpd_fit.fit_gpd_clipped``.
    gof_statistic : {"ad", "cvm"}
        ("gof" method) EDF goodness-of-fit statistic - Anderson-Darling A²
        (tail-weighted) or Cramér-von Mises W².
    gof_significance : float
        ("gof" method) significance level α of the failure-to-reject test.
    tiebreak : {"stability", "lowest_mu"}
        Arbiter WITHIN the chosen candidate set. "stability" → smallest robust
        ξ-dispersion (ties → lowest μ); "lowest_mu" → lowest μ (most data).
    stability_window : int
        Half-width (in candidates) of the neighbour window over which the robust
        ξ-dispersion (scaled MAD) is measured.
    stab_tol : float
        ("stability" method) ξ-dispersion tolerance defining the plateau:
        eligible candidates within ``stab_tol`` of the minimum robust dispersion.
        ξ is dimensionless, so this is station-agnostic.
    tol : float
        ("wmse" method) WMSE-spread fraction defining the WMSE-tolerance set:
        in-band candidates with WMSE <= best + ``tol`` × (upper − best), where best
        is the minimum in-band WMSE and upper the Tukey-robust max (highest in-band
        WMSE <= Q3 + 1.5·IQR).

    Selection methods
    -----------------
    "wmse"  (DEFAULT) - the WMSE-tolerance set is every in-band candidate with
        >= ``min_exceedances`` exceedances whose WMSE is within a fraction
        ``tol`` of the climb from the best fit (floor) to a robust ceiling (the
        highest in-band WMSE that is not a Tukey outlier); ``tiebreak`` picks μ
        within it. The pool is the FULL in-band set, so a genuinely bounded short
        tail (ξ at the lower clip, e.g. hurricane-dominated) stays in the set.
        CAVEAT: WMSE can still shrink into an over-thinned sparse tail, so when the
        pick is ξ-pinned at the clip ``selection_warning`` flags it and suggests
        "stability".
    "stability"  (opt-in) - STABILITY-PRIMARY, lower-clip-guarded. ELIGIBLE
        candidates are in-band, >= ``min_exceedances``, finite, and NOT pinned at
        ``shape_clip_low``; the STABILITY PLATEAU is those within ``stab_tol`` of
        the minimum robust ξ-dispersion (the flat-ξ threshold-stability shelf);
        ``tiebreak`` picks μ within it. WMSE is not used to gate. Robust to the
        degenerate sparse tail with no per-station tuning.
    "mrl"  (opt-in) - automated MEAN-RESIDUAL-LIFE (Langousis et al. 2016, WRR,
        §2.2, eqs 4-6). Non-parametric: the mean excess e(u) is linear in u where
        a GPD holds, so a weighted-least-squares line is fit to the mean-excess
        curve from each candidate upward and μ is the LOWEST in-band order
        statistic that is a local minimum of the fit's weighted MSE. Returns the
        exact order-statistic threshold; ``tiebreak``/``stab_tol``/``tol`` are
        unused. See ``_select_mrl``.
    "gof"  (opt-in) - Choulakian-Stephens FAILURE-TO-REJECT (Langousis §2.3): the
        lowest in-band candidate at which the GPD fit to the exceedances is not
        rejected - the EDF statistic (``gof_statistic`` on the PIT) is <= its
        critical value at ``gof_significance`` (C&S continuous-data asymptotic
        values, interpolated by ξ and clamped to ξ∈[0,0.3]). See ``_select_gof``.
    """
    if selection not in ("wmse", "stability", "mrl", "gof"):
        raise ValueError(
            f"selection must be 'wmse', 'stability', 'mrl', or 'gof'; got "
            f"{selection!r}")
    if fit_method not in ("mle", "mom"):
        raise ValueError(f"fit_method must be 'mle' or 'mom'; got {fit_method!r}")
    if gof_statistic not in ("ad", "cvm"):
        raise ValueError(f"gof_statistic must be 'ad' or 'cvm'; got {gof_statistic!r}")
    if not (0.0 < gof_significance < 1.0):
        raise ValueError(f"gof_significance must be in (0,1); got {gof_significance!r}")
    if tiebreak not in ("stability", "lowest_mu"):
        raise ValueError(
            f"tiebreak must be 'stability' or 'lowest_mu'; got {tiebreak!r}")
    if tol < 0.0:
        raise ValueError(f"tol must be >= 0; got {tol!r}")
    if stab_tol < 0.0:
        raise ValueError(f"stab_tol must be >= 0; got {stab_tol!r}")
    if values_pot.size == 0:
        raise ValueError("values_pot is empty")
    if not np.all(np.isfinite(values_pot)):
        raise ValueError("values_pot contains non-finite entries")
    # Input contract: weibull_aer must pair 1:1 with values_pot, and values_pot
    # must be descending-sorted. The λ_μ math relies on the exceedances being
    # the top-k of a sorted array (so prob = 1 - aer/λ_μ stays in (0,1)); an
    # unsorted array silently mis-pairs values with plotting positions.
    if weibull_aer.shape != values_pot.shape:
        raise ValueError(
            f"weibull_aer shape {weibull_aer.shape} != values_pot shape "
            f"{values_pot.shape}")
    if not np.all(np.isfinite(weibull_aer)):
        raise ValueError("weibull_aer contains non-finite entries")
    if not np.all(weibull_aer > 0.0):
        raise ValueError("weibull_aer must be strictly positive")
    if np.any(np.diff(values_pot) > 0.0):
        raise ValueError(
            "values_pot must be sorted in non-increasing (descending) order")
    if not (record_length > 0.0):
        raise ValueError(f"record_length must be > 0; got {record_length!r}")
    if not (0.0 <= min_percentile < max_percentile <= 100.0):
        raise ValueError("require 0 <= min_percentile < max_percentile <= 100")
    if n_candidates <= 1:
        raise ValueError("n_candidates must be > 1")

    resp_min, resp_max = float(np.min(values_pot)), float(np.max(values_pot))
    resp_range         = resp_max - resp_min
    if not (resp_range > 0.0):
        raise ValueError(
            "all POT values are identical (zero range); cannot scan a "
            "GPD-location band")

    # Band = empirical PERCENTILES of the POT values (data quantiles), not a
    # fraction of the magnitude range - robust to a single extreme value and
    # interpretable as "scan μ between the p_lo and p_hi data quantiles". The
    # count floor (min_exceedances) is the principled upper cap; for typical
    # n_pot it binds before band_hi, which acts as a secondary guardrail.
    band_lo = float(np.percentile(values_pot, min_percentile))
    band_hi = float(np.percentile(values_pot, max_percentile))

    # Evaluate across the FULL range - from the data minimum up to the top of
    # the selection band - so the diagnostics show candidates below the chosen
    # μ, not only the selected location and above. Selection is still confined
    # to the [min_percentile, max_percentile] band below. No rounding: it would
    # collapse candidates for small-magnitude series (e.g. sub-meter NTR).
    candidates = np.linspace(resp_min, band_hi, int(n_candidates))

    n         = candidates.size
    wmse      = np.full(n, np.nan, dtype=np.float64)
    n_exceed  = np.zeros(n, dtype=np.int64)
    shape     = np.full(n, np.nan, dtype=np.float64)
    scale     = np.full(n, np.nan, dtype=np.float64)
    lambda_mu = np.zeros(n, dtype=np.float64)

    for i, th in enumerate(candidates):
        mask_above = values_pot > th
        pot        = values_pot[mask_above]
        aer        = weibull_aer[mask_above]
        n_exceed[i]  = pot.size
        lambda_mu[i] = pot.size / record_length     # λ_μ = n_exc / record_length
        if len(np.unique(pot)) <= 1:
            continue
        try:
            # Shared fit: location fixed at th, ξ clipped to the ensemble's
            # admissible band, σ refit when ξ is clipped - the same model the
            # hazard ensemble uses (see gpd_fit.fit_gpd_clipped).
            c, _loc, sc = fit_gpd_clipped(pot, th, shape_clip_low, shape_clip_high,
                                          method=fit_method)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                # AER → GPD non-exceedance probability via λ_μ (the rate above
                # μ), matching hazard/curve.py. Predict on the ABSOLUTE scale
                # (loc=th). Clip prob into (0,1) for numerical safety.
                prob = np.clip(1.0 - aer / lambda_mu[i], 1e-12, 1.0 - 1e-12)
                pred = genpareto.ppf(prob, c, loc=th, scale=sc)
            shape[i] = c
            scale[i] = sc
            # Weight rare events more (1/AER); drop very frequent events
            # (AER >= 1) and any non-finite prediction (e.g. negative-shape
            # support overshoot).
            weights_mask = (aer < 1.0) & np.isfinite(pred)
            if not np.any(weights_mask):
                continue
            weights        = 1.0 / aer[weights_mask]
            squared_errors = (pot[weights_mask] - pred[weights_mask]) ** 2
            wmse[i]        = float(np.sum(weights * squared_errors)
                                   / np.sum(weights))
        except Exception:
            wmse[i] = np.nan

    # Robust local ξ-dispersion (scaled MAD over the neighbour window): the
    # threshold-stability signal. Used to GATE in the "stability" method and as
    # the default in-set tie-break in both methods.
    shape_stability = _local_dispersion(shape, stability_window)
    in_band = (candidates >= band_lo - 1e-9) & (candidates <= band_hi + 1e-9)

    # MRL / GoF diagnostics (populated only for those methods).
    mrl_u = mrl_excess = mrl_wmse = np.empty(0, dtype=np.float64)
    mrl_slope = mrl_intercept = 0.0
    gof_stat = np.full(n, np.nan, dtype=np.float64)
    gof_crit = np.full(n, np.nan, dtype=np.float64)
    wmse_ceiling = float("nan")   # set by the "wmse" method (plotted accept line)

    # Dispatch to the chosen selection method (default "wmse").
    if selection == "gof":
        best_idx, selected_set, selection_warning, gof_stat, gof_crit = _select_gof(
            values_pot, candidates, shape, scale, n_exceed, in_band,
            int(min_exceedances), gof_statistic, gof_significance, band_lo, band_hi)
        best_threshold = float(candidates[best_idx])
    elif selection == "mrl":
        (best_idx, best_threshold, selected_set, selection_warning,
         mrl_u, mrl_excess, mrl_wmse, mrl_slope, mrl_intercept) = _select_mrl(
            values_pot, candidates, band_lo, band_hi, int(min_exceedances))
    elif selection == "stability":
        best_idx, selected_set, selection_warning = _select_stability(
            shape, shape_stability, n_exceed, in_band, int(min_exceedances),
            shape_clip_low, stab_tol, tiebreak, band_lo, band_hi)
        best_threshold = float(candidates[best_idx])
    else:  # "wmse" (DEFAULT)
        best_idx, selected_set, wmse_ceiling, selection_warning = _select_wmse(
            wmse, shape, shape_stability, n_exceed, in_band,
            int(min_exceedances), shape_clip_low, tol, tiebreak, band_lo, band_hi)
        best_threshold = float(candidates[best_idx])

    return QDOResult(
        best_threshold = float(best_threshold),
        best_idx       = best_idx,
        candidates     = candidates,
        wmse           = wmse,
        n_exceed       = n_exceed,
        shape          = shape,
        scale          = scale,
        shape_stability = shape_stability,
        lambda_mu      = lambda_mu,
        band_lo        = float(band_lo),
        band_hi        = float(band_hi),
        record_length  = float(record_length),
        min_exceedances = int(min_exceedances),
        shape_clip_low = float(shape_clip_low),
        stab_tol       = float(stab_tol),
        tol            = tol,
        wmse_ceiling   = float(wmse_ceiling),
        selection_method = selection,
        tiebreak       = tiebreak,
        selected_set_idx = np.asarray(selected_set, dtype=np.int64),
        selection_warning = selection_warning,
        mrl_u          = np.asarray(mrl_u, dtype=np.float64),
        mrl_excess     = np.asarray(mrl_excess, dtype=np.float64),
        mrl_wmse       = np.asarray(mrl_wmse, dtype=np.float64),
        mrl_slope      = float(mrl_slope),
        mrl_intercept  = float(mrl_intercept),
        gof_stat       = gof_stat,
        gof_crit       = gof_crit,
        gof_statistic  = gof_statistic,
        gof_significance = float(gof_significance),
    )


# ── selection helpers ───────────────────────────────────────────────────────
_MAD_TO_STD = 1.4826        # scales MAD to a normal-consistent standard deviation


def _local_dispersion(series: np.ndarray, window: int) -> np.ndarray:
    """Robust local dispersion per candidate over a ±window slice.

    Uses the scaled median absolute deviation (MAD) of the finite values -
    ``1.4826 · median(|x − median(x)|)`` - so a single anomalous fit in the
    window cannot inflate it (unlike the std). Applied to ξ(μ): a flat (stable)
    region - the GPD threshold-stability signature - has near-zero dispersion.
    Candidates with fewer than three finite neighbours cannot be assessed
    robustly and are marked ``inf`` (treated as maximally unstable), so the
    selection never lands on an isolated, unreliable fit.
    """
    n = series.size
    w = max(int(window), 1)
    disp = np.full(n, np.inf, dtype=np.float64)
    for i in range(n):
        seg    = series[max(0, i - w):min(n, i + w + 1)]
        finite = seg[np.isfinite(seg)]
        if finite.size >= 3:
            med = np.median(finite)
            disp[i] = _MAD_TO_STD * float(np.median(np.abs(finite - med)))
    return disp


_CLIP_EPS    = 1e-3         # ξ within this of the lower clip counts as "pinned"
_FLAT_THRESH = 0.05         # robust ξ-dispersion below this counts as "flat"


def _pick_within(idx_set: np.ndarray, xi_disp: np.ndarray, tiebreak: str) -> int:
    """Choose one index from ``idx_set`` (ascending in μ) per the tie-break.

    "stability" → smallest robust ξ-dispersion; "lowest_mu" → lowest μ. Both
    break ties toward the lowest μ (argmin returns the first occurrence, and
    ``idx_set`` is ascending), matching the EVA-standard data-rich choice.
    """
    if tiebreak == "lowest_mu":
        return int(idx_set[0])
    return int(idx_set[int(np.argmin(xi_disp[idx_set]))])


def _select_wmse(wmse, shape, xi_disp, n_exceed, in_band, min_exceedances,
                 shape_clip_low, tol, tiebreak, band_lo, band_hi):
    """DEFAULT method: WMSE-tolerance set + tie-break.

    The set is every in-band candidate above the exceedance floor whose WMSE is
    within a fraction ``tol`` of the climb from the best fit (the floor, the
    minimum in-band WMSE) up to a robust upper anchor:

        ceiling = best + tol * (upper - best)

    where ``upper`` is the highest in-band WMSE that is NOT a Tukey outlier (i.e.
    the largest value <= Q3 + 1.5*IQR). Anchoring on this robust maximum makes the
    span as high as the data honestly allows while a single freak-high fit cannot
    inflate it, and keeps ``tol`` a single consistent meaning (a fraction of the
    floor->robust-max WMSE spread). The pool is the FULL in-band set, with NO shape
    filtering, so a genuinely bounded short tail (ξ at the lower clip, e.g.
    hurricane-dominated) stays in the set and ``tiebreak`` arbitrates within it.
    ``selection_warning`` flags a degenerate (ξ at the lower clip) or ξ-unstable
    pick, suggesting ``selection='stability'``.
    """
    band_mask = in_band & (n_exceed >= min_exceedances)
    wmse_band = np.where(band_mask, wmse, np.nan)
    if not np.any(np.isfinite(wmse_band)):
        if not np.any(band_mask):
            raise RuntimeError(
                "GPD location search failed: no in-band candidate retains "
                f">= {min_exceedances} exceedances (band [{band_lo:.3f}, "
                f"{band_hi:.3f}]). Lower min_exceedances or raise "
                "threshold_max_percentile.")
        raise RuntimeError(
            "GPD location search failed: in-band candidates meet the exceedance "
            f"floor (>= {min_exceedances}) but none yielded a finite WMSE (all "
            "GPD fits failed/degenerate). Check the POT input or widen the band.")

    best       = float(np.nanmin(wmse_band))
    upper      = _robust_upper_wmse(wmse_band)                 # Tukey-fenced max
    ceiling    = best + tol * (upper - best)
    sel_set    = np.where(wmse_band <= ceiling)[0]             # NaN cmp → False
    if sel_set.size == 0:                                      # defensive
        sel_set = np.array([int(np.nanargmin(wmse_band))], dtype=np.int64)
    best_idx = _pick_within(sel_set, xi_disp, tiebreak)
    return best_idx, sel_set, float(ceiling), _wmse_pick_warning(
        best_idx, shape, xi_disp, shape_clip_low)


def _robust_upper_wmse(wmse_band: np.ndarray) -> float:
    """Highest in-band WMSE that is not a Tukey outlier (<= Q3 + 1.5*IQR).

    The robust upper anchor for the WMSE accept ceiling: as high as the in-band
    fits honestly reach, but immune to a single freak-high WMSE candidate (which
    would otherwise inflate the floor->ceiling span). Falls back to the plain
    maximum if every finite value sits above the fence (degenerate spread).
    """
    finite = wmse_band[np.isfinite(wmse_band)]
    q1, q3 = np.percentile(finite, [25.0, 75.0])
    fence  = q3 + 1.5 * (q3 - q1)
    inlier = finite[finite <= fence]
    return float(np.max(inlier)) if inlier.size else float(np.max(finite))


def _wmse_pick_warning(best_idx, shape, xi_disp, shape_clip_low) -> str:
    """Flag a degenerate / unstable WMSE pick and point to the 'stability' method."""
    if np.isfinite(shape[best_idx]) and shape[best_idx] <= shape_clip_low + _CLIP_EPS:
        return ("selected mu has xi pinned at the lower clip (bounded sparse-tail "
                "over-fit - the WMSE minimum can sit there); consider "
                "GPD_SELECTION='stability'.")
    d = xi_disp[best_idx]
    if np.isfinite(d) and d > _FLAT_THRESH:
        return (f"selected mu sits on an unstable xi region (robust dispersion "
                f"{d:.3g} > {_FLAT_THRESH:g}); consider GPD_SELECTION='stability'.")
    return ""


def _select_stability(shape, xi_disp, n_exceed, in_band, min_exceedances,
                      shape_clip_low, stab_tol, tiebreak, band_lo, band_hi):
    """OPT-IN method: stability plateau (lower-clip guarded) + tie-break.

    ELIGIBLE = in-band, >= floor, finite fit + dispersion, and ξ NOT pinned at
    the lower clip. The PLATEAU = eligible candidates within ``stab_tol`` of the
    minimum robust ξ-dispersion (the flat-ξ shelf); ``tiebreak`` picks within it.
    """
    eligible = (in_band
                & (n_exceed >= min_exceedances)
                & np.isfinite(shape)
                & np.isfinite(xi_disp)
                & (shape > shape_clip_low + _CLIP_EPS))     # not lower-clip-pinned
    if not np.any(eligible):
        if not np.any(in_band & (n_exceed >= min_exceedances)):
            raise RuntimeError(
                "GPD location search failed: no in-band candidate retains "
                f">= {min_exceedances} exceedances (band [{band_lo:.3f}, "
                f"{band_hi:.3f}]). Lower min_exceedances or raise "
                "threshold_max_percentile.")
        raise RuntimeError(
            "GPD location search failed: every in-band candidate above the "
            "exceedance floor is degenerate (xi pinned at the lower clip) or has "
            "no finite fit. The tail is strongly bounded / over-fit; widen the "
            "band downward (lower threshold_min_percentile).")

    elig_idx = np.where(eligible)[0]
    d_min    = float(np.min(xi_disp[elig_idx]))
    plateau  = elig_idx[xi_disp[elig_idx] <= d_min + stab_tol]
    best_idx = _pick_within(plateau, xi_disp, tiebreak)
    return best_idx, plateau, _stability_warning(d_min, stab_tol)


def _stability_warning(d_min: float, stab_tol: float) -> str:
    """Soft flag when no genuinely flat ξ-plateau exists (μ weakly determined)."""
    if np.isfinite(d_min) and d_min > _FLAT_THRESH + stab_tol:
        return (f"no flat xi-plateau found (min robust xi-dispersion {d_min:.3g} "
                f"> {_FLAT_THRESH + stab_tol:.3g}); mu is weakly determined - "
                f"inspect the diagnostics.")
    return ""


def _select_mrl(values_pot, candidates, band_lo, band_hi, min_exceedances):
    """OPT-IN method: automated Mean-Residual-Life (Langousis et al. 2016, §2.2,
    eqs 4-6), band-restricted.

    Candidates are the order statistics u_i = x_(i). The mean excess
    ``e(u_i) = E[X - u_i | X > u_i]`` is linear in u where a GPD holds (eq 5),
    so for each u_i a weighted-least-squares line is fit to the points
    ``(u_j, e(u_j))`` for j >= i (weights ``(n - j)/Var[excess]``). μ* is the
    LOWEST in-band order statistic (>= ``min_exceedances`` excesses) that is a
    local minimum of the fit's weighted MSE; the GPD shape is recovered as
    ``ξ = A/(1 + A)`` from the fitted slope A.

    Returns
    -------
    (best_idx, u_star, selected_set, warning, u, excess, wmse, slope, intercept)
        ``best_idx`` indexes the nearest ``candidates`` entry (for the marker);
        ``u_star`` is the exact selected threshold.
    """
    x = np.sort(np.asarray(values_pot, dtype=np.float64))   # ascending order stats
    n = x.size
    i_max_e = n - 10                          # >= 10 excesses to form e(u_i)
    i_max_fit = n - 20                        # >= 10 regression points (j up to i_max_e)
    if i_max_fit < 1:
        raise RuntimeError(
            "MRL selection failed: need >= ~22 POT values to fit the mean-excess "
            f"plot; got {n}. Use a different GPD_SELECTION or supply more peaks.")

    u   = x[:i_max_e]
    e   = np.empty(i_max_e); var = np.empty(i_max_e); cnt = np.empty(i_max_e)
    for i in range(i_max_e):
        ex     = x[i + 1:] - x[i]             # excesses above u_i = x[i]
        e[i]   = ex.mean()
        cnt[i] = ex.size                      # n - (i+1) excesses above u_i
        var[i] = ex.var(ddof=1) if ex.size > 1 else np.nan
    w = cnt / np.where(var > 0.0, var, np.nan)            # weights (n-j)/Var

    wmse  = np.full(i_max_e, np.nan)
    A_arr = np.full(i_max_e, np.nan)
    B_arr = np.full(i_max_e, np.nan)
    for i in range(i_max_fit):
        uu, ee, ww = u[i:i_max_e], e[i:i_max_e], w[i:i_max_e]
        m = np.isfinite(ww) & np.isfinite(ee)
        if m.sum() < 10:
            continue
        uu, ee, ww = uu[m], ee[m], ww[m]
        W   = ww / ww.sum()                   # weighted least squares for e ~ A·u + B
        Su  = np.sum(W * uu); Se = np.sum(W * ee)
        den = np.sum(W * uu * uu) - Su * Su
        if abs(den) < 1e-15:
            continue
        A = float((np.sum(W * uu * ee) - Su * Se) / den)
        B = float(Se - A * Su)
        r = ee - (A * uu + B)
        wmse[i]  = float(np.sum(ww * r * r) / np.sum(ww))
        A_arr[i] = A; B_arr[i] = B

    # Band + exceedance-floor eligibility (consistency with the other methods).
    eligible = ((u >= band_lo - 1e-9) & (u <= band_hi + 1e-9)
                & (cnt >= min_exceedances) & np.isfinite(wmse))
    elig_idx = np.where(eligible)[0]
    if elig_idx.size == 0:
        raise RuntimeError(
            "MRL selection failed: no in-band order statistic retains "
            f">= {min_exceedances} excesses with a finite mean-excess fit "
            f"(band [{band_lo:.3f}, {band_hi:.3f}]). Lower min_exceedances or "
            "widen the band.")

    # Lowest in-band local minimum of the weighted MSE (eq-5 linearity onset).
    star, warning = None, ""
    for k in elig_idx:
        left  = wmse[k - 1] if (k - 1 >= 0 and np.isfinite(wmse[k - 1])) else np.inf
        right = wmse[k + 1] if (k + 1 < i_max_e and np.isfinite(wmse[k + 1])) else np.inf
        if wmse[k] <= left and wmse[k] <= right:
            star = int(k); break
    if star is None:
        star = int(elig_idx[int(np.argmin(wmse[elig_idx]))])
        warning = ("MRL: no local minimum of the mean-excess fit error in the "
                   "band; used the in-band global minimum (mu weakly determined).")

    u_star   = float(u[star])
    best_idx = int(np.argmin(np.abs(candidates - u_star)))
    return (best_idx, u_star, np.array([best_idx], dtype=np.int64), warning,
            u, e, wmse, A_arr[star], B_arr[star])


# ── GoF (Choulakian-Stephens failure-to-reject) helpers ─────────────────────
# Continuous-data asymptotic critical values of the EDF statistics (both GPD
# parameters ML-estimated), indexed by shape ξ - Langousis et al. (2016) WRR
# Tables 1-2 / Choulakian & Stephens (2001). Columns are the (1-α) quantiles.
_GOF_XI = np.array([0.0, 0.1, 0.2, 0.3])
_GOF_CRIT = {
    "ad":  {0.10: [0.793, 0.763, 0.743, 0.719],
            0.05: [0.971, 0.933, 0.905, 0.876],
            0.01: [1.403, 1.346, 1.299, 1.257]},
    "cvm": {0.10: [0.121, 0.115, 0.112, 0.107],
            0.05: [0.150, 0.143, 0.137, 0.132],
            0.01: [0.220, 0.209, 0.200, 0.191]},
}


def _edf_gof(z: np.ndarray, statistic: str) -> float:
    """EDF goodness-of-fit statistic of PIT values z (Langousis eqs 7-8).

    "ad" = Anderson-Darling A² (tail-weighted); "cvm" = Cramér-von Mises W².
    """
    z = np.clip(np.sort(z), 1e-12, 1.0 - 1e-12)
    nn = z.size
    i  = np.arange(1, nn + 1)
    if statistic == "cvm":
        return float(np.sum((z - (2 * i - 1) / (2 * nn)) ** 2) + 1.0 / (12 * nn))
    return float(-nn - np.sum((2 * i - 1) * (np.log(z) + np.log(1.0 - z[::-1]))) / nn)


def _gof_crit(statistic: str, xi: float, alpha: float) -> float:
    """Critical value for the GoF statistic at shape ξ and significance α.

    ξ is clamped to the tabulated range [0, 0.3] (the values vary little there
    and most accepted thresholds fall inside it); α is interpolated among the
    tabulated levels {0.10, 0.05, 0.01}.
    """
    tbl = _GOF_CRIT[statistic]
    xi_c = min(max(float(xi), 0.0), 0.3)
    c10 = float(np.interp(xi_c, _GOF_XI, tbl[0.10]))
    c05 = float(np.interp(xi_c, _GOF_XI, tbl[0.05]))
    c01 = float(np.interp(xi_c, _GOF_XI, tbl[0.01]))
    a = min(max(float(alpha), 0.01), 0.10)
    return float(np.interp(a, [0.01, 0.05, 0.10], [c01, c05, c10]))


def _select_gof(values_pot, candidates, shape, scale, n_exceed, in_band,
                min_exceedances, gof_statistic, gof_significance, band_lo, band_hi):
    """OPT-IN method: Choulakian-Stephens "failure-to-reject" (Langousis §2.3).

    Scans in-band candidates upward and selects the LOWEST μ at which the GPD
    fit to the exceedances is NOT rejected: the EDF statistic (A²/W² on the PIT
    of the exceedances under the per-candidate fit) is <= its critical value at
    significance ``gof_significance``. Reuses the per-candidate (ξ, σ) fits.
    """
    nC = candidates.size
    stat = np.full(nC, np.nan); crit = np.full(nC, np.nan)
    eligible = (in_band & (n_exceed >= min_exceedances)
                & np.isfinite(shape) & np.isfinite(scale))
    for i in np.where(eligible)[0]:
        th  = candidates[i]
        pot = values_pot[values_pot > th]
        if pot.size < min_exceedances or np.unique(pot).size <= 1:
            continue
        z = genpareto.cdf(pot, shape[i], loc=th, scale=scale[i])
        stat[i] = _edf_gof(z, gof_statistic)
        crit[i] = _gof_crit(gof_statistic, shape[i], gof_significance)
    accept = np.where(np.isfinite(stat) & (stat <= crit))[0]   # NaN cmp → False
    if accept.size:
        best_idx = int(accept[0])                              # lowest μ not rejected
        warning  = ""
    else:
        finite = np.where(np.isfinite(stat))[0]
        if finite.size == 0:
            raise RuntimeError(
                "GoF selection failed: no in-band candidate with "
                f">= {min_exceedances} exceedances yielded a finite fit "
                f"(band [{band_lo:.3f}, {band_hi:.3f}]).")
        best_idx = int(finite[np.argmin(stat[finite] - crit[finite])])  # least-rejected
        warning  = ("GoF: the GPD fit is rejected at every in-band threshold "
                    f"(significance {gof_significance:g}); used the least-rejected "
                    "μ - the tail may not be GP-distributed in the band.")
    return best_idx, np.asarray(accept, dtype=np.int64), warning, stat, crit
