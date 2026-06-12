"""test_smoke - smoke tests for the PST module (config, bootstrap, threshold).

Author / POC : Norberto C. Nadal-Caraballo, PhD  <norberto.c.nadal-caraballo@usace.army.mil>

Validates: (1) config construction + invariants, (2) BootstrapGenerator
contract (C++ when available, pure-Python fallback otherwise), (3) GPD
threshold search runs and returns a value inside its candidate band,
(4) PSTOrchestrator end-to-end on a synthetic POT sample.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")          # headless: no GUI backend in test/CI environments

import numpy as np
import pandas as pd
import pytest
from scipy.stats import genpareto

# Make the in-tree package importable when running tests without `pip install -e`.
_MODULE_ROOT = Path(__file__).resolve().parents[1]
_PKG_PATH    = _MODULE_ROOT / "backend" / "python"
if str(_PKG_PATH) not in sys.path:
    sys.path.insert(0, str(_PKG_PATH))

from probabilistic_simulation_technique import (
    PSTConfig, PSTOrchestrator, CPP_KERNEL_AVAILABLE,
)
from probabilistic_simulation_technique.sampling import (
    BootstrapGenerator, select_gpd_threshold_qdo,
)
from probabilistic_simulation_technique.gpd_fit import fit_gpd_clipped


# ---------------------------------------------------------------------------
# Auto record length: n_pot / events_per_year  (=> lambda_u == events_per_year)
# ---------------------------------------------------------------------------
def test_auto_record_length_from_events_per_year(tmp_path, pot_csv, synthetic_pot):
    cfg = PSTConfig(
        input_csv           = pot_csv,
        output_dir          = tmp_path / "out",
        plots_dir           = tmp_path / "plots",
        record_length_years = None,        # auto = n_pot / events_per_year
        events_per_year     = 10.0,
        num_simulations     = 30,
        random_seed         = 1,
    )
    result = PSTOrchestrator(cfg).run()
    # POT trims to exactly rate × eff_dur peaks, so n_pot/(n_pot/rate) == rate.
    assert result.lambda_val == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def synthetic_pot():
    rng = np.random.default_rng(628)
    return np.sort(rng.gamma(shape=2.0, scale=1.5, size=300))[::-1]


@pytest.fixture
def pot_csv(tmp_path, synthetic_pot):
    path = tmp_path / "synthetic_POT.csv"
    pd.DataFrame({"value": np.sort(synthetic_pot)}).to_csv(path, index=False)
    return path


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
def test_config_rejects_bad_truncation():
    with pytest.raises(Exception):
        PSTConfig(
            input_csv           = Path("a.csv"),
            output_dir          = Path("out"),
            plots_dir           = Path("plots"),
            record_length_years = 100,
            bootstrap           = {"distribution": "gaussian", "truncation": (1.0, -1.0)},
        )


def test_config_rejects_bad_percentile_band():
    with pytest.raises(Exception):
        PSTConfig(
            input_csv                 = Path("a.csv"),
            output_dir                = Path("out"),
            plots_dir                 = Path("plots"),
            record_length_years       = 100,
            threshold_min_percentile  = 80,
            threshold_max_percentile  = 20,
        )


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------
def test_bootstrap_shape_and_order(synthetic_pot):
    gen = BootstrapGenerator(seed=1)
    mat = gen.generate(synthetic_pot, n_simulations=20)
    assert mat.shape == (synthetic_pot.size, 20)
    # Each column is descending-sorted.
    assert np.all(np.diff(mat, axis=0) <= 0)


def test_bootstrap_python_fallback_matches_shape(synthetic_pot):
    gen = BootstrapGenerator(seed=1, use_cpp=False)
    mat = gen.generate(synthetic_pot, n_simulations=10)
    assert mat.shape == (synthetic_pot.size, 10)
    assert np.all(np.diff(mat, axis=0) <= 0)


def test_bootstrap_rejects_ascending():
    gen = BootstrapGenerator(seed=1)
    asc = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        gen.generate(asc, n_simulations=5)


# ---------------------------------------------------------------------------
# GPD location (QDO)
# ---------------------------------------------------------------------------
_QDO_RECORD_LENGTH = 100.0   # lambda_u = n / record_length


def _weibull_aer(values_pot, lambda_val):
    n = values_pot.size
    return (np.arange(1, n + 1) / (n + 1)) * lambda_val


def test_select_threshold_in_band(synthetic_pot):
    lambda_val  = synthetic_pot.size / _QDO_RECORD_LENGTH
    weibull_aer = _weibull_aer(synthetic_pot, lambda_val)
    qdo = select_gpd_threshold_qdo(
        synthetic_pot, weibull_aer, lambda_val, _QDO_RECORD_LENGTH,
        min_percentile=20, max_percentile=80, n_candidates=50,
    )
    # Selection is confined to the band and respects the exceedance floor.
    assert qdo.band_lo <= qdo.best_threshold <= qdo.band_hi
    assert qdo.n_exceed[qdo.best_idx] >= qdo.min_exceedances
    # Per-candidate diagnostics are aligned to the candidate grid.
    for arr in (qdo.wmse, qdo.shape, qdo.scale, qdo.n_exceed, qdo.lambda_mu):
        assert arr.shape == qdo.candidates.shape


def test_qdo_lambda_mu_matches_convention(synthetic_pot):
    """λ_μ per candidate is exactly n_exceed / record_length (hazard convention)."""
    lambda_val  = synthetic_pot.size / _QDO_RECORD_LENGTH
    qdo = select_gpd_threshold_qdo(
        synthetic_pot, _weibull_aer(synthetic_pot, lambda_val),
        lambda_val, _QDO_RECORD_LENGTH, n_candidates=40,
    )
    assert qdo.record_length == pytest.approx(_QDO_RECORD_LENGTH)
    assert np.allclose(qdo.lambda_mu, qdo.n_exceed / _QDO_RECORD_LENGTH)


def test_qdo_shape_is_clipped(synthetic_pot):
    """Fitted ξ never escapes the ensemble's admissible bounds."""
    lambda_val = synthetic_pot.size / _QDO_RECORD_LENGTH
    lo, hi = -0.4, 0.25
    qdo = select_gpd_threshold_qdo(
        synthetic_pot, _weibull_aer(synthetic_pot, lambda_val),
        lambda_val, _QDO_RECORD_LENGTH,
        shape_clip_low=lo, shape_clip_high=hi,
    )
    finite = np.isfinite(qdo.shape)
    assert np.all(qdo.shape[finite] >= lo - 1e-12)
    assert np.all(qdo.shape[finite] <= hi + 1e-12)


def test_qdo_is_deterministic(synthetic_pot):
    lambda_val = synthetic_pot.size / _QDO_RECORD_LENGTH
    aer = _weibull_aer(synthetic_pot, lambda_val)
    a = select_gpd_threshold_qdo(synthetic_pot, aer, lambda_val, _QDO_RECORD_LENGTH)
    b = select_gpd_threshold_qdo(synthetic_pot, aer, lambda_val, _QDO_RECORD_LENGTH)
    assert a.best_threshold == b.best_threshold
    assert np.array_equal(a.wmse, b.wmse, equal_nan=True)


def test_qdo_pure_gpd_is_flat_plateau():
    """For a pure GPD sample, ξ is flat at every threshold (threshold-stability),
    so the selection lands on a low-dispersion plateau with ξ near the true
    value, the lowest-μ candidate (most data), and no instability warning."""
    rng = np.random.default_rng(7)
    xi_true = 0.1
    values_pot = np.sort(genpareto.rvs(xi_true, loc=0.5, scale=1.0,
                                       size=800, random_state=rng))[::-1]
    lambda_val = values_pot.size / 80.0
    q = select_gpd_threshold_qdo(
        values_pot, _weibull_aer(values_pot, lambda_val), lambda_val, 80.0,
        min_percentile=50, max_percentile=95, n_candidates=60, min_exceedances=30,
        selection="stability")
    assert q.selection_method == "stability"
    assert q.selection_warning == ""                       # a flat plateau exists
    assert q.shape_stability[q.best_idx] <= q.stab_tol + 1e-9   # on the plateau
    assert q.shape_clip_low + 1e-3 < q.shape[q.best_idx] <= 0.33  # not clip-pinned
    assert abs(q.shape[q.best_idx] - xi_true) < 0.15       # ξ near the true value
    assert q.best_idx == int(q.selected_set_idx.min())     # lowest-μ on the plateau


def test_qdo_rejects_degenerate_inputs():
    aer = np.linspace(1.0, 0.01, 50)
    # Zero range.
    with pytest.raises(ValueError):
        select_gpd_threshold_qdo(np.full(50, 3.0), aer, 5.0, 100.0)
    # Non-positive record length.
    with pytest.raises(ValueError):
        select_gpd_threshold_qdo(np.linspace(10, 1, 50), aer, 5.0, 0.0)
    # Empty input.
    with pytest.raises(ValueError):
        select_gpd_threshold_qdo(np.empty(0), np.empty(0), 5.0, 100.0)


def test_qdo_rejects_broken_input_contract(synthetic_pot):
    lambda_val = synthetic_pot.size / _QDO_RECORD_LENGTH
    aer = _weibull_aer(synthetic_pot, lambda_val)
    # Ascending (not descending) values.
    with pytest.raises(ValueError):
        select_gpd_threshold_qdo(synthetic_pot[::-1], aer, lambda_val, _QDO_RECORD_LENGTH)
    # Length mismatch between values and plotting positions.
    with pytest.raises(ValueError):
        select_gpd_threshold_qdo(synthetic_pot, aer[:-1], lambda_val, _QDO_RECORD_LENGTH)
    # Non-positive plotting position.
    bad = aer.copy(); bad[0] = 0.0
    with pytest.raises(ValueError):
        select_gpd_threshold_qdo(synthetic_pot, bad, lambda_val, _QDO_RECORD_LENGTH)


def test_gpd_fit_clipped_refits_scale():
    """When ξ is clipped, σ is the constrained MLE (not the unconstrained σ)."""
    rng = np.random.default_rng(11)
    data = 5.0 + genpareto.rvs(0.6, loc=0.0, scale=1.0, size=400, random_state=rng)
    c_u, _, s_u = genpareto.fit(data, floc=5.0)            # unconstrained
    c, loc, s   = fit_gpd_clipped(data, 5.0, -0.5, 0.33)
    _c, _l, s_ref = genpareto.fit(data, fc=0.33, floc=5.0)  # σ | ξ=0.33
    assert c == pytest.approx(0.33) and loc == 5.0
    assert s == pytest.approx(s_ref)        # refit to the constrained MLE
    assert abs(s - s_u) > 1e-6              # and not the stale unconstrained σ
    # No-clip case passes the unconstrained fit through unchanged.
    d2 = 5.0 + genpareto.rvs(0.1, loc=0.0, scale=1.0, size=400, random_state=rng)
    c2, _, s2 = fit_gpd_clipped(d2, 5.0, -0.5, 0.33)
    cu2, _, su2 = genpareto.fit(d2, floc=5.0)
    assert c2 == pytest.approx(cu2) and s2 == pytest.approx(su2)


def test_qdo_selection_is_stability_plateau(synthetic_pot):
    """Selection lands on the stability plateau: the chosen μ is the most stable
    (or lowest-μ) candidate of the within-tol plateau, never lower-clip-pinned."""
    lambda_val = synthetic_pot.size / _QDO_RECORD_LENGTH
    aer = _weibull_aer(synthetic_pot, lambda_val)
    kw  = dict(min_percentile=10, max_percentile=90, n_candidates=60,
               min_exceedances=30, shape_clip_low=-0.5, stab_tol=0.02,
               selection="stability")
    lm = select_gpd_threshold_qdo(synthetic_pot, aer, lambda_val,
                                  _QDO_RECORD_LENGTH, tiebreak="lowest_mu", **kw)
    st = select_gpd_threshold_qdo(synthetic_pot, aer, lambda_val,
                                  _QDO_RECORD_LENGTH, tiebreak="stability", **kw)
    # Both picks are on the (identical) stability plateau and never lower-clip.
    assert lm.selection_method == "stability"
    assert np.array_equal(lm.selected_set_idx, st.selected_set_idx)
    assert lm.best_idx in lm.selected_set_idx and st.best_idx in st.selected_set_idx
    assert lm.best_idx == int(lm.selected_set_idx.min())       # lowest μ
    # stability pick = min dispersion on the plateau (ties -> lowest μ).
    plat = st.selected_set_idx
    assert st.best_idx == int(plat[int(np.argmin(st.shape_stability[plat]))])
    assert st.shape[st.best_idx] > st.shape_clip_low + 1e-3    # not lower-clip-pinned
    # plateau dispersions are all within stab_tol of the minimum.
    d = st.shape_stability[plat]
    assert np.all(d <= d.min() + st.stab_tol + 1e-12)


def test_qdo_excludes_lower_clip_degenerate():
    """The "stability" method must land on a heavy-tail shelf with a finite,
    non-lower-clip ξ - never the bounded sparse tail."""
    rng = np.random.default_rng(5)
    body = genpareto.rvs(0.4, loc=0.5, scale=0.3, size=600, random_state=rng)
    values_pot = np.sort(body)[::-1]
    lambda_val = values_pot.size / 60.0
    q = select_gpd_threshold_qdo(
        values_pot, _weibull_aer(values_pot, lambda_val), lambda_val, 60.0,
        min_percentile=50, max_percentile=95, n_candidates=50, min_exceedances=20,
        selection="stability")
    assert q.shape[q.best_idx] > q.shape_clip_low + 1e-3       # not the bounded tail
    assert q.n_exceed[q.best_idx] >= q.min_exceedances
    assert q.selection_warning == ""                          # a real plateau exists


def test_qdo_mrl_recovers_shape():
    """MRL ("mrl") on a pure GPD recovers ξ from the mean-excess slope and stays
    band-restricted, with its diagnostic arrays populated."""
    rng = np.random.default_rng(3)
    xi_true = 0.2
    vp = np.sort(genpareto.rvs(xi_true, loc=0.5, scale=1.0,
                               size=2000, random_state=rng))[::-1]
    lam = vp.size / 200.0
    q = select_gpd_threshold_qdo(
        vp, _weibull_aer(vp, lam), lam, 200.0,
        min_percentile=5, max_percentile=90, n_candidates=50,
        min_exceedances=30, selection="mrl")
    assert q.selection_method == "mrl"
    assert q.band_lo - 1e-9 <= q.best_threshold <= q.band_hi + 1e-9   # band-restricted
    assert q.mrl_u.size > 0 and q.mrl_wmse.size == q.mrl_u.size
    xi_hat = q.mrl_slope / (1.0 + q.mrl_slope)
    assert abs(xi_hat - xi_true) < 0.2                               # ξ recovered


def test_qdo_mrl_finds_graft():
    """MRL locates the mean-excess linearity onset near a grafted GPD tail."""
    rng = np.random.default_rng(9)
    bulk = rng.uniform(2.0, 5.0, size=1500)
    tail = 5.0 + genpareto.rvs(0.15, loc=0.0, scale=1.0, size=500, random_state=rng)
    vp   = np.sort(np.r_[bulk, tail])[::-1]
    lam  = vp.size / 200.0
    q = select_gpd_threshold_qdo(
        vp, _weibull_aer(vp, lam), lam, 200.0,
        min_percentile=40, max_percentile=95, n_candidates=50,
        min_exceedances=30, selection="mrl")
    assert q.band_lo <= q.best_threshold <= q.band_hi
    assert 3.5 <= q.best_threshold <= 6.0          # neighbourhood of the graft (5)


def test_qdo_gof_failure_to_reject(synthetic_pot):
    """GoF ("gof") returns A²/crit diagnostics and, absent a warning, the
    selected μ is not rejected (statistic <= critical)."""
    lambda_val = synthetic_pot.size / _QDO_RECORD_LENGTH
    q = select_gpd_threshold_qdo(
        synthetic_pot, _weibull_aer(synthetic_pot, lambda_val), lambda_val,
        _QDO_RECORD_LENGTH, min_percentile=10, max_percentile=90,
        n_candidates=60, min_exceedances=30, selection="gof")
    assert q.selection_method == "gof"
    assert q.band_lo <= q.best_threshold <= q.band_hi
    assert q.gof_stat.shape == q.candidates.shape == q.gof_crit.shape
    if not q.selection_warning:                       # μ* not rejected
        assert q.gof_stat[q.best_idx] <= q.gof_crit[q.best_idx] + 1e-9


def test_gpd_fit_mom_matches_formula_and_runs(synthetic_pot):
    """MoM fit matches its closed form, and the pipeline runs with fit_method='mom'."""
    rng = np.random.default_rng(2)
    data = 5.0 + genpareto.rvs(0.2, loc=0.0, scale=1.0, size=500, random_state=rng)
    xi, loc, sg = fit_gpd_clipped(data, 5.0, -0.5, 0.33, method="mom")
    ex = data - 5.0; m = ex.mean(); v = ex.var(ddof=1)
    xi_exp = min(max(0.5 * (1 - m * m / v), -0.5), 0.33)
    assert loc == 5.0 and abs(xi - xi_exp) < 1e-9 and abs(sg - m * (1 - xi_exp)) < 1e-9
    lam = synthetic_pot.size / _QDO_RECORD_LENGTH
    q = select_gpd_threshold_qdo(synthetic_pot, _weibull_aer(synthetic_pot, lam),
                                 lam, _QDO_RECORD_LENGTH, fit_method="mom")
    assert np.isfinite(q.best_threshold)


def test_qdo_default_is_wmse_and_validates(synthetic_pot):
    lambda_val = synthetic_pot.size / _QDO_RECORD_LENGTH
    aer = _weibull_aer(synthetic_pot, lambda_val)
    for bad in (("selection", "nope"), ("tiebreak", "nope"),
                ("fit_method", "nope"), ("gof_statistic", "nope"),
                ("gof_significance", 1.5)):
        with pytest.raises(ValueError):
            select_gpd_threshold_qdo(synthetic_pot, aer, lambda_val,
                                     _QDO_RECORD_LENGTH, **{bad[0]: bad[1]})
    with pytest.raises(ValueError):
        select_gpd_threshold_qdo(synthetic_pot, aer, lambda_val,
                                 _QDO_RECORD_LENGTH, stab_tol=-0.1)
    a = select_gpd_threshold_qdo(synthetic_pot, aer, lambda_val, _QDO_RECORD_LENGTH)
    b = select_gpd_threshold_qdo(synthetic_pot, aer, lambda_val, _QDO_RECORD_LENGTH)
    assert a.best_threshold == b.best_threshold
    assert a.selection_method == "wmse"       # DEFAULT
    assert a.tiebreak == "stability"          # default tie-break
    # WMSE-tolerance set is anchored on the in-band minimum WMSE.
    assert a.selected_set_idx.size >= 1
    assert a.band_lo <= a.best_threshold <= a.band_hi


def test_qdo_floor_too_strict_raises(synthetic_pot):
    lambda_val = synthetic_pot.size / _QDO_RECORD_LENGTH
    with pytest.raises(RuntimeError):
        select_gpd_threshold_qdo(
            synthetic_pot, _weibull_aer(synthetic_pot, lambda_val),
            lambda_val, _QDO_RECORD_LENGTH,
            min_exceedances=synthetic_pot.size + 1,   # impossible floor
        )


# ---------------------------------------------------------------------------
# End-to-end orchestrator
# ---------------------------------------------------------------------------
def test_orchestrator_end_to_end(tmp_path, pot_csv):
    out  = tmp_path / "out"
    plot = tmp_path / "plots"
    cfg  = PSTConfig(
        input_csv           = pot_csv,
        output_dir          = out,
        plots_dir           = plot,
        record_length_years = 100,
        num_simulations     = 50,
        random_seed         = 42,
    )
    result = PSTOrchestrator(cfg).run()

    # Output files exist.
    base = pot_csv.stem.rsplit("_", 1)[0] if "_" in pot_csv.stem else pot_csv.stem
    assert (out  / f"{base}_pst.csv").is_file()
    assert (out  / f"{base}_pst_hc_be_tbl.csv").is_file()
    assert (out  / f"{base}_pst_hc_cb_tbl.csv").is_file()
    assert (out  / f"{base}_pst_hc_be_plt.csv").is_file()
    assert (out  / f"{base}_pst_hc_cb_plt.csv").is_file()
    assert (plot / f"{base}_pst_hc.png").is_file()

    # Result invariants.
    assert result.ensemble.shape[0] == 50
    assert result.aer_table.shape   == (22,)
    assert result.used_cpp_kernel == CPP_KERNEL_AVAILABLE
