"""Smoke tests: package imports + a tiny end-to-end sampling round-trip.

Author / POC : Norberto C. Nadal-Caraballo, PhD  <norberto.c.nadal-caraballo@usace.army.mil>
"""

import sys
from pathlib import Path

_MODULE_ROOT = Path(__file__).resolve().parents[1]
_PKG_PATH    = _MODULE_ROOT / "backend" / "python"
if str(_PKG_PATH) not in sys.path:
    sys.path.insert(0, str(_PKG_PATH))


def test_package_imports():
    import reduced_storm_suite                             # noqa: F401
    from reduced_storm_suite.config.defaults import RTCS_SELECTION_DEFAULTS
    from reduced_storm_suite.config.loader   import load_config           # noqa: F401
    from reduced_storm_suite.io.readers       import load_array            # noqa: F401
    from reduced_storm_suite.io.store         import read_store            # noqa: F401
    from reduced_storm_suite.sampling.pca     import reduce_output         # noqa: F401
    from reduced_storm_suite.sampling.joint_matrix import build_joint_matrix  # noqa: F401
    from reduced_storm_suite.sampling.kmedoids     import select_kmedoids     # noqa: F401
    from reduced_storm_suite.sampling.metrics      import evaluate_sf_metrics # noqa: F401
    from reduced_storm_suite.weights.dsw      import compute_global_dsw    # noqa: F401
    from reduced_storm_suite.weights.qbm      import compute_qbm_bias      # noqa: F401
    from reduced_storm_suite.postproc.plots   import plot_pca_yspace       # noqa: F401
    from reduced_storm_suite.workflows.rtcs_selection import run_rtcs_selection  # noqa: F401
    from reduced_storm_suite.workflows.growth_evaluation import run_growth_evaluation  # noqa: F401
    from reduced_storm_suite.workflows.ingest        import Preprocessor        # noqa: F401
    assert "TBL_AER" in RTCS_SELECTION_DEFAULTS


def test_kmedoids_round_trip():
    import numpy as np
    from reduced_storm_suite.sampling.joint_matrix import build_joint_matrix
    from reduced_storm_suite.sampling.kmedoids import select_kmedoids
    from reduced_storm_suite.sampling.pca import reduce_output

    rng = np.random.default_rng(0)
    X = rng.standard_normal((50, 4))
    Y = rng.standard_normal((50, 30))

    Y_r, _ = reduce_output(Y, variance_threshold=0.95)
    Z, _, _ = build_joint_matrix(X, Y_r, alpha=1.0, beta=1.0)

    idx = select_kmedoids(Z, k=5, seed=0)
    assert len(idx) == 5
    assert len(np.unique(idx)) == 5
    assert idx.min() >= 0 and idx.max() < 50


def test_kmedoids_forced_indices_path():
    """Exercise the forced-medoids dispatch (C++ binding when available,
    Python BUILD+SWAP otherwise)."""
    import numpy as np
    from reduced_storm_suite.sampling.kmedoids import select_kmedoids

    rng = np.random.default_rng(1)
    Z = rng.standard_normal((40, 6))
    forced = np.array([3, 11, 27], dtype=int)

    idx = select_kmedoids(Z, k=8, seed=0, forced_indices=forced)
    assert len(idx) == 8
    assert len(np.unique(idx)) == 8
    for f in forced:
        assert f in idx, f"forced index {f} dropped"


if __name__ == "__main__":
    test_package_imports()
    test_kmedoids_round_trip()
    test_kmedoids_forced_indices_path()
    print("All smoke tests passed.")
