"""solver - thin wrappers around the _pst pybind11 extension.

Author / POC : Norberto C. Nadal-Caraballo, PhD  <norberto.c.nadal-caraballo@usace.army.mil>

CyHAN v2.1 §4.1 specifies the binding role: a conduit that exposes engine
capability without leaking engine internals upward and without absorbing
orchestration responsibility. This file is intentionally minimal - the
higher-level dispatch/fallback (pure-Python bootstrap, configuration mapping)
belongs in ``sampling/bootstrap.py``.

Public API
----------
  CPP_KERNEL_AVAILABLE              bool - True iff ``_pst`` imported
  bootstrap_truncated_cpp(...)      direct call into the C++ kernel
"""

import numpy as np

try:
    from . import _pst  # type: ignore[attr-defined]
    CPP_KERNEL_AVAILABLE = True
except ImportError:
    _pst = None
    CPP_KERNEL_AVAILABLE = False


# Mapping from BootstrapConfig.distribution to the C++ NoiseKind enum value.
_KIND_TO_INT = {"gaussian": 0, "uniform": 1}


def bootstrap_truncated_cpp(
    pot:          np.ndarray,
    n_sims:       int,
    distribution: str,
    trunc_lo:     float,
    trunc_hi:     float,
    seed:         int,
) -> np.ndarray:
    """Call ``_pst.bootstrap_truncated`` and return a ``(n_pot, n_sims)`` array.

    Parameters mirror ``probabilistic_simulation_technique.config.BootstrapConfig``
    with the addition of an explicit integer seed. Raises ``RuntimeError`` if
    the C++ extension is not built or installed.
    """
    if not CPP_KERNEL_AVAILABLE:
        raise RuntimeError(
            "_pst extension is not available; build it with "
            "`python backend/engines/build.py` or use the pure-Python fallback "
            "in sampling/bootstrap.py."
        )

    kind = _KIND_TO_INT.get(distribution.lower())
    if kind is None:
        raise ValueError(
            f"unknown distribution '{distribution}'; expected one of "
            f"{list(_KIND_TO_INT)}"
        )

    pot_arr = np.ascontiguousarray(pot, dtype=np.float64).ravel()
    return _pst.bootstrap_truncated(
        pot_arr, int(n_sims), int(kind), float(trunc_lo), float(trunc_hi), int(seed),
    )
