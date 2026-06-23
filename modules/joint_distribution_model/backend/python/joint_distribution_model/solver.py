"""solver - thin wrapper around the _jdm pybind11 extension (C++ bootstrap kernel).

Author : Norberto C. Nadal-Caraballo, PhD  <norberto.c.nadal-caraballo@usace.army.mil>

Conduit role (CyHAN v2.2 §4.1): exposes the C++ bootstrap kernel without leaking
engine internals upward and without absorbing orchestration. Higher-level dispatch
and the pure-Python fallback live in ``marginals.fit_weibull_boot``.

Public API
----------
  CPP_KERNEL_AVAILABLE                bool - True iff `_jdm` imported
  weibull_bootstrap_cpp(...)          direct call into the C++ kernel
"""

from __future__ import annotations

import numpy as np

try:
    from . import _jdm  # type: ignore[attr-defined]
    CPP_KERNEL_AVAILABLE = True
except ImportError:
    _jdm = None
    CPP_KERNEL_AVAILABLE = False


def weibull_bootstrap_cpp(sample: np.ndarray, n_boot: int, th: float,
                          seed: int) -> np.ndarray:
    """Bootstrap a Weibull fit in C++: returns ``[n_boot, 2]`` of (scale A, shape k).

    Resamples the sample with replacement, jitters by the local order-statistic
    spacing (rejecting any replicate that falls below ``th``), and fits a
    two-parameter Weibull (MLE) to each replicate. Deterministic given ``seed``.
    Raises RuntimeError if the extension is not built.
    """
    if not CPP_KERNEL_AVAILABLE:
        raise RuntimeError(
            "_jdm extension is not available; build it with "
            "`python backend/engines/cpp/build.py` or use the NumPy fallback.")
    s = np.ascontiguousarray(sample, dtype=np.float64).ravel()
    s = s[np.isfinite(s)]
    return _jdm.weibull_bootstrap(s, int(n_boot), float(th), int(seed))
