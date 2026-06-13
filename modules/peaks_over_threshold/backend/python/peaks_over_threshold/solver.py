"""solver - thin wrappers around the _pot pybind11 extension.

Author : Norberto C. Nadal-Caraballo, PhD  <norberto.c.nadal-caraballo@usace.army.mil>

CyHAN v2.2 §4.1 specifies the binding role: a conduit exposing engine
capability without leaking engine internals upward and without absorbing
orchestration responsibility. Higher-level dispatch and fallback live in
``sampling/threshold_search.py``.

Public API
----------
  CPP_KERNEL_AVAILABLE                   bool - True iff `_pot` imported
  find_threshold_for_target_cpp(...)     direct call into the C++ kernel
"""

from typing import Dict

import numpy as np

try:
    from . import _pot  # type: ignore[attr-defined]
    CPP_KERNEL_AVAILABLE = True
except ImportError:
    _pot = None
    CPP_KERNEL_AVAILABLE = False


_METHOD_TO_INT = {"hydrograph": 0, "peak_gap": 1}


def find_threshold_for_target_cpp(
    values:                 np.ndarray,
    times_sec:              np.ndarray,
    interevent_sec:         float,
    method:                 str,
    target_events_per_year: float,
    tolerance:              float,
    start_percentile:       float,
    step_size:              float,
    max_iter:               int,
    record_length_years:    float = 0.0,
) -> Dict:
    """Call ``_pot.find_threshold_for_target`` and return its result dict.

    Raises ``RuntimeError`` if the C++ extension is not built or installed.
    """
    if not CPP_KERNEL_AVAILABLE:
        raise RuntimeError(
            "_pot extension is not available; build it with "
            "`python backend/engines/cpp/build.py` or use the pure-Python "
            "fallback in sampling/threshold_search.py."
        )

    kind = _METHOD_TO_INT.get(method.lower())
    if kind is None:
        raise ValueError(
            f"unknown method '{method}'; expected one of {list(_METHOD_TO_INT)}"
        )

    v = np.ascontiguousarray(values,    dtype=np.float64).ravel()
    t = np.ascontiguousarray(times_sec, dtype=np.float64).ravel()
    return _pot.find_threshold_for_target(
        v, t,
        float(interevent_sec),
        int(kind),
        float(target_events_per_year),
        float(tolerance),
        float(start_percentile),
        float(step_size),
        int(max_iter),
        float(record_length_years),
    )
