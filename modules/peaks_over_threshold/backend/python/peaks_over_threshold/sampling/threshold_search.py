"""threshold_search — iterative percentile-threshold search (C++ or Python).

Author / POC : Norberto C. Nadal-Caraballo, PhD  <norberto.c.nadal-caraballo@usace.army.mil>

Increments a percentile threshold from ``start_percentile`` in steps of
``step_size`` until the segmentation-derived event rate is within
``tolerance`` of the target. Defers the inner per-iteration work to the
``_pot`` C++ kernel when available; falls back to a pure-numpy implementation
that produces identical results.

Public API
----------
  ThresholdSearchResult              dataclass holding the converged state
  IterativeThresholdSearch(config)
      .run(values, times_sec) -> ThresholdSearchResult
"""

from dataclasses import dataclass
from typing      import Optional

import numpy as np

from ..segmentation import segment_hydrograph, segment_peak_gap
from ..solver       import (
    CPP_KERNEL_AVAILABLE,
    find_threshold_for_target_cpp,
)


_SECONDS_PER_YEAR = 365.25 * 86400.0


@dataclass
class ThresholdSearchResult:
    threshold:        float
    peak_indices:     np.ndarray         # int64
    converged:        bool
    iterations:       int
    events_per_year:  float
    final_percentile: float
    used_cpp_kernel:  bool


class IterativeThresholdSearch:
    """Threshold-search dispatcher (C++ kernel preferred, Python fallback)."""

    def __init__(
        self,
        interevent_sec:         float,
        method:                 str,
        target_events_per_year: float,
        tolerance:              float = 0.25,
        start_percentile:       float = 75.0,
        step_size:              float = 0.01,
        max_iter:               Optional[int] = None,
        use_cpp:                bool = True,
    ) -> None:
        if method not in ("hydrograph", "peak_gap"):
            raise ValueError(
                f"method must be 'hydrograph' or 'peak_gap'; got {method!r}"
            )
        self.interevent_sec         = float(interevent_sec)
        self.method                 = method
        self.target_events_per_year = float(target_events_per_year)
        self.tolerance              = float(tolerance)
        self.start_percentile       = float(start_percentile)
        self.step_size              = float(step_size)
        self.max_iter               = max_iter
        self.use_cpp                = use_cpp and CPP_KERNEL_AVAILABLE

    # ──────────────────────────────────────────────────────────────────────
    def run(self, values: np.ndarray, times_sec: np.ndarray) -> ThresholdSearchResult:
        v = np.ascontiguousarray(values,    dtype=np.float64).ravel()
        t = np.ascontiguousarray(times_sec, dtype=np.float64).ravel()
        if v.size != t.size:
            raise ValueError("values and times_sec must have the same length")
        if v.size < 2:
            raise ValueError("need at least two samples")
        if not np.all(np.diff(t) >= 0):
            raise ValueError("times_sec must be sorted ascending")

        max_iter = self.max_iter
        if max_iter is None:
            max_iter = int((100.0 - self.start_percentile) / self.step_size)

        if self.use_cpp:
            r = find_threshold_for_target_cpp(
                values                 = v,
                times_sec              = t,
                interevent_sec         = self.interevent_sec,
                method                 = self.method,
                target_events_per_year = self.target_events_per_year,
                tolerance              = self.tolerance,
                start_percentile       = self.start_percentile,
                step_size              = self.step_size,
                max_iter               = max_iter,
            )
            return ThresholdSearchResult(
                threshold        = float(r["threshold"]),
                peak_indices     = np.asarray(r["peak_indices"], dtype=np.int64),
                converged        = bool(r["converged"]),
                iterations       = int(r["iterations"]),
                events_per_year  = float(r["events_per_year"]),
                final_percentile = float(r["final_percentile"]),
                used_cpp_kernel  = True,
            )
        return self._run_python(v, t, max_iter)

    # ──────────────────────────────────────────────────────────────────────
    def _run_python(
        self,
        v:        np.ndarray,
        t:        np.ndarray,
        max_iter: int,
    ) -> ThresholdSearchResult:
        duration_years = float(t[-1] - t[0]) / _SECONDS_PER_YEAR
        if duration_years <= 0.0:
            raise ValueError("time span must be positive")

        sorted_desc = np.sort(v)[::-1]
        n           = v.size
        segmenter   = (segment_hydrograph if self.method == "hydrograph"
                       else segment_peak_gap)

        last_result = ThresholdSearchResult(
            threshold        = float("nan"),
            peak_indices     = np.empty(0, dtype=np.int64),
            converged        = False,
            iterations       = 0,
            events_per_year  = 0.0,
            final_percentile = self.start_percentile,
            used_cpp_kernel  = False,
        )

        percentile = self.start_percentile
        for it in range(max_iter):
            if percentile >= 100.0:
                break
            frac = 1.0 - percentile / 100.0
            k    = int(np.floor(frac * (n - 1)))
            k    = max(0, min(k, n - 1))
            threshold = float(sorted_desc[k])

            exceed_mask  = v > threshold
            exceed_idx   = np.flatnonzero(exceed_mask)
            if exceed_idx.size == 0:
                percentile += self.step_size
                continue

            peak_idx = segmenter(v, t, exceed_idx, self.interevent_sec)
            if peak_idx.size == 0:
                percentile += self.step_size
                continue

            ev_per_yr = peak_idx.size / duration_years

            last_result = ThresholdSearchResult(
                threshold        = threshold,
                peak_indices     = peak_idx,
                converged        = abs(ev_per_yr - self.target_events_per_year) < self.tolerance,
                iterations       = it + 1,
                events_per_year  = ev_per_yr,
                final_percentile = percentile,
                used_cpp_kernel  = False,
            )
            if last_result.converged:
                return last_result
            percentile += self.step_size

        return last_result
