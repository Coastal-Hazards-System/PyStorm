/**
 * @file        pot_bindings.cpp
 * @brief       pybind11 layer exposing the POT threshold-search kernel.
 *
 * @author      Norberto C. Nadal-Caraballo, PhD  <norberto.c.nadal-caraballo@usace.army.mil>
 *
 * Publishes `_pot.find_threshold_for_target()` for the
 * `peaks_over_threshold` Python package. The kernel itself lives in
 * ``POTThresholdSearch.hpp`` (header-only).
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "POTThresholdSearch.hpp"

namespace py = pybind11;

/**
 * Python-callable wrapper for pot::find_threshold_for_target.
 *
 * Parameters
 * ----------
 * values    : (n,) float64 environmental record (values aligned with times_sec).
 * times_sec : (n,) float64 Unix epoch seconds, ascending.
 * interevent_sec         : inter-event gap (s)
 * method                 : 0 = Hydrograph, 1 = PeakGap
 * target_events_per_year : target event rate (events / yr)
 * tolerance              : acceptance window on |rate − target| (events / yr)
 * start_percentile       : initial threshold percentile in [0, 100)
 * step_size              : percentile step per iteration
 * max_iter               : iteration budget
 *
 * Returns
 * -------
 * dict with keys:
 *   "threshold"        : float
 *   "peak_indices"     : int64 ndarray
 *   "converged"        : bool
 *   "iterations"       : int
 *   "events_per_year"  : float
 *   "final_percentile" : float
 */
py::dict find_threshold_for_target(
    py::array_t<double, py::array::c_style | py::array::forcecast> values,
    py::array_t<double, py::array::c_style | py::array::forcecast> times_sec,
    double interevent_sec,
    int    method,
    double target_events_per_year,
    double tolerance,
    double start_percentile,
    double step_size,
    int    max_iter,
    double record_length_years)
{
    py::buffer_info vb = values.request();
    py::buffer_info tb = times_sec.request();
    if (vb.ndim != 1 || tb.ndim != 1)
        throw std::runtime_error("values and times_sec must be 1-D");
    if (vb.shape[0] != tb.shape[0])
        throw std::runtime_error("values and times_sec must have the same length");

    const std::size_t n = static_cast<std::size_t>(vb.shape[0]);
    const double*     v = static_cast<const double*>(vb.ptr);
    const double*     t = static_cast<const double*>(tb.ptr);

    pot::SegmentationMethod sm;
    switch (method) {
        case 0:  sm = pot::SegmentationMethod::Hydrograph; break;
        case 1:  sm = pot::SegmentationMethod::PeakGap;    break;
        default: throw std::runtime_error(
                     "method must be 0 (hydrograph) or 1 (peak_gap)");
    }

    pot::ThresholdSearchResult r = pot::find_threshold_for_target(
        v, t, n,
        interevent_sec, sm, target_events_per_year, tolerance,
        start_percentile, step_size, max_iter, record_length_years);

    // Copy peak indices to int64 numpy.
    py::array_t<std::int64_t> idx_arr(static_cast<py::ssize_t>(r.peak_indices.size()));
    auto idx_buf = idx_arr.request();
    auto* idx_ptr = static_cast<std::int64_t*>(idx_buf.ptr);
    for (std::size_t i = 0; i < r.peak_indices.size(); ++i)
        idx_ptr[i] = static_cast<std::int64_t>(r.peak_indices[i]);

    py::dict out;
    out["threshold"]        = r.threshold;
    out["peak_indices"]     = idx_arr;
    out["converged"]        = r.converged;
    out["iterations"]       = r.iterations;
    out["events_per_year"]  = r.events_per_year;
    out["final_percentile"] = r.final_percentile;
    return out;
}

PYBIND11_MODULE(_pot, m) {
    m.doc() = "C++ accelerated POT threshold-search kernel "
              "(see POTThresholdSearch.hpp).";

    py::enum_<pot::SegmentationMethod>(m, "SegmentationMethod")
        .value("Hydrograph", pot::SegmentationMethod::Hydrograph)
        .value("PeakGap",    pot::SegmentationMethod::PeakGap);

    m.def("find_threshold_for_target", &find_threshold_for_target,
          py::arg("values"),
          py::arg("times_sec"),
          py::arg("interevent_sec"),
          py::arg("method"),
          py::arg("target_events_per_year"),
          py::arg("tolerance"),
          py::arg("start_percentile"),
          py::arg("step_size"),
          py::arg("max_iter"),
          py::arg("record_length_years") = 0.0,
          "Iteratively search a percentile threshold that produces the target "
          "event rate after segmentation. One-sided: returns the highest "
          "threshold whose rate is still >= target. record_length_years > 0 "
          "overrides the time-span duration (effective duration). Returns a "
          "dict with the chosen threshold, peak indices, and diagnostics.");
}
