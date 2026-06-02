#pragma once
/**
 * @file        POTThresholdSearch.hpp
 * @brief       Iterative percentile-threshold search for POT event detection.
 *
 * @author      Norberto C. Nadal-Caraballo, PhD  <norberto.c.nadal-caraballo@usace.army.mil>
 *
 * Header-only kernel for the POT inner loop. Given a sorted-by-time
 * environmental record (values, times), iterates a percentile-based threshold
 * upward from `start_percentile` in steps of `step_size` until the event rate
 * derived from the chosen segmentation method is within `tolerance` of the
 * target events-per-year. Returns the converged threshold and the indices of
 * the selected peak rows in the original input ordering.
 *
 * Public API
 * ----------
 *   pot::SegmentationMethod  Hydrograph | PeakGap
 *   pot::ThresholdSearchResult
 *   pot::find_threshold_for_target(...)
 *
 * Algorithm
 * ---------
 * Step 1 — Pre-sort values descending (separately) for O(1) percentile lookup.
 * Step 2 — For each iteration `i` of `max_iter`:
 *           p   = start_percentile + i * step_size
 *           thr = sorted_values[k] where k = floor((1 - p/100) * (n - 1))
 *           Filter exceedances (i: values[i] > thr) preserving time order.
 *           Segment into independent events per method.
 *           events_per_year = n_events / duration_years
 *           If |events_per_year - target| < tolerance: converged.
 *
 * Segmentation
 * ------------
 *  - Hydrograph: group consecutive exceedances whose time-gap to the previous
 *    exceedance is <= interevent_sec; take the per-group argmax.
 *  - PeakGap   : within a contiguous block of exceedances, drop any sample
 *    whose preceding neighbour is within interevent_sec AND has a larger
 *    value (the v1 sequential filter).
 *
 * Notes
 * -----
 *  - Times are passed as Unix epoch seconds (float64), already ascending.
 *  - Duration in years is (times.back() - times.front()) / (365.25 * 86400).
 *  - All operations are deterministic; no RNG is used.
 */

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace pot {

enum class SegmentationMethod : int {
    Hydrograph = 0,
    PeakGap    = 1,
};

struct ThresholdSearchResult {
    double                   threshold     {0.0};
    std::vector<std::size_t> peak_indices  {};       // indices into the input arrays
    bool                     converged     {false};
    int                      iterations    {0};
    double                   events_per_year{0.0};
    double                   final_percentile{0.0};
};

namespace detail {

inline std::vector<std::size_t> segment_hydrograph(
    const double* values,
    const double* times_sec,
    const std::vector<std::size_t>& exceed_idx,
    double interevent_sec)
{
    // Groups consecutive exceedances by time-gap > interevent_sec; per group
    // selects the argmax of values. exceed_idx is already in ascending time order.
    std::vector<std::size_t> peaks;
    if (exceed_idx.empty())
        return peaks;

    std::size_t group_start = 0;
    auto flush_group = [&](std::size_t end_exclusive) {
        std::size_t best = exceed_idx[group_start];
        for (std::size_t k = group_start + 1; k < end_exclusive; ++k) {
            const std::size_t idx = exceed_idx[k];
            if (values[idx] > values[best]) best = idx;
        }
        peaks.push_back(best);
    };

    for (std::size_t k = 1; k < exceed_idx.size(); ++k) {
        const double dt = times_sec[exceed_idx[k]] - times_sec[exceed_idx[k - 1]];
        if (dt > interevent_sec) {
            flush_group(k);
            group_start = k;
        }
    }
    flush_group(exceed_idx.size());
    return peaks;
}

inline std::vector<std::size_t> segment_peak_gap(
    const double* values,
    const double* times_sec,
    const std::vector<std::size_t>& exceed_idx,
    double interevent_sec)
{
    // Mirrors v1 sequential filter: keep[i] = true initially; if the gap to
    // the previous kept sample is < interevent_sec and values[i] <= prev, drop it.
    std::vector<std::size_t> peaks;
    if (exceed_idx.empty())
        return peaks;

    std::vector<bool> keep(exceed_idx.size(), true);
    for (std::size_t k = 1; k < exceed_idx.size(); ++k) {
        const double dt =
            times_sec[exceed_idx[k]] - times_sec[exceed_idx[k - 1]];
        if (dt < interevent_sec &&
            values[exceed_idx[k]] <= values[exceed_idx[k - 1]]) {
            keep[k] = false;
        }
    }
    peaks.reserve(exceed_idx.size());
    for (std::size_t k = 0; k < exceed_idx.size(); ++k)
        if (keep[k]) peaks.push_back(exceed_idx[k]);
    return peaks;
}

} // namespace detail

inline ThresholdSearchResult find_threshold_for_target(
    const double*   values,
    const double*   times_sec,
    std::size_t     n,
    double          interevent_sec,
    SegmentationMethod method,
    double          target_events_per_year,
    double          tolerance,
    double          start_percentile,
    double          step_size,
    int             max_iter)
{
    if (n < 2)
        throw std::runtime_error("pot::find_threshold_for_target: need n >= 2");
    if (interevent_sec <= 0.0)
        throw std::runtime_error("interevent_sec must be > 0");
    if (target_events_per_year <= 0.0)
        throw std::runtime_error("target_events_per_year must be > 0");
    if (!(start_percentile >= 0.0 && start_percentile < 100.0))
        throw std::runtime_error("start_percentile must be in [0, 100)");
    if (step_size <= 0.0)
        throw std::runtime_error("step_size must be > 0");
    if (max_iter <= 0)
        throw std::runtime_error("max_iter must be > 0");

    // Verify times are sorted ascending (cheap O(n) check).
    for (std::size_t i = 1; i < n; ++i)
        if (times_sec[i] < times_sec[i - 1])
            throw std::runtime_error("times_sec must be sorted ascending");

    const double duration_sec =
        times_sec[n - 1] - times_sec[0];
    const double duration_years = duration_sec / (365.25 * 86400.0);
    if (duration_years <= 0.0)
        throw std::runtime_error(
            "time span must be positive (last time > first time)");

    // Pre-sort values descending to evaluate percentiles in O(1) per iteration.
    std::vector<double> sorted_desc(values, values + n);
    std::sort(sorted_desc.begin(), sorted_desc.end(), std::greater<double>());

    ThresholdSearchResult result{};
    double percentile = start_percentile;

    for (int iter = 0; iter < max_iter; ++iter, percentile += step_size) {
        if (percentile >= 100.0) break;

        // Percentile threshold = the value such that p% of samples are <= threshold.
        // Equivalent: the (1 - p/100)-th element of the descending sort.
        const double frac = 1.0 - percentile / 100.0;
        std::size_t   k   = static_cast<std::size_t>(
                                std::floor(frac * static_cast<double>(n - 1)));
        if (k >= n) k = n - 1;
        const double threshold = sorted_desc[k];

        // Collect exceedance indices in input (time) order.
        std::vector<std::size_t> exceed_idx;
        exceed_idx.reserve(n / 4);
        for (std::size_t i = 0; i < n; ++i)
            if (values[i] > threshold) exceed_idx.push_back(i);

        if (exceed_idx.empty()) continue;

        std::vector<std::size_t> peaks =
            (method == SegmentationMethod::Hydrograph)
                ? detail::segment_hydrograph(values, times_sec, exceed_idx,
                                             interevent_sec)
                : detail::segment_peak_gap (values, times_sec, exceed_idx,
                                             interevent_sec);

        if (peaks.empty()) continue;

        const double ev_per_yr =
            static_cast<double>(peaks.size()) / duration_years;

        if (std::fabs(ev_per_yr - target_events_per_year) < tolerance) {
            result.threshold        = threshold;
            result.peak_indices     = std::move(peaks);
            result.converged        = true;
            result.iterations       = iter + 1;
            result.events_per_year  = ev_per_yr;
            result.final_percentile = percentile;
            return result;
        }

        // Track the last evaluated state in case we exhaust iterations.
        result.threshold        = threshold;
        result.peak_indices     = std::move(peaks);
        result.iterations       = iter + 1;
        result.events_per_year  = ev_per_yr;
        result.final_percentile = percentile;
    }

    result.converged = false;
    return result;
}

} // namespace pot
