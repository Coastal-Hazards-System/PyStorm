#pragma once
/**
 * @file        PSTBootstrap.hpp
 * @brief       Truncated-noise bootstrap kernel for the Probabilistic Simulation Technique.
 *
 * @author      Norberto C. Nadal-Caraballo, PhD  <norberto.c.nadal-caraballo@usace.army.mil>
 *
 * Header-only C++ implementation of the inner Monte Carlo loop in the PST
 * Generalized Pareto Distribution (GPD) parameter-uncertainty sweep. Each
 * realization (a) resamples the descending-sorted Peaks-Over-Threshold (POT)
 * exceedances with replacement, (b) perturbs each draw by the local descending
 * spacing (delta) scaled by a truncated noise variate, and (c) re-sorts the
 * result in descending order. The Python orchestrator then fits a GPD to each
 * column and aggregates ICDF evaluations into a hazard-curve ensemble.
 *
 * Public API
 * ----------
 *   pst::bootstrap(pot, n_pot, n_sims, kind, lo, hi, seed)
 *       returns a flat row-major (n_pot * n_sims) std::vector<double>;
 *       element (i, j) lives at index i * n_sims + j.
 *
 * Algorithm
 * ---------
 * Given pot[0..n_pot-1] in **descending** order:
 *   delta[i] = pot[i+1] - pot[i]      for i in [0, n_pot - 1)
 *   delta[n_pot - 1] = 0              (no successor)
 * For each simulation j in [0, n_sims):
 *   For each i in [0, n_pot):
 *     idx  = U{0, ..., n_pot-1}
 *     z    = truncated_noise(lo, hi)              (Gaussian or Uniform)
 *     col[i] = pot[idx] + delta[idx] * z
 *   Sort col descending; write col into output column j.
 *
 * Notes
 * -----
 *   - Gaussian noise uses rejection sampling on N(0, 1) against [lo, hi].
 *     Acceptance is ~0.683 for [-1, +1]; expected wasted draws are negligible
 *     compared to the GPD fit cost on the Python side.
 *   - RNG is std::mt19937_64 seeded from the user-supplied seed; identical
 *     seeds produce identical matrices on the same platform.
 */

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <random>
#include <stdexcept>
#include <vector>

namespace pst {

enum class NoiseKind : int {
    Gaussian = 0,
    Uniform  = 1,
};

inline std::vector<double> bootstrap(
    const double* pot,
    std::size_t   n_pot,
    std::size_t   n_sims,
    NoiseKind     kind,
    double        trunc_lo,
    double        trunc_hi,
    std::uint64_t seed)
{
    if (n_pot == 0)
        throw std::runtime_error("pst::bootstrap: n_pot must be > 0");
    if (n_sims == 0)
        throw std::runtime_error("pst::bootstrap: n_sims must be > 0");
    if (!(trunc_lo < trunc_hi))
        throw std::runtime_error("pst::bootstrap: trunc_lo must be < trunc_hi");

    // Verify descending order (cheap O(n) sanity check).
    for (std::size_t i = 0; i + 1 < n_pot; ++i) {
        if (pot[i + 1] > pot[i])
            throw std::runtime_error(
                "pst::bootstrap: pot[] must be sorted in descending order");
    }

    // Descending spacing: delta[i] = pot[i+1] - pot[i] (<= 0), delta[last] = 0.
    std::vector<double> delta(n_pot);
    for (std::size_t i = 0; i + 1 < n_pot; ++i)
        delta[i] = pot[i + 1] - pot[i];
    delta[n_pot - 1] = 0.0;

    std::vector<double> out(n_pot * n_sims);
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<std::size_t>
        idx_dist(0, n_pot - 1);

    // Noise generator dispatch: capture by reference; one draw per call.
    std::normal_distribution<double>       std_normal(0.0, 1.0);
    std::uniform_real_distribution<double> uniform(trunc_lo, trunc_hi);

    std::function<double()> draw_noise;
    if (kind == NoiseKind::Gaussian) {
        draw_noise = [&]() {
            double z;
            do { z = std_normal(rng); } while (z < trunc_lo || z > trunc_hi);
            return z;
        };
    } else {
        draw_noise = [&]() { return uniform(rng); };
    }

    std::vector<double> column(n_pot);

    for (std::size_t j = 0; j < n_sims; ++j) {
        for (std::size_t i = 0; i < n_pot; ++i) {
            std::size_t idx = idx_dist(rng);
            double      z   = draw_noise();
            column[i] = pot[idx] + delta[idx] * z;
        }
        std::sort(column.begin(), column.end(), std::greater<double>());
        // Row-major write: out[i, j] = column[i].
        for (std::size_t i = 0; i < n_pot; ++i)
            out[i * n_sims + j] = column[i];
    }

    return out;
}

} // namespace pst
