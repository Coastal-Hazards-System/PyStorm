#pragma once
/**
 * @file        JDMBootstrap.hpp
 * @brief       Jitter-bootstrap of a two-parameter Weibull fit (JDM Dp marginal).
 *
 * @author      Norberto C. Nadal-Caraballo, PhD  <norberto.c.nadal-caraballo@usace.army.mil>
 *
 * Header-only kernel for the JDM bootstrap inner loop, the module's compute hot
 * spot. Given a peak sample and a truncation floor, it draws n_boot resamples with
 * replacement, jitters each by the local order-statistic spacing (rejecting any
 * replicate that falls below the floor), and fits a two-parameter Weibull (scale A,
 * shape k) to each replicate by maximum likelihood. Deterministic given the seed.
 *
 * Engine contract: arrays in, arrays out. No config, no application I/O; the
 * pybind11 layer in jdm_bindings.cpp handles numpy <-> C++ and the GIL.
 *
 * Public API
 * ----------
 *   jdm::weibull_mle(x, n, A, k)                  single 2-parameter Weibull MLE
 *   jdm::weibull_bootstrap(sample, n, n_boot, th, seed) -> [n_boot*2] (A,k) rows
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <random>
#include <vector>

namespace jdm {

/**
 * Two-parameter Weibull MLE (location fixed at 0) on a positive sample.
 *
 * Solves the shape score g(k) = S1/S0 - 1/k - mean(ln x) = 0 by safeguarded Newton
 * iteration (S0,S1,S2 are the 0th/1st/2nd log-moments of x^k), then sets the scale
 * A = (mean x^k)^(1/k). @param A,k are outputs.
 */
inline void weibull_mle(const double* x, std::size_t n, double& A, double& k,
                        double* lnx_scratch = nullptr) {
    // Precompute ln(x) once and use x^k = exp(k*ln x) (faster than repeated pow()).
    static thread_local std::vector<double> lnx_tl;
    double* lnx = lnx_scratch;
    if (lnx == nullptr) { lnx_tl.resize(n); lnx = lnx_tl.data(); }
    double mlnx = 0.0;
    for (std::size_t i = 0; i < n; ++i) { lnx[i] = std::log(x[i]); mlnx += lnx[i]; }
    mlnx /= static_cast<double>(n);

    auto moments = [&](double kk, double& g, double& gp) -> double {
        double S0 = 0.0, S1 = 0.0, S2 = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            double l = lnx[i];
            double xk = std::exp(kk * l);
            S0 += xk; S1 += xk * l; S2 += xk * l * l;
        }
        g  = S1 / S0 - 1.0 / kk - mlnx;
        gp = (S2 * S0 - S1 * S1) / (S0 * S0) + 1.0 / (kk * kk);
        return S0;
    };

    double kk = 1.0;
    for (int it = 0; it < 100; ++it) {
        double g, gp;
        moments(kk, g, gp);
        double kn = kk - g / gp;
        if (kn < 1e-3) kn = 1e-3;
        if (kn > 50.0) kn = 50.0;
        if (std::fabs(kn - kk) < 1e-10) { kk = kn; break; }
        kk = kn;
    }
    double g, gp;
    double S0 = moments(kk, g, gp);
    k = kk;
    A = std::pow(S0 / static_cast<double>(n), 1.0 / kk);
}

/**
 * Bootstrap a Weibull fit. Returns n_boot*2 doubles, row-major (A, k) per replicate.
 *
 * @param sample  pointer to the peak sample (n values, NaNs already dropped).
 * @param n       sample size (>= 2).
 * @param n_boot  number of bootstrap replicates.
 * @param th      truncation floor; a replicate with any value < th is redrawn.
 * @param seed    RNG seed (reproducible).
 */
inline std::vector<double> weibull_bootstrap(const double* sample, std::size_t n,
                                             int n_boot, double th,
                                             std::uint64_t seed) {
    std::vector<double> s(sample, sample + n);
    std::sort(s.begin(), s.end(), std::greater<double>());   // descending

    std::vector<double> dlt(n);                              // local spacing
    for (std::size_t i = 0; i + 1 < n; ++i) dlt[i] = std::fabs(s[i] - s[i + 1]);
    dlt[n - 1] = (n >= 2) ? dlt[n - 2] : 0.0;

    std::vector<double> out(static_cast<std::size_t>(n_boot) * 2);
    std::mt19937_64 gen(seed);
    std::uniform_int_distribution<std::size_t> uidx(0, n - 1);
    std::normal_distribution<double> norm(0.0, 1.0);
    std::vector<double> y(n);

    for (int b = 0; b < n_boot; ++b) {
        bool ok = false;
        while (!ok) {                                        // reject below the floor
            ok = true;
            for (std::size_t i = 0; i < n; ++i) {
                std::size_t j = uidx(gen);
                y[i] = s[j] + norm(gen) * dlt[j];
                if (y[i] < th) ok = false;
            }
        }
        double A, k;
        weibull_mle(y.data(), n, A, k);
        out[static_cast<std::size_t>(b) * 2 + 0] = A;
        out[static_cast<std::size_t>(b) * 2 + 1] = k;
    }
    return out;
}

}  // namespace jdm
