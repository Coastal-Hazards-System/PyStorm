// gpm_kernel.hpp — power-exponential correlation kernel for the GP metamodel.
//
// Author / POC : Norberto C. Nadal-Caraballo, PhD  <norberto.c.nadal-caraballo@usace.army.mil>
//
// R_ij = exp( -sum_k theta_k * |A_ik - B_jk|^p ).  This is the dominant cost of
// GP prediction (a tall m x n kernel against the support set) and of the
// per-evaluation R build during calibration. It is pure elementwise/reduction
// work with no BLAS, so an OpenMP-parallel C++ loop beats the NumPy broadcast
// (which allocates large temporaries). The O(n^3) Cholesky/solves stay in
// LAPACK (via SciPy) where they are already optimal.

#pragma once
#include <cmath>
#include <cstddef>

namespace gpm {

// Fill R (row-major, m x n) with the power-exponential correlation between the
// rows of A (m x d) and B (n x d). p==1 and p==2 are special-cased to avoid pow.
inline void corr(const double* A, const double* B, const double* theta,
                 double p, std::size_t m, std::size_t n, std::size_t d,
                 double* R) {
    const bool p1 = (p == 1.0);
    const bool p2 = (p == 2.0);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (long i = 0; i < static_cast<long>(m); ++i) {
        const double* ai = A + static_cast<std::size_t>(i) * d;
        double* Ri = R + static_cast<std::size_t>(i) * n;
        for (std::size_t j = 0; j < n; ++j) {
            const double* bj = B + j * d;
            double acc = 0.0;
            for (std::size_t k = 0; k < d; ++k) {
                double delta = ai[k] - bj[k];
                if (delta < 0.0) delta = -delta;
                double t;
                if (p2)       t = delta * delta;
                else if (p1)  t = delta;
                else          t = std::pow(delta, p);
                acc += theta[k] * t;
            }
            Ri[j] = std::exp(-acc);
        }
    }
}

}  // namespace gpm
