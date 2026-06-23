/**
 * @file        jdm_bindings.cpp
 * @brief       pybind11 layer exposing the JDM Weibull-bootstrap kernel.
 *
 * @author      Norberto C. Nadal-Caraballo, PhD  <norberto.c.nadal-caraballo@usace.army.mil>
 *
 * Wraps jdm::weibull_bootstrap() with numpy <-> C++ conversions and publishes it as
 * the `_jdm` extension module in the joint_distribution_model package. Conduit only:
 * it exposes engine capability without adding orchestration (CyHAN v2.2 §4.1). The
 * kernel itself lives in JDMBootstrap.hpp (header-only).
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <cstring>
#include <stdexcept>

#include "JDMBootstrap.hpp"

namespace py = pybind11;

/**
 * Python-callable wrapper for jdm::weibull_bootstrap.
 *
 * Parameters
 * ----------
 * sample : (n,) float64 peak sample (NaNs already dropped), n >= 2.
 * n_boot : number of bootstrap replicates.
 * th     : truncation floor (replicates with any value below it are redrawn).
 * seed   : RNG seed (reproducible).
 *
 * Returns
 * -------
 * (n_boot, 2) float64 array of (scale A, shape k) per replicate.
 */
py::array_t<double> weibull_bootstrap(
    py::array_t<double, py::array::c_style | py::array::forcecast> sample,
    int n_boot, double th, std::uint64_t seed)
{
    py::buffer_info sb = sample.request();
    if (sb.ndim != 1)
        throw std::runtime_error("sample must be 1-D");
    const std::size_t n = static_cast<std::size_t>(sb.shape[0]);
    if (n < 2)
        throw std::runtime_error("sample needs at least two values");
    if (n_boot < 1)
        throw std::runtime_error("n_boot must be positive");
    const double* s = static_cast<const double*>(sb.ptr);

    std::vector<double> res;
    {
        py::gil_scoped_release release;            // pure C++ compute, no Python
        res = jdm::weibull_bootstrap(s, n, n_boot, th, seed);
    }

    py::array_t<double> out({static_cast<py::ssize_t>(n_boot),
                             static_cast<py::ssize_t>(2)});
    std::memcpy(out.request().ptr, res.data(), res.size() * sizeof(double));
    return out;
}

PYBIND11_MODULE(_jdm, m) {
    m.doc() = "C++ accelerated JDM bootstrap kernel (see JDMBootstrap.hpp).";
    m.def("weibull_bootstrap", &weibull_bootstrap,
          py::arg("sample"), py::arg("n_boot"), py::arg("th"), py::arg("seed"),
          "Bootstrap a two-parameter Weibull fit: returns an (n_boot, 2) array of "
          "(scale A, shape k) per replicate. Resample with replacement + "
          "order-statistic-spacing jitter (rejecting replicates below th) + a "
          "per-replicate maximum-likelihood fit. Deterministic given seed; releases "
          "the GIL during compute.");
}
