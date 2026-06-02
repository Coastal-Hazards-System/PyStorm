/**
 * @file        pst_bindings.cpp
 * @brief       pybind11 layer exposing the PSTBootstrap kernel to Python.
 *
 * @author      Norberto C. Nadal-Caraballo, PhD  <norberto.c.nadal-caraballo@usace.army.mil>
 *
 * Wraps pst::bootstrap() with numpy <-> C++ buffer conversions and publishes
 * it as the `_pst` extension module installed into the
 * `probabilistic_simulation_technique` Python package.
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <cstdint>

#include "PSTBootstrap.hpp"

namespace py = pybind11;

/**
 * Python-callable wrapper around pst::bootstrap.
 *
 * Parameters
 * ----------
 * pot      : (n_pot,) float64 POT values sorted in descending order.
 * n_sims   : number of bootstrap realizations to produce.
 * kind     : 0 = Gaussian (rejection-truncated), 1 = Uniform.
 * trunc_lo : truncation lower bound (must be < trunc_hi).
 * trunc_hi : truncation upper bound.
 * seed     : 64-bit RNG seed.
 *
 * Returns
 * -------
 * boot_matrix : (n_pot, n_sims) float64 ndarray. Column j is the j-th
 *               bootstrap realization, descending-sorted within the column.
 */
py::array_t<double> bootstrap_truncated(
    py::array_t<double, py::array::c_style | py::array::forcecast> pot,
    std::size_t   n_sims,
    int           kind,
    double        trunc_lo,
    double        trunc_hi,
    std::uint64_t seed)
{
    py::buffer_info buf = pot.request();
    if (buf.ndim != 1)
        throw std::runtime_error("pot must be a 1-D array");

    const auto n_pot = static_cast<std::size_t>(buf.shape[0]);
    const double* pot_ptr = static_cast<const double*>(buf.ptr);

    pst::NoiseKind nk;
    switch (kind) {
        case 0:  nk = pst::NoiseKind::Gaussian; break;
        case 1:  nk = pst::NoiseKind::Uniform;  break;
        default: throw std::runtime_error("kind must be 0 (gaussian) or 1 (uniform)");
    }

    std::vector<double> flat = pst::bootstrap(
        pot_ptr, n_pot, n_sims, nk, trunc_lo, trunc_hi, seed);

    // Build a (n_pot, n_sims) numpy array, copy from flat row-major buffer.
    py::array_t<double> out({static_cast<py::ssize_t>(n_pot),
                             static_cast<py::ssize_t>(n_sims)});
    auto out_buf = out.request();
    double* out_ptr = static_cast<double*>(out_buf.ptr);
    std::copy(flat.begin(), flat.end(), out_ptr);
    return out;
}

PYBIND11_MODULE(_pst, m) {
    m.doc() = "C++ accelerated truncated-noise bootstrap for the Probabilistic "
              "Simulation Technique (PST).";

    py::enum_<pst::NoiseKind>(m, "NoiseKind")
        .value("Gaussian", pst::NoiseKind::Gaussian)
        .value("Uniform",  pst::NoiseKind::Uniform);

    m.def("bootstrap_truncated", &bootstrap_truncated,
          py::arg("pot"),
          py::arg("n_sims"),
          py::arg("kind"),
          py::arg("trunc_lo"),
          py::arg("trunc_hi"),
          py::arg("seed"),
          "Generate a truncated-noise bootstrap matrix of POT perturbations.\n\n"
          "Parameters\n"
          "----------\n"
          "pot      : (n_pot,) float64, descending-sorted POT values\n"
          "n_sims   : number of bootstrap realizations\n"
          "kind     : 0 = Gaussian (rejection-truncated), 1 = Uniform\n"
          "trunc_lo : truncation lower bound (must be < trunc_hi)\n"
          "trunc_hi : truncation upper bound\n"
          "seed     : 64-bit RNG seed (deterministic given the same platform)\n\n"
          "Returns\n"
          "-------\n"
          "boot_matrix : (n_pot, n_sims) float64 ndarray; column j is one\n"
          "              realization, sorted in descending order.");
}
