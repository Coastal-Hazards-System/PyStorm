"""bootstrap - smoothed bootstrap of descending-sorted POT exceedances.

Author / POC : Norberto C. Nadal-Caraballo, PhD  <norberto.c.nadal-caraballo@usace.army.mil>

Generates an (n_pot x n_sims) matrix of smoothed-bootstrap POT realizations.
Each column is one resample (with replacement) of the exceedances, perturbed by
additive noise from a smoothing kernel whose bandwidth is the local
order-statistic spacing, then sorted descending. The C++ kernel
(``solver.bootstrap_truncated_cpp``) is preferred; a numpy-only fallback is
used when the extension is not available.

Public API
----------
  BootstrapGenerator(distribution, truncation, seed)
      .generate(pot_values, n_simulations) -> ndarray of shape (n_pot, n_sims)

Algorithm
---------
Step 1 - Verify pot_values is sorted descending (x_1 >= x_2 >= ... >= x_n).
         Pre-compute the order-statistic spacing to the NEXT (i+1, adjacent
         smaller) value:  s = diff(pot) ++ [0], i.e. s[i] = pot[i+1] - pot[i]
         (<= 0); the smallest value (no successor) is padded with 0.
Step 2 - For each realization j: resample n_pot indices WITH REPLACEMENT and
         draw n_pot kernel variates z; form column = pot[idx] + s[idx] * z
         (each drawn value is displaced toward its NEXT-smaller neighbour by a
         fraction of its own spacing); sort the column descending; write to
         output column j. The kernel bandwidth is the local order-statistic
         spacing and |z| is bounded by the kernel support.

The C++ and Python implementations are algorithmically identical and produce
numerically equivalent ensembles (not bit-identical RNG sequences, since they
use different generators).
"""

from typing import Optional, Tuple

import numpy as np
from scipy.stats import truncnorm

from ..solver import (
    CPP_KERNEL_AVAILABLE,
    bootstrap_truncated_cpp,
)


class BootstrapGenerator:
    """Truncated-noise bootstrap generator with C++ acceleration when available.

    Parameters
    ----------
    distribution : {"gaussian", "uniform"}
        Smoothing-kernel family for the additive perturbation ("gaussian" =
        truncated normal, "uniform" = uniform).
    truncation : (float, float)
        Kernel support (lo, hi), in units of the local order-statistic spacing;
        must satisfy lo < hi. Bounds each perturbation to its neighbour gap.
    seed : int or None
        RNG seed. ``None`` selects a non-reproducible seed from the OS.
    use_cpp : bool
        If True (default), prefer the C++ kernel when it is available.
    """

    def __init__(
        self,
        distribution: str                       = "gaussian",
        truncation:   Tuple[float, float]       = (-1.0, 1.0),
        seed:         Optional[int]             = None,
        use_cpp:      bool                      = True,
    ) -> None:
        if distribution.lower() not in ("gaussian", "uniform"):
            raise ValueError(
                f"distribution must be 'gaussian' or 'uniform'; got {distribution!r}"
            )
        lo, hi = truncation
        if not (lo < hi):
            raise ValueError(
                f"truncation lower bound must be < upper bound; got ({lo}, {hi})"
            )
        self.distribution = distribution.lower()
        self.truncation   = (float(lo), float(hi))
        self.seed         = seed
        self.use_cpp      = use_cpp and CPP_KERNEL_AVAILABLE
        self._rng         = np.random.default_rng(seed)

    # ──────────────────────────────────────────────────────────────────────
    # Public entry
    # ──────────────────────────────────────────────────────────────────────
    def generate(self, pot_values, n_simulations: int) -> np.ndarray:
        pot = np.asarray(pot_values, dtype=np.float64).ravel()
        if pot.size == 0:
            raise ValueError("pot_values must be non-empty")
        if not np.all(np.diff(pot) <= 0):
            raise ValueError("pot_values must be sorted in descending order")
        if n_simulations <= 0:
            raise ValueError(f"n_simulations must be > 0; got {n_simulations}")

        if self.use_cpp:
            # The C++ kernel needs an explicit 64-bit seed.  Derive one
            # deterministically from the Python RNG so repeated .generate()
            # calls do not yield identical matrices.
            seed = int(self._rng.integers(0, 2**63 - 1, dtype=np.int64))
            return bootstrap_truncated_cpp(
                pot          = pot,
                n_sims       = int(n_simulations),
                distribution = self.distribution,
                trunc_lo     = self.truncation[0],
                trunc_hi     = self.truncation[1],
                seed         = seed,
            )
        return self._generate_python(pot, int(n_simulations))

    # ──────────────────────────────────────────────────────────────────────
    # Pure-Python fallback (algorithmically identical to the C++ kernel)
    # ──────────────────────────────────────────────────────────────────────
    def _generate_python(self, pot: np.ndarray, n_sims: int) -> np.ndarray:
        n_pot = pot.size
        # Spacing to the NEXT (i+1, adjacent smaller) value; last has none -> 0.
        delta = np.append(np.diff(pot), 0.0)

        lo, hi = self.truncation
        if self.distribution == "gaussian":
            noise = truncnorm.rvs(
                lo, hi, size=(n_pot, n_sims), random_state=self._rng,
            )
        else:  # uniform
            noise = self._rng.uniform(low=lo, high=hi, size=(n_pot, n_sims))

        out = np.empty((n_pot, n_sims), dtype=np.float64)
        for j in range(n_sims):
            idx       = self._rng.integers(low=0, high=n_pot, size=n_pot)
            perturbed = pot[idx] + delta[idx] * noise[:, j]
            out[:, j] = np.sort(perturbed)[::-1]
        return out
