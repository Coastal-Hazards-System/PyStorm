"""plots — hazard-curve renderer for PST results.

Author / POC : Norberto C. Nadal-Caraballo, PhD  <norberto.c.nadal-caraballo@usace.army.mil>

Plots the empirical bulk (Weibull plotting positions), the GPD best-estimate
curve, the 10/90% confidence band, and the GPD-threshold anchor on a log-AEF
x-axis (reverse-oriented so high-frequency events sit on the right).
"""

from pathlib import Path
from typing  import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


class HazardCurvePlotter:
    """Render a PST hazard curve onto a matplotlib Axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Destination axes; the caller owns the figure lifecycle.
    lambda_val : float
        Sample intensity (events/yr). Retained for downstream consumers that
        annotate the curve with the population rate.
    """

    def __init__(self, ax, lambda_val: float = 1.0) -> None:
        self.ax         = ax
        self.lambda_val = lambda_val

    def plot_hazard_curve(
        self,
        empirical_cdf: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        gpd_curve:     Optional[Tuple[np.ndarray, np.ndarray]] = None,
        gpd_cb:        Optional[Tuple[np.ndarray, np.ndarray]] = None,
        gpd_threshold: Optional[float]                         = None,
        ylabel:        str                                     = "Response Magnitude",
        output_path:   Optional[Path]                          = None,
    ) -> None:
        """Render the hazard curve.

        Parameters
        ----------
        empirical_cdf : (aef_emp, resp_emp) or None
            Empirical bulk (Weibull plotting positions below the threshold).
        gpd_curve : (aef_gpd, resp_gpd) or None
            GPD best-estimate response above the threshold.
        gpd_cb : (cb10, cb90) or None
            Confidence-bound responses; share x-coordinates with ``gpd_curve``.
        gpd_threshold : float or None
            Response value at the empirical-GPD transition.
        ylabel : str
            Y-axis label.
        output_path : pathlib.Path or None
            If given, the figure is saved at 150 DPI to this path.
        """
        if empirical_cdf is not None:
            aef_emp, resp_emp = empirical_cdf
            self.ax.scatter(aef_emp, resp_emp, color="green", s=20, zorder=3,
                            label="Empirical (WPP)", edgecolors="k")

        aef_gpd = None
        if gpd_curve is not None:
            aef_gpd, resp_gpd = gpd_curve
            self.ax.plot(aef_gpd, resp_gpd, color="blue", linewidth=2,
                         label="GPD Mean")

        if gpd_cb is not None and aef_gpd is not None:
            cb10, cb90 = gpd_cb
            self.ax.fill_between(aef_gpd, cb10, cb90, color="blue", alpha=0.2,
                                 label="GPD 10–90% CBs")

        if gpd_threshold is not None:
            self.ax.axhline(gpd_threshold, linestyle="--", color="gray",
                            linewidth=1.5,
                            label=f"GPD Threshold = {gpd_threshold:.2f}")

        self.ax.set_xscale("log")
        self.ax.set_xlim(1e-3, 10)
        self.ax.set_xticks([1e-3, 1e-2, 1e-1, 1, 10])
        self.ax.set_xlabel("Annual Exceedance Frequency (AEF, yr⁻¹)", fontsize=12)
        self.ax.set_ylabel(ylabel, fontsize=12)
        self.ax.set_title("PyStorm — Probabilistic Simulation Technique (PST)",
                          fontsize=13)
        self.ax.grid(True, which="both", linestyle="--", alpha=0.5)
        self.ax.legend()
        self.ax.invert_xaxis()  # high-frequency events sit on the right

        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=150)
