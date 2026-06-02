"""pot_csv — load POT input and write PST result CSVs.

Author / POC : Norberto C. Nadal-Caraballo, PhD  <norberto.c.nadal-caraballo@usace.army.mil>

Output layout (one set of files per run, ``base_filename`` prefix)::

  <base>_PST.csv             full bootstrap ensemble (n_sims rows, plot-grid cols)
  <base>_PST_HC_BE_tbl.csv   table-grid Best Estimate
  <base>_PST_HC_CB_tbl.csv   table-grid CB10 / CB90
  <base>_PST_HC_BE_plt.csv   plot-grid Best Estimate (merged GPD + empirical)
  <base>_PST_HC_CB_plt.csv   plot-grid CB10 / CB90
"""

from pathlib import Path
from typing  import Optional

import numpy as np
import pandas as pd


def read_pot_csv(path: Path, column: str = "value") -> np.ndarray:
    """Load a POT CSV and return the named column as a float64 numpy array.

    Rows whose value is NaN are dropped.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"POT CSV not found: {path}")
    df = pd.read_csv(path)
    if column not in df.columns:
        raise KeyError(
            f"column '{column}' not in {path.name}; available: {list(df.columns)}"
        )
    return df[column].dropna().to_numpy(dtype=np.float64)


def write_pst_outputs(
    output_dir:    Path,
    base_filename: str,
    ensemble:      np.ndarray,
    aef_plot:      np.ndarray,
    aef_table:     np.ndarray,
    hc_table_be:   np.ndarray,
    hc_table_cb10: np.ndarray,
    hc_table_cb90: np.ndarray,
    hc_plot_aef:   np.ndarray,
    hc_plot_be:    np.ndarray,
    hc_plot_cb10:  np.ndarray,
    hc_plot_cb90:  np.ndarray,
) -> None:
    """Write the full PST output bundle under ``output_dir``."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        ensemble,
        columns=[f"{x:.6f}" for x in aef_plot],
    ).to_csv(output_dir / f"{base_filename}_PST.csv", index=False)

    pd.DataFrame({"AEF": aef_table, "BE": hc_table_be}).to_csv(
        output_dir / f"{base_filename}_PST_HC_BE_tbl.csv", index=False)
    pd.DataFrame({"AEF": aef_table, "CB10": hc_table_cb10, "CB90": hc_table_cb90}).to_csv(
        output_dir / f"{base_filename}_PST_HC_CB_tbl.csv", index=False)

    pd.DataFrame({"AEF": hc_plot_aef, "BE": hc_plot_be}).to_csv(
        output_dir / f"{base_filename}_PST_HC_BE_plt.csv", index=False)
    pd.DataFrame({"AEF": hc_plot_aef, "CB10": hc_plot_cb10, "CB90": hc_plot_cb90}).to_csv(
        output_dir / f"{base_filename}_PST_HC_CB_plt.csv", index=False)
