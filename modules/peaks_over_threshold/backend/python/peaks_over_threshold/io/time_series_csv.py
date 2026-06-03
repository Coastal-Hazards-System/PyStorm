"""time_series_csv — read input time series, write POT peaks.

Author / POC : Norberto C. Nadal-Caraballo, PhD  <norberto.c.nadal-caraballo@usace.army.mil>

The reader normalizes the input columns to ``("datetime", "value")``,
preserving NaN rows so downstream plotting can break the line on gaps.
The writer emits a two-column CSV (``datetime``, ``value``) for the selected
peaks, matching the v1 layout consumed by the PST stage.
"""

from pathlib import Path

import pandas as pd


def read_time_series_csv(
    path:         Path,
    datetime_col: str,
    value_col:    str,
) -> pd.DataFrame:
    """Load and normalize a time-series CSV.

    Returns a DataFrame with columns ``("datetime", "value")`` sorted
    ascending by datetime. Rows with unparseable datetimes are dropped;
    rows with NaN values are preserved (gap plotting needs them).
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"input CSV not found: {path}")

    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    if datetime_col not in df.columns:
        raise KeyError(f"missing datetime column '{datetime_col}' in {path.name}")
    if value_col not in df.columns:
        raise KeyError(f"missing value column '{value_col}' in {path.name}")

    df[datetime_col] = pd.to_datetime(df[datetime_col], errors="coerce")
    df = df.sort_values(datetime_col).reset_index(drop=True)
    df = df[[datetime_col, value_col]].copy()
    df.columns = ["datetime", "value"]
    df.dropna(subset=["datetime"], inplace=True)
    return df.reset_index(drop=True)


def write_pot_peaks(
    output_dir:    Path,
    base_filename: str,
    peaks:         pd.DataFrame,
) -> Path:
    """Write the selected POT peaks to ``<base>_pot.csv``.

    Returns the full output path.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{base_filename}_pot.csv"
    peaks[["datetime", "value"]].to_csv(out_path, index=False)
    return out_path


def write_series_csv(
    df:             pd.DataFrame,
    out_path:       Path,
    value_col:      str = "value",
    datetime_header: str = "Date Time",
    value_header:    str = "Value",
) -> Path:
    """Write a normalized series to CSV with human-friendly headers.

    Reads ``("datetime", <value_col>)`` from ``df`` and emits a two-column CSV
    headed ``(datetime_header, value_header)`` — e.g. ``("Date Time", "NTR")``.
    Blank cells are normalized to ``NaN``. Returns the output path.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out = df[["datetime", value_col]].rename(
        columns={"datetime": datetime_header, value_col: value_header}
    )
    out.to_csv(out_path, index=False, na_rep="NaN")
    return out_path
