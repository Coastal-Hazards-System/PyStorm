"""writer - output writers for the parsed HURDAT2 DataFrame.

Author : Norberto C. Nadal-Caraballo, PhD  <norberto.c.nadal-caraballo@usace.army.mil>
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def write_metrics(metrics: dict, path: Path) -> Path:
    """Write a per-run metrics record (LOOCV scores and settings) as JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")
    return path


def write_csv(df: pd.DataFrame, path: Path) -> Path:
    """Write ``df`` to ``path`` as UTF-8 CSV (creating parent dirs)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")
    return path


def write_parquet(df: pd.DataFrame, path: Path) -> Path:
    """Write ``df`` to ``path`` as Parquet (zstd via pyarrow)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False, engine="pyarrow", compression="zstd")
    return path
