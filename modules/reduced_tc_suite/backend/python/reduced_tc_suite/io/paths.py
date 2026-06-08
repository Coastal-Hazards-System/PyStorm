"""Small path-resolution helpers for user-facing launcher scripts.

Keeps the launcher CONFIG blocks tolerant of naming variations within a
dataset folder (e.g. ``CHS-LA_nodeID.mat`` vs ``CHS-LA_nodeID_probQ.mat``)
without needing the operator to track which exact filename is on disk.
"""

from __future__ import annotations

from pathlib import Path


def resolve_one_file(folder: str | Path, pattern: str, *, label: str | None = None) -> Path:
    """Return the single file in *folder* matching the glob *pattern*.

    Raises FileNotFoundError if zero matches; ValueError if more than one
    (so the operator notices ambiguity instead of silently picking one).
    """
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(
            f"Folder does not exist: {folder}"
            + (f"  (looking for {label})" if label else "")
        )
    matches = sorted(folder.glob(pattern))
    if not matches:
        raise FileNotFoundError(
            f"No file matching '{pattern}' in {folder}"
            + (f"  (needed for {label})" if label else "")
        )
    if len(matches) > 1:
        names = ", ".join(p.name for p in matches)
        raise ValueError(
            f"Pattern '{pattern}' matched {len(matches)} files in {folder}: "
            f"{names}. Disambiguate by removing extras or using a more "
            f"specific pattern."
            + (f"  (needed for {label})" if label else "")
        )
    return matches[0]
