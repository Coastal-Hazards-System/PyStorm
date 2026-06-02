"""segmentation — event segmenters used by the Python fallback.

Author / POC : Norberto C. Nadal-Caraballo, PhD  <norberto.c.nadal-caraballo@usace.army.mil>

The C++ kernel duplicates these algorithms; the Python copies here are used
only when ``_pot`` is unavailable. Their numerical behaviour matches the C++
implementation byte-for-byte on identical inputs.
"""

from .events import segment_hydrograph, segment_peak_gap

__all__ = ["segment_hydrograph", "segment_peak_gap"]
