"""augmented_hurricane_database — NHC HURDAT2 best-track ingest.

Downloads the Atlantic and/or NE-Pacific HURDAT2 best-track files from NHC,
parses them, derives translation speed and heading, and writes a HURDAT-like
CSV per basin.
"""

from augmented_hurricane_database import ebtrk
from augmented_hurricane_database.config import AHDConfig
from augmented_hurricane_database.orchestrator import (
    BasinResult,
    AHDOrchestrator,
    AHDResult,
)
from augmented_hurricane_database.parser import HURDAT2, Storm, TrackPoint
from augmented_hurricane_database.sources import (
    BASINS,
    discover_latest,
    resolve_source,
)

__all__ = [
    "AHDConfig",
    "AHDOrchestrator",
    "AHDResult",
    "BasinResult",
    "HURDAT2",
    "Storm",
    "TrackPoint",
    "BASINS",
    "discover_latest",
    "resolve_source",
    "ebtrk",
]
