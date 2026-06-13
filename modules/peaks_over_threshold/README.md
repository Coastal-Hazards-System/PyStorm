# peaks_over_threshold

*Peaks-Over-Threshold (POT) for sampling of independent events from time series.*

PyStorm module

---

## Overview

POT extracts independent storm peaks from a continuous water-level or non-tidal
residual (NTR) series: the magnitudes and times that exceed an automatically
chosen threshold, declustered into one value per storm event and trimmed to a
fixed average rate. These peaks are the input to the PST module.

The inner threshold search (percentile lookup, exceedance scan, event
segmentation, rate check) runs in C++
(`backend/engines/cpp/POTThresholdSearch.hpp`, exposed as `_pot`). A pure-numpy
fallback runs automatically when the kernel is not built. The launcher builds the
kernel on first run.

## Stages

`STAGES` in the launcher selects which steps run, in canonical order. The primary
use is `["pot"]` (POT only, on a user CSV). The upstream stages are optional and
build the POT input from NOAA data.

| Stage | Produces |
|-------|----------|
| `download` | `data/inputs/raw/<station>/water_level_*.csv`, `tide_prediction_*.csv` (hourly NOAA pulls) |
| `detrend` | `data/inputs/processed/<station>/dwl_<station>.csv` (detrended water level) and `trend_<station>.csv` |
| `ntr` | `data/inputs/processed/<station>/ntr_<station>.csv` (non-tidal residual) |
| `pot` | `data/outputs/<station>/{dwl,ntr}_<station>_pot.csv` (declustered peaks) and plots |

With `pot` in a chain, peaks are extracted from both processed series (`dwl` and
`ntr`). All output and plot filenames are lower case.

**NTR** (non-tidal residual) is the detrended water level minus the hourly
astronomical tide on the same grid (tide is downloaded with `interval=h`). The
tide aligns exactly on matched hours and interpolates only to fill a missing
tide hour.

**Detrending** removes a linear sea-level trend by least squares. `midpoint`
centers time on the National Tidal Datum Epoch (NTDE) midpoint, matching NOAA
datum convention; `ordinary` centers on the record mean. The slope is fitted
from the record or imposed with `DETREND_SLOPE`. NTDE and slope can be set per
station: a single value applies to every station, or a list parallel to
`STATION_IDS` gives one value each. NTDE years may be fractional (for example,
2012.42).

## Method

**1. Effective duration.** The record length used for all rates is the effective
duration: (number of non-NaN hourly steps) / (365.25 × 24) years. Gaps and
missing data do not count, so a 100-year span that is 50 percent complete counts
as about 50 years.

**2. Iterative threshold search.** The threshold is raised from `START_PERCENTILE`
upward in `STEP_SIZE` increments (percentiles of the series). At each level the
exceedances are declustered into independent events by `METHOD`:

- `hydrograph`: consecutive exceedances within `INTEREVENT_HOURS` of one another
  form a single event whose peak is the event maximum.
- `peak_gap`: a sample is dropped if it lies within `INTEREVENT_HOURS` of, and is
  no larger than, the previous retained peak.

The event rate is (number of events) / effective_duration. The search is
one-sided: it keeps the highest threshold whose rate is still at least
`TARGET_EVENTS_PER_YEAR`, and flags convergence when that rate lands in
`[target, target + TOLERANCE]`.

**3. Rank-trim to an exact count.** The retained peaks are rank-ordered and
trimmed to exactly `round(TARGET_EVENTS_PER_YEAR × effective_duration)` of the
largest, so the written sample has an effective rate of exactly the target. This
is also what lets PST recover the record length from the peak count, so keep
PST's `EVENTS_PER_YEAR` equal to `TARGET_EVENTS_PER_YEAR`.

The C++ and Python implementations are algorithmically identical.

## Outputs

| File | Contents |
|------|----------|
| `<series>_<station>_pot.csv` | columns `datetime`, `value` (one declustered peak per row) |
| `<series>_<station>_pot.png` | time series, peaks, and threshold |

Data files go to `data/outputs/<station>/`; all plots go to the shared
`data/outputs/plots/`.

## Quickstart

```bash
cd modules/peaks_over_threshold

# Edit the USER OPTIONS block, then run (the C++ kernel builds on first run):
python run_peaks_over_threshold.py

# CLI batch over explicit input CSVs (POT-only on each):
python run_peaks_over_threshold.py path/to/ntr_8518750.csv another.csv

# Tests:
pytest tests/
```

The USER OPTIONS block covers the stages, station list, POT parameters
(interevent hours, method, target events per year, tolerance, start percentile,
step size), and the NOAA-to-NTR pipeline settings (years, datum, NTDE, slope).
Positional CLI paths force POT-only on each file; `--help` lists options.

## Programmatic API

Like every PyStorm module, POT exposes one entry point in
`backend/python/api_peaks_over_threshold.py`:

```python
run(config) -> POTResult | PipelineResult
```

The launcher (`run_peaks_over_threshold.py`) only assembles `config` (stations,
stages, parameters) and calls `run`. To drive POT from your own code:

```python
import sys
sys.path.insert(0, "modules/peaks_over_threshold/backend/python")
from api_peaks_over_threshold import run

result = run(config)   # config: a dict (or a POTConfig / PreprocessConfig)
```

It returns a **`POTResult`** for a single POT-only extraction, or a
**`PipelineResult`** when preprocessing stages (and/or multiple stations and
targets) run. `POTResult` carries `threshold`, `peaks_df`, `events_per_year`,
`final_percentile`, `effective_duration_years`, `converged`, `iterations`, and
`used_cpp_kernel`. `PipelineResult` bundles `preprocess` and a `pot` map of
`{target: POTResult}` (e.g. `{"dwl": ..., "ntr": ...}`).

## Layout

```
peaks_over_threshold/
├── run_peaks_over_threshold.py             Launcher (user options only)
├── pyproject.toml                          Installable orchestrator package
├── ENGINE_MANIFEST.toml                    Structured module manifest
├── backend/
│   ├── engines/cpp/                        C++ threshold-search kernel (_pot)
│   │   ├── POTThresholdSearch.hpp          Header-only kernel
│   │   ├── pot_bindings.cpp                pybind11 → _pot
│   │   ├── CMakeLists.txt
│   │   └── build.py
│   └── python/
│       ├── api_peaks_over_threshold.py    Orchestrator entry (stage dispatch)
│       └── peaks_over_threshold/           Orchestration package
│           ├── config.py                   POTConfig + PreprocessConfig (pydantic)
│           ├── orchestrator.py             POTOrchestrator
│           ├── solver.py                   Thin _pot binding wrapper
│           ├── sampling/                   IterativeThresholdSearch
│           ├── segmentation/               hydrograph + peak_gap segmenters
│           ├── preprocessing/              download / detrend / ntr chain
│           ├── postproc/                   NaN-aware time-series plot
│           └── io/                         CSV reader and writers
├── tests/                                  Smoke + preprocessing tests
└── data/                                   inputs/{raw,processed}/ & outputs/ (gitignored)
```

Per CyHAN v2.2, the launcher (`run_peaks_over_threshold.py`) holds user options
only and calls the orchestrator (`backend/python/api_peaks_over_threshold.py`),
which dispatches the stages. The module is self-contained.

## Acronyms

| Acronym | Expansion |
|---------|-----------|
| NTDE | National Tidal Datum Epoch |
| NTR | Non-Tidal Residual |
| POT | Peaks Over Threshold |
| PST | Probabilistic Simulation Technique |
| WL | Water Level |
