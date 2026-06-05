# peaks_over_threshold

**Peaks-Over-Threshold (POT) for Sampling of Independent Events from Time Series**

PyStorm Module: POT (CyHAN v2.0 §5)

---

## Introduction

The **Peaks-Over-Threshold (POT)** method extracts independent extreme events
from an environmental time series (e.g. water levels, storm surge) by:

1. Selecting a percentile-based magnitude threshold,
2. Filtering exceedances,
3. Segmenting them into independent events using one of two
   inter-event-time rules.

The module ships an **iterative threshold search** that tunes the percentile
until the segmentation-derived event rate matches a target events/year within
a tolerance. The inner per-iteration kernel (percentile lookup +
exceedance scan + event segmentation + rate check) is implemented in C++
(`backend/engines/cpp/POTThresholdSearch.hpp`) and exposed as `_pot`. A
pure-numpy fallback activates automatically when the extension is not built.

---

## 1. Module Layout (CyHAN v2.0 §16.1)

```
peaks_over_threshold/
├── run_peaks_over_threshold.py                 ← launcher (user-facing, §5.3)
├── README.md
├── pyproject.toml
├── requirements.txt
├── ENGINE_MANIFEST.toml
├── backend/
│   ├── engines/cpp/
│   │   ├── POTThresholdSearch.hpp              header-only threshold-search kernel
│   │   ├── pot_bindings.cpp                    pybind11 → _pot
│   │   ├── CMakeLists.txt
│   │   ├── build.py
│   │   └── README.md
│   └── python/
│       ├── main_peaks_over_threshold.py        ← orchestrator entry (§5.3)
│       └── peaks_over_threshold/               ← expanded package (§5.3)
│           ├── __init__.py
│           ├── config.py            pydantic POTConfig + PreprocessConfig
│           ├── orchestrator.py      POTOrchestrator workflow runner
│           ├── solver.py            thin _pot binding wrapper
│           ├── sampling/
│           │   └── threshold_search.py   IterativeThresholdSearch
│           ├── segmentation/
│           │   └── events.py            hydrograph + peak-gap segmenters
│           ├── preprocessing/                upstream NOAA → NTR chain
│           │   ├── noaa_download.py        download_noaa_wl_data
│           │   ├── detrend.py              detrend_time_series
│           │   ├── ntr.py                  estimate_ntr (NTR)
│           │   └── orchestrator.py         PreprocessOrchestrator (stage runner)
│           ├── postproc/
│           │   └── plots.py             NaN-aware TimeSeriesPlotter
│           └── io/
│               └── time_series_csv.py   CSV reader + peaks/series writers
├── tests/
│   ├── test_smoke.py
│   └── test_preprocessing.py
├── data/                                       § 16.7
│   ├── inputs/
│   │   ├── raw/<station>/                     raw NOAA pulls (per gauge)
│   │   └── processed/<station>/               detrended WL + NTR series
│   └── outputs/<station>/                     POT peaks + plots/
├── research/
└── docs/
```

The two mandatory entry artifacts per CyHAN v2.0 §5.3:

| Artifact     | Location                                       | Role               |
|--------------|------------------------------------------------|--------------------|
| Launcher     | `run_peaks_over_threshold.py`                  | user-facing entry  |
| Orchestrator | `backend/python/main_peaks_over_threshold.py`  | non-user-facing    |

---

## 1a. Stages (CyHAN v2.0 §5.3)

The launcher's `STAGES` list selects which steps run, in canonical order:

| Stage      | Engine                                   | Produces                                   |
|------------|------------------------------------------|--------------------------------------------|
| `download` | `preprocessing.download_noaa_wl_data`    | `raw/<station>/water_level_*.csv`, `tide_prediction_*.csv` |
| `detrend`  | `preprocessing.detrend_time_series`      | `processed/<station>/dwl_<station>.csv` (+ `trend_<station>.csv`) |
| `ntr`      | `preprocessing.estimate_ntr`             | `processed/<station>/ntr_<station>.csv`    |
| `pot`      | `orchestrator.POTOrchestrator`           | `outputs/<station>/dwl_<station>_pot.csv` **and** `ntr_<station>_pot.csv` (+ plots) |

In a chain, the `pot` stage extracts peaks from **both** processed series it
finds — the detrended water level (`dwl_*`) and the non-tidal residual
(`ntr_*`). All output and plot filenames are lower case.

* **Primary use — POT only.** Default `STAGES = ["pot"]`: POT runs on the
  user-provided `INPUT_CSV`. (A bare `POTConfig` or a dict without `stages`
  behaves identically, preserving the original single-purpose entry.)
* **Secondary use — NOAA → NTR pipeline.** Add any of `download`, `detrend`,
  `ntr` to build the POT input from raw gauge data. When `pot` is also present
  the chain feeds straight into extraction (`download → detrend → ntr → pot`).

`NTR` (non-tidal residual) = detrended water level − astronomical tide, both on
the same hourly grid (tide is downloaded with `interval=h`); it replaces the v1
"storm surge" naming throughout outputs and labels. Tide alignment to the WL
timestamps is exact on matched hours and only interpolates as a fallback for a
missing tide hour.

Detrending uses a resolution-independent epoch-seconds cast, so it is correct
under both nanosecond and microsecond pandas datetime resolutions.

---

## 2. Methods

### 2.1 Iterative Threshold Search

Let `values[0..n-1]` and `times_sec[0..n-1]` be the time series (values
aligned with ascending Unix epoch seconds), `λ_target` the target events per
year, `τ` the tolerance, `p₀` the starting percentile, `Δp` the percentile
step, and `T = (times[-1] − times[0]) / (365.25 · 86400)` the record length
in years. Pre-sort `values` descending so percentile lookup is `O(1)`.

For iteration `i = 0, 1, …`:

1. `p = p₀ + i · Δp`. Stop if `p ≥ 100`.
2. `threshold = sorted_desc[k]` where `k = ⌊(1 − p/100) · (n − 1)⌋`.
3. Compute exceedance indices `E = { j : values[j] > threshold }` in time
   order.
4. Segment `E` into independent peaks `P` via the chosen method (§2.2).
5. `λ = |P| / T`. If `|λ − λ_target| < τ`, return `(threshold, P)`.

Implementation: `sampling/threshold_search.py` (Python dispatcher) →
`_pot.find_threshold_for_target` (C++) or a pure-numpy fallback.

### 2.2 Event Segmentation

Two segmentation rules are provided, both operating on the time-ordered
exceedance index list `E`:

**Hydrograph** (`method = "hydrograph"`): scan `E`; start a new group when
the time gap to the previous exceedance exceeds `interevent_sec`. The selected
peak per group is `argmax(values)`.

**Peak-Gap** (`method = "peak_gap"`): sequential filter — drop a sample whose
preceding (chronological) exceedance is within `interevent_sec` AND has
value ≥ the current one. The legacy alias `"peaks"` is normalized to
`"peak_gap"`.

The C++ and Python implementations are algorithmically identical.

---

## 3. Workflow

```
┌──────────────────────────────────────────────────────────────────────┐
│  data/inputs/processed/<base>.csv      (DATETIME_COL, VALUE_COL)     │
└──────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌──────────────────────────────────────────────────────────────────────┐
│  [1]  read_time_series_csv  →  (datetime, value)                     │
├──────────────────────────────────────────────────────────────────────┤
│  [2]  Convert to (values, times_sec) numpy arrays                    │
├──────────────────────────────────────────────────────────────────────┤
│  [3]  IterativeThresholdSearch (_pot or numpy fallback)              │
│       → ThresholdSearchResult (threshold, peak_indices, ...)         │
├──────────────────────────────────────────────────────────────────────┤
│  [4]  Materialize peaks DataFrame from indices                       │
├──────────────────────────────────────────────────────────────────────┤
│  [5]  write_pot_peaks  +  TimeSeriesPlotter                          │
│       → data/outputs/<base>_POT.csv                                  │
│       → data/outputs/plots/<base>_POT.png                            │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 4. Outputs

| File                | Contents                                                   |
|---------------------|------------------------------------------------------------|
| `<base>_POT.csv`    | Two columns: `datetime`, `value` for each selected peak    |
| `<base>_POT.png`    | Time series + peaks + threshold line                       |

---

## 5. Quickstart

```bash
cd modules/peaks_over_threshold

# (Optional) Build the C++ kernel. Pure-Python fallback works without this.
python backend/engines/cpp/build.py

# Make the package importable
pip install -e .

# Edit USER OPTIONS in run_peaks_over_threshold.py, then run:
python run_peaks_over_threshold.py

# Or override on the CLI for ad-hoc runs:
python run_peaks_over_threshold.py \
    --input data/inputs/processed/storm_surge_8518750_1920_2025.csv \
    --method hydrograph \
    --target-events 10 \
    --start-percentile 75
```

Smoke tests:

```bash
pytest tests/
```

---

## 6. CyHAN v2.0 Compliance

| Requirement                                                       | Status                                                         |
|-------------------------------------------------------------------|----------------------------------------------------------------|
| §1   API → Orchestrator → Engine; one-way dependency              | ✓ engine is header-only; orchestrator owns side effects        |
| §4.1 Binding is a conduit, not authority                          | ✓ `_pot` exposes one function; orchestration lives in Python   |
| §4.2 Orchestration in Python, non-user-facing                     | ✓ `main_<name>.py` + expanded package                          |
| §5.1 Module ships engine + orchestrator + launcher                | ✓                                                              |
| §5.2 Self-contained; no sibling-module imports                    | ✓                                                              |
| §5.3 Launcher `run_<name>.py` at module root, user-facing         | ✓                                                              |
| §5.3 Orchestrator `main_<name>.py` at `backend/python/`           | ✓                                                              |
| §5.3 Launcher contains no orchestration logic                     | ✓ delegates to `main_<name>.run`                               |
| §16.1 / §16.2 Recommended folder layout + layer mapping           | ✓                                                              |
| §16.4 Engine isolated under `backend/engines/cpp/`                | ✓                                                              |
| §16.7 Data convention                                             | ✓                                                              |

---

## 7. Acronyms

| Acronym | Expansion                                              |
|---------|--------------------------------------------------------|
| CyHAN   | C++/Python Hybrid Architecture Network                 |
| POT     | Peaks Over Threshold                                   |
