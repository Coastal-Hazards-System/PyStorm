# Augmented Hurricane Database (AHD)

Downloads NHC's **HURDAT2** best-track archive, parses it, derives **heading
direction** and **translation speed**, and writes a HURDAT-like CSV - one file
per basin (Atlantic and/or NE/NC Pacific).

This is the Python re-implementation of the legacy MATLAB house-format
pre-processor (`CHS_TC_HURDAT_Atlantic.m`) and the GP-metamodel imputation of
Taflanidis et al.

## What it produces

One CSV per basin (`data/outputs/hurdat2_<basin>_<end_year>.csv`), one row per
storm time-snapshot, with these columns:

| column | meaning | units |
|---|---|---|
| `tc_no` | storm index within the file (1-based) | |
| `snap_no` | snapshot index within the storm (1-based) | |
| `year` | season year (from the NHC id) | |
| `nhc_id` | NHC cyclone id (e.g. `AL092021`) | |
| `basin` | `atlantic` or `pacific` | |
| `name` | storm name (`UNNAMED` if none) | |
| `ymd` / `hhmm` | date `YYYYMMDD` / time `HHMM` | UTC |
| `time_utc` | exact timestamp | UTC |
| `lat` / `lon` | center position (lon wrapped <-180 → +360) | ° |
| `landflag` | record-identifier code (L=4 landfall, …) | |
| `status` | system status code (HU=3 hurricane, …) | |
| `vmax_kmh` | maximum sustained wind | km/h |
| `pmin_hpa` | minimum central pressure | hPa |
| **`trans_kmh`** | **translation (forward) speed** | **km/h** |
| **`heading_deg`** | **heading (forward azimuth), (-180, 180]** | **°** |
| `radii{34,50,64}_{ne,se,sw,nw}_km` | wind radii by quadrant | km |
| `rmax_km` | radius of maximum wind | km |

**Unit & sentinel conventions** (matching the MATLAB house format): wind is
converted knots → km/h, wind-radii and Rmax nautical miles → km, pressure stays
in hPa, and HURDAT2 sentinels (`-99` / `-999`) become `NaN`. `trans_kmh` and
`heading_deg` come from consecutive 6-hourly fixes via WGS-84 geodesic
distance/azimuth (the first fix is forward-filled from the second). A stationary
fix (rounded speed 0 km/h) gets `NaN` speed and heading.

## Usage

```bash
pip install -r requirements.txt          # or: pip install -e .[dev]

# Edit the USER OPTIONS block, then:
python run_augmented_hurricane_database.py

# …or override on the command line:
python run_augmented_hurricane_database.py --basin atlantic
python run_augmented_hurricane_database.py --basin both --no-download
```

### Options (USER OPTIONS block in `run_augmented_hurricane_database.py`)

- **`BASIN`** - `"atlantic"`, `"pacific"`, or `"both"`.
- **`DOWNLOAD`** - `True` fetches the newest dated file per basin from the NHC
  directory (`https://www.nhc.noaa.gov/data/hurdat/`) into `data/inputs/`,
  ranked by record end-year then date-stamp. `False` uses the newest
  `hurdat2-*.txt` already on disk (no network).
- **`OVERWRITE`** - re-download even if the dated file is already local.
- **`ATLANTIC_FILE` / `PACIFIC_FILE`** - pin an explicit local HURDAT2 file per
  basin (absolute, or relative to `data/inputs/`). Bypasses **both** discovery
  and download: the named file is used as-is, with no network even when
  `DOWNLOAD = True`, and it overrides `ATLANTIC_URL` / `PACIFIC_URL`. `None`
  resolves automatically.
- **`ATLANTIC_URL` / `PACIFIC_URL`** - per-basin HURDAT2 download URL override.
  `None` auto-discovers the newest NHC file; set a full file URL to fetch that
  exact file instead (mirror/proxy/moved file). Honored only when
  `DOWNLOAD = True`; a local `ATLANTIC_FILE` / `PACIFIC_FILE` pin still wins.
- **`APPEND_EBTRK_RMAX`** - backfill missing `rmax_km` from the Extended Best
  Track dataset (see below).
- **`EBTRK_FILE`** - pin explicit local EBTRK file(s). EBTRK ships as **three**
  files, one per cyclone-id basin: AL (Atlantic), EP (East Pacific), CP (Central
  Pacific); the Atlantic build needs AL, the Pacific build needs EP and CP. Give
  a single path or a **list** (absolute, or relative to `data/inputs/`). Files
  are pooled and matched to HURDAT storms by cyclone id, so order is irrelevant
  and unused files are harmless - for `BASIN = "both"` just list all three.
  Bypasses **both** discovery and download (no network, overrides the
  `EBTRK_*_URL` options). `None` selects the correct file(s) per basin
  automatically.
- **`EBTRK_AL_URL` / `EBTRK_EP_URL` / `EBTRK_CP_URL`** - per-file EBTRK download
  URL overrides (one per cyclone-id code). `None` auto-discovers the newest file
  from the CIRA listing; set a full file URL to fetch that exact file. Honored
  only when `DOWNLOAD = True`; an `EBTRK_FILE` local pin still wins.
- **`PLOT_IMPUTATION` + `PLOT_ATLANTIC_CP` / `PLOT_ATLANTIC_RMAX` /
  `PLOT_PACIFIC_CP` / `PLOT_PACIFIC_RMAX`** - per-TC imputation plots (off by
  default; see below). `PLOT_IMPUTATION` is a master switch (`True` = all four on,
  `False` = all off, `None` = use the four flags). `PLOT_JOBS` sets plotting
  worker processes (`None`/`0` = auto, `1` = serial); `PLOT_DIR` sets the output
  folder.
- **`WRITE_PARQUET`** - also write a Parquet copy beside each CSV (needs
  `pyarrow`).
- **`OUTPUT_STEM`** - output filename stem. Substituted from the source file:
  `{basin}`, `{start_year}`, `{end_year}`, and `{created}` (the NHC file date as
  `YYYYMMDD`). The default marks the file as augmented HURDAT2 and carries the
  record span and NHC vintage, e.g.
  `augmented_hurdat2_atlantic_1851-2025_20260227.csv`.

## EBTRK Rmax backfill (optional)

HURDAT2 records the radius of maximum wind (`rmax_km`) only for recent seasons.
The **Extended Best Track (EBTRK)** dataset (CIRA) carries Rmax back to the late
1980s. With `APPEND_EBTRK_RMAX = True` (or `--ebtrk-rmax`) the module
downloads/parses the EBTRK file(s) for a basin and fills the `rmax_km` of HURDAT
rows where it is **missing**; HURDAT-provided values are never overwritten.

Both basins are supported. The Atlantic uses the `EBTRK_AL` file. The HURDAT
nepac (`pacific`) record contains both East Pacific (EP) and Central Pacific (CP)
storms, which have separate EBTRK files, so the Pacific basin uses both
(`EBTRK_EP` and `EBTRK_CP`).

Like the HURDAT2 resolver, the EBTRK file is **discovered**, not pinned: when
`DOWNLOAD = True` the module reads the CIRA listing page, selects the newest
"new format" file per code (ranked by record end-year then `DD-Mon-YYYY` publish
stamp), and fetches it. If the listing cannot be reached it falls back to the
last-known default file. Per-file URL overrides (`EBTRK_AL_URL` / `EBTRK_EP_URL`
/ `EBTRK_CP_URL`) bypass discovery for a specific file.

This ports `ebtrk/CHS_TC_HURDAT_Atlantic_with_ebtrk_Rm_append_v3.m`. One
improvement: the original matched EBTRK to HURDAT on synoptic **datetime alone**,
which is ambiguous when two storms share a time. Because an EBTRK record's first
eight characters are exactly the HURDAT cyclone id (e.g. `AL071988`,
`EP061990`, `CP011992`), the join here is on **(nhc_id, synoptic time)**, exact
per storm.

Typical Atlantic coverage after backfill: about 0% before 1988, about 79% for
1988-2020 (EBTRK), 100% for 2021 onward (HURDAT). EBTRK ends in 2021, so storms
after 2021 are not backfilled and are left to the GP metamodel.

```bash
python run_augmented_hurricane_database.py --basin atlantic --ebtrk-rmax
```

## GP-metamodel imputation (optional)

Values still missing after the steps above - central pressure (`pmin_hpa`) and
radius of maximum wind (`rmax_km`) - can be filled with **Gaussian-process
metamodels** (a Python re-implementation of the GP metamodel of *Taflanidis et
al.*; ports the CHS `gp_metamodel/` MATLAB). Enable with `IMPUTE_GPM = True` or
`--impute-gpm`.

Method (per basin, self-trained on the observed rows - no external data):
1. **Central pressure** (response = `1013 − pmin`, the deficit). Two universal-
   kriging GPs: **Cp6** `[lat, lon, vmax, Vf, sin Hd, cos Hd]` for fixes with
   known motion, **Cp3** `[lat, lon, vmax]` for single-point / first / stationary
   fixes (no `Vf`/`Hd`). Predict and fill the missing rows.
2. **Radius of max wind** (response = `rmax`), using the pressure-completed data:
   **Rm7** `[lat, lon, vmax, Cp-deficit, Vf, sin Hd, cos Hd]` and **Rm4**
   `[lat, lon, vmax, Cp-deficit]`. Fill missing rows, clamp to [8, 600] km.

The GP is universal kriging with an anisotropic power-exponential kernel
`exp(−Σ θ_k|Δx_k|^p)` + nugget, hyperparameters by MLE (concentrated NLL, LHS +
analytic-gradient L-BFGS-B). Each model reports leave-one-out CV R²/RMSE.
Observed values are kept; only the missing rows are filled.

### Quality upgrades over the original MATLAB (all on by default)

| flag | what it does |
|---|---|
| `GPM_VECCHIA` | **NNGP** - each prediction conditions on its nearest neighbors among *all* training fixes (not just the capped support set), using the whole dataset at O(n·m³) cost. |
| `GPM_PHYSICAL_MEAN` | **Physics-informed trend**: Cp uses a wind-pressure mean `[vmax, lat, vmax²]` (Δp ∝ V²); Rmax uses `[lat, Cp-deficit, vmax]` (size grows with latitude, shrinks with intensity). The GP models the residual around it - better extrapolation than a constant mean. |
| `GPM_LOG_RMAX` | Fit the **Rmax** models in **log space** - Rmax is positive and ~lognormal, so this matches the log-linear size physics and stabilizes the intensity-dependent scatter. (Cp is left untransformed - its `vmax²` basis already captures the curvature and the deficit is near-homoscedastic.) |
| `GPM_PARALLEL` | Train the two models per target concurrently (speed only; the C++ kernel releases the GIL). |

**Turn the three quality flags off** (`GPM_VECCHIA=GPM_PHYSICAL_MEAN=GPM_LOG_RMAX=False`)
to closely reproduce the **original MATLAB** GP metamodel: constant-mean kriging,
raw response, prediction over the support set. (Validated: in that configuration
the imputed central pressure matches the MATLAB to r=0.98, MAE ~2 hPa, zero bias.)

**Why nearest-neighbor GP, not a dense GP.** The training sets reach 15k to 24k
fixes, where an exact dense GP is a roughly 4.7 GB covariance and is infeasible to
refit during calibration. Instead each prediction conditions on its nearest
neighbors among all training fixes (NNGP), at O(n*m^3) cost. This is preferred
over the MATLAB's sparse-Cholesky route: central pressure is smooth and
long-range, so its correlation matrix is not sparse and sparse Cholesky also fills
in badly in 6-7 dimensions. The physics-informed trend absorbs the long-range
structure, leaving a short-range residual that a small neighbor set captures, so
NNGP reaches the dense-GP accuracy ceiling for central pressure while scaling
linearly. The full GP capability remains (set `GPM_VECCHIA=False` with a large
support count). Full reasoning and the file-by-file MATLAB-to-Python map are in
`backend/python/augmented_hurricane_database/gp_metamodel/README.md`.

**Per-target settings** (empirically tuned). Cp is smooth and long-range, Rmax is
short-range and noisy, so they use different defaults:

| target | `..._MAX_SUPPORT` | `..._NEIGHBORS` | `..._N_CAL` / `..._N_LHS` |
|---|---|---|---|
| Cp (`GPM_CP_*`) | 6000 | 30 | 4000 / 250 |
| Rmax (`GPM_RMAX_*`) | 8000 | 30 | 4000 / 250 |

**Acceleration.** An analytic likelihood gradient (about 3x faster calibration,
better hyperparameters), a smaller calibration subset, a response-stratified
support set, parallel two-model training, and an OpenMP C++ correlation kernel
(`_gpm`, about 10x to 34x over the NumPy broadcast; builds on first run, NumPy
fallback). The cubic-cost Cholesky and solves stay in LAPACK through SciPy. Net on
real Atlantic Cp: training about 240 s to about 40 s, prediction of 55k rows about
36 s to about 4 s, accuracy improving at the same time.

### Model cache

Training is the expensive step, so fitted models are cached and reused. With
`MODEL_DIR` set (default `data/models/`), each model is written there as a
compressed `.npz` named `{basin}_{model}_{signature}.npz`, e.g.
`pacific_Rm7_458253b9be.npz`:

- **`{basin}`** - `atlantic` or `pacific`.
- **`{model}`** - which of the four: `Cp6` / `Cp3` (central pressure, full /
  reduced) or `Rm7` / `Rm4` (radius of max wind, full / reduced).
- **`{signature}`** - the first 10 hex digits of an MD5 over the model's settings
  **and** a fingerprint of its training data (row count + target sum). It is a
  cache-invalidation key, not a readable label: change any `GPM_*` setting or the
  underlying data (a newer HURDAT vintage, EBTRK toggled) and the signature
  changes → a new filename → a guaranteed cache miss → retrain. An identical
  config on identical data reproduces the signature, so the cached model is
  reused.

`GPM_RETRAIN = False` (default) reuses a model whose signature matches and
otherwise trains and caches one; `GPM_RETRAIN = True` always retrains and
overwrites. The files are safe to delete - you just pay one retrain to
regenerate them.

### Validation against the MATLAB

The primary comparison is the recommended (default) configuration against the
MATLAB on the IDENTICAL data (the same HURDAT file, and for Rmax the same
EBTRK-augmented training set), both scored by leave-one-out on the deployed
predictor, the metric the MATLAB reports:

| model | recommended NNGP (LOOCV) R^2 / RMSE | MATLAB (LOOCV) R^2 / RMSE |
|---|---|---|
| Cp6 | 0.937 / 4.81 hPa | 0.932 / 4.97 hPa |
| Cp3 | 0.920 / 5.45 hPa | 0.918 / 5.49 hPa |
| Rm7 | 0.607 / 35.2 km  | 0.603 / 36.0 km  |
| Rm4 | 0.447 / 41.8 km  | 0.401 / 44.3 km  |

The recommended configuration beats the MATLAB on all four models. Two levers
drive it: a deep calibration (n_cal=4000, n_lhs=250), which lifts Cp6 to 0.937,
and a support set scaled to each target's signal (not its raw fix count). Cp is
smooth and high-skill, so 6000 (about 25% of its ~24k fixes) saturates it; Rmax
has fewer fixes (~15k) but is noisier and low-skill, so it needs a larger
fraction - 8000 rather than the earlier 3000. Rmax follows the MATLAB workflow:
it is trained on
observed pressure (before central-pressure imputation) and predicts the missing
Rmax with the completed pressure. The full head-to-head, the per-target sweeps,
and the calibration-depth discussion are in
`backend/python/augmented_hurricane_database/gp_metamodel/README.md`.

```bash
python run_augmented_hurricane_database.py --basin atlantic --ebtrk-rmax --impute-gpm
```

## Per-TC imputation plots (optional)

To visually inspect the imputed data along each storm's time history, the module
can write **one PNG per tropical cyclone**: the GP-metamodel-completed series as a
line (`GPM`) with the originally observed values as red dots (`Obs`). Two targets:
central-pressure deficit (`Δp = 1013 − pmin`, hPa) and radius of maximum wind
(km). `Obs` marks the rows that were known before imputation (HURDAT2, plus EBTRK
backfill for Rmax); every other point on the line is GP-imputed.

The feature is **off by default**, with four independently switchable groups -
Atlantic Cp, Atlantic Rmax, Pacific Cp, Pacific Rmax - plus a master switch
(`PLOT_IMPUTATION = True/False` to flip all four, `None` to use the individual
flags). It needs `IMPUTE_GPM = True` (the plots show what the GP filled) and
`matplotlib` (`pip install -e .[plots]`).

A full basin is ~1300-2000 storms, so this writes thousands of PNGs. The renderer
is built for it (Agg backend, one reused figure, numeric dates, low-compression
PNGs) and `PLOT_JOBS` spreads the storms over worker processes (`None`/`0` =
auto). Files land under `PLOT_DIR/imputation_<basin>_<target>/` as
`DataImputation_{Cp,Rm}_HURDAT_<basin>_<AL|EP|CP><year>_<NN>.png` (the NHC basin,
year, and storm number, e.g. `..._atlantic_AL1880_02.png`).

## Layout

```
augmented_hurricane_database/
├── run_augmented_hurricane_database.py        # launcher (USER OPTIONS)
├── backend/python/
│   ├── api_augmented_hurricane_database.py   # orchestrator entry: run(config)
│   └── augmented_hurricane_database/
│       ├── config.py          # AHDConfig (pydantic)
│       ├── sources.py         # NHC discovery + download
│       ├── parser.py          # HURDAT2 → tidy DataFrame, motion columns
│       ├── ebtrk.py           # EBTRK CIRA discovery/download/parse + Rmax backfill
│       ├── plots.py           # optional per-TC imputation diagnostic plots
│       ├── writer.py          # CSV / Parquet writers
│       └── orchestrator.py    # AHDOrchestrator: per-basin resolve → parse → (EBTRK) → write
├── tests/                     # pytest smoke tests (offline)
├── data/{inputs,outputs}/
├── backend/engines/cpp/       # _gpm C++ correlation kernel (pybind11 + build.py)
├── backend/python/augmented_hurricane_database/gp_metamodel/  # GP imputation (+ README)
└── pyproject.toml, requirements.txt, ENGINE_MANIFEST.toml
```

## Tests

```bash
python -m pytest tests -q      # offline; uses a synthetic best-track fixture
```
