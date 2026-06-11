# tc_climatological_analysis

CRL-based tropical-cyclone **storm recurrence rates** for the U.S. Atlantic and
Pacific coasts. For each CHS **Coastal Reference Location (CRL)**, the module
selects the tropical cyclones from the augmented HURDAT2 best-track that pass
within a cutoff distance, then computes — with the **Gaussian Kernel Function
(GKF)** — the omnidirectional and directional recurrence rates, both annually and
per calendar month, for four intensity bins.

It is a Python port of the CHS MATLAB drivers `CHS_Atlantic_StormSelection.m` and
`CHS_SRR_GKF.m` (with `GaussianWeights.m`, `AzimuthDiff.m`,
`headingZeroDegree_adj.m`).

## What it produces

- **SRR** — omnidirectional storm recurrence rate, `(1/Nyrs) · Σ Wi` with the
  distance kernel `Wi = 1/(√(2π)·K)·exp(−½(D/K)²)`. **Units: storms / km / year.**
- **DSRR** — directional rate `Ld(θ) = (1/Nyrs) · Σ Wd_i(θ)·Wi` with the heading
  kernel (σ = 30°). **Units: storms / degree / year.** The normalized shape is
  recentered on its circular mean into a heading pdf/cdf with a mean and stdv.
- Both **annually and per calendar month** (Jan–Dec); the twelve monthly rates
  sum exactly to the annual rate (the month is the storm's closest approach).
- For four intensity bins on the deficit `dp = 1013 − Cp`:
  `All (dp ≥ 8)` · `Low [8, 28)` · `Med [28, 48)` · `High [48, ∞)` hPa.

Run per basin (Atlantic and Pacific; Pacific is enabled now that a Pacific CRL
set exists).

## Inputs

- **CRL sets** under `data/inputs/raw/` — the Atlantic CSV
  (`CHS_Atl_CRLs_v1.6.csv`: `ID,lat,lon`) and the Pacific tab-delimited file
  (`CHS_PAC_CRLs_v1.2.txt`: `Latitude,Longitude,Region,ID`). The loader
  auto-detects the delimiter and column names.
- **Augmented HURDAT2** — the `augmented_hurricane_database` (AHD) module output.
  By default the newest `augmented_hurdat2_<basin>_*.csv` under the sibling AHD
  module's `data/outputs` is linked automatically; set
  `ATLANTIC_HURDAT_FILE` / `PACIFIC_HURDAT_FILE` to pin your own.

Inputs follow the CyHAN **raw / processed** convention: original source files in
`data/inputs/raw/`, any derived inputs in `data/inputs/processed/`.

## Run

```bash
pip install -r requirements.txt
python run_tc_climatological_analysis.py                 # USER OPTIONS block
python run_tc_climatological_analysis.py --basin atlantic
python run_tc_climatological_analysis.py --basin both --plots
```

### Key options (USER OPTIONS block)

- **`BASIN`** — `"atlantic"`, `"pacific"`, or `"both"`.
- **`ATLANTIC_CRL_FILE` / `PACIFIC_CRL_FILE`** — CRL files under `data/inputs/raw/`.
- **`ATLANTIC_HURDAT_FILE` / `PACIFIC_HURDAT_FILE` / `AHD_OUTPUTS_DIR`** — pin the
  augmented-HURDAT source, or let it auto-link to the AHD module.
- **`K_SIZE` (200 km), `DIR_KERNEL` (30°), `MAX_DIST` (600 km), `MAX_CP`
  (1005 hPa), `START_YEAR` (1938), `MIN_DP`/`DP_LOW`/`DP_MED`** — GKF/selection
  parameters (defaults match the CHS MATLAB). The effective start year is clamped
  to each basin's first season (Atlantic 1938, Pacific 1949).
- **`PLOT_SELECTION`** — per-CRL selected-TC maps (off by default; see below).

## Per-CRL selected-TC maps (optional)

With `PLOT_SELECTION = True` (or `--plots`) the module writes one map per CRL —
the selected TCs colored by intensity (High red, Med yellow, Low green) with the
CRL in blue — over a **Natural Earth** basemap (coastline + country + state/
province lines), replacing the legacy low-resolution NOAA coastline. A full basin
is ~1,000+ CRLs, so it is opt-in and parallelized (`PLOT_JOBS`); the basemap
downloads once into `data/inputs/raw/naturalearth/`. Needs `matplotlib` + `pyshp`
(`pip install -e .[plots]`).

## Outputs (`data/outputs/`)

Every non-plot output carries the HURDAT vintage `<start>-<end>_<created>` (the
record start/end years and the NHC file date), matching the AHD source file —
e.g. `srr_atlantic_1851-2025_20260227.csv`. Below, `<v>` = that tag.

| File | Contents |
|---|---|
| `selection_<basin>_<v>.csv` | per-CRL selected TCs (representative point + closest approach) |
| `srr_<basin>_<v>.csv` | annual + monthly omnidirectional SRR per intensity bin (storms/km/yr) |
| `dsrr_<basin>_<v>.csv` | directional heading mean/stdv per bin (deg) |
| `dsrr_<basin>_<v>.npz` | full DSRR arrays — rate/pdf/cdf, annual + monthly, per bin |
| `srr_<R>km/srr_<R>km_<basin>_<v>.csv` | optional **SRR_<R>km** variant — SRR · 2R (TC/yr), separate folder |
| `plots/selection_<basin>/…` | optional annual per-CRL maps (SRR box) |
| `plots/selection_monthly_<basin>/…` | optional per-CRL × month maps |
| `plots/selection_<R>km_<basin>/…` | optional SRR_<R>km map variants (separate folders) |

### SRR_<R>km variant (within a radius)

A second variant of the **SRR results only** (not DSRR), off by default
(`SRR_RADIAL`): **SRR_<R>km = SRR · (2·R)** — the rate (storms/km/yr) times the
2R-km diameter, giving the **expected storms / year within `SRR_RADIUS_KM` of each
CRL** (TC/yr). The radius is user-set (default **200 km** → ×400). It is written to
its own `srr_<R>km/` folder and, when maps are enabled, plotted in separate
`selection_<R>km_<basin>/` folders with the SRR box relabeled `SRR_<R>km (TC/yr)`.
The per-CRL maps already print the SRR in the order **All, High, Med, Low**.

## Method

1. **Load** the CRL set and the augmented HURDAT2 best-track.
2. **Select** per CRL: every TC within `MAX_DIST`; the representative point is the
   fix maximizing `GaussianWeights(K_SIZE)·(1013 − Cp)` (proximity × intensity);
   the closest-approach distance drives the rate kernel. Fixes with missing or
   `> MAX_CP` central pressure are dropped.
3. **SRR / DSRR** via the GKF, per intensity bin and per month of closest approach.
4. **Write** the selection, SRR/DSRR tables, and full DSRR arrays; optionally map
   each CRL.

## Layout

```
tc_climatological_analysis/
├── run_tc_climatological_analysis.py        # launcher (USER OPTIONS)
├── backend/python/
│   ├── main_tc_climatological_analysis.py   # orchestrator entry: run(config)
│   └── tc_climatological_analysis/
│       ├── config.py        # TCAConfig (pydantic)
│       ├── crls.py          # CRL loader (CSV + tab-delimited)
│       ├── hurdat_source.py # locate/load augmented HURDAT2 (links to AHD module)
│       ├── selection.py     # Gaussian-weighted per-CRL storm selection
│       ├── gkf.py           # SRR + DSRR + monthly (GKF)
│       ├── basemap.py       # Natural Earth coastline/boundaries via pyshp
│       ├── plots.py         # per-CRL selected-TC maps
│       ├── writer.py        # SRR/DSRR table + array writers
│       └── orchestrator.py  # TCAOrchestrator: per-basin pipeline
├── data/{inputs/{raw,processed},outputs}/
└── tests/
```
