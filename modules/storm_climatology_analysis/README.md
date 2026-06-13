# storm_climatology_analysis

CRL-based tropical-cyclone **storm recurrence rates** for the U.S. Atlantic and
Pacific coasts. For each CHS **Coastal Reference Location (CRL)**, the module
selects the tropical cyclones from the augmented HURDAT2 best-track that pass
within a cutoff distance, then computes - with the **Gaussian Kernel Function
(GKF)** - the omnidirectional and directional recurrence rates, both annually and
per calendar month, for four intensity bins, plus a **continuous daily** (day-of-year)
seasonal cycle of the omnidirectional rate.

> **Storm type (TC / ETC).** SCA runs in two storm-type modes, set by `STORM_TYPE`
> (`--storm-type`): **`tc`** (tropical cyclones, the implemented analysis) and
> **`etc`** (extratropical cyclones, a *placeholder* that raises `NotImplementedError`
> for now; the same GKF recurrence-rate machinery would run on an ETC track source).
> The rest of this README documents the `tc` mode.

The companion **whitepaper** (`SRR_GKF_Whitepaper.md`) documents the method in full.

It is a Python port of the CHS MATLAB drivers `CHS_Atlantic_StormSelection.m` and
`CHS_SRR_GKF.m` (with `GaussianWeights.m`, `AzimuthDiff.m`,
`headingZeroDegree_adj.m`).

## What it produces

- **SRR** - omnidirectional storm recurrence rate, `(1/Nyrs) · Σ Wi` with the
  distance kernel `Wi = 1/(√(2π)·K)·exp(−½(D/K)²)`. **Units: storms / km / year.**
- **DSRR** - directional rate `Ld(θ) = (1/Nyrs) · Σ Wd_i(θ)·Wi` with the heading
  kernel (σ = 30°). **Units: storms / degree / year.** The normalized shape is
  recentered on its circular mean into a heading pdf/cdf with a mean and stdv.
- **Daily SRR** - the omnidirectional rate as a continuous seasonal cycle over
  day-of-year 1..365, `SRR_daily(d) = (1/Nyrs) · Σ Wt_i(d)·Wi` with a circular
  (period-365) day-of-year kernel `Wt` (σ = `DAY_KERNEL`, default 14 days).
  **Units: TC/km/yr per day-of-year** - a *rate density* across the calendar, i.e.
  the annual SRR (TC/km/yr) spread over the season, so the 365 daily values **sum to
  the annual SRR**. The two time terms mean different things: `/yr` is the rate per
  hurricane season (averaged over the record), and `per day-of-year` is the density
  across the calendar. (See "Reading the daily-SRR units" below.)
- Both **annually and per calendar month** (Jan-Dec); the twelve monthly rates
  sum exactly to the annual rate (the month is the storm's closest approach).
- For four intensity bins on the deficit `dp = 1013 − Cp`:
  `All (dp ≥ 8)` · `Low [8, 28)` · `Med [28, 48)` · `High [48, ∞)` hPa.

Run per basin (Atlantic and Pacific; Pacific is enabled now that a Pacific CRL
set exists).

## Inputs

- **CRL sets** under `data/inputs/raw/` - the Atlantic CSV
  (`CHS_Atl_CRLs_v1.6.csv`: `ID,lat,lon`) and the Pacific tab-delimited file
  (`CHS_PAC_CRLs_v1.2.txt`: `Latitude,Longitude,Region,ID`). The loader
  auto-detects the delimiter and column names.
- **Augmented HURDAT2** - the `augmented_hurricane_database` (AHD) module output.
  By default the newest `augmented_hurdat2_<basin>_*.csv` under the sibling AHD
  module's `data/outputs` is linked automatically; set
  `ATLANTIC_HURDAT_FILE` / `PACIFIC_HURDAT_FILE` to pin your own.

Inputs follow the CyHAN **raw / processed** convention: original source files in
`data/inputs/raw/`, any derived inputs in `data/inputs/processed/`.

## Run

```bash
pip install -r requirements.txt
python run_storm_climatology_analysis.py                 # USER OPTIONS block
python run_storm_climatology_analysis.py --basin atlantic
python run_storm_climatology_analysis.py --basin both --plots
```

### Key options (USER OPTIONS block)

- **`BASIN`** - `"atlantic"`, `"pacific"`, or `"both"`.
- **`ATLANTIC_CRL_FILE` / `PACIFIC_CRL_FILE`** - CRL files under `data/inputs/raw/`.
- **`ATLANTIC_HURDAT_FILE` / `PACIFIC_HURDAT_FILE` / `AHD_OUTPUTS_DIR`** - pin the
  augmented-HURDAT source, or let it auto-link to the AHD module.
- **`K_SIZE` (200 km), `DIR_KERNEL` (30°), `MAX_DIST` (600 km), `MAX_CP`
  (1005 hPa), `START_YEAR` (1938), `MIN_DP`/`DP_LOW`/`DP_MED`** - GKF/selection
  parameters (defaults match the CHS MATLAB). `START_YEAR = None` uses the entire
  HURDAT record; otherwise it is clamped up to each basin's first season (Atlantic
  1938, Pacific 1949). The effective start year appears in the output filenames.
- **`DAY_KERNEL` (14 days)** - bandwidth of the daily-SRR day-of-year kernel
  (see the daily-SRR section below for how it is chosen).
- **`PLOT_SELECTION` / `PLOT_SELECTION_MONTHLY` / `PLOT_SELECTION_DAILY`** - per-CRL
  plots (off by default in the config; see below). The `--plots` / `--no-plots` CLI
  flags toggle all three together.

## Per-CRL plots (optional)

Three optional per-CRL products, all parallelized by `PLOT_JOBS`:

- **Annual + monthly maps** (`PLOT_SELECTION`, `PLOT_SELECTION_MONTHLY`) - one map per
  CRL (and per calendar month) of the selected TCs colored by intensity (High red, Med
  yellow, Low green) with the CRL in blue, over a **Natural Earth** basemap (coastline +
  country + state/province lines). A full basin is ~1,000+ CRLs (monthly is ~12x that),
  so they are opt-in. The basemap downloads once into `data/inputs/raw/naturalearth/`;
  needs `matplotlib` + `pyshp` (`pip install -e .[plots]`).
- **Daily SRR curves** (`PLOT_SELECTION_DAILY`) - one line plot per CRL of the
  continuous daily SRR over day-of-year 1..365, with the **All (black), High (red), Med
  (gold), Low (green)** curves on one set of axes (months on the x-axis). This is an XY
  curve, not a map, so it needs only `matplotlib` (no basemap/pyshp). With `SRR_RADIAL`
  a second `SRR_<R>km` daily plot (TC/yr within R, per day-of-year) is written to its
  own folder.

The `--plots` / `--no-plots` CLI flags enable or disable all three products at once.

### Reading the daily-SRR units

The daily SRR is a **rate density over the calendar**, not a count on a given day. Read
it as: *the annual SRR, spread across day-of-year*. Two distinct time terms appear:

- **`/yr`** - the rate per hurricane season, averaged over the `Nyrs`-year record (the
  same `/yr` as the annual SRR).
- **`per day-of-year`** - the density across the calendar; "per day" is a position in
  the season, **not** a second time axis.

So a value at, say, day 250 is the share of the annual rate accruing at that point in
the season. **Summing all 365 daily values returns the annual SRR** (`Σ_d SRR_daily(d)
= SRR`), exactly, because the day-of-year kernel integrates to one over the year. The
base units are `TC/km/yr per day-of-year`; the within-radius variant is
`TC/yr within R, per day-of-year` (it sums to the annual `SRR_<R>km` of `TC/yr`).

## Outputs (`data/outputs/`)

Every non-plot output is tagged `<start>-<end>_<created>`, where `<start>` is the
effective rate start year (`START_YEAR`, clamped up to the basin's first season;
or the full record start when `START_YEAR` is None), `<end>` is the last season,
and `<created>` is the NHC HURDAT file date (from the AHD source). E.g.
`srr_atlantic_1938-2025_20260227.csv`. Below, `<v>` = that tag.

| File | Contents |
|---|---|
| `selection_<basin>_<v>.csv` | per-CRL selected TCs (representative point + closest approach) |
| `srr_<basin>_<v>.csv` | annual + monthly omnidirectional SRR per intensity bin (storms/km/yr) |
| `srr_daily_<basin>_<v>.csv` | continuous daily SRR per bin, long form (`crl_id,lat,lon,doy,srr_daily_*`; TC/km/yr per day-of-year, summing over doy to the annual SRR) |
| `dsrr_<basin>_<v>.csv` | directional heading mean/stdv per bin (deg) |
| `dsrr_<basin>_<v>.npz` | full arrays - DSRR rate/pdf/cdf (annual + monthly) and `srr_daily_*` (CRL×365), per bin |
| `srr_<R>km/srr_<R>km_<basin>_<v>.csv` | optional **SRR_<R>km** variant - SRR · 2R (TC/yr), separate folder |
| `plots/selection_<basin>/…` | optional annual per-CRL maps (SRR box) |
| `plots/selection_monthly_<basin>/…` | optional per-CRL × month maps |
| `plots/daily_<basin>/…` | optional per-CRL daily SRR curves (All/High/Med/Low vs day 1..365) |
| `plots/selection_<R>km_<basin>/…`, `plots/daily_<R>km_<basin>/…` | optional SRR_<R>km map / daily-curve variants |

In the SRR CSVs each row is one CRL, and the data columns hold the SRR for each
of the four intensity levels (All, High, Med, Low) as both an **annual** value and
the **twelve monthly** values (Jan-Dec, which sum to the annual). Units are
**TC/km/yr** for `srr_<basin>` and **TC/yr** for the `srr_<R>km` variant.

### SRR_<R>km variant (within a radius)

A second variant of the **SRR results only** (not DSRR), off by default
(`SRR_RADIAL`): **SRR_<R>km = SRR · (2·R)** - the rate (storms/km/yr) times the
2R-km diameter, giving the **expected storms / year within `SRR_RADIUS_KM` of each
CRL** (TC/yr). The radius is user-set (default **200 km** → ×400). It is written to
its own `srr_<R>km/` folder and, when maps are enabled, plotted in separate
`selection_<R>km_<basin>/` folders with the SRR box relabeled `SRR_<R>km (TC/yr)`.
The per-CRL maps already print the SRR in the order **All, High, Med, Low**.

## Daily SRR (the continuous seasonal cycle)

The daily SRR is the omnidirectional rate resolved over day-of-year. A few points
explain how it is built; the **whitepaper** has the full treatment.

- **Two kernels, multiplied (not sequential).** Each storm contributes a single
  **scalar** spatial weight `Wi` (the distance kernel, fixed by its closest-approach
  distance) times a **365-long curve** `Wt_i(d)` (the day-of-year kernel, fixed by its
  closest-approach day). `SRR_daily(d) = (1/Nyrs)·Σ_i Wt_i(d)·Wi`. The spatial kernel
  sets *how much* a storm counts; the temporal kernel sets *when* in the year it is
  deposited. There is no intermediate smoothed field. This mirrors the DSRR heading
  kernel `Wd(θ)·Wi`.
- **It is a separable 2-D (distance, time) kernel, not a 3-D (x, y, t) one.** The map
  geometry is reduced to the scalar closest-approach distance during selection, so the
  estimator only resolves distance and time: `K(D, t) = W(D)·W(t)`, a tensor product
  with no cross term.
- **The temporal kernel is circular (wrapped, period 365).** Day-of-year is a loop, so
  the day difference is wrapped to (−182.5, 182.5]: Dec 31 and Jan 1 are 1 day apart,
  not 364. Mass that spills past one end of the year reappears at the other, so `Wt`
  integrates to 1 and the **365 daily values sum exactly to the annual SRR**, with no
  edge effect at the year boundary.
- **365- vs 366-day grid.** The grid is a fixed 365-day calendar keyed by **calendar
  date** (the CF `noleap` convention). The decisive reason is **uniform year-exposure**:
  every date except Feb 29 occurs once per year (`Nyrs` times), so on the 365-day grid
  all days share the same exposure and dividing by `Nyrs` is unbiased. A 366-day grid
  gives Feb 29 a structurally undersampled bin (~`Nyrs/4`) that would show a spurious
  trough unless separately exposure-corrected. Feb 29 is folded onto the Feb 28 / Mar 1
  boundary; the TC season (~May-Nov) never reaches it, so the effect is immaterial.
- **Bandwidth `DAY_KERNEL` (default 14 days).** Chosen by weighted leave-one-out
  likelihood cross-validation on the closest-approach day-of-year
  (`analysis/day_kernel_sensitivity.py`): the data-driven optimum is ~10-11 days
  (Atlantic) and ~13-18 days (Pacific); 14 days sits between the basins' optima, inside
  both per-CRL inter-quartile ranges, at a negligible cross-validated cost
  (< 0.006 nats/storm). It is tunable.

## Method

1. **Load** the CRL set and the augmented HURDAT2 best-track.
2. **Select** per CRL: every TC within `MAX_DIST`; the representative point is the
   fix maximizing `GaussianWeights(K_SIZE)·(1013 − Cp)` (proximity × intensity);
   the closest-approach distance drives the rate kernel, and its date supplies the
   month and day-of-year. Fixes with missing or `> MAX_CP` central pressure are dropped.
3. **SRR / DSRR** via the GKF, per intensity bin and per month of closest approach;
   plus the **daily SRR** via the circular day-of-year kernel (see above).
4. **Write** the selection, SRR/DSRR/daily tables, and full arrays; optionally map
   each CRL.

## Layout

```
storm_climatology_analysis/
├── run_storm_climatology_analysis.py        # launcher (USER OPTIONS)
├── SRR_GKF_Whitepaper.md                    # companion whitepaper (method + validation)
├── analysis/
│   └── day_kernel_sensitivity.py            # DAY_KERNEL bandwidth LOO-CV sensitivity
├── backend/python/
│   ├── api_storm_climatology_analysis.py   # orchestrator entry: run(config)
│   └── storm_climatology_analysis/
│       ├── config.py        # SCAConfig (pydantic)
│       ├── crls.py          # CRL loader (CSV + tab-delimited)
│       ├── hurdat_source.py # locate/load augmented HURDAT2 (links to AHD module)
│       ├── selection.py     # Gaussian-weighted per-CRL storm selection
│       ├── gkf.py           # SRR + DSRR + monthly + daily (GKF)
│       ├── basemap.py       # Natural Earth coastline/boundaries via pyshp
│       ├── plots.py         # per-CRL selected-TC maps
│       ├── writer.py        # SRR/DSRR table + array writers
│       └── orchestrator.py  # SCAOrchestrator: per-basin pipeline
├── data/{inputs/{raw,processed},outputs}/
└── tests/
```
