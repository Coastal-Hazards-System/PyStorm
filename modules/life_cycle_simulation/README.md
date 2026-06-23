# life_cycle_simulation (LCS)

*Monte-Carlo synthetic tropical-cyclone life cycles for a Coastal Reference
Location, driven by the storm_climatology_analysis (SCA) storm recurrence rate.*

For a chosen CRL and a radius of influence, LCS draws a synthetic catalog of
tropical cyclones over a requested life-cycle length (e.g. 100 years) and a
requested number of realizations (e.g. 1000). The annual SRR sets how often TCs
occur, the Low/Med/High SRR ratios set each TC's intensity stratum, and the
seasonal SRR shape sets each TC's calendar day of closest approach. Every catalog
row is one synthetic TC, ready for downstream event-based life-cycle hazard work.

This module focuses on tropical cyclones (`storm_type="tc"`). Extratropical
cyclones (`"etc"`) are a scaffolded placeholder.

## Method

For the chosen CRL, with the SRR read from the SCA tables:

1. **Poisson rate.** `lambda = SRR_all * (2 * R)` [TC / yr], where `SRR_all` is the
   omnidirectional annual rate density [TC / km / yr] and `R` is the radius of
   influence [km]. The `2R`-km band turns the rate density into an expected count.
2. **Annual activity.** `N(y) ~ Poisson(lambda)` for each year `y = 1..SIM_YEARS`,
   independently for each of `N_REALIZATIONS` life cycles.
3. **Intensity stratum.** Each TC draws Low/Med/High `~ Categorical(p_low, p_med,
   p_high)`, `p_s = SRR_s / (SRR_low + SRR_med + SRR_high)`.
4. **Day of occurrence.** Each TC draws a day-of-year `1..365` from its stratum's
   seasonal SRR shape, then maps to a calendar month and day. The shape is either
   the smooth daily SRR curve (`day_method="daily"`) or the monthly SRR spread
   uniformly within each calendar month (`day_method="monthly"`).

The whole `(realization x year)` grid is drawn in one vectorized pass, so cost
scales with the number of TCs produced, not with Python loops over years or
realizations. No C++ engine: the work is light and fully NumPy-vectorized.

## Inputs

LCS consumes the SCA outputs (it does not read HURDAT or any track data directly):

| Input | What it is |
|-------|------------|
| `srr_<basin>_<v>.csv` (required) | Per-CRL annual + monthly SRR in TC/km/yr. The `all` rate sets `lambda`; the Low/Med/High rates set the stratum split. **Use this file, not the `srr_<R>km` variant** (already multiplied by the diameter). |
| `srr_daily_<basin>_<v>.csv` (optional) | Companion continuous daily SRR (long form `crl_id,lat,lon,doy,...`). Auto-located next to the input; used by `day_method="daily"` for the seasonal shape. |

## Outputs

`data/outputs/`, one pair per CRL (tag = `crl<NNNN>_R<R>km_<Y>yr_<N>real`):

```text
lcs_catalog_<tag>.csv      one row per synthetic TC:
                             realization, year, event, intensity, month, day, doy
lcs_summary_<tag>.csv      per-realization TC counts overall and by stratum
plots/lcs_diag_<tag>.png   optional QC figure (count / stratum / seasonality)
```

## Quickstart

```bash
cd modules/life_cycle_simulation
python run_life_cycle_simulation.py
```

Edit the USER OPTIONS block at the top of the launcher (CRL id, radius,
years, realizations), or override on the command line:

```bash
python run_life_cycle_simulation.py --crl 844 --radius-km 200 \
    --years 100 --realizations 1000 --plots
```

## Programmatic API

Every module exposes a single `run(config)` entry point (see the root README).

```python
import sys
sys.path.insert(0, "modules/life_cycle_simulation/backend/python")
from api_life_cycle_simulation import run

result = run({
    "input_csv": "modules/storm_climatology_analysis/data/outputs/srr_atlantic_1938-2025_20260227.csv",
    "crl_ids": [844],
    "radius_km": 200.0,
    "sim_years": 100,
    "n_realizations": 1000,
    "day_method": "daily",   # or "monthly"
    "seed": 12345,
})
```

`run(config)` returns an `LCSResult`:

| Field | Type | Meaning |
|-------|------|---------|
| `results` | `dict[int, CRLResult]` | One entry per CRL id requested |
| `sim_years` | `int` | Life-cycle length used |
| `n_realizations` | `int` | Realization count used |

Each `CRLResult` carries `crl_id`, `lat`, `lon`, `lam` (the Poisson rate), the
stratum probabilities `p_low/p_med/p_high`, `n_events` (total synthetic TCs),
`daily_used` (whether the smooth daily SRR backed the day draw), and the
`catalog_path` / `summary_path` / `plot_path` written.

`config` accepts a plain dict (validated into an `LCSConfig`) or an `LCSConfig`
directly.

## Configuration

| Option | Default | Meaning |
|--------|---------|---------|
| `input_csv` | (required) | SCA `srr_<basin>_<v>.csv` (TC/km/yr) |
| `daily_csv` | `None` | Daily companion; `None` auto-locates next to `input_csv` |
| `crl_ids` | `[844]` | CRL id, or a list of ids (one catalog each) |
| `radius_km` | `200.0` | Radius of influence (km); `lambda = SRR * 2 * radius_km` |
| `sim_years` | `100` | Life-cycle length (years) |
| `n_realizations` | `1000` | Independent realizations |
| `day_method` | `"daily"` | `"daily"` (smooth) or `"monthly"` (month + uniform day) |
| `seed` | `12345` | Reproducible RNG; `None` = nondeterministic |
| `make_plots` | `False` | Write the per-CRL diagnostic figure |
| `output_dir` | `data/outputs` | Where catalogs and summaries land |

## Tests

```bash
cd modules/life_cycle_simulation
pytest -q
```

Smoke tests cover the config validators, the 365-day calendar maps, SRR prefix
detection and daily-companion location, the Poisson / stratum formulas, the
vectorized simulator (counts expansion, stratum/day draws, reproducibility), the
catalog/summary writers, the `tc`/`etc` dispatch, and an end-to-end `run(config)`.
