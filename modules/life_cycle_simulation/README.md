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

### Serial correlation and clustering (optional)

With `correlation=True` the independent Poisson in step 2 is replaced by a
serially-correlated, overdispersed annual rate, so active and quiet years cluster
(the annual mean is preserved, so the catalog still matches the SRR):

```text
S_y      = ar_phi * S_{y-1} + sqrt(1 - ar_phi^2) * eps_y      (AR(1), N(0,1))
lambda_y = lambda * exp(ar_beta * S_y - ar_beta^2 / 2)        (mean-preserving)
N(y)     ~ Poisson(lambda_y * G_y),  G_y ~ Gamma(mean 1, var overdispersion)
```

`ar_beta` drives the year-to-year memory (lag-1 autocorrelation); `overdispersion`
lifts the count variance (`Fano = 1 + lambda * overdispersion`).

Each parameter left as `None` (the default) is **calibrated from that CRL's
historical annual counts** in the SCA selection table (`overdispersion =
(Fano - 1)/mean`, the AR(1) terms from the lag-1/lag-2 count autocorrelation); set a
number to override. A sparse, low-rate CRL typically calibrates to ~0 (Poisson),
which is the statistically appropriate result; the clustering signal is
basin/regional. The selection table is auto-located next to `input_csv`.

Because that signal is regional, set **`regional_pool_km`** to calibrate from every
CRL within that great-circle distance of the target (pooling their mean-weighted
second moments, with no storm double-counting) instead of the one sparse record. This
surfaces the modest regional overdispersion that a single CRL cannot resolve (e.g. a
Gulf CRL moves from `Fano ~ 1.00` per-CRL to `~ 1.08` pooled over ~70 neighbours).
`None` (default) keeps the per-CRL calibration.

By default the pool weights every member CRL uniformly (a hard cutoff at
`regional_pool_km`). Set **`regional_pool_sigma_km`** to instead taper the weights by
distance, `w = exp(-d^2 / (2 sigma^2))`, so CRLs nearer the target (which share more
of its climate state) count more. Here `sigma` is the kernel bandwidth, ideally the
climate decorrelation length: the half-weight distance is `~1.18 sigma`. The fade at
the cutoff is set by the ratio `R/sigma`, not by `R` alone, the edge weight being
`exp(-0.5 (R/sigma)^2)` (about `0.14` at `R = 2 sigma`, `0.01` at `R = 3 sigma`), so
set `regional_pool_km` to roughly `2-3 sigma` to truncate only a negligible tail.
`None` keeps the uniform weighting.

### Within-season clustering (optional, intra-year)

The correlation layer above acts on the annual *count*; it does not change *when*
within a year storms land. `intra_year_correlation=False` (default) places a year's
storms independently from the seasonal shape (an inhomogeneous Poisson process). When
**`intra_year_correlation=True`**, a year's storms bunch into a sub-seasonal active
window (an MJO-phase or persistent-pattern effect) via a **shared-factor Gaussian
copula** on the event days, which preserves **both** the annual count and the seasonal
day-of-year marginal exactly; only the within-year inter-arrival gaps tighten. This is
the *intra*-year analogue of `correlation` (the *inter*-year, count-level layer).

The strength `within_season_rho` in `[0, 1)` defaults to `None` = **calibrated** from
the historical within-year storm-day correlation: each selected storm's `doy` is
mapped to its seasonal quantile and then to a latent normal, and rho is the within-year
correlation of those normals (with a one-standard-error small-sample shrinkage so a
sparse CRL is not given spurious clustering). A number overrides. A branching
Hawkes/Neyman-Scott process is deliberately *not* used: it would change the
SRR-calibrated count, whereas the copula keeps the count fixed.

Setting `regional_pool_km` pools the same neighbouring CRLs (and optional Gaussian
taper) the count calibration uses, sharpening the estimate from a handful of per-CRL
multi-storm years to thousands of CRL-years. The **`within_season`** figure validates
the result: the simulated distribution of same-year inter-arrival gaps against the
historical one from the selection (pdf + ECDF); when rho is right the two overlie.

### Sequencing

With `sequencing=True` (default) the catalog gains a chronological event timeline:
`event_time` (continuous years = `(year-1) + (doy-1)/365`), `seq` (the
per-realization chronological order), and `wait_yr` (the inter-arrival waiting time
from the previous event). Rows are ordered by realization then `event_time`.

## Inputs

LCS consumes the SCA outputs (it does not read HURDAT or any track data directly):

| Input | What it is |
|-------|------------|
| `srr_<basin>_<v>.csv` (required) | Per-CRL annual + monthly SRR in TC/km/yr. The `all` rate sets `lambda`; the Low/Med/High rates set the stratum split. **Use this file, not the `srr_<R>km` variant** (already multiplied by the diameter). |
| `srr_daily_<basin>_<v>.csv` (optional) | Companion continuous daily SRR (long form `crl_id,lat,lon,doy,...`). Auto-located next to the input; used by `day_method="daily"` for the seasonal shape. |

## Outputs

`data/outputs/`, one pair per CRL (tag = `crl<NNNN>_R<R>km_<Y>yr_<N>real`):

```text
lcs_catalog_<tag>.csv      one row per synthetic TC: realization, year, event,
                             intensity, month, day, doy, and (when sequencing)
                             event_time, seq, wait_yr
lcs_summary_<tag>.csv      per-realization TC counts overall and by stratum
plots/crl<NNNN>/lcs_<key>_<tag>.png   optional figures, one folder per CRL holding
                             all its plots (incl. clustering: annual-count ACF +
                             trajectories when correlation is on)
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
| `correlation` | `False` | Serial correlation + overdispersion of annual counts (else independent Poisson) |
| `ar_phi` | `None` | AR(1) persistence `[0, 1)`; `None` = calibrate from history |
| `ar_beta` | `None` | Log-rate sensitivity to the state (lag-1 ACF); `None` = calibrate |
| `overdispersion` | `None` | Rate-multiplier variance (`Fano = 1 + lambda*overdispersion`); `None` = calibrate |
| `regional_pool_km` | `None` | Pool CRLs within this many km for the calibration (regional); `None` = per-CRL |
| `regional_pool_sigma_km` | `None` | Gaussian distance taper for the pool (`exp(-d^2/2 sigma^2)`); `None` = uniform |
| `selection_csv` | `None` | SCA selection table for calibration; `None` auto-locates next to `input_csv` |
| `intra_year_correlation` | `False` | Within-season (intra-year) day clustering (else independent placement) |
| `within_season_rho` | `None` | Intra-year clustering strength `[0, 1)`; `None` = calibrate from history |
| `sequencing` | `True` | Add the chronological event timeline (`event_time`, `seq`, `wait_yr`) |
| `make_plots` | `False` | Write the per-CRL diagnostic figures |
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
