# joint_distribution_model (JDM)

*Per-CRL Joint Probability Method (JPM) characterization of tropical-cyclone
parameters: distance-weighted marginal distributions and a meta-Gaussian copula.*

For each Coastal Reference Location (CRL) and intensity bin (All / High / Med / Low
by central-pressure deficit `Dp = 1013 - Cp`), JDM characterizes the joint
distribution of the four JPM storm parameters **[Heading, Dp, Rmax, forward
translation Vt]**: the **marginal** distribution of each parameter, plus a
**meta-Gaussian copula** that captures their dependence. This is the
parameter-characterization layer that synthetic-storm generation builds on.

JDM is a downstream consumer of `storm_climatology_analysis` (SCA), like LCS. It is
fitting-only (it characterizes the distributions); a synthetic-parameter sampler
belongs downstream (in LCS). Tropical cyclones only (`storm_type="tc"`).

## Method

Per CRL, in three stages:

1. **Distance-weighted adjustment + intensity binning.** Each storm's parameter is
   rescaled so the sample mean/std become the Gaussian-distance-weighted mean/std
   (`X_adj = z*sigma_dw + mu_dw`, `z = (X-mu)/sigma`); heading is recentered on the
   SCA DSRR circular mean. Storms are then binned by adjusted `Dp` into
   High (`>=48`), Med (`[28,48)`), Low (`[8,28)`), and All (`>=8`).
2. **Marginal distributions** per CRL per intensity: `Dp` = truncated Weibull (the
   `>=28` body is jitter-bootstrapped, the `[8,28)` tail a point fit, each truncated
   to its band); `Rmax` = lognormal; `Vt` = normal (High/Med) or lognormal (Low);
   `Heading` = the SCA DSRR directional distribution.
3. **Meta-Gaussian copula:** `rho = sin(pi/2 * tau)`, where `tau` is the Kendall's
   tau matrix of `[Hd, Dp, Rmax, Vt]`, per intensity.

Pure Python (scipy); the Dp bootstrap is parallelized over CRLs. The Weibull fit
uses a fast root-finding MLE so the 10000x bootstrap stays tractable.

## Inputs

JDM reads two **SCA outputs** (run SCA first); it does not read HURDAT directly:

| Input | What it is |
|-------|------------|
| `selection_<basin>_<v>.csv` | Per-CRL selected TCs: `heading_deg, trans_kmh, vmax_kmh, cp_mindist, cp_gauss, gaussW, dist, rmax_km, dp, year`. |
| `dsrr_<basin>_<v>.npz` | Per-CRL directional heading mean/stdv/cdf per intensity bin, and CRL lat/lon. |

Both auto-locate (newest) under the sibling SCA outputs dir, or pin explicit files.

## Outputs

`data/outputs/`, tagged `<v>` from the SCA selection filename:

```text
jdm_marginals_<basin>_<v>.csv   per CRL x intensity x parameter marginal params
                                  (crl_id, intensity, param, dist, p1, p2, trunc_lo,
                                   trunc_hi, n)
jdm_copula_<basin>_<v>.npz      per CRL x intensity Kendall tau + Gaussian rho (4x4,
                                  axes = [Hd, Dp, Rmax, Vt])
jdm_adjusted_<basin>_<v>.csv    adjusted, stratum-labeled per-storm [Hd, Dp, Rmax, Vt]
plots/marginals_<basin>/CHS_<Basin>_CRL_<NNNN>.png   optional per-CRL marginal fits
plots/copula_<basin>/CHS_<Basin>_CRL_<NNNN>.png      optional per-CRL copula figure
                                                       (rho heatmaps + All-bin pairs)
```

## Quickstart

```bash
cd modules/joint_distribution_model
python run_joint_distribution_model.py                       # Atlantic, full bootstrap
python run_joint_distribution_model.py --n-boot 500 --plots  # quick run + figures
python run_joint_distribution_model.py --cp-source cp_gauss  # use SCA representative Cp
```

The Dp bootstrap is the heavy step; lower `--n-boot` for a quick pass and use
`--n-jobs` to parallelize over CRLs.

## Programmatic API

```python
import sys
sys.path.insert(0, "modules/joint_distribution_model/backend/python")
from api_joint_distribution_model import run

result = run({
    "basins": "atlantic",
    "sca_outputs_dir": "modules/storm_climatology_analysis/data/outputs",
    "n_boot": 10000, "cp_source": "cp_mindist", "seed": 12345,
})
```

`run(config)` returns a `JDMResult` with `results: dict[str, BasinResult]`; each
`BasinResult` carries the input files, `n_crls`, `n_records`, and the
`marginals_path` / `copula_path` / `adjusted_path` written. `config` accepts a dict
(validated into a `JDMConfig`) or a `JDMConfig`.

## Configuration

| Option | Default | Meaning |
|--------|---------|---------|
| `basins` | `["atlantic"]` | `atlantic` / `pacific` / `both` (Pacific scaffolded) |
| `sca_outputs_dir` | sibling SCA `data/outputs` | Where to auto-locate the SCA inputs |
| `cp_source` | `cp_gauss` | Cp for Dp (`cp_gauss` = SCA representative-point Cp; `cp_mindist` = Cp at closest approach) |
| `start_year` | `1938` | Drop selected TCs before this season |
| `min_dp` / `dp_low` / `dp_med` | `8 / 28 / 48` | Intensity-bin deficit thresholds (hPa) |
| `n_boot` | `10000` | Dp jitter-bootstrap count per CRL |
| `seed` | `12345` | Reproducible bootstrap; `None` = nondeterministic |
| `n_jobs` | `None` | CRL parallelism (`None`/0 = auto, 1 = serial) |
| `make_plots` | `False` | Write per-CRL marginal diagnostic figures |

## Tests

```bash
cd modules/joint_distribution_model
pytest -q
```

Smoke tests cover the config validators, the distance-weighted adjustment, heading
recenter, intensity binning, the jitter bootstrap, the Weibull/lognormal/normal fits
and truncated-Weibull bounds, the copula `tau -> rho`, the SCA source accessors, and
an end-to-end `run(config)` on a synthetic SCA selection + DSRR.
