# storm_surge_hydrograph (SSH)

Per-save-point **unit (scalable) storm-surge hydrographs**. For each coastal save
point, SSH reduces an ensemble of synthetic-TC surge time series to a single
dimensionless surge shape (peak = 1) that is **scaled by two parameters, a peak surge
elevation and an equivalent width** (a timescale), to reconstruct a full hydrograph. One
shape is derived per save point.

**Why two parameters.** Normalizing by peak alone leaves a large temporal spread
(coefficient of variation ~0.8) that is nearly independent of the peak, so a peak-only
shape blurs the crest and tails. The module therefore uses **double normalization**:
the shape is a function of dimensionless time `s = tau/W`, where `W` is the equivalent
width (area/peak). The companion **whitepaper** (`SSH_Whitepaper.md`, local, not versioned)
compares four shape models (amplitude-only, double normalization, duration clustering,
functional PCA); double normalization is the most accurate and parameter-efficient. The
comparison is reproducible via `analysis/shape_model_comparison.py`.

## Inputs (`data/inputs/raw/`)

The Coastal Texas Study (CTXS) save-point dataset near Galveston, TX (the delivered files carry a `CTXCS_` prefix):

- **`CTXCS_staID.csv`** - one row per save point: `id, lat, lon, depth`. The depth
  column is **positive-down**, so ground elevation above NAVD88 is `-depth`.
- **`CTXCS_TP_SYN_Tides_0_SLC_0_surge_SP#####.csv`** - one file per save point;
  each column is one synthetic TC's surge time series (water-surface elevation,
  **metres above NAVD88**), rows are **15-min** steps. `-99999` = **dry** (water
  below the point's ground); trailing `NaN` = padding (storms of unequal length).
- **`CTXCS_TP_SYN_Tides_0_SLC_0_time.csv`** - matching timestamps (used only to
  confirm the 15-min step).

Inputs follow the CyHAN **raw / processed** convention.

## Method (per save point)

1. **Ground & surge-above-ground.** Ground elevation `G = -depth` (m NAVD88).
   For each storm, surge above ground `a(t) = E(t) - G`; dry (`-99999`) and any
   sub-ground samples are `0` (no water on the point).
2. **Peak-align + double-normalize.** Each storm is normalized by its own peak surge
   `n(t) = a(t)/max(a)`, shifted so **`tau = 0` is the time of the peak**, and its time
   is scaled by an **equivalent width** `W = (integral of a)/peak`, giving a shape in
   **dimensionless time `s = tau/W`**. (Peak alignment keeps the crest sharp; time
   normalization removes the temporal spread.)
3. **Aggregate.** The doubly-normalized storms are averaged (mean or median) into one
   **canonical unit hydrograph** `C(s)`, with `C(0) = 1`.
4. **Parametric limbs.** Separate curves are fit to the **rising** (`s <= 0`) and
   **falling** (`s >= 0`) limbs, each a generalized Gaussian
   `C(s) = exp(-0.5 (|s|/sigma)^p)`, continuous at the peak.
5. **Scale (two parameters).** A hydrograph for a target **peak elevation `P`** and
   **equivalent width `W`** is `E(tau) = G + C(tau/W) * (P - G)`. Because `W` is independent
   of `P`, it is a genuine second input: supply a storm-specific `W`, or use the reported
   width distribution / P25-P50-P75 envelope. (`METHOD = "amplitude"` selects the
   legacy peak-only shape over physical time `tau`.)

### Equivalent width vs actual duration

The equivalent width `W = area/peak` is a *characteristic width* (the peak-height rectangle
with the same area), not a literal event length; it is the timescale the model scales by. A
comparison (`analysis/timescale_comparison.py`) shows it collapses the ensemble marginally
better than the FWHM or a second-moment width and is always defined. It maps to a physical
**actual duration** - the time the surge exceeds `z0 = max(ground, MHHW) + 0.30 m` (0.30 m
above ground for overland points, 0.30 m above MHHW for overwater) - through the canonical
level-width `Phi(f)`: `actual_duration = W * Phi(f)`, `f = (z0 - G)/(P - G)`. On the CTXS
data the actual duration is ~1.4x the equivalent width (corr 0.95), and converting an
observed duration + peak back to `W` is accurate to ~10% (`analysis/actual_duration_relationship.py`).
So the model can be driven by an equivalent width or by an observed inundation duration.

### A note on the data

Save-point wetness varies strongly with ground elevation. In the CTXS set the
deeper points (SP4149-4153, ground +2.0 to +2.4 m) are wet for **71-110** of the
660 storms and give tight shapes (fit RMSE ~0.01); the higher points
(SP3911-3915, ground +0.4 to +1.3 m) are wet for only **3-5** storms, so their unit
hydrographs are data-limited and noisier. Each shape is still built from that
point's own storms, as configured.

## Run

```bash
pip install -r requirements.txt      # numpy, pandas, pydantic, scipy, matplotlib
python run_storm_surge_hydrograph.py
python run_storm_surge_hydrograph.py --aggregate median --no-plots
```

### Key options (USER OPTIONS block)

- **`STAID_FILE` / `SURGE_FILE_GLOB` / `TIME_FILE`** - input files under
  `data/inputs/raw/` (`{sp}` is the 5-digit save-point id).
- **`SAVE_POINTS`** - subset of ids, or `None` for all found.
- **`METHOD` ("double_norm")** - shape model: `double_norm` (peak + equivalent width;
  recommended) or `amplitude` (legacy peak-only). `--method` overrides.
- **`DT_HOURS` (0.25), `DRY_VALUE` (-99999), `DEPTH_POS_DOWN` (True)** - data
  semantics (ground elevation = `-depth`).
- **`MIN_WET_SAMPLES` (5)** - skip a storm with fewer above-ground samples.
- **`WINDOW_HOURS` (auto), `MAX_WINDOW_HOURS` (72)** - peak-aligned half-window;
  auto fits the wet extent.
- **`AGGREGATE` ("mean")** - mean or median across a point's storms.
- **`PARAMETRIC` (True)** - fit rising/falling generalized-Gaussian limbs.
- **`ACTUAL_DUR_OFFSET_M` (0.30), `MHHW_NAVD88` (None)** - actual-duration threshold
  `z0 = max(ground, MHHW) + offset`. `None` MHHW treats all points as overland (0.30 m
  above ground); set MHHW (m NAVD88) when overwater points are present.
- **`SCALE_PEAKS` ("auto")** - example scaled hydrographs at each point's observed
  median and max peak, or a list of peak elevations, or `None`.

## Outputs (`data/outputs/`)

| File | Contents |
|---|---|
| `unit_hydrograph_SP#####.csv` | canonical shape: `s_dimensionless, u_empirical, u_parametric` (`tau_hours` for the legacy amplitude method) |
| `scaled/hydrograph_SP#####_peak<P>m.csv` | peak-scaling examples at the median duration (`tau_hours, elevation_m_navd88, surge_above_ground_m`) |
| `scaled/hydrograph_SP#####_widthenv_p25/p50/p75.csv` | equivalent-width envelope at the median peak |
| `ssh_parameters.csv` | per save point: geometry, ground elev, overwater flag, method, n_storms, peak min/median/max, equiv-width P25/P50/P75, actual-duration threshold + median, corr(peak,width), limb fit |
| `plots/SSH_SP#####.png` | 3 panels: canonical unit hydrograph + ensemble + fit; peak scaling; equivalent-width envelope |
| `plots/SSH_ensemble_SP#####.png` | every storm's unnormalized hydrograph (m NAVD88) peak-aligned, colored by peak elevation |
| `analysis/*.py` | comparison studies (whitepaper): `shape_model_comparison.py`, `timescale_comparison.py`, `actual_duration_relationship.py` + metrics CSVs and figures |

To scale to a target peak and duration in code:

```python
from storm_surge_hydrograph.hydrograph import scale_to_peak, width_stats
W = width_stats(unit_hydrograph)["p50"]                     # or a storm-specific equiv width
tau, elev = scale_to_peak(unit_hydrograph, peak_elev=4.5, equiv_width=W)        # m NAVD88
# or drive it from an observed inundation duration (time above 0.30 m) + the peak:
tau, elev = scale_to_peak(unit_hydrograph, peak_elev=4.5, actual_duration=9.0)  # h
```

## Layout

```
storm_surge_hydrograph/
|- run_storm_surge_hydrograph.py            # launcher (USER OPTIONS)
|- analysis/                                # whitepaper comparison studies
|  |- shape_model_comparison.py             # amplitude/double-norm/cluster/FPCA
|  |- timescale_comparison.py               # equivalent width vs FWHM vs 2nd-moment
|  |- actual_duration_relationship.py       # actual duration <-> equivalent width
|- backend/python/
|  |- main_storm_surge_hydrograph.py        # orchestrator entry: run(config)
|  |- storm_surge_hydrograph/
|     |- config.py        # SSHConfig (pydantic)
|     |- io.py            # staID + surge/time matrix loaders
|     |- hydrograph.py    # normalize, unit hydrograph, parametric limbs, scaling
|     |- writer.py        # unit / scaled / parameter CSV writers
|     |- plots.py         # per-save-point diagnostic figure
|     |- orchestrator.py  # SSHOrchestrator: per-save-point pipeline
|- data/{inputs/{raw,processed},outputs}/
|- tests/
```
