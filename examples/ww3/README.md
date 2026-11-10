# Reproduce the WW3 comparisons

These examples configure the **installed waden public API**. They contain no
alternative solver and need no NeuralSWM, Gin, neural checkpoint or PyTorch.
Run from the WADEN project root, in either a checkout or an unpacked source distribution.

## Install and obtain the reference

```bash
python -m pip install .
python -m pip install -r examples/ww3/requirements.txt
git clone --no-checkout https://github.com/GaryYang77/waden.git /tmp/waden-ww3-reference
git -C /tmp/waden-ww3-reference checkout --detach b80fdce679414345134a25275421db439c185349
```

The clone supplies archived **data only**; do not install that PyTorch project.
Official experiment descriptions are in
[NOAA-EMC/WW3](https://github.com/NOAA-EMC/WW3/tree/develop/regtests).
The NetCDF files identify WW3 **6.07**. This workflow uses fixed outputs and
requires no Fortran/MPI build. To rebuild WW3, use the archive's selected
`ww3_grid.inp`, `ww3_strt.inp`, `ww3_shel.inp` and propagation settings.
A different build or output dataset constitutes a different reference.
The configs select specific files among the archive's 1968 and 2000 runs.
Missing assets fail; `import waden` never downloads or installs anything.

## Run, compare, visualize

```bash
python examples/ww3/run_case.py \
  --config examples/ww3/configs/tp2_1.json \
  --data-root /tmp/waden-ww3-reference/data --output /tmp/waden-ww3/tp2_1
python examples/ww3/compare.py \
  --config examples/ww3/configs/tp2_1.json \
  --data-root /tmp/waden-ww3-reference/data --output /tmp/waden-ww3/tp2_1
python examples/ww3/plot.py \
  --config examples/ww3/configs/tp2_1.json \
  --data-root /tmp/waden-ww3-reference/data --output /tmp/waden-ww3/tp2_1
```

Replace `tp2_1` with `tp1_1`, `tp1_2` or `tp1_4` for the other runs/comparisons.
For the merged propagation figure, first run **both** tp1.1 and tp1.2 into
sibling directories `/tmp/waden-ww3/tp1_1` and `/tmp/waden-ww3/tp1_2`.
Then call `plot.py` with either configuration; it reads the other saved run
automatically. `--companion-output PATH` selects a different partner directory.
`run_case.py --dtype float64` enables JAX x64 before construction. CPU and
GPU use the same example; `run.json` records devices. Published illustrations
use float32 on one RTX 4080 SUPER. Example-only dependencies are separate
from the core's JAX/NumPy dependencies.

Each output directory contains:

- `prediction.npz`: full predicted spectra, finite-band moments, actual time,
  coordinates, masks, depth, integral and clipping audit.
- `initial_arrays.npz`: an initial wave field for the neural-source example.
- `run.json`: configuration, reference paths/revision, installed package path,
  environment, devices, command and all-grid CFL diagnostic.
- `comparison.json`: float64 full-window and per-time metrics, maximum-error
  indices, station coverage and pulse motion.
- Case-specific animated WebP and a lossless 300-dpi snapshot at step 50:
  tp1.1/tp1.2 share two wave-height profile rows; tp1.4 has wave-height and
  direction rows; tp2.1 has only WW3/WADEN filled-contour panels. The two
  profile figures use identical 12 × 6 inch proportions; the map uses the
  archived 18 × 6 inch proportions. There are no difference/error panels.
  `plot.py --assets PATH` selects an image directory. Animation is 150 dpi.
- A `*_visualization.json` records the actual font file, dimensions, labels,
  shown steps and physical elapsed times. By default, tp1.1/tp1.2 show
  steps 0–100, tp1.4 shows 0–18000 s (steps 0–60), and tp2.1 retains its
  complete 50 steps. `--max-step N` overrides the displayed window only;
  saved trajectories and accuracy reports always cover the complete run.
  Curve values are not fitted or clipped.

The plots require **Times New Roman**. Installed fonts are used directly;
under WSL, existing `/mnt/c/Windows/Fonts/times*.ttf` files are registered
without copying them. Else install the font or pass `--font-path /path/to/times.ttf`.
No alternative font is silently substituted, and font files are not distributed.
Titles, labels and legends follow the old waden figures with NWW/NWM changed
to WADEN. tp1.4 overlays the bathymetry on separate right-hand axes, with
depth increasing downward. tp1.1/tp1.2 annotate the initial-peak deep-water
phase speed, group speed and grid spacing. tp1.4's displayed direction is coming-from, matching that figure;
stored predictions and scientific comparison metrics keep their original
conventions and are unchanged by plotting.

## Conditions and interface mapping

| Case | Grid | Active transport | Output interval × advances | Reference spectra |
|---|---|---|---|---|
| tp1.1 | 360 × 3 spherical, equatorial channel | x, periodic | 3600 s × 144 | 15 stations, first 10 frames |
| tp1.2 | 3 × 123 spherical, meridional channel | y | 3600 s × 144 | 1 station, 145 frames |
| tp1.4 | 13 × 3 Cartesian, varying depth | x and direction | 300 s × 144 | 11 stations, 145 frames |
| tp2.1 | 43 × 43 Cartesian, irregular domain | x and y | 360 s × 50 | 6 stations, 51 frames |

Saved trajectories include the initial frame in addition to the advances.
Spherical grid coordinates are degrees; Cartesian coordinates are metres.
The archive's Cartesian station coordinates contain kilometre-valued numbers
despite `units="m"`; configs explicitly multiply by 1000 and check grid
alignment. Spectral direction is **going-to compass degrees**, as declared
by the NetCDF attributes; bulk `dir` is **coming-from**, converted by 180°.
Axes change from `[time,y,x]` and `[time,station,freq,direction]` to waden's
`[batch,x,y,direction,freq]` convention.

The common initial station spectral shape is combined with the full **t=0**
`hs` envelope; observed nonzero initial station shapes are checked for
separability. No later interior reference initializes or corrects predictions.
`MAPSTA=1/2` selects reference ocean cells; `MAPSTA=2` supplies constant
prescribed initial spectra. Original physical bathymetry is checked against
the NetCDF depth. Sources, frequency shift and the tail are off. GSE is off;
tp1.4/tp2.1 explicitly archive `PRO3 WDTHCG=WDTHTH=0`. The tp1.1/tp1.2 text
inputs do not explicitly specify PRO3 widths, so their binary preprocessing
history is not reconstructed here. Compiled switches are recorded from the
NetCDF metadata, which can differ from the adjacent text `switch` file.
A compiled ST module does not itself enable sources in these no-source
propagation experiments. Historical velocity factors of 1.05 are not used.

tp1.4 needs an **open outgoing edge**. Its last excluded x column is a
zero-spectrum ghost with the adjacent physical depth continued into it.
`Environment.boundary_mask` freezes it while the public transport carries
flux out. Physical depths and scored cells stay unchanged. A zero-depth
ghost selects an impermeable shoreline and traps energy; that is a different
boundary problem. No flux formula or core implementation is changed.

`physical_cfl` includes inactive cells and frozen ghosts. tp1.4's all-grid
angular maximum is about 3.23 at an excluded ghost, not an evolved ocean
cell. Diagnostics are recorded without automatic timestep adjustment or
stability certification. WW3's internal `CFLTM` target is not an acceptance
threshold for this all-grid diagnostic. Archived timesteps are retained.

## Interpreting the measurements

Principal scores cover **all MAPSTA=1 cells at all saved times**, including
the initial frame. Prescribed values are displayed but not scored. SWH
errors use unweighted cells/times; the integral audit is area weighted.
Station spectra are compared only at exact common times. tp1.1 station
coverage ends at hour 9 despite its 144-hour field trajectory; sparse station
agreement must not be described as full-field spectral agreement.

Relative L2 is `norm(prediction-reference)/norm(reference)`; peak-normalized
error is the maximum absolute difference divided by the reference peak.
For a zero reference, absolute metrics are used and relative metrics are
null; no artificial denominator is added. Direction/period validity uses
reference SWH ≥0.01 m. Angular differences are circular and never clipped.
If the prediction has zero directional resultant on that reference-selected
domain, it has no defined direction. The report counts these cases separately;
the numeric angular score retains NumPy's `atan2(0,0)=0` convention, so its
maximum must not be interpreted as a physical angular discrepancy there.

Reports compare both archived bulk diagnostics and moments recomputed from
both station spectra with the same midpoint finite-band quadrature. WW3's
diagnostic endpoint/tail conventions can differ from this common quadrature.
Pulse motion fits cover the first 48 hours, before exit affects the pulse.

## Measured agreement

The following measurements use float32 on one RTX 4080 SUPER and the complete
saved windows, including the initial frame. Prescribed cells are excluded;
station spectra use only actual common output times.

| Case | Advances | Full-field SWH relative L2 | SWH RMSE (m) | Maximum SWH difference (m) | Station spectrum relative L2 |
|---|---:|---:|---:|---:|---:|
| tp1.1 | 144 | 3.09e-5 | 4.08e-6 | 0.000309 | 4.13e-7 |
| tp1.2 | 144 | 0.0433 | 0.00905 | 0.193 | 5.92e-5 |
| tp1.4 | 144 | 0.00828 | 0.00734 | 0.0456 | 0.0785 |
| tp2.1 | 50 | 0.0227 | 0.0474 | 0.304 | 5.48e-6 |

tp1.1's periodic-domain spectrum integral changes by about 1e-7 relative to
its initial value. The other cases have open or prescribed boundaries.
tp1.1's small station error covers only the first nine hours, not its full
144-hour field trajectory. tp1.4 retains a station spectral difference of
about 7.85%, while tp2.1 has a local maximum SWH difference of 0.304 m.

These comparisons describe the archived experiments rather than exact WW3
equivalence. Boundary treatment, splitting, precision and diagnostic
definitions can contribute residual differences; the cause of every
residual has not been isolated. Full float64 runs give similar SWH errors
for tp1.2, tp1.4 and tp2.1, so those differences cannot be attributed solely
to float32 precision.
