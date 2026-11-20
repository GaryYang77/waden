# Reproduce the WW3 comparisons

These examples use the **installed waden public API** with the bundled
WW3 reference data.
Run from the WADEN project root, in either a checkout or an unpacked source distribution.

## Install

```bash
python -m pip install .
python -m pip install -r examples/ww3/requirements.txt
```

The reference data are included in [`data/`](data/README.md), both in the
repository and the source distribution. No separate download is required.
Official experiment descriptions are in
[NOAA-EMC/WW3](https://github.com/NOAA-EMC/WW3/tree/develop/regtests).
The NetCDF files identify WW3 **6.07**. This workflow uses fixed outputs and
requires no Fortran/MPI build. To regenerate the WW3 reference files,
use the optional instructions below.
A different build or output dataset constitutes a different reference.
The configs select specific files among the archive's 1968 and 2000 runs.
Missing assets fail; `import waden` never downloads or installs anything.

## Run, compare, visualize

```bash
for case in tp1_1 tp1_2 tp1_4 tp2_1; do
  python examples/ww3/run_case.py \
    --config "examples/ww3/configs/$case.json" \
    --data-root examples/ww3/data --output "outputs/ww3/$case"
  python examples/ww3/compare.py \
    --config "examples/ww3/configs/$case.json" \
    --data-root examples/ww3/data --output "outputs/ww3/$case"
done

for case in tp1_1 tp1_4 tp2_1; do
  python examples/ww3/plot.py \
    --config "examples/ww3/configs/$case.json" \
    --data-root examples/ww3/data --output "outputs/ww3/$case"
done
```

The first plot combines tp1.1 and tp1.2 from their sibling output directories.
`--companion-output PATH` selects a different partner directory.
`run_case.py --dtype float64` enables JAX x64 before construction. CPU and
GPU use the same example; `run.json` records devices. Published illustrations
use float32 on one RTX 4080 SUPER. Example-only dependencies are separate
from the core's JAX/NumPy dependencies.

Each output directory contains:

- `prediction.npz`: full predicted spectra, finite-band moments, actual time,
  coordinates, masks, depth, integral and clipping audit.
- `initial_arrays.npz`: an initial wave field for the neural-source example.
- `run.json`: configuration, reference paths and WW3 version, installed package path,
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
Titles, labels and legends identify WW3 and WADEN consistently. tp1.4 overlays the bathymetry on separate right-hand axes, with
depth increasing downward. tp1.1/tp1.2 annotate the initial-peak deep-water
phase speed, group speed and grid spacing. tp1.4's displayed direction is coming-from, matching that figure;
stored predictions and scientific comparison metrics keep their original
conventions and are unchanged by plotting.

## Optional: regenerate the reference with WW3

The bundled NetCDF files are sufficient for the WADEN comparisons above.
To run WW3 itself, install WAVEWATCH III **6.07.1** with NetCDF support and
compile `ww3_grid`, `ww3_strt`, `ww3_shel`, `ww3_ounf` and `ww3_ounp` using
the supplied `data/ww3_tp*/input/switch` (identical across the four cases).
This is a serial `SHRD`, `PR3 UQ`, `NC4` configuration; no MPI launcher is
needed. The external WW3/NetCDF-Fortran installation is separate from pip.

From the WADEN project root, set the absolute executable-directory path and
run in fresh output directories:

```bash
WW3_EXE="/absolute/path/to/WW3-6.07.1/model/exe"
for case in ww3_tp1.1 ww3_tp1.2 ww3_tp1.4 ww3_tp2.1; do
  work="outputs/ww3_reference/$case"
  mkdir -p "$work"
  cp examples/ww3/data/"$case"/input/*.inp "$work/"
  cp examples/ww3/data/"$case"/input/*.depth "$work/"
  (
    cd "$work" || exit
    for program in ww3_grid ww3_strt ww3_shel ww3_ounf ww3_ounp; do
      "$WW3_EXE/$program" > "$program.log" 2>&1 || exit
    done
  ) || exit
done
```

The generated `.nc` filenames match the bundled reference filenames.
tp1.1 intentionally exports only the first 10 station-spectrum frames;
tp1.2 and tp1.4 export 145 frames, and tp2.1 exports 51 frames. The bundled
files retain their original WW3 build metadata; source-term modules are
inactive in these propagation cases.

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
the NetCDF depth at wet cells; excluded cells then receive zero depth.
Sources, frequency shift and the tail are off. GSE is off;
tp1.4/tp2.1 explicitly archive `PRO3 WDTHCG=WDTHTH=0`. The tp1.1/tp1.2 text
inputs do not explicitly specify PRO3 widths, so their binary preprocessing
history is not reconstructed here. Compiled switches are recorded from the
NetCDF metadata, which can differ from the adjacent text `switch` file.
A compiled ST module does not itself enable sources in these no-source
propagation experiments.

Shoreline faces use the incident sea-cell speed, absorb outgoing energy and
supply zero land inflow. Only faces directly touching land or prescribed
cells use first-order upwind reconstruction; interior sea faces retain
QUICKEST/TVD. tp1.4's last excluded x column is restored as a positive-depth,
zero-spectrum outgoing ghost using the adjacent physical depth.
`Environment.boundary_mask` prescribes this ghost while transport carries
flux out. Physical wet-cell depths and scored cells retain their input values.

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
| tp1.2 | 144 | 0.0434 | 0.00908 | 0.196 | 5.92e-5 |
| tp1.4 | 144 | 0.00243 | 0.00215 | 0.00896 | 0.0620 |
| tp2.1 | 50 | 4.63e-6 | 9.66e-6 | 0.00103 | 2.84e-6 |

tp1.1's periodic-domain spectrum integral changes by about 1e-7 relative to
its initial value. The other cases have open or prescribed boundaries.
tp1.1's small station error covers only the first nine hours, not its full
144-hour field trajectory. tp1.4 retains a station spectral difference of
about 6.20%. tp2.1's full-field comparison covers 929 free sea cells at all
51 saved times; its six spectral stations supply the station comparison.
