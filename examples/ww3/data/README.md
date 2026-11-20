# WW3 reference data

This directory supplies the WW3 6.07 reference outputs and experiment inputs
used by the four examples in this project. No additional download is needed.

The reference outputs correspond to WAVEWATCH III regression cases tp1.1,
tp1.2, tp1.4 and tp2.1. The complete inputs for regenerating them with WW3
6.07.1 are included; see the [example guide](../README.md) and
[NOAA-EMC/WW3](https://github.com/NOAA-EMC/WW3/tree/develop/regtests).

Each `ww3_tp*/input/` contains the grid and station-spectrum NetCDF files
selected by `../configs/`, together with the archived grid, initial-state,
time-stepping, depth, station-list, switch and NetCDF export inputs.
The runnable text inputs include the hourly/station output settings for
tp1.2, the station-list terminator for tp2.1, and export settings matching
the supplied filenames and station grouping. Each `switch` specifies the
supported serial WW3 6.07.1 build. The NetCDF files retain
their original metadata. The experiment switches and output conventions are
described in the [example guide](../README.md).

These are prescribed synthetic propagation/refraction experiments. Only the
initial fields and declared boundary conditions initialize WADEN; subsequent
WW3 fields serve as the comparison reference.
