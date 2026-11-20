# WW3 reference data

This directory supplies the WW3 6.07 reference outputs and experiment inputs
used by the four examples in this project. No additional download is needed.

The files are an unchanged subset of the original PyTorch WADEN data archive
at revision `b80fdce679414345134a25275421db439c185349`. This revision identifies
their provenance; reproducing these examples does not require checking it out.
The source experiments are WAVEWATCH III regression cases tp1.1, tp1.2,
tp1.4 and tp2.1; see [NOAA-EMC/WW3](https://github.com/NOAA-EMC/WW3/tree/develop/regtests).

Each `ww3_tp*/input/` contains the grid and station-spectrum NetCDF files
selected by `../configs/`, together with the archived grid, initial-state,
time-stepping, depth, station-list and switch inputs. The NetCDF files retain
their original metadata. The experiment switches and output conventions are
described in the [example guide](../README.md).

These are prescribed synthetic propagation/refraction experiments. Only the
initial fields and declared boundary conditions initialize WADEN; subsequent
WW3 fields serve as the comparison reference.
