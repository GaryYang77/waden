# -*- coding: utf-8 -*-
"""Grid and spectral-coordinate construction for the waden v2 core.

The formulas mirror the v1 ``ModelArchitecture.set_wave_environment`` path
without importing ``neuralwave_jax``.  The public input ``theta_deg`` is a
compass direction in degrees; the spectral grid stores radians.
"""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

from .typing import Grid, SpectralGrid, StepConfig


def _orientation(values) -> str:
    values = np.asarray(values)
    return "increasing" if values.size < 2 or values[1] > values[0] else "decreasing"


def _round5(value):
    return jnp.round(value * 1e5) * 1e-5


def _spacing(coords):
    padded = jnp.concatenate((coords[-2:-1], coords, coords[0:1]))
    spacing = jnp.abs(0.5 * (jnp.diff(padded[1:]) + jnp.diff(padded[:-1])))
    return spacing.at[0].set(spacing[1]).at[-1].set(spacing[-2])


def frequency_cell_widths(freq: jax.Array) -> jax.Array:
    """Positive widths in Hz for midpoint cells, extrapolated half a bin at ends."""
    freq = jnp.ravel(freq)
    if freq.size < 2:
        raise ValueError("frequency cell widths require at least two bin centers")
    edges = jnp.concatenate((freq[:1] - 0.5 * (freq[1:2] - freq[:1]),
                             0.5 * (freq[1:] + freq[:-1]),
                             freq[-1:] + 0.5 * (freq[-1:] - freq[-2:-1])))
    return jnp.diff(edges)


def cell_area(metadata: dict) -> jax.Array:
    """Return finite-volume cell area in m², shape (1,Nlon,Nlat,1,1)."""
    area = jnp.asarray(metadata["Dlon"]) * jnp.asarray(metadata["Dlat"])
    if metadata.get("coordinate_type", "Cartesian") == "Spherical":
        area = area * metadata["r_earth"]**2 * jnp.abs(jnp.asarray(metadata["cosLAT"]))
    return area


def prepare_environment(
    lon,
    lat,
    theta_deg,
    freq,
    *,
    coordinate_type: str = "Spherical",
    r_earth: float = 6366197.7236758135,
    lon_orientation: str | None = None,
    lat_orientation: str | None = None,
    dtype=jnp.float32,
) -> tuple[Grid, SpectralGrid]:
    """Build the v2 grid and spectral grid from production coordinates.

    ``lon``/``lat`` are degrees, ``theta_deg`` is the going-to compass direction
    in degrees, and ``freq`` is in Hz. For spherical grids the stored
    ``lon``/``lat`` are radians. The default remains float32; requesting
    float64 requires enabling JAX x64 before construction. Geometry is still
    rounded to the documented five decimals for the trigonometric fields.
    """
    lon_deg = jnp.asarray(lon, dtype=dtype)
    lat_deg = jnp.asarray(lat, dtype=dtype)
    theta_deg = jnp.asarray(theta_deg, dtype=dtype)
    freq = jnp.asarray(freq, dtype=dtype)

    if coordinate_type == "Spherical":
        lon_grid = lon_deg / 180.0 * jnp.pi
        lat_grid = lat_deg / 180.0 * jnp.pi
    else:
        lon_grid = lon_deg
        lat_grid = lat_deg

    ntheta = int(theta_deg.shape[0])
    dtheta = jnp.asarray(2 * jnp.pi / ntheta, dtype=dtype)
    feature_freq = jnp.diff(jnp.pad(freq, (0, 1), constant_values=0.0).at[-1].set(1.0))
    quadrature_df = frequency_cell_widths(freq)

    lon_orientation = lon_orientation or _orientation(lon_deg)
    lat_orientation = lat_orientation or _orientation(lat_deg)
    host_dtype = np.dtype(theta_deg.dtype)
    theta_data = jnp.asarray(np.deg2rad(np.asarray(theta_deg, dtype=host_dtype)), dtype=dtype)
    if lon_orientation == "increasing" and lat_orientation == "decreasing":
        theta_flux = jnp.asarray(
            ((180.0 - np.asarray(theta_deg, dtype=host_dtype)) % 360.0) * np.pi / 180.0,
            dtype=dtype,
        )
    else:
        sx = 1.0 if lon_orientation == "increasing" else -1.0
        sy = 1.0 if lat_orientation == "increasing" else -1.0
        theta_flux = jnp.mod(
            jnp.arctan2(sx * jnp.sin(theta_data), sy * jnp.cos(theta_data)),
            2.0 * jnp.pi,
        )

    grid = Grid(
        lon=lon_grid,
        lat=lat_grid,
        dlon=_spacing(lon_grid),
        dlat=_spacing(lat_grid),
        cos_lat=_round5(jnp.cos(lat_grid)),
        r_earth=float(r_earth),
        coordinate_type=coordinate_type,
        lon_orientation=lon_orientation,
        lat_orientation=lat_orientation,
    )
    spectral_grid = SpectralGrid(
        theta_data=theta_data,
        theta_flux=theta_flux,
        sin_theta=_round5(jnp.sin(theta_flux)),
        cos_theta=_round5(jnp.cos(theta_flux)),
        freq=freq,
        dtheta=dtheta,
        quadrature_df=quadrature_df,
        feature_df=feature_freq,
        theta_periodic=True,
    )
    return grid, spectral_grid


def build_legacy_metadata(
    grid: Grid,
    spectral_grid: SpectralGrid,
    config: StepConfig,
    wavenum=None,
    Cg=None,
) -> dict:
    """Restore the v1 metadata dictionary used by ``model_advance`` and the NN.

    The returned shapes and dtypes match v1 ``metadata`` exactly for the
    official demo path.  ``wavenum``/``Cg`` are optional so ``step`` can build
    the base metadata before solving them from depth.
    """
    lon = grid.lon.reshape(1, -1, 1, 1, 1)
    lat = grid.lat.reshape(1, 1, -1, 1, 1)
    theta = spectral_grid.theta_flux.reshape(1, 1, 1, -1, 1)
    freq = spectral_grid.freq.reshape(1, 1, 1, 1, -1)
    metadata = {
        "gravity": config.gravity,
        "coordinate_type": grid.coordinate_type,
        "r_earth": grid.r_earth,
        "lon": lon,
        "lat": lat,
        "theta": theta,
        "freq": freq,
        "Ntheta": int(theta.shape[-2]),
        "Nk": int(freq.shape[-1]),
        "DTH": spectral_grid.dtheta,
        "DFreq": spectral_grid.feature_df,
        "quadrature_df": spectral_grid.quadrature_df,
        "Dlon": grid.dlon.reshape(1, -1, 1, 1, 1),
        "Dlat": grid.dlat.reshape(1, 1, -1, 1, 1),
        "sinTheta": spectral_grid.sin_theta.reshape(1, 1, 1, -1, 1),
        "cosTheta": spectral_grid.cos_theta.reshape(1, 1, 1, -1, 1),
        "cosLAT": grid.cos_lat.reshape(1, 1, -1, 1, 1),
        "dt_global": config.dt_global,
        "dt_x": config.dt_x,
        "dt_y": config.dt_y,
        "dt_th": config.dt_th,
        "dt_k": config.dt_k,
        "dt_stot": config.dt_stot,
        "dt_gse": config.dt_gse,
        "yfirst": config.yfirst,
        "FACHFA": config.FACHFA,
        "FACVX": config.FACVX,
        "FACVY": config.FACVY,
        "FACTH": config.FACTH,
        "FACK": config.FACK,
        "FDG": config.FDG,
        "FULCR": config.FULCR,
        "khf": config.khf,
        "m": config.m,
        "latitude_sign": 1 if grid.lat_orientation == "increasing" else -1,
        # Flux divergence uses a positive bin spacing, while theta_flux may
        # decrease along that axis (official grid: pi - theta_data).
        "theta_axis_sign": jnp.sign(jnp.sin(spectral_grid.theta_flux[1] - spectral_grid.theta_flux[0])),
    }
    if wavenum is not None:
        metadata["wavenum"] = wavenum
    if Cg is not None:
        metadata["Cg"] = Cg
    return metadata


def from_legacy_metadata(metadata: dict) -> tuple[Grid, SpectralGrid]:
    """Map a v1 metadata dictionary back to the v2 structures (test bridge)."""
    lon = jnp.ravel(jnp.asarray(metadata["lon"]))
    lat = jnp.ravel(jnp.asarray(metadata["lat"]))
    theta_flux = jnp.ravel(jnp.asarray(metadata["theta"]))
    theta_deg = (180.0 - jnp.rad2deg(theta_flux)) % 360.0
    freq = jnp.ravel(jnp.asarray(metadata["freq"]))
    grid = Grid(
        lon=lon,
        lat=lat,
        dlon=jnp.ravel(jnp.asarray(metadata["Dlon"])),
        dlat=jnp.ravel(jnp.asarray(metadata["Dlat"])),
        cos_lat=jnp.ravel(jnp.asarray(metadata["cosLAT"])),
        r_earth=float(metadata["r_earth"]),
        coordinate_type=metadata["coordinate_type"],
        lon_orientation=_orientation(lon),
        lat_orientation=_orientation(lat),
    )
    spectral_grid = SpectralGrid(
        theta_data=theta_deg / 180.0 * jnp.pi,
        theta_flux=theta_flux,
        sin_theta=jnp.ravel(jnp.asarray(metadata["sinTheta"])),
        cos_theta=jnp.ravel(jnp.asarray(metadata["cosTheta"])),
        freq=freq,
        dtheta=jnp.asarray(metadata["DTH"]),
        quadrature_df=frequency_cell_widths(freq),
        feature_df=jnp.asarray(metadata["DFreq"]),
        theta_periodic=True,
    )
    return grid, spectral_grid
