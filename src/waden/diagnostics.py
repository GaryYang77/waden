# -*- coding: utf-8 -*-
"""Physical finite-volume measures and CFL preflight for the waden core."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .geometry import build_legacy_metadata, cell_area
from .physics import compute_group_velocity, solve_wavenumber
from .typing import Grid, SpectralGrid, StepConfig


def spectral_integral(spectra: jax.Array, grid: Grid, spectral: SpectralGrid, mask: jax.Array) -> jax.Array:
    """Integrate E(f,theta) over physical area, frequency and direction, per batch."""
    metadata = build_legacy_metadata(grid, spectral, StepConfig())
    weights = cell_area(metadata) * spectral.quadrature_df * spectral.dtheta
    return jnp.sum(jnp.where(mask[..., None, None], spectra, 0) * weights, axis=(1, 2, 3, 4))


def physical_cfl(depth: jax.Array, grid: Grid, spectral: SpectralGrid, config: StepConfig) -> dict:
    """Return worst physical x/y/theta CFL with the chosen static subdivisions.

    Frequency shifting is zero in stationary water without currents. Learned
    theta/frequency closures have their own dimensional contract, so this
    diagnostic does not certify neural closures or enable adaptive stepping.
    """
    from .integrator import _compute_depth_gradients
    from .tendencies import theta_transport_velocity

    metadata = build_legacy_metadata(grid, spectral, config)
    k = solve_wavenumber(metadata["freq"], depth, metadata=metadata)
    cg = compute_group_velocity(metadata["freq"], k, depth)
    metadata.update(wavenum=k, Cg=cg)
    dx, dy = metadata["Dlon"], metadata["Dlat"]
    if grid.coordinate_type == "Spherical":
        dx = dx * grid.r_earth * metadata["cosLAT"]
        dy = dy * grid.r_earth
    gx, gy = _compute_depth_gradients(depth, metadata)
    velocity = theta_transport_velocity(k, metadata, depth, gx, gy)
    return {
        "x": jnp.max(jnp.abs(config.FACVX * cg * metadata["sinTheta"] / dx)) * config.dt_x / config.substeps,
        "y": jnp.max(jnp.abs(config.FACVY * cg * metadata["cosTheta"] / dy)) * config.dt_y / config.substeps,
        "theta": jnp.max(jnp.abs(velocity / spectral.dtheta)) * config.dt_th / config.substeps,
        "frequency": jnp.asarray(0, dtype=depth.dtype),
    }
