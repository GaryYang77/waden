# -*- coding: utf-8 -*-
"""JAX array types and public state, environment and configuration structures.

The dataclasses are registered as JAX pytrees with dynamic array leaves
and static metadata.
"""

from __future__ import annotations

import dataclasses
import numbers
from typing import Callable, Dict, Tuple, Union

import jax


# =============================================================================
# Core Spectral Field Types
# =============================================================================

# Primary spectral field: [B, Nlon, Nlat, D, F]
#   B = batch, Nlon = longitude, Nlat = latitude, D = directions, F = frequencies
FieldSpectra = jax.Array

# Station/masked spectra: [B, N, D, F] where N = masked(Nlon*Nlat)
StationSpectra = jax.Array

# Group velocity: same shape as FieldSpectra [B, Nlon, Nlat, D, F]
GroupVelocity = jax.Array

# Valid mask: [B, Nlon, Nlat], includes boundary and land/sea mask
ValidMask = jax.Array

# Sea mask: [B, Nlon, Nlat], 1 for sea, 0 for land
SeaMask = jax.Array

# Boundary condition spectra: [B, Nlon, Nlat, D, F]
BoundaryCondition = jax.Array

# Source/sink terms: [B, Nlon, Nlat, D, F]
StotSpectra = jax.Array


# =============================================================================
# Coordinate Types
# =============================================================================

SpatialCoords = Tuple[jax.Array, jax.Array]   # (lon, lat)
SpectraCoords = Tuple[jax.Array, jax.Array]   # (freqs, wavenumbers)

Lon = jax.Array           # Longitude array
Lat = jax.Array           # Latitude array
Theta = jax.Array         # Direction array (radians)
Freqs = jax.Array         # Frequency array
Wavenumbers = jax.Array   # Wavenumber array (shape varies)
Depth = jax.Array         # Water depth


# =============================================================================
# Public state and geometry
# =============================================================================

@dataclasses.dataclass(frozen=True)
class Grid:
    """Spatial grid; coordinate arrays are dynamic leaves, geometry settings static."""

    lon: jax.Array
    lat: jax.Array
    dlon: jax.Array
    dlat: jax.Array
    cos_lat: jax.Array
    r_earth: float = 6366197.7236758135
    coordinate_type: str = "Spherical"
    lon_orientation: str = "increasing"
    lat_orientation: str = "decreasing"


jax.tree_util.register_dataclass(
    Grid,
    data_fields=("lon", "lat", "dlon", "dlat", "cos_lat"),
    meta_fields=("r_earth", "coordinate_type", "lon_orientation", "lat_orientation"),
)


@dataclasses.dataclass(frozen=True)
class SpectralGrid:
    """Spectral coordinates in radians/Hz; ``dtheta`` stays a traced scalar."""

    theta_data: jax.Array
    theta_flux: jax.Array
    sin_theta: jax.Array
    cos_theta: jax.Array
    freq: jax.Array
    dtheta: jax.Array
    quadrature_df: jax.Array
    feature_df: jax.Array
    theta_periodic: bool = True


jax.tree_util.register_dataclass(
    SpectralGrid,
    data_fields=(
        "theta_data", "theta_flux", "sin_theta", "cos_theta",
        "freq", "dtheta", "quadrature_df", "feature_df",
    ),
    meta_fields=("theta_periodic",),
)


@dataclasses.dataclass(frozen=True)
class State:
    """Energy density (B,Nlon,Nlat,D,F) plus parity of spatial update events."""

    spectra: jax.Array
    spatial_phase: jax.Array | int = 0


jax.tree_util.register_dataclass(State, data_fields=("spectra", "spatial_phase"), meta_fields=())


@dataclasses.dataclass(frozen=True)
class Environment:
    """Per-step forcing arrays.

    ``mask`` is the effective model mask (boundary-hit cells are False), not the
    evaluation wet mask. ``source`` is optional; the official path computes the
    parameterization source online and leaves it as ``None``.
    Sources have SI dE/dt units. boundary_mask selects prescribed spatial
    cells independently of the boundary spectrum's value, including zero.
    """

    depth: jax.Array
    mask: jax.Array
    boundary: jax.Array
    wind_u: jax.Array
    wind_v: jax.Array
    source: jax.Array | None = None
    boundary_mask: jax.Array | None = None

    def to_legacy_dict(self) -> Dict[str, jax.Array]:
        """Reshape environmental fields for dictionary-based model callbacks."""
        depth = self.depth[..., 0, 0]
        return {
            "depth": depth,
            "bathymetry": depth,
            "wind_u": self.wind_u,
            "wind_v": self.wind_v,
        }


jax.tree_util.register_dataclass(
    Environment,
    data_fields=("depth", "mask", "boundary", "wind_u", "wind_v", "source", "boundary_mask"),
    meta_fields=(),
)


@dataclasses.dataclass(frozen=True)
class StepConfig:
    """Static integration configuration; all fields are hashable metadata."""

    dt_global: int = 3600
    dt_x: int = 3600
    dt_y: int = 3600
    dt_th: int = 3600
    dt_k: int = 3600
    dt_stot: int = 3600
    dt_gse: int = 0
    yfirst: bool = True
    FACHFA: float = 1.0
    FACVX: float = 1.0
    FACVY: float = 1.0
    FACTH: float = 1.0
    FACK: float = 1.0
    FDG: float = 1.0
    FULCR: bool = False
    khf: float = 10.0
    m: float = 5.0
    gravity: float = 9.806
    substeps: int = 1
    tail_start_index: int | None = None
    checkpoint_level: str = "none"

    def __post_init__(self):
        """Reject schedules that silently lose time or omit enabled operators."""
        names = ("dt_global", "dt_x", "dt_y", "dt_th", "dt_k", "dt_stot", "dt_gse")
        for name in names:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < 0:
                raise ValueError(f"{name} must be an integer number of seconds, with 0 disabling a component")
        if self.dt_global <= 0:
            raise ValueError("dt_global must be positive")
        if not isinstance(self.substeps, numbers.Integral) or self.substeps <= 0:
            raise ValueError("substeps must be a positive integer")
        for name in names:
            value = getattr(self, name)
            if value and (self.dt_global % value or value % self.substeps):
                raise ValueError(f"{name} must divide dt_global and be divisible by substeps")


jax.tree_util.register_dataclass(
    StepConfig,
    data_fields=(),
    meta_fields=(
        "dt_global", "dt_x", "dt_y", "dt_th", "dt_k", "dt_stot", "dt_gse",
        "yfirst", "FACHFA", "FACVX", "FACVY", "FACTH", "FACK", "FDG",
        "FULCR", "khf", "m", "gravity", "substeps", "tail_start_index", "checkpoint_level",
    ),
)


@dataclasses.dataclass(frozen=True)
class RolloutResult:
    """Output of a multi-step rollout; ``times`` stays outside the scanned pytree."""

    spectra: jax.Array
    final_state: State
    times: list
    diagnostics: dict | None = None


# =============================================================================
# Dictionary Types
# =============================================================================

# Environment data: wind (unnormed, std, mean, ...), bathymetry, time, etc.
EnvironmentDataDict = Dict[str, jax.Array]

# Metadata: baseinfo from gin configs (gravity, dt_global, wavenum, Cg, etc.)
MetadataDict = Dict[str, Union[jax.Array, float, int, str, bool]]

LossDict = Dict[str, jax.Array]
MetricsDict = Dict[str, float]


# =============================================================================
# Functional Types
# =============================================================================

FluxCalculator = Callable[..., Tuple[jax.Array, jax.Array]]
BoundaryUpdater = Callable[[jax.Array], jax.Array]
TendencyCalculator = Callable[..., jax.Array]
