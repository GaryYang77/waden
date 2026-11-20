# -*- coding: utf-8 -*-
"""Pure JAX operator-splitting integrators for NeuralWave.

Supports configurable gradient rematerialization (``jax.checkpoint``).
Its memory and runtime tradeoffs depend on the workload.
"""

import dataclasses
import math
from functools import partial
from typing import Callable, Optional

import jax
import jax.numpy as jnp

from .boundary import apply_boundary_condition, update_absorb_boundary
from .geometry import build_legacy_metadata, cell_area, frequency_cell_widths
from .physics import compute_group_velocity, solve_wavenumber
from .tendencies import k_tendency, theta_tendency, x_tendency, y_tendency
from .typing import Environment, Grid, RolloutResult, SpectralGrid, State, StepConfig


TendencyFn = Callable[..., jax.Array]
BoundaryFn = Callable[[jax.Array], jax.Array]
ApplyBoundaryFn = Callable[[jax.Array, jax.Array, jax.Array], jax.Array]
StotAdvanceFn = Callable[..., jax.Array]
GSEAlleviationFn = Callable[..., jax.Array]

CHECKPOINT_NONE = "none"
"""No rematerialization."""

CHECKPOINT_PER_STEP = "per_step"
"""Rematerialize each full integrator step."""

CHECKPOINT_PER_PHASE = "per_phase"
"""Rematerialize each physics phase."""

_VALID_CHECKPOINT_LEVELS = frozenset(
    {CHECKPOINT_NONE, CHECKPOINT_PER_STEP, CHECKPOINT_PER_PHASE}
)


def _gcd(a: int, b: int) -> int:
    return math.gcd(a, b)


def _as_grid_field(value, dtype) -> jax.Array:
    array = jnp.asarray(value, dtype=dtype)
    while array.ndim < 5:
        array = jnp.expand_dims(array, axis=-1)
    if array.shape[-2] != 1 or array.shape[-1] != 1:
        array = array[..., :1, :1]
    return array


def _compute_depth_gradients(depth: jax.Array, metadata: dict) -> tuple[jax.Array, jax.Array]:
    depth_3d = jnp.squeeze(jnp.squeeze(depth, axis=-1), axis=-1)

    depthx = jnp.pad(depth_3d, ((0, 0), (1, 1), (0, 0)), mode="wrap")
    depthy = jnp.pad(depth_3d, ((0, 0), (0, 0), (1, 1)), mode="edge")

    Dlon = _as_grid_field(metadata["Dlon"], depth.dtype)
    Dlat = _as_grid_field(metadata["Dlat"], depth.dtype)

    DdDx = 0.5 * ((depthx[:, 2:, :] - depthx[:, 1:-1, :]) + (depthx[:, 1:-1, :] - depthx[:, :-2, :]))
    DdDy = 0.5 * ((depthy[:, :, 2:] - depthy[:, :, 1:-1]) + (depthy[:, :, 1:-1] - depthy[:, :, :-2]))

    DdDx = jnp.expand_dims(jnp.expand_dims(DdDx, axis=-1), axis=-1) / Dlon
    DdDy = jnp.expand_dims(jnp.expand_dims(DdDy, axis=-1), axis=-1) / Dlat

    if metadata["coordinate_type"] == "Spherical":
        cosLAT = _as_grid_field(metadata["cosLAT"], depth.dtype)
        r_earth = jnp.asarray(metadata["r_earth"], dtype=depth.dtype)
        DdDx = DdDx / (r_earth * cosLAT)
        DdDy = DdDy / r_earth

    land_mask = depth > 0
    return DdDx * land_mask, DdDy * land_mask


def _sanitize_physics_fields(metadata: dict, land_mask: jax.Array) -> dict:
    expanded_land_mask = jnp.broadcast_to(land_mask, jnp.asarray(metadata["wavenum"]).shape)
    wavenum = jnp.where(expanded_land_mask, jnp.asarray(metadata["wavenum"]), 0.0)
    Cg = jnp.where(expanded_land_mask, jnp.asarray(metadata["Cg"]), 0.0)
    return {
        **metadata,
        "wavenum": wavenum,
        "Cg": Cg,
    }


def _physical_wet_mask(depth, mask, boundary, boundary_mask):
    """Include prescribed wet cells excluded from the free-evolution mask."""
    if boundary_mask is None:
        boundary_mask = jnp.sum(boundary, axis=(-2, -1)) > 0
    return (depth > 0) & (mask | boundary_mask)[..., None, None]


def _compute_should_mask(num_steps: int, dt_min: int, dt: float, dtype) -> jax.Array:
    if dt <= 0:
        return jnp.zeros((num_steps,), dtype=bool)
    step_indices = jnp.arange(1, num_steps + 1, dtype=dtype)
    return jnp.mod(step_indices * jnp.asarray(dt_min, dtype=dtype), jnp.asarray(dt, dtype=dtype)) < 1e-5


def _update_interior(spectra: jax.Array, update: jax.Array, metadata: dict) -> tuple:
    # Full physical-cell tendencies are native; retain interior callbacks.
    updated = spectra + update if update.shape == spectra.shape else spectra.at[:, 1:-1, 1:-1, :, :].add(update)
    prescribed = metadata.get("boundary_mask")
    if prescribed is not None:
        updated = jnp.where(prescribed[..., None, None], spectra, updated)
    return _project_nonnegative(updated), _clipping_mass(updated, metadata)


def _project_nonnegative(spectra: jax.Array) -> jax.Array:
    """Positivity projection, kept explicit for independent clipping-budget audits."""
    return jnp.maximum(spectra, 0)


def _clipping_mass(before: jax.Array, metadata: dict) -> jax.Array:
    """Measure actual pre-projection negative mass, never infer it from drift."""
    if not metadata.get("_audit_clipping", False):
        return jnp.zeros((before.shape[0],), dtype=before.dtype)
    dtype = jnp.float64 if jax.config.x64_enabled else before.dtype
    area = cell_area(metadata).astype(dtype)
    df = metadata.get("quadrature_df")
    df = frequency_cell_widths(metadata["freq"]) if df is None else df
    weights = area * jnp.asarray(df, dtype=dtype) * jnp.asarray(metadata["DTH"], dtype=dtype)
    negative = jnp.where(metadata["_wet_mask"], jnp.maximum(-before, 0), 0).astype(dtype)
    return jnp.sum(negative * weights, axis=(1,2,3,4))


def simple_stot_integrator(
    spectra: jax.Array,
    stot: jax.Array,
    dt: float,
    metadata: dict,
    mask: jax.Array,
) -> jax.Array:
    """Apply an SI source dE/dt; application adapters own NN calibration factors."""
    del metadata
    update_term = stot * jnp.asarray(dt, dtype=spectra.dtype)
    updated_spectra = spectra + update_term
    mask_expanded = jnp.expand_dims(jnp.expand_dims(mask.astype(bool), axis=-1), axis=-1)
    mask_expanded = jnp.broadcast_to(mask_expanded, spectra.shape)
    return jnp.where(mask_expanded, updated_spectra, spectra)


# ---------------------------------------------------------------------------
# Per-phase functions for operator-splitting integrator
# ---------------------------------------------------------------------------


def _phase_spectral_half_step(
    spectra: jax.Array,
    should_integrate_th: jax.Array,
    should_integrate_k: jax.Array,
    dt_th: float,
    dt_k: float,
    th_tendency_fn: TendencyFn,
    k_tendency_fn: TendencyFn,
    physics_metadata: dict,
    depth: jax.Array,
    DdDx: jax.Array,
    DdDy: jax.Array,
    boundary_condition: jax.Array,
    mask: jax.Array,
    apply_boundary_func: ApplyBoundaryFn,
) -> tuple[jax.Array, jax.Array]:
    """Apply spectral half-steps on all physical cells.

    Interior-shaped external callbacks remain supported. Explicit Euler
    component updates do not establish overall second-order accuracy.

    Parameters
    ----------
    spectra : jax.Array
        Wave energy density spectra ``E(f, theta)``, shape
        ``(B, Nlon, Nlat, D, F)``, in ``m^2 s rad^-1``.
    should_integrate_th : jax.Array
        Scalar boolean — whether to apply the theta half-step.
    should_integrate_k : jax.Array
        Scalar boolean — whether to apply the wavenumber half-step.
    dt_th : float
        Theta refraction time step [s].
    dt_k : float
        Wavenumber refraction time step [s].
    th_tendency_fn : TendencyFn
        Theta tendency calculator.
    k_tendency_fn : TendencyFn
        Wavenumber tendency calculator.
    physics_metadata : dict
        Sanitized physics metadata.
    depth : jax.Array
        Bathymetry, ``(B, H, W, 1, 1)``.
    DdDx, DdDy : jax.Array
        Depth gradients, ``(B, H, W, 1, 1)``.

    Returns
    -------
    jax.Array
        Updated spectra, shape ``(B, H, W, D, F)``.
    """
    if physics_metadata.get("boundary_mask") is not None:
        spectra = apply_boundary_func(spectra, boundary_condition, mask)
    th_tend = th_tendency_fn(
        spectra, dt=dt_th, metadata=physics_metadata,
        depth=depth, DdDx=DdDx, DdDy=DdDy,
    )
    update_th = jnp.where(
        should_integrate_th, 0.5 * dt_th * th_tend, jnp.zeros_like(th_tend),
    )
    spectra, clip_th = _update_interior(spectra, update_th, physics_metadata)

    k_tend = k_tendency_fn(
        spectra, dt=dt_k, metadata=physics_metadata,
        depth=depth, DdDx=DdDx, DdDy=DdDy,
    )
    update_k = jnp.where(
        should_integrate_k, 0.5 * dt_k * k_tend, jnp.zeros_like(k_tend),
    )
    spectra, clip_k = _update_interior(spectra, update_k, physics_metadata)
    return spectra, clip_th + clip_k


def _phase_spatial_single_direction(
    spectra: jax.Array,
    should_integrate_first: jax.Array,
    should_integrate_second: jax.Array,
    dt_first: float,
    dt_second: float,
    first_tendency_fn: TendencyFn,
    second_tendency_fn: TendencyFn,
    physics_metadata: dict,
    land_mask: jax.Array,
    boundary_condition: jax.Array,
    mask: jax.Array,
    boundary_update_func: BoundaryFn,
    apply_boundary_func: ApplyBoundaryFn,
) -> tuple[jax.Array, jax.Array]:
    """Execute one spatial propagation path (first-then-second direction).

    Parameters
    ----------
    spectra : jax.Array
        Spectra, ``(B, H, W, D, F)``.
    should_integrate_first, should_integrate_second : jax.Array
        Scalar booleans controlling whether each direction is active.
    dt_first, dt_second : float
        Time step sizes [s] for each direction.
    first_tendency_fn, second_tendency_fn : TendencyFn
        Tendency calculators for first and second directions.
    physics_metadata : dict
        Sanitized physics metadata.
    land_mask : jax.Array
        Ocean mask, ``(B, H, W, 1, 1)``; True where sea.
    boundary_condition : jax.Array
        External boundary spectra, ``(B, H, W, D, F)``.
    mask : jax.Array
        Sea/land mask, ``(B, H, W)``.
    boundary_update_func : BoundaryFn
        Boundary refresh (e.g. absorb zeros at edges).
    apply_boundary_func : ApplyBoundaryFn
        Boundary condition injection.

    Returns
    -------
    jax.Array
        Spectra after two-direction propagation, ``(B, H, W, D, F)``.
    """
    spectra = apply_boundary_func(spectra, boundary_condition, mask)
    tend_1 = first_tendency_fn(
        spectra, dt=dt_first, metadata=physics_metadata, land_mask=land_mask,
    )
    update_1 = jnp.where(
        should_integrate_first, dt_first * tend_1, jnp.zeros_like(tend_1),
    )
    clip_first = _clipping_mass(spectra + update_1, physics_metadata)
    spectra = boundary_update_func(_project_nonnegative(spectra + update_1))

    spectra = apply_boundary_func(spectra, boundary_condition, mask)
    tend_2 = second_tendency_fn(
        spectra, dt=dt_second, metadata=physics_metadata, land_mask=land_mask,
    )
    update_2 = jnp.where(
        should_integrate_second, dt_second * tend_2, jnp.zeros_like(tend_2),
    )
    clip_second = _clipping_mass(spectra + update_2, physics_metadata)
    return boundary_update_func(_project_nonnegative(spectra + update_2)), clip_first + clip_second


def _phase_spatial_propagation(
    spectra: jax.Array,
    yfirst_current: jax.Array,
    should_integrate_x: jax.Array,
    should_integrate_y: jax.Array,
    dt_x: float,
    dt_y: float,
    x_tendency_fn: TendencyFn,
    y_tendency_fn: TendencyFn,
    physics_metadata: dict,
    land_mask: jax.Array,
    boundary_condition: jax.Array,
    mask: jax.Array,
    boundary_update_func: BoundaryFn,
    apply_boundary_func: ApplyBoundaryFn,
) -> tuple[jax.Array, jax.Array]:
    """Dual-path spatial propagation with alternating Y/X-first order.

    Computes both Y-then-X and X-then-Y orderings and selects via
    ``jnp.where`` to eliminate data-dependent branching (XLA-friendly).
    Alternating the propagation order across steps removes the operator-
    splitting direction bias (WW3 Manual v6.07, §3.3.1).

    Parameters
    ----------
    spectra : jax.Array
        Spectra, ``(B, H, W, D, F)``.
    yfirst_current : jax.Array
        Scalar boolean — True if Y should be propagated before X.
    should_integrate_x, should_integrate_y : jax.Array
        Scalar booleans.
    dt_x, dt_y : float
        Time step sizes [s] for X and Y propagation.
    x_tendency_fn, y_tendency_fn : TendencyFn
        Tendency calculators.
    physics_metadata : dict
        Sanitized physics metadata.
    land_mask : jax.Array
        Ocean mask, ``(B, H, W, 1, 1)``.
    boundary_condition : jax.Array
        External boundary spectra, ``(B, H, W, D, F)``.
    mask : jax.Array
        Sea/land mask, ``(B, H, W)``.
    boundary_update_func : BoundaryFn
        Boundary refresh.
    apply_boundary_func : ApplyBoundaryFn
        Boundary condition injection.

    Returns
    -------
    jax.Array
        Spectra after spatial propagation, ``(B, H, W, D, F)``.
    """
    common = dict(
        physics_metadata=physics_metadata,
        land_mask=land_mask,
        boundary_condition=boundary_condition,
        mask=mask,
        boundary_update_func=boundary_update_func,
        apply_boundary_func=apply_boundary_func,
    )
    spectra_y_first = _phase_spatial_single_direction(
        spectra,
        should_integrate_first=should_integrate_y,
        should_integrate_second=should_integrate_x,
        dt_first=dt_y, dt_second=dt_x,
        first_tendency_fn=y_tendency_fn,
        second_tendency_fn=x_tendency_fn,
        **common,
    )
    spectra_x_first = _phase_spatial_single_direction(
        spectra,
        should_integrate_first=should_integrate_x,
        should_integrate_second=should_integrate_y,
        dt_first=dt_x, dt_second=dt_y,
        first_tendency_fn=x_tendency_fn,
        second_tendency_fn=y_tendency_fn,
        **common,
    )
    return jax.tree_util.tree_map(lambda y, x: jnp.where(yfirst_current, y, x), spectra_y_first, spectra_x_first)


def _phase_source_integration(
    spectra: jax.Array,
    should_integrate_stot: jax.Array,
    stot: jax.Array,
    dt_stot: float,
    physics_metadata: dict,
    mask: jax.Array,
    boundary_condition: jax.Array,
    stot_advance: StotAdvanceFn,
    boundary_update_func: BoundaryFn,
    apply_boundary_func: ApplyBoundaryFn,
) -> tuple[jax.Array, jax.Array]:
    """Conditionally integrate source/sink terms.

    Parameters
    ----------
    spectra : jax.Array
        Spectra, ``(B, H, W, D, F)``.
    should_integrate_stot : jax.Array
        Scalar boolean — whether to apply source terms this sub-step.
    stot : jax.Array
        Source term, same shape as *spectra*.
    dt_stot : float
        Source integration time step [s].
    physics_metadata : dict
        Sanitized physics metadata.
    mask : jax.Array
        Sea/land mask, ``(B, H, W)``.
    boundary_condition : jax.Array
        External boundary spectra.
    stot_advance : StotAdvanceFn
        Source-term integrator function.
    boundary_update_func : BoundaryFn
        Boundary refresh.
    apply_boundary_func : ApplyBoundaryFn
        Boundary condition injection.

    Returns
    -------
    jax.Array
        Updated spectra, ``(B, H, W, D, F)``.
    """
    def _integrate(s: jax.Array) -> jax.Array:
        s = boundary_update_func(s)
        s = apply_boundary_func(s, boundary_condition, mask)
        s = stot_advance(s, stot, dt=dt_stot, metadata=physics_metadata, mask=mask)
        return _project_nonnegative(s), _clipping_mass(s, physics_metadata)

    return jax.lax.cond(should_integrate_stot, _integrate,
                        lambda s: (s, _clipping_mass(jnp.zeros_like(s), physics_metadata)), spectra)


def _phase_gse_averaging(
    spectra: jax.Array,
    should_integrate_gse: jax.Array,
    dt_gse: float,
    physics_metadata: dict,
    mask: jax.Array,
    boundary_condition: jax.Array,
    gse_alleviation: GSEAlleviationFn,
    boundary_update_func: BoundaryFn,
    apply_boundary_func: ApplyBoundaryFn,
) -> tuple[jax.Array, jax.Array]:
    """Conditionally apply conservative GSE spatial averaging."""

    def _integrate(s: jax.Array) -> jax.Array:
        s = gse_alleviation(s, metadata=physics_metadata, mask=mask, dt=dt_gse)
        correction = _clipping_mass(s, physics_metadata)
        s = boundary_update_func(_project_nonnegative(s))
        return apply_boundary_func(s, boundary_condition, mask), correction

    return jax.lax.cond(should_integrate_gse, _integrate,
                        lambda s: (s, _clipping_mass(jnp.zeros_like(s), physics_metadata)), spectra)


# ---------------------------------------------------------------------------
# Integration schedule
# ---------------------------------------------------------------------------


def _prepare_integration_schedule(
    dt_global: int,
    dt_x: int,
    dt_y: int,
    dt_th: int,
    dt_k: int,
    dt_stot: int,
    dtype,
    yfirst_initial: bool,
    dt_gse: int = 0,
    spatial_phase=0,
) -> dict:
    """Compute the sub-step schedule and per-component execution masks.

    Determines the minimum common time step (GCD of all active component
    time steps), the total number of sub-steps, and boolean masks indicating
    which components should execute at each sub-step.

    Parameters
    ----------
    dt_global : int
        Global integration time step [s].
    dt_x, dt_y, dt_th, dt_k, dt_stot, dt_gse : int
        Component time steps [s]; 0 means the component is disabled.
    dtype
        Floating-point dtype for mask computation.
    yfirst_initial : bool
        Initial Y-first flag for alternating spatial order.

    Returns
    -------
    dict
        Keys: ``num_steps``, ``should_integrate_*`` boolean masks of shape
        ``(num_steps,)``, and ``yfirst_per_step`` boolean array.
    """
    component_dts = (dt_x, dt_y, dt_th, dt_k, dt_stot, dt_gse)
    if dt_global <= 0 or any(dt < 0 or int(dt) != dt or (dt and dt_global % dt) for dt in component_dts):
        raise ValueError("component integer time steps must divide the positive global interval")
    active_dts = [int(dt) for dt in component_dts if dt > 0]
    if not active_dts:
        return {"num_steps": 0}

    dt_min = active_dts[0]
    for dt_val in active_dts[1:]:
        dt_min = _gcd(dt_min, dt_val)
    num_steps = int(dt_global // dt_min)

    should_x = _compute_should_mask(num_steps, dt_min, dt_x, dtype)
    should_y = _compute_should_mask(num_steps, dt_min, dt_y, dtype)
    should_th = _compute_should_mask(num_steps, dt_min, dt_th, dtype)
    should_k = _compute_should_mask(num_steps, dt_min, dt_k, dtype)
    should_stot = _compute_should_mask(num_steps, dt_min, dt_stot, dtype)
    should_gse = _compute_should_mask(num_steps, dt_min, dt_gse, dtype)

    spatial_events = should_x | should_y
    prior_events = jnp.concatenate([jnp.zeros((1,), dtype=jnp.int32),
                                    jnp.cumsum(spatial_events[:-1].astype(jnp.int32))])
    yfirst_per_step = ((prior_events + spatial_phase) % 2 == 0) == yfirst_initial

    return {
        "num_steps": num_steps,
        "should_integrate_x": should_x,
        "should_integrate_y": should_y,
        "should_integrate_th": should_th,
        "should_integrate_k": should_k,
        "should_integrate_stot": should_stot,
        "should_integrate_gse": should_gse,
        "yfirst_per_step": yfirst_per_step,
    }


# ---------------------------------------------------------------------------
# Main integrator
# ---------------------------------------------------------------------------


def model_advance(
    spectra: jax.Array,
    stot: jax.Array,
    depth: jax.Array,
    metadata: dict,
    mask: jax.Array,
    boundary_condition: jax.Array,
    x_tendency_fn: TendencyFn = x_tendency,
    y_tendency_fn: TendencyFn = y_tendency,
    th_tendency_fn: TendencyFn = theta_tendency,
    k_tendency_fn: TendencyFn = k_tendency,
    stot_advance: StotAdvanceFn = simple_stot_integrator,
    boundary_update_func: BoundaryFn = update_absorb_boundary,
    apply_boundary_func: ApplyBoundaryFn = apply_boundary_condition,
    gse_alleviation: Optional[GSEAlleviationFn] = None,
    checkpoint_level: str = CHECKPOINT_NONE,
    audit_clipping: bool = False,
) -> jax.Array | tuple[jax.Array, jax.Array]:
    """Integrate wave spectra one global time step using operator splitting.

    Uses spectral half-steps around spatial propagation per sub-step::

        θ half → k half → spatial(X/Y) → optional GSE → θ half → k half → source

    The source and component updates are explicit Euler; this arrangement
    does not imply second-order accuracy. The supported overall order is
    established by refinement tests. Source terms are in SI dE/dt units.
    The total number of sub-steps is ``dt_global / gcd(active_dts)``.
    ``jax.lax.scan`` drives the loop; each sub-step (or each physics phase
    within it) can be wrapped with ``jax.checkpoint`` for gradient
    rematerialization, controlled by *checkpoint_level*.

    References: WW3 Manual v6.07, §3.3.

    Parameters
    ----------
    spectra : jax.Array
        Initial wave energy density spectra ``E(f, theta)``,
        ``(B, Nlon, Nlat, D, F)``, in ``m^2 s rad^-1``.
    stot : jax.Array
        Source/sink term, same shape as *spectra*.
    depth : jax.Array
        Bathymetry, ``(B, H, W, 1, 1)``; positive where sea.
    metadata : dict
        Physics metadata (grid spacings, wavenumber, Cg, coordinate type,
        time step configuration, etc.).
    mask : jax.Array
        Sea/land mask, ``(B, H, W)``; True where sea.
    boundary_condition : jax.Array
        External lateral boundary spectra, ``(B, H, W, D, F)``.
    x_tendency_fn : TendencyFn
        Longitude propagation tendency calculator.
    y_tendency_fn : TendencyFn
        Latitude propagation tendency calculator.
    th_tendency_fn : TendencyFn
        Direction refraction tendency calculator.
    k_tendency_fn : TendencyFn
        Wavenumber refraction tendency calculator.
    stot_advance : StotAdvanceFn
        Source-term integrator.
    boundary_update_func : BoundaryFn
        Boundary refresh callback (e.g. absorb zeros at edges).
    apply_boundary_func : ApplyBoundaryFn
        Boundary condition injection callback.
    gse_alleviation : GSEAlleviationFn, optional
        Optional conservative GSE spatial averaging operator.
    checkpoint_level : str
        Gradient rematerialization strategy.  One of ``"none"`` (default,
        no rematerialization), ``"per_step"`` (rematerialize each full
        integrator sub-step), or ``"per_phase"`` (rematerialize each
        physics phase — maximum memory savings, recommended for global
        training).

    Returns
    -------
    jax.Array
        Updated spectra after one global time step, ``(B, H, W, D, F)``.

    Raises
    ------
    ValueError
        If *checkpoint_level* is not a recognised value.
    """
    if checkpoint_level not in _VALID_CHECKPOINT_LEVELS:
        raise ValueError(
            f"checkpoint_level must be one of {sorted(_VALID_CHECKPOINT_LEVELS)}, "
            f"got {checkpoint_level!r}")

    dt_global = metadata.get("dt_global", 3600)
    dt_x = metadata.get("dt_x", 1800)
    dt_y = metadata.get("dt_y", 1800)
    dt_th = metadata.get("dt_th", 600)
    dt_k = metadata.get("dt_k", 600)
    dt_stot = metadata.get("dt_stot", 300)
    dt_gse = metadata.get("dt_gse", 0)
    yfirst_initial = metadata.get("yfirst", True)
    gse_enabled = (
        gse_alleviation is not None
        and getattr(gse_alleviation, "enabled", False)
        and dt_gse > 0
    )
    active_dt_gse = dt_gse if gse_enabled else 0

    if "_spatial_wet_mask" in metadata:
        land_mask = metadata["_spatial_wet_mask"]
    else:
        land_mask = _physical_wet_mask(depth, mask, boundary_condition, metadata.get("boundary_mask"))
    if "_depth_gradients" in metadata:
        DdDx, DdDy = metadata["_depth_gradients"]
    else:
        DdDx, DdDy = _compute_depth_gradients(depth, metadata)
    physics_metadata = _sanitize_physics_fields(metadata, land_mask)
    physics_metadata.update(_audit_clipping=audit_clipping, _wet_mask=land_mask)
    if metadata.get("boundary_mask") is not None:
        physics_metadata["_wet_mask"] = land_mask & ~metadata["boundary_mask"][..., None, None]
    zero_clipping = _clipping_mass(jnp.zeros_like(spectra), physics_metadata)

    schedule = _prepare_integration_schedule(
        dt_global,
        dt_x,
        dt_y,
        dt_th,
        dt_k,
        dt_stot,
        spectra.dtype,
        yfirst_initial,
        dt_gse=active_dt_gse,
        spatial_phase=metadata.get("spatial_phase", 0),
    )
    num_steps = schedule["num_steps"]
    if num_steps == 0:
        return (spectra, zero_clipping) if audit_clipping else spectra

    # --- Phase closures (capture pre-computed state, checkpoint-friendly) ---

    def _spectral_half(spectra_in: jax.Array, step_idx: jax.Array) -> tuple[jax.Array, jax.Array]:
        return _phase_spectral_half_step(
            spectra_in,
            should_integrate_th=schedule["should_integrate_th"][step_idx],
            should_integrate_k=schedule["should_integrate_k"][step_idx],
            dt_th=dt_th, dt_k=dt_k,
            th_tendency_fn=th_tendency_fn, k_tendency_fn=k_tendency_fn,
            physics_metadata=physics_metadata, depth=depth, DdDx=DdDx, DdDy=DdDy,
            boundary_condition=boundary_condition, mask=mask, apply_boundary_func=apply_boundary_func)

    def _spatial(spectra_in: jax.Array, step_idx: jax.Array) -> tuple[jax.Array, jax.Array]:
        return _phase_spatial_propagation(
            spectra_in,
            yfirst_current=schedule["yfirst_per_step"][step_idx],
            should_integrate_x=schedule["should_integrate_x"][step_idx],
            should_integrate_y=schedule["should_integrate_y"][step_idx],
            dt_x=dt_x, dt_y=dt_y,
            x_tendency_fn=x_tendency_fn, y_tendency_fn=y_tendency_fn,
            physics_metadata=physics_metadata, land_mask=land_mask,
            boundary_condition=boundary_condition, mask=mask,
            boundary_update_func=boundary_update_func,
            apply_boundary_func=apply_boundary_func)

    def _source(spectra_in: jax.Array, step_idx: jax.Array) -> tuple[jax.Array, jax.Array]:
        return _phase_source_integration(
            spectra_in,
            should_integrate_stot=schedule["should_integrate_stot"][step_idx],
            stot=stot, dt_stot=dt_stot, physics_metadata=physics_metadata,
            mask=mask, boundary_condition=boundary_condition,
            stot_advance=stot_advance, boundary_update_func=boundary_update_func,
            apply_boundary_func=apply_boundary_func)

    if gse_enabled:
        def _gse(spectra_in: jax.Array, step_idx: jax.Array) -> tuple[jax.Array, jax.Array]:
            return _phase_gse_averaging(
                spectra_in,
                should_integrate_gse=schedule["should_integrate_gse"][step_idx],
                dt_gse=dt_gse,
                physics_metadata=physics_metadata,
                mask=mask,
                boundary_condition=boundary_condition,
                gse_alleviation=gse_alleviation,
                boundary_update_func=boundary_update_func,
                apply_boundary_func=apply_boundary_func,
            )
    else:
        def _gse(spectra_in: jax.Array, step_idx: jax.Array) -> tuple[jax.Array, jax.Array]:
            del step_idx
            return spectra_in, zero_clipping

    if checkpoint_level == CHECKPOINT_PER_PHASE:
        _spectral_half = jax.checkpoint(_spectral_half)
        _spatial = jax.checkpoint(_spatial)
        _gse = jax.checkpoint(_gse)
        _source = jax.checkpoint(_source)

    def _step_body(carry, step_idx):
        s, total = carry
        s, c1 = _spectral_half(s, step_idx)
        s, c2 = _spatial(s, step_idx)
        s, c3 = _gse(s, step_idx)
        s, c4 = _spectral_half(s, step_idx)
        s, c5 = _source(s, step_idx)
        return (s, total+c1+c2+c3+c4+c5), None

    if checkpoint_level == CHECKPOINT_PER_STEP:
        _step_body = jax.checkpoint(_step_body)

    step_indices = jnp.arange(num_steps, dtype=jnp.int32)
    (spectra, clipped), _ = jax.lax.scan(_step_body, (spectra, zero_clipping), step_indices)

    spectra = boundary_update_func(spectra)
    spectra = apply_boundary_func(spectra, boundary_condition, mask)
    clipped = clipped + _clipping_mass(spectra, physics_metadata)
    spectra = _project_nonnegative(spectra)
    return (spectra, clipped) if audit_clipping else spectra


def apply_frequency_tail(spectra: jax.Array, freq: jax.Array, start_index: int, exponent: float = 5.0) -> jax.Array:
    """Prescribe E(f,theta) proportional to f^-exponent above an explicit anchor.

    This is a prescribed tail closure (WW3 manual, spectral discretization),
    not k-space transport and not a conservative internal redistribution.
    """
    if not 0 <= start_index < spectra.shape[-1]:
        raise ValueError("tail_start_index must select a frequency bin")
    frequency = jnp.ravel(freq)
    anchor = spectra[..., start_index:start_index + 1]
    tail = anchor * (frequency / frequency[start_index])**(-exponent)
    return jnp.where(jnp.arange(frequency.size) > start_index, tail, spectra)


def _spatial_events(metadata: dict) -> int:
    """Count each simultaneous x/y event once, preserving cross-call parity."""
    x, y, duration = metadata["dt_x"], metadata["dt_y"], metadata["dt_global"]
    both = duration // math.lcm(x, y) if x and y else 0
    return (duration // x if x else 0) + (duration // y if y else 0) - both


def step(
    state: State,
    environment: Environment,
    grid: Grid,
    spectral_grid: SpectralGrid,
    config: StepConfig,
    *,
    source_fn: Optional[Callable[..., jax.Array]] = None,
    encode_fn: Optional[Callable[..., jax.Array]] = None,
    decode_fn: Optional[Callable[..., jax.Array]] = None,
    x_tendency_fn: TendencyFn = x_tendency,
    y_tendency_fn: TendencyFn = y_tendency,
    th_tendency_fn: TendencyFn = theta_tendency,
    k_tendency_fn: TendencyFn = k_tendency,
    stot_advance: StotAdvanceFn = simple_stot_integrator,
    boundary_update_func: BoundaryFn = update_absorb_boundary,
    apply_boundary_func: ApplyBoundaryFn = apply_boundary_condition,
    gse_alleviation: Optional[GSEAlleviationFn] = None,
    audit_clipping: bool = False,
) -> State | tuple[State, dict[str, jax.Array]]:
    """Advance one global interval with explicit source and boundary contracts.

    ``source_fn`` returns SI dE/dt once per global interval. None uses the
    supplied Environment.source, or zero when that is also None. Applications
    own calibration factors for learned sources. Static ``config.substeps``
    subdivides each component interval; State carries the spatial-order phase.
    ``encode_fn`` and ``decode_fn`` are identity by default.

    With audit_clipping=True, return (State, diagnostics), where clipping_mass
    measures the integrator's actual positivity additions per batch. It does
    not account for arbitrary external encoder/decoder transformations.
    """
    base_metadata = build_legacy_metadata(grid, spectral_grid, config)
    wavenum = solve_wavenumber(
        freq=base_metadata["freq"], depth=environment.depth, metadata=base_metadata)
    group_velocity = compute_group_velocity(
        freq=base_metadata["freq"], wavenum=wavenum, depth=environment.depth)
    metadata = {**base_metadata, "wavenum": wavenum, "Cg": group_velocity}
    return _step_with_metadata(
        state, environment, grid, spectral_grid, config, metadata,
        source_fn=source_fn, encode_fn=encode_fn, decode_fn=decode_fn,
        x_tendency_fn=x_tendency_fn, y_tendency_fn=y_tendency_fn,
        th_tendency_fn=th_tendency_fn, k_tendency_fn=k_tendency_fn,
        stot_advance=stot_advance, boundary_update_func=boundary_update_func,
        apply_boundary_func=apply_boundary_func, gse_alleviation=gse_alleviation,
        audit_clipping=audit_clipping,
    )


def _step_with_metadata(
    state, environment, grid, spectral_grid, config, metadata, *,
    source_fn=None, encode_fn=None, decode_fn=None,
    x_tendency_fn=x_tendency, y_tendency_fn=y_tendency,
    th_tendency_fn=theta_tendency, k_tendency_fn=k_tendency,
    stot_advance=simple_stot_integrator,
    boundary_update_func=update_absorb_boundary,
    apply_boundary_func=apply_boundary_condition,
    gse_alleviation=None, audit_clipping=False,
):
    """One step with physics prepared from this call's differentiable depth."""
    operator_metadata = {**metadata, **{name: metadata[name] // config.substeps for name in (
        "dt_global", "dt_x", "dt_y", "dt_th", "dt_k", "dt_stot", "dt_gse")},
        "neural_dt_th": config.dt_th, "neural_dt_k": config.dt_k}
    operator_metadata["boundary_mask"] = environment.boundary_mask
    operator_metadata["_spatial_wet_mask"] = _physical_wet_mask(
        environment.depth, environment.mask, environment.boundary, environment.boundary_mask)
    free_mask = environment.mask if environment.boundary_mask is None else (
        environment.mask & ~environment.boundary_mask)
    boundary_apply = apply_boundary_func if environment.boundary_mask is None else partial(
        apply_boundary_func, boundary_mask=environment.boundary_mask)

    encoded = state.spectra if encode_fn is None else encode_fn(
        state.spectra, environment, metadata)
    if environment.boundary_mask is not None:
        encoded = boundary_apply(encoded, environment.boundary, free_mask)
    stot = environment.source if source_fn is None else source_fn(encoded, environment, metadata)
    stot = jnp.zeros_like(encoded) if stot is None else stot

    def advance(carry, _):
        current, phase, cumulative = carry
        updated = model_advance(
            spectra=current,
            stot=stot,
            depth=environment.depth,
            metadata={**operator_metadata, "spatial_phase": phase},
            mask=free_mask,
            boundary_condition=environment.boundary,
            x_tendency_fn=x_tendency_fn,
            y_tendency_fn=y_tendency_fn,
            th_tendency_fn=th_tendency_fn,
            k_tendency_fn=k_tendency_fn,
            stot_advance=stot_advance,
            boundary_update_func=boundary_update_func,
            apply_boundary_func=boundary_apply,
            gse_alleviation=gse_alleviation,
            checkpoint_level=config.checkpoint_level,
            audit_clipping=audit_clipping,
        )
        if audit_clipping:
            updated, correction = updated
        else:
            correction = jnp.zeros_like(cumulative)
        return (updated, (phase + _spatial_events(operator_metadata)) % 2, cumulative+correction), None

    dtype = jnp.float64 if audit_clipping and jax.config.x64_enabled else encoded.dtype
    initial = (encoded, state.spatial_phase, jnp.zeros((encoded.shape[0],), dtype=dtype))
    (integrated, phase, clipped), _ = jax.lax.scan(advance, initial, None, length=config.substeps)
    decoded = integrated if decode_fn is None else decode_fn(
        integrated, environment, metadata)
    if config.tail_start_index is not None:
        decoded = apply_frequency_tail(decoded, spectral_grid.freq, config.tail_start_index, config.m)
    result = State(spectra=boundary_apply(decoded, environment.boundary, environment.mask), spatial_phase=phase)
    return (result, {"clipping_mass": clipped}) if audit_clipping else result


def rollout(
    state: State,
    environments: Environment,
    grid: Grid,
    spectral_grid: SpectralGrid,
    config: StepConfig,
    *,
    source_fn: Optional[Callable[..., jax.Array]] = None,
    encode_fn: Optional[Callable[..., jax.Array]] = None,
    decode_fn: Optional[Callable[..., jax.Array]] = None,
    x_tendency_fn: TendencyFn = x_tendency,
    y_tendency_fn: TendencyFn = y_tendency,
    th_tendency_fn: TendencyFn = theta_tendency,
    k_tendency_fn: TendencyFn = k_tendency,
    stot_advance: StotAdvanceFn = simple_stot_integrator,
    boundary_update_func: BoundaryFn = update_absorb_boundary,
    apply_boundary_func: ApplyBoundaryFn = apply_boundary_condition,
    gse_alleviation: Optional[GSEAlleviationFn] = None,
    times=None,
    rng_key=None,
    carry_phase: bool = False,
    audit_clipping: bool = False,
    fixed_depth: bool = False,
) -> RolloutResult:
    """Run a leading-time environment sequence with one compiled scan body.

    Spatial phase always travels with State across calls. The G2 carry_phase
    keyword is retained but no longer resets phase. ``times`` stays outside
    the scanned pytree; sampling/PRNG carry remains unsupported. Clipping
    audit results, when requested, are returned as per-time/per-batch arrays.

    With fixed_depth=True, environments.depth must be a single (B,H,W,1,1)
    field; all other non-None fields retain their leading time axis. Dispersion,
    group velocity and depth gradients are computed once within this call's
    AD graph and shared across time. A changed depth/configuration is prepared
    anew on every call. Use the default stacked depth for time-varying water.
    """
    if rng_key is not None:
        raise NotImplementedError("sampling PRNG carry is deferred to G4")
    # Retained keyword for G2 callers: phase now always travels with State.
    del carry_phase

    prepared = None
    shared_depth = None
    if fixed_depth:
        if environments.depth.ndim != 5:
            raise ValueError("fixed_depth requires one depth field of shape (B,H,W,1,1), without a time axis")
        shared_depth = environments.depth
        steps = int(environments.wind_u.shape[0])
        base = build_legacy_metadata(grid, spectral_grid, config)
        k = solve_wavenumber(base["freq"], shared_depth, metadata=base)
        prepared = {
            **base, "wavenum": k,
            "Cg": compute_group_velocity(base["freq"], k, shared_depth),
            "_depth_gradients": _compute_depth_gradients(shared_depth, base),
        }
        scan_environments = dataclasses.replace(environments, depth=None)
    else:
        steps = int(environments.depth.shape[0])
        scan_environments = environments
    times = list(range(steps)) if times is None else list(times)
    if len(times) != steps:
        raise ValueError(f"times must have {steps} entries, got {len(times)}")

    def body(carry: State, environment: Environment):
        if fixed_depth:
            environment = dataclasses.replace(environment, depth=shared_depth)
        advance = step if prepared is None else partial(_step_with_metadata, metadata=prepared)
        updated = advance(
            carry,
            environment,
            grid,
            spectral_grid,
            config,
            source_fn=source_fn,
            encode_fn=encode_fn,
            decode_fn=decode_fn,
            x_tendency_fn=x_tendency_fn,
            y_tendency_fn=y_tendency_fn,
            th_tendency_fn=th_tendency_fn,
            k_tendency_fn=k_tendency_fn,
            stot_advance=stot_advance,
            boundary_update_func=boundary_update_func,
            apply_boundary_func=apply_boundary_func,
            gse_alleviation=gse_alleviation,
            audit_clipping=audit_clipping,
        )
        if audit_clipping:
            updated, diagnostics = updated
            return updated, (updated.spectra, diagnostics["clipping_mass"])
        return updated, updated.spectra

    final_state, spectra = jax.lax.scan(body, state, scan_environments)
    diagnostics = None
    if audit_clipping:
        spectra, clipping = spectra
        diagnostics = {"clipping_mass": clipping}
    return RolloutResult(spectra=spectra, final_state=final_state, times=times, diagnostics=diagnostics)


__all__ = [
    "CHECKPOINT_NONE",
    "CHECKPOINT_PER_PHASE",
    "CHECKPOINT_PER_STEP",
    "apply_frequency_tail",
    "model_advance",
    "rollout",
    "simple_stot_integrator",
    "step",
]
