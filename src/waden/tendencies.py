# -*- coding: utf-8 -*-
# Origin: v1 at 47a6558; independent v2 scientific candidates evolve here.
"""Pure JAX tendency calculations for NeuralWave."""

from typing import Callable, Optional

import jax
import jax.numpy as jnp

from .flux import get_stencils, quickest_flux, wavenumber_flux
from .physics import calculate_depth_dispersion_derivative


ArrayFn = Callable[..., jax.Array]
FluxFn = Callable[..., tuple[jax.Array, jax.Array]]


def _as_array(value, dtype=None):
    array = jnp.asarray(value)
    if dtype is not None:
        array = array.astype(dtype)
    return array


def _broadcast_like(value, reference: jax.Array) -> jax.Array:
    return jnp.broadcast_to(_as_array(value, reference.dtype), reference.shape)


def _safe_divisor(value, dtype, eps: float) -> jax.Array:
    return jnp.clip(_as_array(value, dtype=dtype), min=eps)


def _broadcast_metadata_fields(spectra: jax.Array, metadata: dict):
    return {
        "Cg": _broadcast_like(metadata["Cg"], spectra),
        "wavenum": _broadcast_like(metadata["wavenum"], spectra),
        "sinTheta": _broadcast_like(metadata["sinTheta"], spectra),
        "cosTheta": _broadcast_like(metadata["cosTheta"], spectra),
    }


def _coordinate_type(metadata: dict) -> str:
    coordinate_type = metadata["coordinate_type"]
    if coordinate_type not in {"Spherical", "Cartesian"}:
        raise ValueError("coordinate_type should be Spherical or Cartesian.")
    return coordinate_type


def _depth_gradients_like_spectra(
    spectra: jax.Array,
    depth: jax.Array,
    DdDx: jax.Array,
    DdDy: jax.Array,
):
    del depth
    return (
        _broadcast_like(DdDx, spectra),
        _broadcast_like(DdDy, spectra),
    )


def x_tendency(
    spectra: jax.Array,
    dt: float,
    metadata: dict,
    land_mask: Optional[jax.Array] = None,
    flux_fn: FluxFn = quickest_flux,
    eps: float = 1e-9,
    stencil_mode: Optional[str] = None,
) -> jax.Array:
    """Calculate the x-direction propagation tendency."""
    coordinate_type = _coordinate_type(metadata)
    fields = _broadcast_metadata_fields(spectra, metadata)
    spec = _as_array(_broadcast_like(metadata["FACHFA"], spectra) * spectra)

    if metadata.get("FULCR", False):
        raise NotImplementedError("Current handling is not implemented yet.")

    if coordinate_type == "Spherical":
        dxgrid = _as_array(metadata["Dlon"], dtype=spectra.dtype) * (
            _as_array(metadata["r_earth"], dtype=spectra.dtype)
            * _as_array(metadata["cosLAT"], dtype=spectra.dtype)
        )
    else:
        dxgrid = _as_array(metadata["Dlon"], dtype=spectra.dtype)

    lamda = _broadcast_like(metadata["FACVX"], spectra) * fields["Cg"] * fields["sinTheta"]
    fln_negx, fln_posx = flux_fn(
        N=spec,
        Cg=lamda,
        land_mask=land_mask,
        Dgrid=dxgrid,
        target_dim=1,
        dt=dt,
        **({"mode": stencil_mode} if stencil_mode is not None else {}),
    )
    return (fln_negx - fln_posx) / _safe_divisor(dxgrid, spectra.dtype, eps)


def y_tendency(
    spectra: jax.Array,
    dt: float,
    metadata: dict,
    land_mask: Optional[jax.Array] = None,
    flux_fn: FluxFn = quickest_flux,
    eps: float = 1e-9,
    stencil_mode: Optional[str] = None,
) -> jax.Array:
    """Calculate the y-direction propagation tendency."""
    coordinate_type = _coordinate_type(metadata)
    fields = _broadcast_metadata_fields(spectra, metadata)

    if coordinate_type == "Spherical":
        spec = _broadcast_like(metadata["FACHFA"], spectra) * spectra
        dygrid = _as_array(metadata["Dlat"], dtype=spectra.dtype) * _as_array(
            metadata["r_earth"], dtype=spectra.dtype
        ) * _as_array(metadata["cosLAT"], dtype=spectra.dtype)
        phi = (
            _broadcast_like(metadata["FACVY"], spectra)
            * fields["Cg"]
            * fields["cosTheta"]
            * _broadcast_like(metadata["cosLAT"], spectra)
        )
    else:
        spec = _broadcast_like(metadata["FACHFA"], spectra) * spectra
        dygrid = _as_array(metadata["Dlat"], dtype=spectra.dtype)
        phi = _broadcast_like(metadata["FACVY"], spectra) * fields["Cg"] * fields["cosTheta"]

    spec = _as_array(spec)
    fln_negy, fln_posy = flux_fn(
        N=spec,
        Cg=phi,
        land_mask=land_mask,
        Dgrid=dygrid,
        target_dim=2,
        dt=dt,
        **({"mode": stencil_mode} if stencil_mode is not None else {}),
    )
    return (fln_negy - fln_posy) / _safe_divisor(dygrid, spectra.dtype, eps)


def theta_transport_velocity(
    spectra: jax.Array,
    metadata: dict,
    depth: jax.Array,
    DdDx: jax.Array,
    DdDy: jax.Array,
    dsdd_fn: ArrayFn = calculate_depth_dispersion_derivative,
    eps: float = 1e-9,
) -> jax.Array:
    """Return d(theta_flux)/dt, before conversion to the array's bin direction."""
    del spectra
    coordinate_type = _coordinate_type(metadata)
    DdDx_full, DdDy_full = _depth_gradients_like_spectra(metadata["Cg"], depth, DdDx, DdDy)
    Cg = _as_array(metadata["Cg"])
    wavenum = _as_array(metadata["wavenum"])
    sinTheta = _as_array(metadata["sinTheta"])
    cosTheta = _as_array(metadata["cosTheta"])

    DsDd = dsdd_fn(Cg, wavenum, metadata["freq"], depth, eps=eps)
    safe_wavenum = jnp.clip(wavenum, min=eps)
    VCTH = -_as_array(metadata["FACTH"], dtype=Cg.dtype) * (1.0 / safe_wavenum) * (
        DsDd * (cosTheta * DdDx_full - sinTheta * DdDy_full)
    )
    if coordinate_type == "Spherical":
        # WW3 (2016), Eq. (2.15), transformed from east-axis angles to the
        # grid's compass convention (latitude index may point south).
        VCTH = VCTH + metadata.get("latitude_sign", -1) * metadata["FDG"] * (
            Cg * sinTheta * jnp.tan(metadata["lat"]) / metadata["r_earth"])

    return VCTH


def theta_tendency(
    spectra: jax.Array,
    dt: float,
    metadata: dict,
    depth: jax.Array,
    DdDx: jax.Array,
    DdDy: jax.Array,
    flux_fn: FluxFn = quickest_flux,
    dsdd_fn: ArrayFn = calculate_depth_dispersion_derivative,
    eps: float = 1e-9,
) -> jax.Array:
    """Calculate the theta-direction tendency on all physical grid points."""
    fields = _broadcast_metadata_fields(spectra, metadata)
    metadata_full = dict(metadata)
    metadata_full.update(fields)
    VCTH = theta_transport_velocity(spectra, metadata_full, depth, DdDx, DdDy, dsdd_fn=dsdd_fn, eps=eps)
    # A grid-angle rate is not necessarily a positive-index angular rate.
    # For the official grid theta_flux decreases as the direction bin grows.
    VCTH = metadata.get("theta_axis_sign", 1) * VCTH
    fln_negth, fln_posth = flux_fn(
        N=spectra,
        Cg=VCTH,
        Dgrid=metadata["DTH"],
        target_dim=3,
        dt=dt,
    )
    theta_pro = (fln_negth - fln_posth) / _safe_divisor(metadata["DTH"], spectra.dtype, eps)
    return theta_pro


def wavenumber_transport_velocity(
    spectra: jax.Array,
    metadata: dict,
    depth: jax.Array,
    DdDx: jax.Array,
    DdDy: jax.Array,
    dsdd_fn: ArrayFn = calculate_depth_dispersion_derivative,
    eps: float = 1e-9,
) -> jax.Array:
    """Calculate the wavenumber-direction transport velocity VCWN."""
    del spectra
    DdDx_full, DdDy_full = _depth_gradients_like_spectra(metadata["Cg"], depth, DdDx, DdDy)
    Cg = _as_array(metadata["Cg"])
    wavenum = _as_array(metadata["wavenum"])
    sinTheta = _as_array(metadata["sinTheta"])
    cosTheta = _as_array(metadata["cosTheta"])

    DsDd = dsdd_fn(Cg, wavenum, metadata["freq"], depth, eps=eps)
    # Ray k-dot (rad m^-1 s^-1), WW3 (2016), Eq. (2.10).
    VCWN = -_as_array(metadata["FACK"], dtype=Cg.dtype) * DsDd * (
        sinTheta * DdDx_full + cosTheta * DdDy_full
    )
    return _as_array(VCWN)


def k_tendency(
    spectra: jax.Array,
    dt: float,
    metadata: dict,
    depth: jax.Array,
    DdDx: jax.Array,
    DdDy: jax.Array,
    flux_fn: FluxFn = wavenumber_flux,
    dsdd_fn: ArrayFn = calculate_depth_dispersion_derivative,
    eps: float = 1e-9,
) -> jax.Array:
    """Return zero physical frequency shift for E(f,theta) in stationary water.

    On the supported stationary-depth, zero-current path, absolute frequency
    is conserved along rays (SWAN technical manual, wave kinematics).
    k-dot is a ray diagnostic, not a divergence acting on E(f,theta).
    Time-varying depth/current frequency shifts are outside this first release.
    Neural frequency redistribution remains a separate learned closure.
    """
    del dt, depth, DdDx, DdDy, flux_fn, dsdd_fn, eps
    if metadata.get("FULCR", False):
        raise NotImplementedError("frequency shifting by currents is not implemented")
    return jnp.zeros_like(spectra)


def _neural_inputs(
    spectra: jax.Array,
    metadata: dict,
    depth: jax.Array,
    DdDx: jax.Array,
    DdDy: jax.Array,
    dsdd_fn: ArrayFn,
    eps: float,
):
    del spectra
    wavenum = _as_array(metadata["wavenum"])
    Cg = _as_array(metadata["Cg"])
    DdDx_full, DdDy_full = _depth_gradients_like_spectra(wavenum, depth, DdDx, DdDy)
    DsDd = dsdd_fn(Cg, wavenum, metadata["freq"], depth, eps=eps)
    safe_wavenum = jnp.clip(wavenum, min=eps)
    b, h, w, d, f = safe_wavenum.shape
    inputs = jnp.concatenate(
        [
            (safe_wavenum * DdDx_full).reshape(b * h * w, d * f),
            (safe_wavenum * DdDy_full).reshape(b * h * w, d * f),
            DsDd.reshape(b * h * w, d * f),
        ],
        axis=-1,
    )
    return inputs, {
        "wavenum": wavenum,
        "Cg": Cg,
        "sinTheta": _as_array(metadata["sinTheta"]),
        "cosTheta": _as_array(metadata["cosTheta"]),
    }, (b, h, w, d, f)


def _run_network_blocked(network_fn: ArrayFn, inputs: jax.Array, num_intra_batch_block: Optional[int]) -> jax.Array:
    if num_intra_batch_block is None or num_intra_batch_block <= 1:
        return jnp.asarray(network_fn(inputs))

    total_points = inputs.shape[0]
    blocks = int(num_intra_batch_block)
    block_size = (total_points + blocks - 1) // blocks
    outputs = []
    for start in range(0, total_points, block_size):
        end = min(start + block_size, total_points)
        outputs.append(jnp.asarray(network_fn(inputs[start:end])))
    return jnp.concatenate(outputs, axis=0)


def _build_sample_mask(valid_mask: jax.Array, rng_key, sample_ratio: float, min_samples: int) -> jax.Array:
    total_points = valid_mask.shape[0]
    valid_count = jnp.sum(valid_mask.astype(jnp.int32))
    sample_count = jnp.maximum(
        jnp.asarray(min_samples, dtype=jnp.int32),
        jnp.floor(valid_count.astype(jnp.float32) * sample_ratio).astype(jnp.int32),
    )
    sample_count = jnp.minimum(sample_count, valid_count)

    perm = jax.random.permutation(rng_key, total_points)
    valid_perm = valid_mask[perm]
    sampled_perm = valid_perm & (jnp.cumsum(valid_perm.astype(jnp.int32)) <= sample_count)
    return jnp.zeros_like(valid_mask).at[perm].set(sampled_perm)


def _sampled_network_output(
    inputs: jax.Array,
    depth: jax.Array,
    network_fn: ArrayFn,
    sample_ratio: Optional[float],
    min_samples: int,
    num_intra_batch_block: Optional[int],
    *,
    rng_key: jax.Array | None,
    default_rng_seed: int,
) -> jax.Array:
    full_output = _run_network_blocked(network_fn, inputs, num_intra_batch_block)
    if sample_ratio is None or sample_ratio <= 0:
        return full_output

    if rng_key is None:
        rng_key = jax.random.key(default_rng_seed)

    valid_mask = jnp.reshape(depth > 0, (-1,))
    sample_mask = _build_sample_mask(valid_mask, rng_key, sample_ratio, min_samples)
    sample_mask = jnp.reshape(sample_mask, (-1, 1))
    return jnp.where(sample_mask, full_output, jax.lax.stop_gradient(full_output))


def neural_theta_transport_velocity(
    spectra: jax.Array,
    metadata: dict,
    depth: jax.Array,
    DdDx: jax.Array,
    DdDy: jax.Array,
    network_fn: ArrayFn,
    dsdd_fn: ArrayFn = calculate_depth_dispersion_derivative,
    eps: float = 1e-9,
) -> jax.Array:
    """Calculate neural theta transport velocity from a pure network function."""
    inputs, fields, shape = _neural_inputs(spectra, metadata, depth, DdDx, DdDy, dsdd_fn, eps)
    b, h, w, d, f = shape
    hat_A = jnp.asarray(network_fn(inputs)).reshape(b, h, w, d, f, 2)
    VCTH = hat_A[..., 0] * fields["cosTheta"] - hat_A[..., 1] * fields["sinTheta"]
    return jnp.tanh(VCTH)


def neural_k_transport_velocity(
    spectra: jax.Array,
    metadata: dict,
    depth: jax.Array,
    DdDx: jax.Array,
    DdDy: jax.Array,
    network_fn: ArrayFn,
    dsdd_fn: ArrayFn = calculate_depth_dispersion_derivative,
    eps: float = 1e-9,
) -> jax.Array:
    """Calculate neural wavenumber transport velocity from a pure network function."""
    inputs, fields, shape = _neural_inputs(spectra, metadata, depth, DdDx, DdDy, dsdd_fn, eps)
    b, h, w, d, f = shape
    hat_A = jnp.asarray(network_fn(inputs)).reshape(b, h, w, d, f, 2)
    VCWN = hat_A[..., 0] * fields["cosTheta"] + hat_A[..., 1] * fields["sinTheta"]
    return jnp.tanh(VCWN)


def sampling_neural_theta_transport_velocity(
    spectra: jax.Array,
    metadata: dict,
    depth: jax.Array,
    DdDx: jax.Array,
    DdDy: jax.Array,
    network_fn: ArrayFn,
    sample_ratio: Optional[float] = None,
    min_samples: int = 1024,
    num_intra_batch_block: Optional[int] = None,
    rng_key: jax.Array | None = None,
    default_rng_seed: int = 0,
    dsdd_fn: ArrayFn = calculate_depth_dispersion_derivative,
    eps: float = 1e-9,
) -> jax.Array:
    """Calculate sampled neural theta transport velocity with stop-gradient on unsampled points."""
    inputs, fields, shape = _neural_inputs(spectra, metadata, depth, DdDx, DdDy, dsdd_fn, eps)
    b, h, w, d, f = shape
    hat_A_full = _sampled_network_output(
        inputs,
        depth,
        network_fn,
        sample_ratio,
        min_samples,
        num_intra_batch_block,
        rng_key=rng_key,
        default_rng_seed=default_rng_seed,
    )
    hat_A = hat_A_full.reshape(b, h, w, d, f, 2)
    VCTH = hat_A[..., 0] * fields["cosTheta"] - hat_A[..., 1] * fields["sinTheta"]
    return jnp.tanh(VCTH)


def sampling_neural_k_transport_velocity(
    spectra: jax.Array,
    metadata: dict,
    depth: jax.Array,
    DdDx: jax.Array,
    DdDy: jax.Array,
    network_fn: ArrayFn,
    sample_ratio: Optional[float] = None,
    min_samples: int = 1024,
    num_intra_batch_block: Optional[int] = None,
    rng_key: jax.Array | None = None,
    default_rng_seed: int = 0,
    dsdd_fn: ArrayFn = calculate_depth_dispersion_derivative,
    eps: float = 1e-9,
) -> jax.Array:
    """Calculate sampled neural wavenumber transport velocity with stop-gradient on unsampled points."""
    inputs, fields, shape = _neural_inputs(spectra, metadata, depth, DdDx, DdDy, dsdd_fn, eps)
    b, h, w, d, f = shape
    hat_A_full = _sampled_network_output(
        inputs,
        depth,
        network_fn,
        sample_ratio,
        min_samples,
        num_intra_batch_block,
        rng_key=rng_key,
        default_rng_seed=default_rng_seed,
    )
    hat_A = hat_A_full.reshape(b, h, w, d, f, 2)
    VCWN = hat_A[..., 0] * fields["cosTheta"] + hat_A[..., 1] * fields["sinTheta"]
    return jnp.tanh(VCWN)


def _first_order_tendency(
    spectra: jax.Array,
    transport_velocity: jax.Array,
    dt: float,
    target_dim: int,
    mode: str,
) -> jax.Array:
    N_im1, N_i = get_stencils(spectra, target_dim=target_dim, pad_left=1, pad_right=0, mode=mode)
    is_positive = transport_velocity >= 0
    N_upwind = jnp.where(is_positive, N_im1, N_i)
    dt_array = jnp.asarray(dt, dtype=spectra.dtype)
    safe_dt = jnp.where(dt_array == 0, jnp.asarray(3600.0, dtype=spectra.dtype), dt_array)
    fln_neg = transport_velocity * N_upwind / safe_dt
    fln_pos = get_stencils(fln_neg, target_dim=target_dim, pad_left=0, pad_right=1, mode=mode)[1]
    return fln_neg - fln_pos


def neural_theta_tendency(
    spectra: jax.Array,
    dt: float,
    metadata: dict,
    depth: jax.Array,
    DdDx: jax.Array,
    DdDy: jax.Array,
    network_fn: ArrayFn,
    dsdd_fn: ArrayFn = calculate_depth_dispersion_derivative,
    eps: float = 1e-9,
) -> jax.Array:
    """Calculate neural theta tendency using first-order upwind flux."""
    del eps
    velocity = neural_theta_transport_velocity(
        spectra,
        metadata,
        depth,
        DdDx,
        DdDy,
        network_fn=network_fn,
        dsdd_fn=dsdd_fn,
    )
    theta_pro = _first_order_tendency(spectra, velocity, dt=metadata.get("neural_dt_th", dt), target_dim=3, mode="circular")
    return theta_pro


def neural_k_tendency(
    spectra: jax.Array,
    dt: float,
    metadata: dict,
    depth: jax.Array,
    DdDx: jax.Array,
    DdDy: jax.Array,
    network_fn: ArrayFn,
    dsdd_fn: ArrayFn = calculate_depth_dispersion_derivative,
    eps: float = 1e-9,
) -> jax.Array:
    """Calculate neural wavenumber tendency using first-order upwind flux."""
    del eps
    velocity = neural_k_transport_velocity(
        spectra,
        metadata,
        depth,
        DdDx,
        DdDy,
        network_fn=network_fn,
        dsdd_fn=dsdd_fn,
    )
    wavenum_pro = _first_order_tendency(spectra, velocity, dt=metadata.get("neural_dt_k", dt), target_dim=4, mode="replicate")
    return wavenum_pro


def sampling_neural_theta_tendency(
    spectra: jax.Array,
    dt: float,
    metadata: dict,
    depth: jax.Array,
    DdDx: jax.Array,
    DdDy: jax.Array,
    network_fn: ArrayFn,
    sample_ratio: Optional[float] = None,
    min_samples: int = 1024,
    num_intra_batch_block: Optional[int] = None,
    rng_key: jax.Array | None = None,
    default_rng_seed: int = 0,
    dsdd_fn: ArrayFn = calculate_depth_dispersion_derivative,
    eps: float = 1e-9,
) -> jax.Array:
    """Calculate sampled neural theta tendency using first-order upwind flux."""
    del eps
    velocity = sampling_neural_theta_transport_velocity(
        spectra,
        metadata,
        depth,
        DdDx,
        DdDy,
        network_fn=network_fn,
        sample_ratio=sample_ratio,
        min_samples=min_samples,
        num_intra_batch_block=num_intra_batch_block,
        rng_key=rng_key,
        default_rng_seed=default_rng_seed,
        dsdd_fn=dsdd_fn,
    )
    theta_pro = _first_order_tendency(spectra, velocity, dt=metadata.get("neural_dt_th", dt), target_dim=3, mode="circular")
    return theta_pro


def sampling_neural_k_tendency(
    spectra: jax.Array,
    dt: float,
    metadata: dict,
    depth: jax.Array,
    DdDx: jax.Array,
    DdDy: jax.Array,
    network_fn: ArrayFn,
    sample_ratio: Optional[float] = None,
    min_samples: int = 1024,
    num_intra_batch_block: Optional[int] = None,
    rng_key: jax.Array | None = None,
    default_rng_seed: int = 0,
    dsdd_fn: ArrayFn = calculate_depth_dispersion_derivative,
    eps: float = 1e-9,
) -> jax.Array:
    """Calculate sampled neural wavenumber tendency using first-order upwind flux."""
    del eps
    velocity = sampling_neural_k_transport_velocity(
        spectra,
        metadata,
        depth,
        DdDx,
        DdDy,
        network_fn=network_fn,
        sample_ratio=sample_ratio,
        min_samples=min_samples,
        num_intra_batch_block=num_intra_batch_block,
        rng_key=rng_key,
        default_rng_seed=default_rng_seed,
        dsdd_fn=dsdd_fn,
    )
    wavenum_pro = _first_order_tendency(spectra, velocity, dt=metadata.get("neural_dt_k", dt), target_dim=4, mode="replicate")
    return wavenum_pro


__all__ = [
    "k_tendency",
    "neural_k_tendency",
    "neural_k_transport_velocity",
    "neural_theta_tendency",
    "neural_theta_transport_velocity",
    "sampling_neural_k_tendency",
    "sampling_neural_k_transport_velocity",
    "sampling_neural_theta_tendency",
    "sampling_neural_theta_transport_velocity",
    "theta_tendency",
    "theta_transport_velocity",
    "wavenumber_transport_velocity",
    "x_tendency",
    "y_tendency",
]
