# -*- coding: utf-8 -*-
"""
Flux calculations for NeuralWave JAX.

JAX equivalent of waden_hpc/flux_calculation.py.
All calculations are implemented as pure functions and are compatible with
future use from the tendency and integrator stages.
"""

from typing import Optional, Tuple

import jax
import jax.numpy as jnp


def _normalize_mode(mode: str) -> str:
    if mode == "circular":
        return "wrap"
    if mode == "replicate":
        return "edge"
    raise ValueError(f"Unsupported padding mode: {mode}")


def _expand_trailing_dims(array: jax.Array, ndim: int) -> jax.Array:
    while array.ndim < ndim:
        array = jnp.expand_dims(array, axis=-1)
    return array


def _broadcast_like(array: jax.Array, reference: jax.Array) -> jax.Array:
    array = _expand_trailing_dims(array, reference.ndim)
    return jnp.broadcast_to(array, reference.shape)


def _index_at_axis(ndim: int, target_dim: int, index: int) -> Tuple[slice | int, ...]:
    if target_dim < 0:
        target_dim = ndim + target_dim
    idx = [slice(None)] * ndim
    idx[target_dim] = index
    return tuple(idx)

# Ultimate JAX-optimized version (even moveaxis is omitted) 
# As padding in JAX is efficient and supports specifying the axis directly, 
# we can avoid moving axes around.
def get_stencils(
    tensor: jax.Array,
    target_dim: int,
    pad_left: int = 2,
    pad_right: int = 1,
    mode: str = "circular",
) -> Tuple[jax.Array, ...]:
    if target_dim < 0:
        target_dim = tensor.ndim + target_dim
        
    # Directly specify padding on the target_dim, with all other dimensions set to (0,0).
    pad_width = [(0, 0)] * tensor.ndim
    pad_width[target_dim] = (pad_left, pad_right)
    
    # Directly pad the original tensor—no need to moveaxis back and forth!
    padded = jnp.pad(tensor, pad_width, mode=_normalize_mode(mode))
    
    original_size = tensor.shape[target_dim]
    stencils = []
    for offset in range(pad_left + pad_right + 1):
        stencils.append(
            jax.lax.slice_in_dim(
                padded,
                start_index=offset,
                limit_index=offset + original_size,
                axis=target_dim,
            )
        )
    return tuple(stencils)


def first_order_flux(
    N: jax.Array,
    Cg: jax.Array,
    land_mask: jax.Array,
    target_dim: int = 1,
    mode: Optional[str] = None,
) -> Tuple[jax.Array, jax.Array]:
    """First-order upwind flux, matching waden_hpc.FirstOrderFlux.

    Parameters
    ----------
    mode : {"circular", "replicate"}, optional
        Stencil padding along ``target_dim`` for spectra, velocity and mask.
        None preserves the original axis-dependent padding defaults.
        This selects stencil neighbors; it does not refresh ghost cells.
    """
    if mode is None:
        mode = "circular" if target_dim in [1] else "replicate"
    N_im1, N_i = get_stencils(N, target_dim, pad_left=1, pad_right=0, mode=mode)
    Cg_i, Cg_ip1 = get_stencils(Cg, target_dim, pad_left=0, pad_right=1, mode=mode)

    land_mask = land_mask.astype(jnp.float32)
    lm_i, lm_ip1 = get_stencils(land_mask, target_dim, pad_left=0, pad_right=1, mode=mode)
    land2sea_mask = _broadcast_like((lm_i == 0) & (lm_ip1 == 1), Cg_i)

    Cg_eff = jnp.where(land2sea_mask, Cg_ip1, Cg_i)
    Cg_eff_im1 = get_stencils(Cg_eff, target_dim, pad_left=1, pad_right=0, mode=mode)[0]

    Cg_b = 0.5 * (Cg_eff_im1 + Cg_eff)
    N_upwind = jnp.where(Cg_b >= 0, N_im1, N_i)
    FLN_minus = Cg_b * N_upwind
    FLN_plus = get_stencils(FLN_minus, target_dim, pad_left=0, pad_right=1, mode=mode)[1]
    return FLN_minus, FLN_plus


def _quickest_tvd_limiter(
    N: jax.Array,
    Cg_b: jax.Array,
    Nb: jax.Array,
    CFL: jax.Array,
    target_dim: int,
    mode: str,
    eps: float,
    N_stencils: Optional[Tuple[jax.Array, jax.Array, jax.Array, jax.Array]] = None,
) -> jax.Array:
    if N_stencils is None:
        N_im2, N_im1, N_i, N_ip1 = get_stencils(N, target_dim, pad_left=2, pad_right=1, mode=mode)
    else:
        N_im2, N_im1, N_i, N_ip1 = N_stencils

    is_positive = Cg_b >= 0
    N_c = jnp.where(is_positive, N_im1, N_i)
    N_u = jnp.where(is_positive, N_im2, N_ip1)
    N_d = jnp.where(is_positive, N_i, N_im1)

    DQ = N_d - N_u
    DQNZ = jnp.sign(DQ) * jnp.clip(jnp.abs(DQ), min=eps)
    DQNZ = jnp.where(DQ == 0, -eps, DQNZ)

    Nc_tilde = jnp.clip((N_c - N_u) / DQNZ, -0.1, 1.1)
    Nb_tilde = jnp.maximum((Nb - N_u) / DQNZ, Nc_tilde)
    CFL = jnp.clip(jnp.abs(CFL), min=eps)

    Nclimit = Nc_tilde / CFL
    Nb_tilde = jnp.minimum(Nb_tilde, jnp.clip(Nclimit, max=1.0))

    Nclimit2 = ((N_c / CFL) - N_u) / DQNZ
    Nc_tilde = jnp.where(
        DQ < 0,
        jnp.maximum(Nc_tilde, Nclimit2),
        jnp.minimum(Nc_tilde, Nclimit2),
    )
    Nc_rev = N_u + Nc_tilde * DQ
    Nb_rev = N_u + Nb_tilde * DQ

    # Definition of CFAC see Page 129 of WW3 Manual v6.07, it helps to judge the monotonicity.
    # If |Nc_tilde - 0.5| < 0.5 means 0 < Nc_tilde <= 1, then use Nb_rev; else use Nc_rev.
    CFAC = jnp.clip(jnp.floor(2.0 * jnp.abs(Nc_tilde - 0.5)), 0.0, 1.0)

    N_c = jnp.where(CFL > 1.0, Nc_rev, N_c)
    return (1.0 - CFAC) * Nb_rev + CFAC * N_c


def _theta_tvd_limiter(
    N: jax.Array,
    Cg_b: jax.Array,
    Nb: jax.Array,
    CFL: jax.Array,
    target_dim: int,
    mode: str,
    eps: float,
    N_stencils: Optional[Tuple[jax.Array, jax.Array, jax.Array, jax.Array]] = None,
) -> jax.Array:
    if N_stencils is None:
        N_im2, N_im1, N_i, N_ip1 = get_stencils(N, target_dim, pad_left=2, pad_right=1, mode=mode)
    else:
        N_im2, N_im1, N_i, N_ip1 = N_stencils

    is_positive = Cg_b >= 0
    N_c = jnp.where(is_positive, N_im1, N_i)
    N_u = jnp.where(is_positive, N_im2, N_ip1)
    N_d = jnp.where(is_positive, N_i, N_im1)

    DQ = N_d - N_u
    DQNZ = jnp.sign(DQ) * jnp.clip(jnp.abs(DQ), min=eps)
    DQNZ = jnp.where(DQ == 0, -eps, DQNZ)

    Nc_tilde = (N_c - N_u) / DQNZ
    Nb_tilde = jnp.maximum((Nb - N_u) / DQNZ, jnp.clip(Nc_tilde, min=-0.1, max=1.1))
    CFL = jnp.clip(jnp.abs(CFL), min=eps)

    Nclimit = ((N_c / CFL) - N_u) / DQNZ
    Nb_tilde = jnp.minimum(Nb_tilde, jnp.clip(Nclimit, max=1.0))
    Nb_tilde = jnp.where(
        DQ < 0,
        jnp.maximum(Nb_tilde, Nclimit),
        jnp.minimum(Nb_tilde, Nclimit),
    )

    Nc_tilde = jnp.where(
        DQ < 0,
        jnp.maximum(Nc_tilde, Nclimit),
        jnp.minimum(Nc_tilde, Nclimit),
    )

    Nc_rev = N_u + Nc_tilde * DQ
    Nb_rev = N_u + Nb_tilde * DQ
    # Definition of CFAC see Page 129 of WW3 Manual v6.07, it helps to judge the monotonicity.
    # If |Nc_tilde - 0.5| < 0.5 means 0 < Nc_tilde <= 1, then use Nb_rev; else use Nc_rev.
    CFAC = jnp.clip(jnp.floor(2.0 * jnp.abs(Nc_tilde - 0.5)), 0.0, 1.0)
    return (1.0 - CFAC) * Nb_rev + CFAC * Nc_rev


def _handle_land_boundaries(
    N: jax.Array,
    Cg: jax.Array,
    Cg_b: jax.Array,
    Nb: jax.Array,
    land_mask: jax.Array,
    target_dim: int,
    mode: str,
    N_stencils: Optional[Tuple[jax.Array, jax.Array, jax.Array, jax.Array]] = None,
    Cg_stencils: Optional[Tuple[jax.Array, jax.Array]] = None,
    boundary_mask: Optional[jax.Array] = None,
) -> Tuple[jax.Array, jax.Array]:
    if N_stencils is None:
        N_im1, N_i = get_stencils(N, target_dim, pad_left=1, pad_right=0, mode=mode)
    else:
        _, N_im1, N_i, _ = N_stencils

    wet_left, wet_right = get_stencils(land_mask.astype(bool), target_dim, 1, 0, mode)
    if Cg_stencils is None:
        speed_left, speed_right = get_stencils(Cg, target_dim, 1, 0, mode)
    else:
        speed_left, speed_right = Cg_stencils
    # WW3 shoreline: sea-cell speed, zero land inflow, absorbed outgoing waves.
    speed = jnp.where(~wet_left & wet_right, speed_right,
                      jnp.where(wet_left & ~wet_right, speed_left, Cg_b))
    donor = jnp.where(speed >= 0, jnp.where(wet_left, N_im1, 0),
                      jnp.where(wet_right, N_i, 0))
    boundary_face = ~wet_left | ~wet_right
    if boundary_mask is not None:
        prescribed = _expand_trailing_dims(boundary_mask, N.ndim)
        fixed_left, fixed_right = get_stencils(prescribed, target_dim, 1, 0, mode)
        boundary_face = boundary_face | fixed_left | fixed_right
    FLN_minus = jnp.where(boundary_face, speed * donor, Cg_b * Nb)
    FLN_plus = get_stencils(FLN_minus, target_dim, pad_left=0, pad_right=1, mode=mode)[1]
    return FLN_minus, FLN_plus


def quickest_flux(
    N: jax.Array,
    Cg: jax.Array,
    Dgrid: jax.Array,
    target_dim: int = 1,
    land_mask: Optional[jax.Array] = None,
    dt: float = 1.0,
    eps: float = 1e-9,
    mode: Optional[str] = None,
    boundary_mask: Optional[jax.Array] = None,
) -> Tuple[jax.Array, jax.Array]:
    """QUICKEST/TVD flux with absorbing, first-order shoreline faces.

    Parameters
    ----------
    mode : {"circular", "replicate"}, optional
        Stencil padding along ``target_dim`` for spectra, velocity and mask.
        None preserves the original axis-dependent padding defaults.
        This selects stencil neighbors; it does not refresh ghost cells.
    boundary_mask : array, optional
        Prescribed cells, including zero-spectrum ghosts. Only their incident
        faces use first-order upwind fluxes.
    """
    if mode is None:
        mode = "circular" if target_dim in [1, 3] else "replicate"

    N_im2, N_im1, N_i, N_ip1 = get_stencils(N, target_dim, pad_left=2, pad_right=1, mode=mode)
    Cg_im1, Cg_i = get_stencils(Cg, target_dim, pad_left=1, pad_right=0, mode=mode)
    N_stencils = (N_im2, N_im1, N_i, N_ip1)

    Dgrid_expanded = _broadcast_like(Dgrid, N_i)
    safe_Dgrid = jnp.clip(Dgrid_expanded, min=eps)
    Cg_b = 0.5 * (Cg_im1 + Cg_i)
    CFL = (Cg_b * dt) / safe_Dgrid

    is_positive = Cg_b >= 0
    CU_positive = (N_im2 - 2.0 * N_im1 + N_i) / (safe_Dgrid ** 2)
    CU_negative = (N_im1 - 2.0 * N_i + N_ip1) / (safe_Dgrid ** 2)
    CU = jnp.where(is_positive, CU_positive, CU_negative)

    Nb = 0.5 * ((1 + CFL) * N_im1 + (1 - CFL) * N_i)
    Nb = Nb - (1 - CFL ** 2) / 6.0 * CU * (Dgrid_expanded ** 2)
    Nb = _quickest_tvd_limiter(N, Cg_b, Nb, CFL, target_dim, mode, eps, N_stencils=N_stencils)

    if land_mask is not None:
        return _handle_land_boundaries(
            N,
            Cg,
            Cg_b,
            Nb,
            land_mask,
            target_dim,
            mode,
            N_stencils=N_stencils,
            Cg_stencils=(Cg_im1, Cg_i),
            boundary_mask=boundary_mask,
        )

    FLN_minus = Cg_b * Nb
    FLN_plus = get_stencils(FLN_minus, target_dim, pad_left=0, pad_right=1, mode=mode)[1]
    return FLN_minus, FLN_plus


def theta_flux(
    N: jax.Array,
    Cg: jax.Array,
    Dgrid: jax.Array,
    target_dim: int = 3,
    land_mask: Optional[jax.Array] = None,
    dt: float = 1.0,
    eps: float = 1e-9,
) -> Tuple[jax.Array, jax.Array]:
    """Theta-dimension flux with the specialized limiter from waden_hpc."""
    mode = "circular" if target_dim in [1, 3] else "replicate"

    N_im2, N_im1, N_i, N_ip1 = get_stencils(N, target_dim, pad_left=2, pad_right=1, mode=mode)
    Cg_im1, Cg_i, Cg_ip1 = get_stencils(Cg, target_dim, pad_left=1, pad_right=1, mode=mode)
    N_stencils = (N_im2, N_im1, N_i, N_ip1)

    Dgrid_expanded = _broadcast_like(Dgrid, N_i)
    safe_Dgrid = jnp.clip(Dgrid_expanded, min=eps)
    Cg_b = 0.5 * (Cg_im1 + Cg_i)
    CFL = (jnp.abs(Cg_b) * dt) / safe_Dgrid

    is_positive = Cg_b >= 0
    CU_positive = (N_im2 - 2.0 * N_im1 + N_i) / (safe_Dgrid ** 2)
    CU_negative = (N_im1 - 2.0 * N_i + N_ip1) / (safe_Dgrid ** 2)
    CU = jnp.where(is_positive, CU_positive, CU_negative)

    Nb = 0.5 * ((1 + CFL) * N_im1 + (1 - CFL) * N_i)
    Nb = Nb - (1 - CFL ** 2) / 6.0 * CU * (Dgrid_expanded ** 2)
    Nb = _theta_tvd_limiter(N, Cg_b, Nb, CFL, target_dim, mode, eps, N_stencils=N_stencils)

    if land_mask is not None:
        return _handle_land_boundaries(
            N,
            Cg,
            Cg_b,
            Nb,
            land_mask,
            target_dim,
            mode,
            N_stencils=N_stencils,
            Cg_stencils=(Cg_im1, Cg_i),
        )

    FLN_minus = Cg_b * Nb
    FLN_plus = get_stencils(FLN_minus, target_dim, pad_left=0, pad_right=1, mode=mode)[1]
    return FLN_minus, FLN_plus


def wavenumber_flux(
    N: jax.Array,
    Cg: jax.Array,
    wavenum: jax.Array,
    target_dim: int = 4,
    land_mask: Optional[jax.Array] = None,
    dt: float = 1.0,
    khf: Optional[float] = None,
    m: float = 5.0,
    khf_index: Optional[int] = None,
    eps: float = 1e-9,
) -> Tuple[jax.Array, jax.Array]:
    """Shared upwind faces for a density per k, never directly for E(f,theta).

    ``Cg`` retains its argument name for callers, but denotes k-dot in
    rad/m/s. Divide the returned divergence by k-cell widths to obtain a
    tendency. The low-k boundary is closed; high-k outflow is open. Explicit
    incoming tail data use WW3 (2016), Eq. (2.18), with f/f_hf=sqrt(k/k_hf)
    in deep water. A caller converting E(f) must use N_k=E_f*Cg/(2*pi*sigma).
    """
    del dt, eps
    velocity = jnp.broadcast_to(Cg, N.shape)
    previous, current = get_stencils(N, target_dim, 1, 0, "replicate")
    v_previous, _ = get_stencils(velocity, target_dim, 1, 0, "replicate")
    face_velocity = 0.5 * (v_previous + velocity)
    left = face_velocity * jnp.where(face_velocity >= 0, previous, current)
    left = left.at[_index_at_axis(N.ndim, target_dim, 0)].set(0)
    if land_mask is not None:
        left = jnp.where(_broadcast_like(land_mask.astype(bool), N), left, 0)
    right = get_stencils(left, target_dim, 0, 1, "replicate")[1]
    high_velocity = jnp.take(velocity, -1, axis=target_dim)
    incoming = jnp.zeros_like(high_velocity)
    if khf is not None and khf_index is not None:
        k = jnp.broadcast_to(wavenum, N.shape)
        face_k = 1.5 * jnp.take(k, -1, axis=target_dim) - 0.5 * jnp.take(k, -2, axis=target_dim)
        anchor_k = jnp.asarray(khf)
        ratio = jnp.where(anchor_k > 0, face_k / jnp.where(anchor_k > 0, anchor_k, 1), 1)
        incoming = jnp.take(N, khf_index, axis=target_dim) * ratio**(-(m + 2) / 2)
    high = high_velocity * jnp.where(high_velocity >= 0, jnp.take(N, -1, axis=target_dim), incoming)
    if land_mask is not None:
        high = jnp.where(jnp.take(_broadcast_like(land_mask.astype(bool), N), -1, axis=target_dim), high, 0)
    right = right.at[_index_at_axis(N.ndim, target_dim, -1)].set(high)
    return left, right


def calculate_flux(
    N: jax.Array,
    Cg: jax.Array,
    scheme: str = "first_order",
    target_dim: int = 1,
    **kwargs,
) -> Tuple[jax.Array, jax.Array]:
    """Factory function for all flux variants."""
    scheme = scheme.lower()
    if scheme == "first_order":
        return first_order_flux(N=N, Cg=Cg, target_dim=target_dim, **kwargs)
    if scheme == "quickest":
        return quickest_flux(N=N, Cg=Cg, target_dim=target_dim, **kwargs)
    if scheme == "theta":
        return theta_flux(N=N, Cg=Cg, target_dim=target_dim, **kwargs)
    if scheme == "wavenumber":
        return wavenumber_flux(N=N, Cg=Cg, target_dim=target_dim, **kwargs)
    raise ValueError(f"Unknown flux scheme: {scheme}")


def pr1flux(
    N: jax.Array,
    Cg: jax.Array,
    land_mask: jax.Array,
    target_dim: int = 1,
    mode: Optional[str] = None,
) -> Tuple[jax.Array, jax.Array]:
    return first_order_flux(
        N=N, Cg=Cg, land_mask=land_mask, target_dim=target_dim,
        **({"mode": mode} if mode is not None else {}),
    )


def qckflux(
    N: jax.Array,
    Cg: jax.Array,
    land_mask: jax.Array,
    Dgrid: jax.Array,
    dt: float = 1.0,
    target_dim: int = 1,
    mode: Optional[str] = None,
) -> Tuple[jax.Array, jax.Array]:
    return quickest_flux(
        N=N, Cg=Cg, Dgrid=Dgrid, land_mask=land_mask, dt=dt, target_dim=target_dim,
        **({"mode": mode} if mode is not None else {}),
    )


def qckflux4k(
    spectra: jax.Array,
    Cg: jax.Array,
    wavenum: jax.Array,
    mask: Optional[jax.Array] = None,
    dt: float = 1.0,
    khf: Optional[float] = None,
    m: float = 5.0,
    target_dim: int = 4,
    khf_index: Optional[int] = None,
) -> Tuple[jax.Array, jax.Array]:
    return wavenumber_flux(
        N=spectra,
        Cg=Cg,
        wavenum=wavenum,
        land_mask=mask,
        dt=dt,
        khf=khf,
        m=m,
        target_dim=target_dim,
        khf_index=khf_index,
    )
