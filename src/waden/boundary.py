# -*- coding: utf-8 -*-
# v2 bootstrap copy of the v1 physics core at 47a6558; Gin registration removed.
"""Pure JAX boundary utilities for NeuralWave."""

import jax
import jax.numpy as jnp


def _zeros_like_boundary(array: jax.Array) -> jax.Array:
    return jnp.zeros((), dtype=array.dtype)


def identity_boundary_update(array: jax.Array) -> jax.Array:
    """Keep all cells unchanged for a grid without ghost cells (all cells are physical).

    Parameters
    ----------
    array : jax.Array
        Field with shape ``(B, H, W, ...)``; all H x W cells are physical.

    Returns
    -------
    jax.Array
        The input array, including its physical edge cells.

    Notes
    -----
    Select periodic stencil modes separately in the spatial tendencies.
    This callback does not disable external boundary forcing or land masking.
    For periodic x/y transport on an all-physical-cell grid, select periodic
    stencil modes in the spatial tendencies and use this callback; the v2
    compatibility adapter supplies the corresponding configuration bindings.
    """
    return array


def update_period_boundary(array: jax.Array) -> jax.Array:
    """Refresh one periodic ghost layer on both spatial axes.

    The first/last row and column are ghost cells; physical cells occupy
    ``array[:, 1:-1, 1:-1, ...]``. Velocity, depth and mask fields must also
    follow this layout with consistent ghost values. For a grid containing
    only physical cells, use ``identity_boundary_update`` with periodic
    flux stencils instead.
    """
    updated = array
    updated = updated.at[:, 0:1, ...].set(updated[:, -2:-1, ...])
    updated = updated.at[:, -1:, ...].set(updated[:, 1:2, ...])
    updated = updated.at[:, :, 0:1, ...].set(updated[:, :, -2:-1, ...])
    updated = updated.at[:, :, -1:, ...].set(updated[:, :, 1:2, ...])
    return updated


def update_close_boundary(array: jax.Array) -> jax.Array:
    """Apply replicate boundary updates on all spatial edges."""
    updated = array
    updated = updated.at[:, 0:1, ...].set(updated[:, 1:2, ...])
    updated = updated.at[:, -1:, ...].set(updated[:, -2:-1, ...])
    updated = updated.at[:, :, 0:1, ...].set(updated[:, :, 1:2, ...])
    updated = updated.at[:, :, -1:, ...].set(updated[:, :, -2:-1, ...])
    return updated


def update_absorb_boundary(array: jax.Array) -> jax.Array:
    """Keep the interior region and zero out all spatial boundaries."""
    interior = jnp.zeros_like(array, dtype=bool)
    interior = interior.at[:, 1:-1, 1:-1, ...].set(True)
    return jnp.where(interior, array, _zeros_like_boundary(array))


def update_absorb_boundary_global(array: jax.Array) -> jax.Array:
    """Keep the interior latitude band and zero out the top and bottom edges."""
    interior = jnp.zeros_like(array, dtype=bool)
    interior = interior.at[:, :, 1:-1, ...].set(True)
    return jnp.where(interior, array, _zeros_like_boundary(array))


def update_simple_boundary(array: jax.Array) -> jax.Array:
    """Apply absorb boundaries in x and periodic boundaries in y."""
    updated = update_absorb_boundary(array)
    updated = updated.at[:, :, 0:1, ...].set(array[:, :, -2:-1, ...])
    updated = updated.at[:, :, -1:, ...].set(array[:, :, 1:2, ...])
    return updated


def apply_boundary_condition(
    spectra: jax.Array,
    boundary_condition: jax.Array,
    mask: jax.Array,
    boundary_mask: jax.Array | None = None,
) -> jax.Array:
    """Apply external spectra, including explicit zero Dirichlet values.

    ``boundary_mask`` selects prescribed spatial cells. None retains the
    existing positive-spectrum inference for callers without an explicit mask.
    """
    mask_expanded = jnp.expand_dims(jnp.expand_dims(mask.astype(bool), axis=-1), axis=-1)
    mask_expanded = jnp.broadcast_to(mask_expanded, spectra.shape)
    if boundary_mask is None:
        boundary_mask = jnp.sum(boundary_condition, axis=(-1, -2)) > 0
    boundary_mask_expanded = jnp.expand_dims(jnp.expand_dims(boundary_mask, axis=-1), axis=-1)
    boundary_mask_expanded = jnp.broadcast_to(boundary_mask_expanded, spectra.shape)
    spectra_masked = jnp.where(mask_expanded, spectra, _zeros_like_boundary(spectra))
    return jnp.where(boundary_mask_expanded, boundary_condition, spectra_masked)


__all__ = [
    "apply_boundary_condition",
    "identity_boundary_update",
    "update_absorb_boundary",
    "update_absorb_boundary_global",
    "update_close_boundary",
    "update_period_boundary",
    "update_simple_boundary",
]
