# -*- coding: utf-8 -*-
# v2 bootstrap copy of the v1 physics core at 47a6558; Gin registration removed.
"""GSE spatial averaging for the JAX integrator (v2 bootstrap copy)."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from .geometry import cell_area


def _axis_source_target_slices(size: int, offset: int, periodic: bool) -> tuple[slice, slice]:
    if periodic:
        return slice(None), slice(None)
    return (
        slice(max(0, -offset), size - max(0, offset)),
        slice(max(0, offset), size - max(0, -offset)),
    )


def _target_valid_mask_for_source(
    valid_mask: jax.Array,
    offset_x: int,
    offset_y: int,
    periodic_x: bool = False,
    periodic_y: bool = False,
) -> jax.Array:
    _, nx, ny = valid_mask.shape
    src_x, tgt_x = _axis_source_target_slices(nx, offset_x, periodic_x)
    src_y, tgt_y = _axis_source_target_slices(ny, offset_y, periodic_y)
    target_lookup = valid_mask
    if periodic_x and offset_x != 0:
        target_lookup = jnp.roll(target_lookup, shift=-offset_x, axis=1)
    if periodic_y and offset_y != 0:
        target_lookup = jnp.roll(target_lookup, shift=-offset_y, axis=2)
    return jnp.zeros_like(valid_mask, dtype=bool).at[:, src_x, src_y].set(
        target_lookup[:, tgt_x, tgt_y]
    )


def _add_constant_offset(
    weights: jax.Array,
    offset_x: int,
    offset_y: int,
    value: jax.Array,
) -> jax.Array:
    return weights.at[..., offset_x + 1, offset_y + 1].add(value)


class GSESpatialAveraging:
    """Nine-point GSE transport of area-weighted spectral mass.

    Use validate_displacement during preparation and choose static substeps
    so all corners lie within the nine-point footprint (Tolman, 2002).
    No post-hoc global rescaling is used to claim conservation.
    """

    def __init__(
        self,
        enabled: bool = False,
        alpha_s: float = 1.5,
        alpha_n: float = 1.5,
        renormalize_land_blocked: bool = True,
        preserve_spectral_totals: bool = True,
        periodic_x: bool = False,
        periodic_y: bool = False,
        eps: float = 1e-8,
        max_corner_extent: float = 1.0,
        substeps: int = 1,
    ):
        self.enabled = enabled
        self.alpha_s = alpha_s
        self.alpha_n = alpha_n
        self.renormalize_land_blocked = renormalize_land_blocked
        self.preserve_spectral_totals = preserve_spectral_totals
        self.periodic_x = periodic_x
        self.periodic_y = periodic_y
        self.eps = eps
        self.max_corner_extent = max_corner_extent
        if not 0 < max_corner_extent <= 1 or not isinstance(substeps, int) or substeps < 1:
            raise ValueError("GSE requires 0 < max_corner_extent <= 1 and positive integer substeps")
        if alpha_s < 0 or alpha_n < 0:
            raise ValueError("GSE spread coefficients must be nonnegative")
        self.substeps = substeps

    def __call__(
        self,
        spectra: jax.Array,
        metadata: dict[str, Any],
        mask: jax.Array,
        dt: float,
    ) -> jax.Array:
        if not self.enabled or dt <= 0:
            return spectra

        valid_mask = self._normalize_mask(mask)
        Cg = self._expand_to_spectra(jnp.asarray(metadata["Cg"], dtype=spectra.dtype), spectra)

        weights = self.compute_average_weights(Cg, metadata, dt=dt / self.substeps)
        effective_weights = self.prepare_valid_scatter_weights(weights, valid_mask)
        area = cell_area(metadata)

        def body(_, current):
            mass = self.scatter_by_weights(current * area, effective_weights, valid_mask)
            return mass / area

        return jax.lax.fori_loop(0, self.substeps, body, spectra)

    def _normalize_mask(self, mask: jax.Array) -> jax.Array:
        mask = jnp.asarray(mask)
        if mask.ndim == 5:
            mask = mask[..., 0, 0]
        elif mask.ndim == 4 and mask.shape[-1] == 1:
            mask = mask[..., 0]
        return mask.astype(bool)

    def _expand_to_spectra(self, value: jax.Array, spectra: jax.Array) -> jax.Array:
        return jnp.broadcast_to(jnp.asarray(value, dtype=spectra.dtype), spectra.shape)

    def _frequency_ratio(self, metadata: dict[str, Any], ref: jax.Array) -> jax.Array:
        if "freq_ratio" in metadata:
            return jnp.clip(jnp.asarray(metadata["freq_ratio"], dtype=ref.dtype), min=self.eps)

        freq = jnp.asarray(metadata["freq"], dtype=ref.dtype).reshape(-1)
        if freq.size < 2:
            return jnp.asarray(1.1, dtype=ref.dtype)
        ratios = freq[1:] / jnp.clip(freq[:-1], min=self.eps)
        ratios = jnp.where(jnp.isfinite(ratios) & (ratios > self.eps), ratios, jnp.ones_like(ratios))
        return jnp.median(ratios)

    def _physical_grid_steps(self, metadata: dict[str, Any], ref: jax.Array) -> tuple[jax.Array, jax.Array]:
        dx = jnp.asarray(metadata["Dlon"], dtype=ref.dtype)
        dy = jnp.asarray(metadata["Dlat"], dtype=ref.dtype)
        if metadata.get("coordinate_type", "Cartesian") == "Spherical":
            r_earth = jnp.asarray(metadata.get("r_earth", 6366197.7236758135), dtype=ref.dtype)
            cos_lat = jnp.abs(jnp.asarray(metadata["cosLAT"], dtype=ref.dtype))
            dx = dx * r_earth * jnp.clip(cos_lat, min=self.eps)
            dy = dy * r_earth
        return jnp.clip(dx, min=self.eps), jnp.clip(dy, min=self.eps)

    def _corners(
        self,
        Cg: jax.Array,
        metadata: dict[str, Any],
        dt: float,
    ) -> tuple:
        Cg, theta = jnp.broadcast_arrays(Cg, jnp.asarray(metadata["theta"], dtype=Cg.dtype))
        dtheta = jnp.asarray(metadata["DTH"], dtype=Cg.dtype)
        dx, dy = self._physical_grid_steps(metadata, Cg)
        freq_ratio = self._frequency_ratio(metadata, Cg)

        sin_t = jnp.sin(theta)
        cos_t = jnp.cos(theta)
        stream_scale = (
            0.5
            * self.alpha_s
            * (freq_ratio - 1.0 / jnp.clip(freq_ratio, min=self.eps))
            * Cg
            * jnp.asarray(dt, dtype=Cg.dtype)
        )
        normal_scale = self.alpha_n * dtheta * Cg * jnp.asarray(dt, dtype=Cg.dtype)

        sx = stream_scale * sin_t / dx
        sy = stream_scale * cos_t / dy
        nx = -normal_scale * cos_t / dx
        ny = normal_scale * sin_t / dy

        corners = (
            (sx + nx, sy + ny),
            (-sx + nx, -sy + ny),
            (-sx - nx, -sy - ny),
            (sx - nx, sy - ny),
        )
        return corners

    def required_substeps(self, Cg: jax.Array, metadata: dict[str, Any], dt: float) -> jax.Array:
        """Return the static subdivision count to choose before compiling."""
        extent = jnp.max(jnp.stack([jnp.maximum(jnp.abs(x), jnp.abs(y))
                                    for x, y in self._corners(Cg, metadata, dt)]))
        return jnp.maximum(1, jnp.ceil(extent / self.max_corner_extent)).astype(jnp.int32)

    def validate_displacement(self, Cg: jax.Array, metadata: dict[str, Any], dt: float) -> None:
        """Check a concrete environment outside JIT; never silently clip offsets.

        dt is the interval of one GSE call, before this object's internal
        subdivision. With step/rollout use config.dt_gse/config.substeps;
        do not divide by self.substeps again here. __call__ is JIT-compatible
        and assumes this preflight was performed for the supplied environment.
        """
        required = int(self.required_substeps(Cg, metadata, dt))
        if self.substeps < required:
            raise ValueError(f"GSE footprint requires at least {required} substeps, got {self.substeps}")

    def compute_average_weights(self, Cg: jax.Array, metadata: dict[str, Any], dt: float) -> jax.Array:
        """Return nonnegative normalized weights within the validated footprint."""
        Cg = jnp.broadcast_arrays(Cg, jnp.asarray(metadata["theta"], dtype=Cg.dtype))[0]
        weights = jnp.zeros((*Cg.shape, 3, 3), dtype=Cg.dtype)
        weights = _add_constant_offset(weights, 0, 0, jnp.full(Cg.shape, 1.0 / 3.0, dtype=Cg.dtype))

        for rx, ry in self._corners(Cg, metadata, dt):
            abs_x = jnp.abs(rx)
            abs_y = jnp.abs(ry)
            max_abs = jnp.maximum(abs_x, abs_y)
            min_abs = jnp.minimum(abs_x, abs_y)

            center_w = (1.0 - max_abs) / 6.0
            axis_w = (max_abs - min_abs) / 6.0
            diag_w = min_abs / 6.0
            weights = _add_constant_offset(weights, 0, 0, center_w)

            sign_x = jnp.sign(rx)
            sign_y = jnp.sign(ry)
            x_major = abs_x >= abs_y
            y_major = ~x_major

            for ox in (-1, 1):
                mask_x = (x_major & (sign_x == ox)).astype(Cg.dtype)
                weights = _add_constant_offset(weights, ox, 0, axis_w * mask_x)
            for oy in (-1, 1):
                mask_y = (y_major & (sign_y == oy)).astype(Cg.dtype)
                weights = _add_constant_offset(weights, 0, oy, axis_w * mask_y)
            for ox in (-1, 1):
                for oy in (-1, 1):
                    mask_xy = ((sign_x == ox) & (sign_y == oy)).astype(Cg.dtype)
                    weights = _add_constant_offset(weights, ox, oy, diag_w * mask_xy)

        return weights

    def prepare_valid_scatter_weights(
        self,
        weights: jax.Array,
        valid_mask: jax.Array,
    ) -> jax.Array:
        effective = jnp.zeros_like(weights)
        availability = jnp.zeros_like(weights[..., 0, 0])

        for ix, ox in enumerate((-1, 0, 1)):
            for iy, oy in enumerate((-1, 0, 1)):
                target_mask = _target_valid_mask_for_source(
                    valid_mask,
                    ox,
                    oy,
                    periodic_x=self.periodic_x,
                    periodic_y=self.periodic_y,
                )[..., None, None]
                weighted = weights[..., ix, iy] * target_mask.astype(weights.dtype)
                effective = effective.at[..., ix, iy].set(weighted)
                availability = availability + weighted

        if self.renormalize_land_blocked:
            normalizer = jnp.clip(availability, min=self.eps)
            effective = jnp.where(
                availability[..., None, None] > self.eps,
                effective / normalizer[..., None, None],
                jnp.zeros_like(effective),
            )
        elif self.preserve_spectral_totals:
            # Blocked transfers remain in the source cell, rather than being
            # lost and restored by a global rescaling of unrelated cells.
            effective = effective.at[..., 1, 1].add(jnp.maximum(0, 1 - availability))
        return effective

    def scatter_by_weights(
        self,
        spectra: jax.Array,
        effective_weights: jax.Array,
        valid_mask: jax.Array,
    ) -> jax.Array:
        sea_source = valid_mask[..., None, None]
        source = jnp.where(sea_source, spectra, jnp.zeros_like(spectra))
        _, nx, ny, _, _ = spectra.shape
        out = jnp.zeros_like(spectra)

        for ix, ox in enumerate((-1, 0, 1)):
            for iy, oy in enumerate((-1, 0, 1)):
                contribution = source * effective_weights[..., ix, iy]
                if self.periodic_x and ox != 0:
                    contribution = jnp.roll(contribution, shift=ox, axis=1)
                if self.periodic_y and oy != 0:
                    contribution = jnp.roll(contribution, shift=oy, axis=2)
                src_x, tgt_x = _axis_source_target_slices(nx, ox, self.periodic_x)
                src_y, tgt_y = _axis_source_target_slices(ny, oy, self.periodic_y)
                out = out.at[:, tgt_x, tgt_y, :, :].add(contribution[:, src_x, src_y, :, :])

        return jnp.where(sea_source, out, jnp.zeros_like(out))

    def match_spectral_totals(
        self,
        reference: jax.Array,
        candidate: jax.Array,
        valid_mask: jax.Array,
        cell_areas: jax.Array | None = None,
    ) -> jax.Array:
        """Explicit rescaling utility; not used by conservative GSE transport.

        Supply cell_areas for nonuniform/spherical cells. None denotes unit
        Cartesian cells. Conservation tests exercise transport without this.
        """
        sea = valid_mask[..., None, None].astype(candidate.dtype)
        weights = sea if cell_areas is None else sea * cell_areas
        reference_total = jnp.sum(reference * weights, axis=(1, 2), keepdims=True)
        candidate_total = jnp.sum(candidate * weights, axis=(1, 2), keepdims=True)
        scale = jnp.where(
            jnp.abs(candidate_total) > self.eps,
            reference_total / jnp.clip(candidate_total, min=self.eps),
            jnp.ones_like(candidate_total),
        )
        corrected = candidate * scale
        return jnp.where(sea.astype(bool), corrected, jnp.zeros_like(corrected))


__all__ = ["GSESpatialAveraging"]
