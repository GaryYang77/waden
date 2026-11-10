# -*- coding: utf-8 -*-
"""Run a periodic zero-source physical step on caller-supplied wave arrays."""

import argparse
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import waden


def advance(arrays: dict[str, np.ndarray], steps: int = 1) -> np.ndarray:
    """Advance a real wave field using the standalone demo control settings.

    Parameters
    ----------
    arrays : dict[str, np.ndarray]
        Arrays documented in the package README. Directions are going-to
        compass degrees; depth is in metres, frequency in Hz, wind in m/s.

    Returns
    -------
    np.ndarray
        Final spectrum after ``steps * 600`` seconds, shape (B, Nlon, Nlat, D, F).
    """
    grid, spectral = waden.prepare_environment(
        arrays["lon"], arrays["lat"], arrays["theta_deg"], arrays["freq"]
    )
    depth = jnp.asarray(arrays["depth"])[..., None, None]
    wet = depth[..., 0, 0] > 0
    state = waden.State(jnp.where(wet[..., None, None], jnp.asarray(arrays["spectra"]), 0))
    environment = waden.Environment(
        depth=depth, mask=wet, boundary=jnp.zeros_like(state.spectra),
        wind_u=jnp.asarray(arrays["wind_u"]), wind_v=jnp.asarray(arrays["wind_v"]),
    )
    config = waden.StepConfig(
        dt_global=600, dt_x=600, dt_y=600, dt_th=600, dt_k=600,
        dt_stot=0, substeps=20,
    )
    options = dict(
        boundary_update_func=waden.identity_boundary_update,
        x_tendency_fn=partial(waden.x_tendency, stencil_mode="circular"),
        y_tendency_fn=partial(waden.y_tendency, stencil_mode="circular"),
    )
    if steps < 1:
        raise ValueError("steps must be positive")
    if steps == 1:
        result = jax.jit(lambda s, e: waden.step(s, e, grid, spectral, config, **options))(state, environment)
    else:
        # A stationary zero-source control, not a time-varying forecast.
        sequence = jax.tree.map(lambda value: jnp.broadcast_to(value, (steps,) + value.shape), environment)
        result = jax.jit(lambda s, e: waden.rollout(s, e, grid, spectral, config, **options).final_state)(state, sequence)
    return np.asarray(result.spectra)


def main() -> None:
    """Read an NPZ wave field and write its advanced spectrum."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--steps", type=int, default=1, help="Number of 600-second stationary-control steps")
    args = parser.parse_args()
    with np.load(args.input) as values:
        spectra = advance(dict(values), steps=args.steps)
    np.savez_compressed(args.output, spectra=spectra)


if __name__ == "__main__":
    main()
