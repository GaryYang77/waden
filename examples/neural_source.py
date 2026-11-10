"""Differentiate a physical step through a small JAX neural source on real arrays.

The target is deliberately synthetic: a 0.1% spectral amplification of a
zero-source step. This demonstrates an end-to-end parameter gradient and
one update; it is not a calibrated source package or a forecast skill test.
"""

import argparse
from functools import partial
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import waden


def demonstrate(arrays, learning_rate=10.0, checkpoint_in=None, checkpoint_out=None):
    grid, spectral = waden.prepare_environment(arrays["lon"], arrays["lat"],
                                               arrays["theta_deg"], arrays["freq"],
                                               coordinate_type=str(arrays.get("coordinate_type", "Spherical")))
    depth = jnp.asarray(arrays["depth"])[..., None, None]
    mask = depth[..., 0, 0] > 0
    state = waden.State(jnp.where(mask[..., None, None], jnp.asarray(arrays["spectra"]), 0))
    environment = waden.Environment(depth=depth, mask=mask,
                                    boundary=jnp.zeros_like(state.spectra),
                                    wind_u=jnp.asarray(arrays["wind_u"]), wind_v=jnp.asarray(arrays["wind_v"]))
    config = waden.StepConfig(dt_global=60, dt_x=60, dt_y=60, dt_th=60,
                              dt_k=0, dt_stot=60, substeps=4)
    options = dict(boundary_update_func=waden.identity_boundary_update,
                   x_tendency_fn=partial(waden.x_tendency, stencil_mode="circular"),
                   y_tendency_fn=partial(waden.y_tendency, stencil_mode="circular"))
    key1, key2 = jax.random.split(jax.random.key(42))
    params = dict(w1=jax.random.normal(key1, (3, 8)) * 0.01, b1=jnp.zeros(8),
                  w2=jax.random.normal(key2, (8, 1)) * 0.01, b2=jnp.zeros(1))
    completed_updates = 0
    if checkpoint_in is not None:
        with np.load(checkpoint_in) as saved:
            params = {name: jnp.asarray(saved[name]) for name in params}
            completed_updates = int(saved["completed_updates"])

    def source(parameters, spectrum, env, metadata):
        del metadata
        wind = jnp.sqrt(env.wind_u**2 + env.wind_v**2)[..., None, None] / 10
        features = jnp.stack([jnp.log1p(spectrum),
                              jnp.broadcast_to(wind, spectrum.shape),
                              jnp.broadcast_to(jnp.log1p(env.depth / 100), spectrum.shape)], axis=-1)
        hidden = jnp.tanh(features @ parameters["w1"] + parameters["b1"])
        rate = jnp.tanh(hidden @ parameters["w2"] + parameters["b2"])[..., 0] / 3600
        # Output is dE/dt in SI units. The application owns this parametrization.
        return rate * spectrum

    baseline = jax.jit(lambda s: waden.step(s, environment, grid, spectral, config, **options))(state).spectra
    target = baseline * 1.001
    scale = jnp.sum(target**2)
    if float(scale) <= 0:
        raise ValueError("The example needs a nonzero real initial spectrum")

    def objective(parameters):
        result = waden.step(state, environment, grid, spectral, config,
                            source_fn=partial(source, parameters), **options)
        return jnp.sum((result.spectra - target)**2) / scale

    loss, gradients = jax.jit(jax.value_and_grad(objective))(params)
    # This rate belongs to this dimensionless toy objective. At 0.1, the
    # sparse WW3 pulse's source update can fall below float32 resolution.
    updated = jax.tree.map(lambda p, g: p - learning_rate * g, params, gradients)
    new_loss = jax.jit(objective)(updated)
    norms = {name: float(jnp.linalg.norm(value)) for name, value in gradients.items()}
    if not np.isfinite([float(loss), float(new_loss), *norms.values()]).all():
        raise ValueError("Nonfinite neural source objective or gradients")
    if not any(value > 0 for value in norms.values()) or not new_loss < loss:
        raise ValueError("The illustrative parameter update did not reduce the objective")
    if checkpoint_out is not None:
        checkpoint_out = Path(checkpoint_out)
        checkpoint_out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(checkpoint_out, **jax.tree.map(np.asarray, updated),
                            completed_updates=np.int64(completed_updates+1))
    return dict(loss_before=float(loss), loss_after=float(new_loss), learning_rate=learning_rate,
                parameter_gradient_l2=norms,
                checkpoint_in=str(checkpoint_in) if checkpoint_in is not None else None,
                checkpoint_out=str(checkpoint_out) if checkpoint_out is not None else None,
                completed_updates=completed_updates+1,
                scope="Illustrative synthetic target on supplied real arrays; no training-scope certification")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Same NPZ contract as physical_step.py")
    parser.add_argument("--learning-rate", type=float, default=10.0,
                        help="Rate for this normalized demonstration objective, not a training recommendation")
    parser.add_argument("--checkpoint-in", type=Path, help="Resume this example's four-array MLP checkpoint")
    parser.add_argument("--checkpoint-out", type=Path, help="Save updated MLP parameters and update count as NPZ")
    args = parser.parse_args()
    with np.load(args.input) as arrays:
        print(json.dumps(demonstrate(dict(arrays), args.learning_rate,
                                     args.checkpoint_in, args.checkpoint_out), indent=2))


if __name__ == "__main__":
    main()
