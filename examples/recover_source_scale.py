"""Recover a known source coefficient through WADEN on supplied wave arrays.

Synthetic observations are generated with coefficient 0.7. This controlled
source-only inverse problem illustrates parameter learning, not bathymetry
inversion or forecast skill. No neural checkpoint is needed.
"""

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import waden


def recover(arrays):
    with jax.experimental.enable_x64():
        grid, spectral = waden.prepare_environment(
            arrays["lon"], arrays["lat"], arrays["theta_deg"], arrays["freq"],
            coordinate_type=str(arrays.get("coordinate_type", "Spherical")), dtype=jnp.float64)
        depth = jnp.asarray(arrays["depth"], dtype=jnp.float64)[..., None, None]
        wet = depth[..., 0, 0] > 0
        initial = jnp.where(wet[..., None, None], jnp.asarray(arrays["spectra"], dtype=jnp.float64), 0)
        state = waden.State(initial)
        environment = waden.Environment(depth, wet, jnp.zeros_like(initial),
                                        jnp.asarray(arrays["wind_u"]), jnp.asarray(arrays["wind_v"]))
        config = waden.StepConfig(dt_global=3600, dt_x=0, dt_y=0, dt_th=0, dt_k=0,
                                  dt_stot=3600, dt_gse=0)
        normalization = jnp.sum(initial**2)
        if float(normalization) <= 0:
            raise ValueError("Source recovery needs a nonzero initial spectrum")

        def predict(coefficient):
            def source(spectrum, env, metadata):
                del spectrum, env, metadata
                return coefficient * initial / config.dt_global
            return waden.step(state, environment, grid, spectral, config, source_fn=source,
                               boundary_update_func=waden.identity_boundary_update).spectra

        target = predict(jnp.float64(0.7))

        def objective(coefficient):
            return jnp.sum((predict(coefficient)-target)**2) / normalization

        def update(coefficient, _):
            loss, gradient = jax.value_and_grad(objective)(coefficient)
            return coefficient-0.1*gradient, loss

        recovered, losses = jax.jit(lambda: jax.lax.scan(update, jnp.float64(0.2), None, length=100))()
        final_loss = float(objective(recovered))
        error = float(abs(recovered-0.7)/0.7)
        if not np.isfinite([float(recovered), final_loss]).all() or error > 0.01 or final_loss > 0.1*float(losses[0]):
            raise ValueError("The controlled source coefficient was not recovered")
        return dict(initial_coefficient=0.2, true_coefficient=0.7, recovered_coefficient=float(recovered),
                    steps=100, learning_rate=0.1, loss_before=float(losses[0]), loss_after=final_loss,
                    relative_parameter_error=error, dtype="float64",
                    scope="Synthetic source-only observations on supplied arrays; no depth-inversion or forecast-skill claim")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="NPZ using the physical-step example array contract")
    args = parser.parse_args()
    with np.load(args.input) as arrays:
        print(json.dumps(recover(dict(arrays)), indent=2))


if __name__ == "__main__":
    main()
