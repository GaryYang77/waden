"""Run an archived WW3 propagation experiment through the installed waden API."""

import argparse
from functools import partial
import json
from importlib.metadata import version
import os
from pathlib import Path
import platform
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
import waden

from case_io import load_case, moments, read_config


def setup(case, config, dtype):
    grid, spectral = waden.prepare_environment(
        case["x"], case["y"], case["theta"], case["freq"],
        coordinate_type=config["coordinate_type"], dtype=dtype)
    state = waden.State(jnp.asarray(case["initial"][None], dtype=dtype))
    environment = waden.Environment(
        depth=jnp.asarray(case["depth"][None, ..., None, None], dtype=dtype),
        mask=jnp.asarray(case["wet"][None]),
        boundary=jnp.where(jnp.asarray(case["boundary_mask"])[None, ..., None, None], state.spectra, 0),
        wind_u=jnp.zeros(state.spectra.shape[:3], dtype=dtype),
        wind_v=jnp.zeros(state.spectra.shape[:3], dtype=dtype),
        boundary_mask=jnp.asarray(case["boundary_mask"][None]))
    step_config = waden.StepConfig(**config["step_config"])
    options = dict(boundary_update_func=waden.identity_boundary_update,
                   x_tendency_fn=partial(waden.x_tendency, stencil_mode=config["x_stencil"]),
                   y_tendency_fn=partial(waden.y_tendency, stencil_mode="replicate"))
    return grid, spectral, state, environment, step_config, options


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True, help="Directory containing ww3_tp*/input")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    args = parser.parse_args()
    config = read_config(args.config)
    case = load_case(config, args.data_root)
    if args.dtype == "float64":
        jax.config.update("jax_enable_x64", True)
    dtype = getattr(jnp, args.dtype)
    grid, spectral, state, environment, step_config, options = setup(case, config, dtype)
    intervals = np.diff(case["times"]).astype("timedelta64[s]").astype(int)
    if not np.all(intervals == step_config.dt_global):
        raise ValueError("Case step must match every actual WW3 output interval")
    cfl = {k: float(v) for k, v in waden.physical_cfl(environment.depth, grid, spectral, step_config).items()}
    steps = len(intervals)
    def stack(value):
        return jnp.broadcast_to(value, (steps,) + value.shape)
    environments = jax.tree.map(stack, environment)
    # Static-depth reuse is explicit; all other fields retain the time axis.
    import dataclasses
    environments = dataclasses.replace(environments, depth=environment.depth)

    @jax.jit
    def advance(start, sequence):
        result = waden.rollout(start, sequence, grid, spectral, step_config,
                               fixed_depth=True, audit_clipping=True, **options)
        return result.spectra, result.diagnostics["clipping_mass"], result.final_state.spatial_phase

    start = time.perf_counter()
    spectra, clipping, phase = advance(state, environments)
    spectra = np.concatenate([np.asarray(state.spectra), np.asarray(spectra)[:, 0]], axis=0)
    elapsed = time.perf_counter() - start
    if not np.isfinite(spectra).all() or np.any(spectra < 0):
        raise ValueError("Nonfinite or negative predicted spectrum")
    diagnostics = moments(spectra, case["theta"], case["freq"])
    integral = np.asarray(waden.spectral_integral(jnp.asarray(spectra), grid, spectral,
                                                jnp.broadcast_to(environment.mask, spectra.shape[:3])))
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output / "initial_arrays.npz",
                        spectra=np.asarray(state.spectra),
                        depth=np.where(case["wet"], case["depth"], 0)[None],
                        lon=case["x"], lat=case["y"], theta_deg=case["theta"], freq=case["freq"],
                        wind_u=np.asarray(environment.wind_u), wind_v=np.asarray(environment.wind_v),
                        coordinate_type=config["coordinate_type"])
    np.savez_compressed(args.output / "prediction.npz", spectra=spectra,
                        time=case["times"].astype("datetime64[s]").astype(np.int64),
                        x=case["x"], y=case["y"], theta_deg=case["theta"], freq=case["freq"],
                        wet=case["wet"], boundary_mask=case["boundary_mask"], depth=case["depth"],
                        clipping_mass=np.asarray(clipping)[:, 0], integral=integral,
                        final_spatial_phase=np.asarray(phase), **diagnostics)
    record = dict(case=config["case"], config=config, reference=case["provenance"],
                  command=sys.argv, python=platform.python_version(), jax=jax.__version__,
                  numpy=np.__version__, waden_version=version("waden"),
                  waden_file=waden.__file__, dtype=args.dtype, prng="none (deterministic propagation)",
                  compilation={name: os.environ.get(name) for name in ("JAX_PLATFORMS", "XLA_FLAGS", "XLA_PYTHON_CLIENT_PREALLOCATE")},
                  devices=[str(d) for d in jax.devices()], cfl=cfl,
                  cfl_scope="Public all-grid diagnostic, including inactive/ghost cells; measured, not a stability certificate",
                  elapsed_compile_and_run_seconds=elapsed,
                  intervals=steps, initial_and_output_frames=len(spectra),
                  source="disabled", gse="disabled; see the case notes for archived WW3 settings",
                  numerical_acceptance="MEASURED; WW3 scientific thresholds are not frozen")
    (args.output / "run.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(dict(case=config["case"], frames=len(spectra), cfl=cfl,
                          seconds=elapsed, output=str(args.output)), indent=2))


if __name__ == "__main__":
    main()
