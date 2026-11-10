# Using WADEN

WADEN supplies the dynamics; applications supply wave fields, environmental
data, source terms and objectives. Neural networks, checkpoint restoration,
Gin and NetCDF readers are not core runtime dependencies.

## Input arrays

The [physical-step example](examples/physical_step.py) reads an NPZ with:

| Key | Shape / units |
|---|---|
| `spectra` | `(B, Nlon, Nlat, D, F)`, spectral energy density |
| `depth` | `(B, Nlon, Nlat)`, metres; nonpositive values denote land |
| `lon`, `lat` | one-dimensional coordinates, degrees |
| `theta_deg` | `D` going-to compass directions, degrees clockwise from north |
| `freq` | `F` frequency centres, Hz |
| `wind_u`, `wind_v` | `(B, Nlon, Nlat)`, m/s |

It uses a zero-source periodic configuration of 600 seconds and 20 substeps,
validated on NeuralSWM's real demo grid. Different grids and timesteps need
their own CFL/stability checks. The default `StepConfig()` setting of
3600 seconds and one substep has not been certified as a stable standalone
physical configuration on that grid.

`physical_step.py --steps N` repeats this stationary zero-source control
through `rollout`, preserving the spatial phase and writing the final spectrum.

The WW3 runner also exports `initial_arrays.npz`, with an additional
`coordinate_type` value. The neural-source example accepts this field to
select spherical or Cartesian coordinates.

## Single and multiple steps

`prepare_environment` constructs `Grid` and `SpectralGrid`. `State` contains
the spectrum and its spatial integration phase. `Environment` contains
depth, mask, prescribed boundary spectrum, winds and optional source and
`boundary_mask`. Explicit boundary masks can prescribe zero spectra.

Pass these objects and a `StepConfig` to `step`. For multiple steps, stack
environment fields along a leading time axis and call `rollout`. Preserve
the returned `final_state`, including its spatial phase, when resuming a
window. Callers supply and retain output times.

The grid, state, environment and configuration objects are JAX pytrees.
Use `jax.jit` around a function returning arrays or `State`;
`RolloutResult.times` is a host-side list.

### Fixed-depth reuse

If depth is constant throughout a window, keep it without a time axis and
enable reuse explicitly:

```python
import dataclasses
import waden

shared = dataclasses.replace(environments, depth=environments.depth[0])
result = waden.rollout(
    state, shared, grid, spectral_grid, config,
    fixed_depth=True, times=output_times,
)
```

Depth has shape `(B, Nlon, Nlat, 1, 1)`; other non-None fields retain their
time axis. `fixed_depth` defaults to `False`. Reuse stays inside the
differentiation graph, including dependence on depth.

## External source terms

`source_fn(spectrum, environment, metadata)` returns SI `dE/dt`.
It is evaluated once per global interval and its output is used by the
source integration. The application owns network parameters and calibration.

Inside a scalar training objective, close over the network parameters:

```python
from functools import partial
import jax
import waden

def source(parameters, spectrum, environment, metadata):
    return neural_source(parameters, spectrum, environment)  # SI dE/dt

def loss(parameters):
    advanced = waden.step(
        state, environment, grid, spectral_grid, config,
        source_fn=partial(source, parameters),
    )
    return objective(advanced.spectra, target)

loss_value, gradients = jax.value_and_grad(loss)(parameters)
```

This fragment assumes application-defined arrays, network and objective.
The complete [MLP example](examples/neural_source.py) uses a target equal to
1.001 times a zero-source step, computes parameter gradients and makes one
gradient-descent update. It illustrates coupling, not a calibrated source
model or forecast-skill result.

The example can save and resume its own four-array MLP parameters:

```bash
python examples/neural_source.py input.npz --checkpoint-out /tmp/source_step1.npz
python examples/neural_source.py input.npz --checkpoint-in /tmp/source_step1.npz \
  --checkpoint-out /tmp/source_step2.npz
```

These NPZ files belong to this small example. Applications retain their own
network architecture, pretrained-checkpoint format, optimizer and training loop.
The [source-coefficient recovery example](examples/recover_source_scale.py)
uses float64 to recover a known coefficient from synthetic source-only
observations. It does not invert water depth or establish forecast accuracy.

## Numerical and differentiation notes

Enable JAX x64 before requesting
`prepare_environment(..., dtype=jax.numpy.float64)`. Dispersion has an
implicit derivative. Limiters, clipping, masks and the deep-water cutoff
restrict differentiability to local regions away from switching surfaces.
Full float32 bathymetry inversion has not been validated.

`checkpoint_level` selects `none`, `per_step` or `per_phase`; the default is
`none`. The `none`/`per_step` training comparison covers one GPU, the official
NeuralSWM checkpoint/configuration, a fixed real float32 batch, three steps,
spectral MSE and one AdamW update from initialized optimizer state. It does
not establish long-term training agreement, nonzero optimizer-history
agreement, or support for other weights and multi-device training.

The high-frequency tail closure is off by default. Its cutoff and exponent
need application-specific scientific validation. GSE requires displacement
validation outside JIT at `dt=config.dt_gse/config.substeps`; its own
`substeps` further divides that interval.

## NeuralSWM compatibility

WADEN's distribution version is 0.2.0; its scientific model is referred to
as v2 in NeuralSWM. The original `physics_jax` v1 retains legacy-checkpoint
prediction behaviour. WADEN includes scientific corrections and has a
separate reference trajectory. Loading the same network parameters does
not make its predictions identical to v1. Existing NeuralSWM production
commands continue to use v1.
