# -*- coding: utf-8 -*-
"""Public API for WADEN's differentiable JAX wave dynamics core."""

from .boundary import (
	apply_boundary_condition,
	identity_boundary_update,
	update_absorb_boundary,
	update_absorb_boundary_global,
	update_close_boundary,
	update_period_boundary,
	update_simple_boundary,
)
from .flux import calculate_flux, first_order_flux, get_stencils, quickest_flux, theta_flux, wavenumber_flux
from .geometry import build_legacy_metadata, from_legacy_metadata, prepare_environment
from .diagnostics import physical_cfl, spectral_integral
from .gse_alleviation import GSESpatialAveraging
from .integrator import (
	CHECKPOINT_NONE,
	CHECKPOINT_PER_PHASE,
	CHECKPOINT_PER_STEP,
	apply_frequency_tail,
	model_advance,
	rollout,
	simple_stot_integrator,
	step,
)
from .physics import calculate_depth_dispersion_derivative, compute_group_velocity, solve_wavenumber
from .tendencies import (
	k_tendency,
	neural_k_tendency,
	neural_k_transport_velocity,
	neural_theta_tendency,
	neural_theta_transport_velocity,
	theta_tendency,
	theta_transport_velocity,
	wavenumber_transport_velocity,
	x_tendency,
	y_tendency,
)
from .typing import Environment, Grid, RolloutResult, SpectralGrid, State, StepConfig

__all__ = [
	"apply_frequency_tail",
	"physical_cfl",
	"spectral_integral",
	"CHECKPOINT_NONE",
	"CHECKPOINT_PER_PHASE",
	"CHECKPOINT_PER_STEP",
	"calculate_depth_dispersion_derivative",
	"calculate_flux",
	"compute_group_velocity",
	"apply_boundary_condition",
	"build_legacy_metadata",
	"Environment",
	"from_legacy_metadata",
	"identity_boundary_update",
	"first_order_flux",
	"get_stencils",
	"GSESpatialAveraging",
	"Grid",
	"k_tendency",
	"model_advance",
	"neural_k_tendency",
	"neural_k_transport_velocity",
	"neural_theta_tendency",
	"neural_theta_transport_velocity",
	"prepare_environment",
	"quickest_flux",
	"rollout",
	"RolloutResult",
	"solve_wavenumber",
	"simple_stot_integrator",
	"SpectralGrid",
	"State",
	"step",
	"StepConfig",
	"theta_tendency",
	"theta_flux",
	"theta_transport_velocity",
	"update_absorb_boundary",
	"update_absorb_boundary_global",
	"update_close_boundary",
	"update_period_boundary",
	"update_simple_boundary",
	"wavenumber_transport_velocity",
	"wavenumber_flux",
	"x_tendency",
	"y_tendency",
]
