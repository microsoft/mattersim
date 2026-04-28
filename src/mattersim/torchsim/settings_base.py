"""Base settings classes for TorchSim runners (without dependencies on torch or torchsim).

Essentially annotated dictionaries with validation logic which can be used
in database schemas.
"""

from dataclasses import dataclass

# Temperature is automatically passed as a separate keyword argument to the
# integrate function, therefore not present here.
INTEGRATOR_PARAMS: dict[str, set[str]] = {
    "nve": set(),
    "nvt_langevin": {"gamma"},
    "npt_langevin": {"alpha", "b_tau", "pressure_bar"},
    "nvt_nose_hoover": {"t_tau"},
    "npt_nose_hoover": {"t_tau", "b_tau", "pressure_bar"},
}


# Type alias for temperature — can be a scalar or 1D array for time-varying
# temperature schedules.  Shape is [n_timesteps] for a schedule, or a scalar
# for constant temperature.
TemperatureLike = float | list[float]


@dataclass(frozen=True)
class IntegratorSettingsBase:
    """Base settings for molecular dynamics integrators.

    This class contains only the pure data settings without any torch
    dependencies, making it suitable for use in database schemas and
    serialization.
    """

    # NOTE: TorchSim works in metal units internally.
    name: str  # Name of the integrator (e.g., 'nvt_langevin')
    temperature_K: TemperatureLike
    num_steps: int
    timestep_ps: float = 1e-3
    t_tau: float | None = None
    b_tau: float | None = None
    gamma: float | None = None
    alpha: float | None = None
    pressure_bar: float | None = None
    seed: int | None = None
    save_checkpoint_every: int = 100
    autobatcher: dict | bool = True

    @property
    def simulation_time_ps(self) -> float:
        """Total simulation time in picoseconds."""
        return self.num_steps * self.timestep_ps

    @property
    def has_barostat(self) -> bool:
        return self.name in ["npt_nose_hoover", "npt_langevin"]

    def __post_init__(self) -> None:
        """Validate integrator settings."""
        if self.has_barostat:
            if self.pressure_bar is None:
                raise ValueError(
                    f"Pressure must be specified for integrator {self.name}"
                )

        if self.num_steps % self.save_checkpoint_every != 0:
            raise ValueError(
                "num_steps must be a multiple of save_checkpoint_every, "
                "otherwise the final state won't be saved."
            )

        # Validate parameters
        allowed_params = INTEGRATOR_PARAMS.get(self.name, set())
        provided_params = {
            param
            for param in ["gamma", "alpha", "t_tau", "b_tau", "pressure_bar"]
            if getattr(self, param, None) is not None
        }
        invalid = provided_params - allowed_params
        if invalid:
            raise ValueError(
                f"Parameters {invalid} not valid for {self.name}. "
                f"Allowed: {allowed_params}"
            )


@dataclass(frozen=True)
class OptimizerSettingsBase:
    """Base settings for structure optimization.

    This class contains only the pure data settings without any torch
    dependencies, making it suitable for use in database schemas and
    serialization.
    """

    name: str
    max_steps: int
    steps_between_swaps: int = 1
    fmax: float = 1e-1
    cell_filter_name: str = "frechet"
    constrain_symmetry: bool = False
    scalar_pressure: float = 0.0
    save_checkpoint_every: int = 1
    autobatcher: dict | bool = True

    def __post_init__(self) -> None:
        """Validate optimizer settings."""
        if self.max_steps % self.save_checkpoint_every != 0:
            raise ValueError(
                "max_steps must be a multiple of save_checkpoint_every, "
                "otherwise the final state won't be saved."
            )
