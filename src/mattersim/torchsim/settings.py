"""Settings classes for TorchSim batch runners."""

from dataclasses import dataclass, field
from typing import Callable, Literal, TypeAlias

import torch
import torch_sim as ts
from torch_sim.models.interface import ModelInterface
from torch_sim.units import MetalUnits

from mattersim.torchsim.settings_base import (
    IntegratorSettingsBase,
    OptimizerSettingsBase,
    TemperatureLike,
)

# Accept scalar temperatures, tensor temperatures, and 1D/2D torch tensors
TemperatureInput: TypeAlias = float | torch.Tensor

THERMOSTAT_SETTINGS = {
    "strong": 10,
    "default": 100,
}

BAROSTAT_SETTINGS = {
    "strong": 100,
    "default": 1000,
}


DTYPE = torch.float64

unit_system = ts.units.UnitSystem.metal

BYTES_PER_GB = 1024**3


def _cuda_if_available() -> Literal["cpu", "cuda"]:
    """Return ``"cuda"`` if available, otherwise ``"cpu"``."""
    return "cuda" if torch.cuda.is_available() else "cpu"


AUTOBATCHER_MAX_MEMORY_SCALER_PER_GB = {
    "mattersim-v1.0.0-1M": {
        "inflight": 11702.282205867403,
        "binning": 11702.282205867403,
    },
    "mattersim-v1.0.0-5M": {
        "inflight": 4569.9429648775285,
        "binning": 4569.9429648775285,
    },
}


def make_atom_count_batcher(max_natoms_per_batch: int) -> dict:
    """Return autobatcher config dict that limits batches by total atom count.

    Note: if any single structure has more atoms than *max_natoms_per_batch*,
    the autobatcher will raise ``ValueError`` when loading states.  Either
    increase the limit or filter out oversized structures beforehand.

    Args:
        max_natoms_per_batch: Maximum total atom count per batch.

    Returns:
        A dict suitable for ``OptimizerSettings(autobatcher=...)`` or
        ``IntegratorSettings(autobatcher=...)``.
    """
    return {
        "memory_scales_with": "n_atoms",
        "max_memory_scaler": max_natoms_per_batch,
    }


@dataclass(frozen=True)
class OptimizerSettings(OptimizerSettingsBase):
    """OptimizerSettings with torch-specific convenience methods.

    Inherits all settings from OptimizerSettingsBase and adds methods that
    require torch for tensor conversion and torch_sim for optimizer access.
    """

    device: Literal["cpu", "cuda"] = field(default_factory=_cuda_if_available)

    @property
    def cell_filter(self) -> ts.CellFilter:
        return ts.CellFilter[self.cell_filter_name]

    @property
    def optimizer(self) -> ts.Optimizer:
        return ts.Optimizer[self.name]

    @property
    def init_kwargs(self) -> dict:
        kwargs: dict = {"cell_filter": self.cell_filter}
        if self.scalar_pressure != 0.0:
            kwargs["scalar_pressure"] = self.scalar_pressure
        return kwargs

    def get_autobatcher(self, model: ModelInterface) -> bool | ts.InFlightAutoBatcher:
        if not torch.cuda.is_available():
            return False
        if isinstance(self.autobatcher, dict):
            return ts.InFlightAutoBatcher(model=model, **self.autobatcher)
        elif self.autobatcher is True:
            version = getattr(getattr(model, "model", None), "version", None)
            if version and version in AUTOBATCHER_MAX_MEMORY_SCALER_PER_GB:
                current_gpu_gb = (
                    torch.cuda.get_device_properties(0).total_memory / BYTES_PER_GB
                )
                max_memory_scaler = (
                    AUTOBATCHER_MAX_MEMORY_SCALER_PER_GB[version]["inflight"]
                    * current_gpu_gb
                )
                return ts.InFlightAutoBatcher(
                    model=model, max_memory_scaler=max_memory_scaler
                )
            # Fall back to torch-sim's default auto-detection
            return ts.InFlightAutoBatcher(model=model)
        return self.autobatcher

    @property
    def prop_calculators(self) -> dict[int, dict[str, Callable]]:
        return {
            self.save_checkpoint_every: {
                "stress": lambda state: state.stress,
                "potential_energy": lambda state: state.energy,
            }
        }


@dataclass(frozen=True)
class IntegratorSettings(IntegratorSettingsBase):
    """IntegratorSettings with torch-specific convenience methods.

    Inherits all settings from IntegratorSettingsBase and adds methods that
    require torch for tensor conversion and torch_sim for integrator access.
    """

    device: Literal["cpu", "cuda"] = field(default_factory=_cuda_if_available)

    def _to_tensor(
        self, value: float | None, dtype: torch.dtype = DTYPE
    ) -> torch.Tensor | None:
        """Convert float to tensor if not None."""
        if value is None:
            return None
        return torch.full([], fill_value=value, device=self.device, dtype=dtype)

    def get_autobatcher(self, model: ModelInterface) -> bool | ts.BinningAutoBatcher:
        if not torch.cuda.is_available():
            return False
        if isinstance(self.autobatcher, dict):
            return ts.BinningAutoBatcher(model=model, **self.autobatcher)
        elif self.autobatcher is True:
            version = getattr(getattr(model, "model", None), "version", None)
            if version and version in AUTOBATCHER_MAX_MEMORY_SCALER_PER_GB:
                current_gpu_gb = (
                    torch.cuda.get_device_properties(0).total_memory / BYTES_PER_GB
                )
                max_memory_scaler = (
                    AUTOBATCHER_MAX_MEMORY_SCALER_PER_GB[version]["binning"]
                    * current_gpu_gb
                )
                return ts.BinningAutoBatcher(
                    model=model, max_memory_scaler=max_memory_scaler
                )
            return ts.BinningAutoBatcher(model=model)
        return self.autobatcher

    @property
    def init_kwargs(self) -> dict:
        kwargs = {}
        param_map = {
            "npt_langevin": {"b_tau": self.b_tau, "alpha": self.alpha},
            "nvt_nose_hoover": {"tau": self.t_tau},
            "npt_nose_hoover": {"b_tau": self.b_tau, "t_tau": self.t_tau},
        }
        params = param_map.get(self.name, {})
        kwargs.update({k: self._to_tensor(v) for k, v in params.items()})

        if self.seed is not None:
            kwargs["seed"] = self.seed
        return kwargs

    @property
    def step_kwargs(self) -> dict:
        kwargs = {
            "n_steps": self.num_steps,
            "temperature": self.temperature_K,
            "timestep": self.timestep_ps,
        }
        if self.has_barostat:
            assert self.pressure_bar is not None
            kwargs["external_pressure"] = self.pressure_bar * unit_system.pressure
        if self.integrator == ts.Integrator.nvt_langevin and self.gamma is not None:
            kwargs["gamma"] = self.gamma
        return kwargs

    @property
    def integrator(self) -> ts.Integrator:
        return ts.Integrator[self.name]

    @classmethod
    def from_thermostat_barostat_settings(
        cls,
        name: str,
        temperature_K: TemperatureInput,
        num_steps: int,
        timestep_ps: float = 1e-3,
        thermostat_setting_name: str = "default",
        barostat_setting_name: str | None = None,
        pressure_bar: float | None = None,
        seed: int | None = None,
        save_checkpoint_every: int = 100,
        device: Literal["cpu", "cuda"] | None = None,
        autobatcher: dict | bool = True,
    ) -> "IntegratorSettings | list[IntegratorSettings]":
        """Convenience constructor to create IntegratorSettings from thermostat
        and barostat settings.

        Args:
            name: Name of the integrator (e.g., 'nvt_langevin', 'npt_nose_hoover').
            temperature_K: Temperature specification. Can be:
                - float: Constant temperature (returns single IntegratorSettings)
                - 1D Tensor: Temperature schedule (returns single IntegratorSettings)
                - 2D Tensor: Different schedule per system, shape [n_systems, n_steps]
                  (returns list of IntegratorSettings, one per system)
            num_steps: Number of MD steps to run.
            timestep_ps: Timestep in picoseconds.
            thermostat_setting_name: Thermostat coupling strength ("default" or "strong").
            barostat_setting_name: Barostat coupling strength ("default" or "strong"), or None.
            pressure_bar: External pressure in bar (required for NPT integrators).
            seed: Random seed for reproducibility.
            save_checkpoint_every: Save state every N steps.
            device: Device to run on. Defaults to CUDA if available, else CPU.
            autobatcher: Autobatcher configuration.

        Returns:
            Single IntegratorSettings or list of IntegratorSettings.
        """
        dt = timestep_ps * MetalUnits.time
        t_tau = None
        b_tau = None
        gamma = None
        alpha = None

        if thermostat_setting_name not in THERMOSTAT_SETTINGS:
            raise ValueError(
                f"Invalid thermostat setting name: {thermostat_setting_name}. "
                f"Valid options are: {list(THERMOSTAT_SETTINGS.keys())}"
            )
        if (
            barostat_setting_name is not None
            and barostat_setting_name not in BAROSTAT_SETTINGS
        ):
            raise ValueError(
                f"Invalid barostat setting name: {barostat_setting_name}. "
                f"Valid options are: {list(BAROSTAT_SETTINGS.keys())}"
            )
        thermostat_setting = THERMOSTAT_SETTINGS[thermostat_setting_name]
        barostat_setting = (
            BAROSTAT_SETTINGS[barostat_setting_name]
            if barostat_setting_name is not None
            else None
        )

        if name in ["nvt_nose_hoover", "npt_nose_hoover"]:
            t_tau = thermostat_setting * dt
            if barostat_setting is not None:
                b_tau = barostat_setting * dt
        elif name == "nvt_langevin":
            gamma = 1 / (thermostat_setting * dt)
        elif name == "npt_langevin":
            alpha = 1 / (thermostat_setting * dt)
            if barostat_setting is not None:
                b_tau = 1 / (barostat_setting * dt)

        return cls.with_per_system_temperatures(
            temperatures_K=temperature_K,
            name=name,
            num_steps=num_steps,
            timestep_ps=timestep_ps,
            t_tau=t_tau,
            b_tau=b_tau,
            gamma=gamma,
            alpha=alpha,
            pressure_bar=pressure_bar,
            seed=seed,
            save_checkpoint_every=save_checkpoint_every,
            device=device,
            autobatcher=autobatcher,
        )

    @classmethod
    def with_per_system_temperatures(
        cls,
        temperatures_K: TemperatureInput,
        name: str,
        num_steps: int,
        timestep_ps: float = 1e-3,
        t_tau: float | None = None,
        b_tau: float | None = None,
        gamma: float | None = None,
        alpha: float | None = None,
        pressure_bar: float | None = None,
        seed: int | None = None,
        save_checkpoint_every: int = 100,
        device: Literal["cpu", "cuda"] | None = None,
        autobatcher: dict | bool = True,
    ) -> "IntegratorSettings | list[IntegratorSettings]":
        """Create IntegratorSettings with the specified temperature(s).

        Args:
            temperatures_K: Temperature specification in Kelvin. Can be:
                - float: Constant temperature (returns single IntegratorSettings)
                - 1D Tensor: Temperature schedule (returns single IntegratorSettings)
                - 2D Tensor: Different schedule per system, shape [n_systems, n_steps]
                  (returns list of IntegratorSettings, one per system)
            name: Name of the integrator (e.g., 'nvt_langevin').
            num_steps: Number of MD steps to run.
            timestep_ps: Timestep in picoseconds.
            t_tau: Thermostat coupling time constant.
            b_tau: Barostat coupling time constant.
            gamma: Langevin friction coefficient.
            alpha: Langevin barostat coupling parameter.
            pressure_bar: External pressure in bar.
            seed: Random seed for reproducibility.
            save_checkpoint_every: Save state every N steps.
            device: Device to run on. Defaults to CUDA if available, else CPU.
            autobatcher: Autobatcher configuration.

        Returns:
            Single IntegratorSettings or list of IntegratorSettings.
        """
        resolved_device: Literal["cpu", "cuda"] = device or _cuda_if_available()

        def _make_settings(temp: TemperatureLike) -> "IntegratorSettings":
            return cls(
                temperature_K=temp,
                name=name,
                num_steps=num_steps,
                timestep_ps=timestep_ps,
                t_tau=t_tau,
                b_tau=b_tau,
                gamma=gamma,
                alpha=alpha,
                pressure_bar=pressure_bar,
                seed=seed,
                save_checkpoint_every=save_checkpoint_every,
                device=resolved_device,
                autobatcher=autobatcher,
            )

        def _validate_schedule(schedule: list[float], context: str = "") -> None:
            if len(schedule) != num_steps:
                raise ValueError(
                    f"Temperature schedule length ({len(schedule)}) must match "
                    f"num_steps ({num_steps}){context}."
                )

        if isinstance(temperatures_K, torch.Tensor):
            if temperatures_K.ndim == 0:
                return _make_settings(float(temperatures_K.item()))
            elif temperatures_K.ndim == 1:
                schedule = temperatures_K.tolist()
                _validate_schedule(schedule)
                return _make_settings(schedule)
            elif temperatures_K.ndim == 2:
                settings_list = []
                for i in range(temperatures_K.shape[0]):
                    schedule = temperatures_K[i].tolist()
                    _validate_schedule(schedule, f" for system {i}")
                    settings_list.append(_make_settings(schedule))
                return settings_list
            else:
                raise ValueError(
                    f"Tensor must be 0D (scalar), 1D (schedule), or 2D "
                    f"(per-system schedules), got {temperatures_K.ndim}D"
                )

        if isinstance(temperatures_K, (int, float)):
            return _make_settings(float(temperatures_K))

        raise ValueError(
            f"Invalid temperature type: {type(temperatures_K)}. "
            "Expected float or Tensor."
        )
