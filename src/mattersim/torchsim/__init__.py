"""TorchSim integration for MatterSim.

Provides batch molecular dynamics and structure relaxation using TorchSim
as the simulation backend.

Example usage::

    from mattersim.torchsim import TorchSimBatchRelaxer, OptimizerSettings

    settings = OptimizerSettings(name="fire", max_steps=500, fmax=0.01)
    relaxer = TorchSimBatchRelaxer.from_structures(structures, settings=settings)
    trajectories = relaxer.relax()
"""

from mattersim.torchsim.base import TorchSimBatchRunner
from mattersim.torchsim.batch_relax import TorchSimBatchRelaxer
from mattersim.torchsim.md import TorchSimBatchMD
from mattersim.torchsim.model_loading import get_torchsim_wrapper
from mattersim.torchsim.settings import IntegratorSettings, OptimizerSettings
from mattersim.torchsim.settings_base import (
    IntegratorSettingsBase,
    OptimizerSettingsBase,
    TemperatureLike,
)
from mattersim.torchsim.torchsim_wrapper import TorchSimWrapper

__all__ = [
    "IntegratorSettings",
    "IntegratorSettingsBase",
    "OptimizerSettings",
    "OptimizerSettingsBase",
    "TemperatureLike",
    "TorchSimBatchMD",
    "TorchSimBatchRelaxer",
    "TorchSimBatchRunner",
    "TorchSimWrapper",
    "get_torchsim_wrapper",
]
