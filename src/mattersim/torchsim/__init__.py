"""TorchSim integration for MatterSim.

Provides a TorchSim-compatible model wrapper for MatterSim potentials,
enabling GPU-accelerated atomistic simulations via TorchSim.

Example usage::

    from mattersim.torchsim import TorchSimWrapper, get_torchsim_wrapper

    wrapper = get_torchsim_wrapper(potential="mattersim-v1.0.0-1M", device="cuda")
    result = wrapper(state)
"""

from mattersim.torchsim.model_loading import get_torchsim_wrapper
from mattersim.torchsim.torchsim_wrapper import TorchSimWrapper

__all__ = [
    "TorchSimWrapper",
    "get_torchsim_wrapper",
]
