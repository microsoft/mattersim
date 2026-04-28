"""TorchSim wrapper loading utilities.

Provides :func:`get_torchsim_wrapper` for creating a TorchSim-compatible
model interface from various MatterSim potential inputs.
"""

from __future__ import annotations

import logging
from typing import Union

from mattersim.forcefield.aoti_compile import (
    AOTISettings,
    compile_m3gnet_aoti,
    load_aoti_model,
)
from mattersim.forcefield.potential import Potential, load_mattersim
from mattersim.torchsim.settings import DTYPE
from mattersim.torchsim.torchsim_wrapper import TorchSimWrapper

LOG = logging.getLogger(__name__)


TorchSimPotentialLike = Union[Potential, TorchSimWrapper, str]
"""Type alias for inputs accepted by :func:`get_torchsim_wrapper`.

- ``Potential``: an already-loaded MatterSim potential.
- ``TorchSimWrapper``: an already-wrapped model (returned as-is).
- ``str``: a checkpoint path or model identifier (e.g. ``"mattersim-v1.0.0-5M"``).
"""


def _apply_aoti(potential: Potential, aoti: AOTISettings) -> None:
    """Compile and swap the model with an AOTI-compiled version in-place."""
    if not aoti.enabled:
        return
    version_str = potential.version or "custom"
    package_path = compile_m3gnet_aoti(
        potential.model,
        version=version_str,
        device=potential.device,
        settings=aoti,
    )
    potential.model = load_aoti_model(
        package_path=package_path,
        model_args=potential.model.model_args,
        settings=aoti,
    )


def get_torchsim_wrapper(
    potential: TorchSimPotentialLike | None,
    device: str,
    gradient_checkpointing: bool = False,
    aoti: AOTISettings = AOTISettings(enabled=False),
    sanitize_nan: bool = False,
    max_neighbors: int = 0,
) -> TorchSimWrapper:
    """Get a TorchSimWrapper from various potential inputs.

    This is the main entry point for creating a TorchSim-compatible model
    from a MatterSim potential.

    Args:
        potential: The potential model to wrap. Can be:
            - ``None``: load default pre-trained model
            - ``str``: checkpoint path or model identifier
            - ``Potential``: an already-loaded model
            - ``TorchSimWrapper``: returned as-is (with updated settings)
        device: Device to run the model on.
        gradient_checkpointing: Enable gradient checkpointing.
        aoti: AOTI compilation settings.
        sanitize_nan: When True, replace non-finite model outputs with NaN.
        max_neighbors: Maximum number of neighbors per atom. 0 = no limit.
    """
    if isinstance(potential, TorchSimWrapper):
        if gradient_checkpointing:
            potential.model.enable_gradient_checkpointing(True)
        potential._sanitize_nan = sanitize_nan
        potential._max_neighbors = max_neighbors
        return potential
    if potential is None:
        potential = load_mattersim(
            gradient_checkpointing=gradient_checkpointing,
        )
    elif isinstance(potential, str):
        potential = load_mattersim(
            potential,
            gradient_checkpointing=gradient_checkpointing,
        )
    elif gradient_checkpointing:
        potential.enable_gradient_checkpointing(True)
    _apply_aoti(potential, aoti)
    return TorchSimWrapper(
        model=potential,
        device=device,
        dtype=DTYPE,
        sanitize_nan=sanitize_nan,
        max_neighbors=max_neighbors,
    )
