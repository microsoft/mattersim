"""TorchSim wrapper for MatterSim models."""
# Inspired from https://github.com/TorchSim/torch-sim/blob/main/torch_sim/models/mattersim.py

from __future__ import annotations

import logging

import torch
import torch_sim as ts
from torch_sim.models.interface import ModelInterface
from torch_sim.units import MetalUnits

from mattersim.forcefield.potential import Potential
from mattersim.torchsim.graph_construction import build_graph_from_simstate

LOG = logging.getLogger(__name__)


class TorchSimWrapper(ModelInterface):
    """Computes atomistic energies, forces and stresses using a MatterSim model.

    This class wraps a MatterSim model to compute energies, forces, and
    stresses for atomistic systems.  It handles model initialization,
    configuration, and provides a forward pass that accepts a SimState object
    and returns model predictions.

    Examples:
        >>> model = TorchSimWrapper(model=loaded_mattersim_model)
        >>> results = model(state)
    """

    # Convert from GPa (MatterSim stress units) to eV/Angstrom^3 (TorchSim)
    GPa_to_eV_per_A3 = MetalUnits.pressure * 1e4

    def __init__(
        self,
        model: Potential,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        max_neighbors: int = 0,
    ) -> None:
        """Initialize the TorchSimWrapper.

        Args:
            model: The MatterSim Potential to wrap.
            device: Device to run the model on.
            dtype: Data type for outputs and optimizer state.
                The model weights always stay in float32; only the outputs are
                cast to this dtype.
            max_neighbors: Maximum number of neighbors per atom in the radius
                graph.  0 means no limit.
        """
        super().__init__()

        self._device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        if isinstance(self._device, str):
            self._device = torch.device(self._device)

        self._dtype = dtype or torch.float32
        self._memory_scales_with = "n_atoms_x_density"
        self._compute_stress = True
        self._compute_forces = True
        self._max_neighbors = max_neighbors

        self.model = model.to(self._device)
        self.model = self.model.eval()

        # Detect the model's native dtype from its parameters.
        # AOTI-compiled models have no nn.Parameters, so default to float32.
        first_param = next(self.model.parameters(), None)
        self._model_dtype = (
            first_param.dtype if first_param is not None else torch.float32
        )

        model_args = self.model.model.model_args
        self.two_body_cutoff = model_args["cutoff"]
        self.three_body_cutoff = model_args["threebody_cutoff"]

        self.implemented_properties = [
            "energy",
            "forces",
            "stress",
        ]

    def forward(self, state: ts.SimState) -> dict[str, torch.Tensor]:
        """Perform forward pass to compute energies, forces, and stresses.

        Builds the MatterSim graph input directly from SimState tensors,
        avoiding intermediate conversion to ase.Atoms objects.

        Args:
            state: SimState object containing positions, cells, atomic numbers,
                and other system information.

        Returns:
            dict: Model predictions containing:
                - energy (torch.Tensor): [batch_size]
                - forces (torch.Tensor): [n_atoms, 3]
                - stress (torch.Tensor): [batch_size, 3, 3] in eV/Angstrom^3
        """
        if state.device != self._device:
            state = state.to(self._device)

        output_dtype = state.dtype

        graph_input = build_graph_from_simstate(
            state,
            twobody_cutoff=self.two_body_cutoff,
            threebody_cutoff=self.three_body_cutoff,
            max_num_neighbors_threshold=self._max_neighbors,
        )

        # Cast graph tensors to model dtype without touching the SimState
        if output_dtype != self._model_dtype:
            graph_input = {
                k: v.to(dtype=self._model_dtype) if v.is_floating_point() else v
                for k, v in graph_input.items()
            }

        result = self.model.forward(
            graph_input,
            include_forces=self._compute_forces,
            include_stresses=self._compute_stress,
        )

        output = {
            "energy": result["total_energy"].to(dtype=output_dtype).detach(),
            "forces": (result["forces"].to(dtype=output_dtype).detach().reshape(-1, 3)),
            "stress": (
                self.GPa_to_eV_per_A3
                * result["stresses"].to(dtype=output_dtype).detach().reshape(-1, 3, 3)
            ),
        }

        return output
