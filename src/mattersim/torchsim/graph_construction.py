"""Build a MatterSim graph input dict directly from a TorchSim SimState.

This avoids the intermediate conversion to ``ase.Atoms`` objects and is used
by :class:`~mattersim.torchsim.torchsim_wrapper.TorchSimWrapper`.
"""

from __future__ import annotations

import torch
import torch_sim as ts

from mattersim.datasets.utils.converter import create_batch_graph_dict


def build_graph_from_simstate(
    sim_state: ts.SimState,
    *,
    twobody_cutoff: float = 5.0,
    threebody_cutoff: float = 4.0,
    max_num_neighbors_threshold: int = 0,
) -> dict[str, torch.Tensor]:
    """Build a MatterSim graph input dict directly from a TorchSim SimState.

    This is a thin wrapper around :func:`create_batch_graph_dict` that
    extracts the relevant tensors from a TorchSim SimState.

    Args:
        sim_state: A TorchSim SimState object containing positions, cell,
            atomic_numbers, pbc, and system_idx tensors.
        twobody_cutoff: Cutoff radius for two-body interactions, in Angstrom.
        threebody_cutoff: Cutoff radius for three-body interactions, in Angstrom.
        max_num_neighbors_threshold: Maximum number of neighbors per atom.
            0 means no limit.

    Returns:
        A dictionary containing the graph representation expected by
        MatterSim's ``Potential.forward()`` method.
    """
    device = sim_state.positions.device

    n_atoms_per_graph = torch.bincount(
        sim_state.system_idx, minlength=sim_state.n_systems
    ).to(device)

    # Expand pbc to [n_graphs, 3] if needed
    if sim_state.pbc.dim() == 1:
        pbc = sim_state.pbc.unsqueeze(0).expand(sim_state.n_systems, -1)
    else:
        pbc = sim_state.pbc

    return create_batch_graph_dict(
        pos=sim_state.wrap_positions,
        cell=sim_state.row_vector_cell,
        atomic_numbers=sim_state.atomic_numbers,
        num_atoms=n_atoms_per_graph,
        twobody_cutoff=twobody_cutoff,
        threebody_cutoff=threebody_cutoff,
        pbc=pbc,
        max_num_neighbors_threshold=max_num_neighbors_threshold,
    )
