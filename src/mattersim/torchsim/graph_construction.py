"""Build a MatterSim graph input dict directly from a TorchSim SimState.

This avoids the intermediate conversion to ``ase.Atoms`` objects and is used
by :class:`~mattersim.torchsim.torchsim_wrapper.TorchSimWrapper`.
"""

from __future__ import annotations

import torch
import torch_sim as ts

from mattersim.datasets.utils.converter import create_batch_graph_dict


def _normalize_nonperiodic_systems(
    pos: torch.Tensor,
    cell: torch.Tensor,
    pbc: torch.Tensor,
    system_idx: torch.Tensor,
    n_systems: int,
    twobody_cutoff: float,
    threebody_cutoff: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Normalize non-periodic systems for graph construction.

    For fully non-periodic systems (pbc=False in all directions), creates a
    large fake cubic cell and wraps positions into it, matching the behavior
    of :func:`~mattersim.datasets.utils.converter._normalize_atoms` in the
    ASE calculator code path.

    Args:
        pos: [total_atoms, 3] raw Cartesian positions.
        cell: [n_systems, 3, 3] unit cell matrices.
        pbc: [n_systems, 3] periodic boundary condition flags.
        system_idx: [total_atoms] system index per atom.
        n_systems: Number of systems in the batch.
        twobody_cutoff: Two-body cutoff radius in Angstrom.
        threebody_cutoff: Three-body cutoff radius in Angstrom.

    Returns:
        Tuple of (pos, cell, pbc) with non-periodic systems normalized.
    """
    non_periodic = ~pbc.any(dim=1)  # [n_systems]
    if not non_periodic.any():
        return pos, cell, pbc

    device = pos.device
    pos = pos.clone()
    cell = cell.clone()
    pbc = pbc.clone()
    pad = max(twobody_cutoff, threebody_cutoff) * 5.0

    for i in range(n_systems):
        if non_periodic[i]:
            mask = system_idx == i
            sys_pos = pos[mask]

            extent = sys_pos.max(dim=0).values - sys_pos.min(dim=0).values
            box_len = max(extent.max().item() + pad, pad)

            cell[i] = torch.eye(3, device=device, dtype=cell.dtype) * box_len
            pbc[i] = True

            # Wrap positions into the fake cell (equivalent to ASE atoms.wrap())
            frac = sys_pos / box_len
            pos[mask] = (frac % 1.0) * box_len

    return pos, cell, pbc


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

    # Normalize non-periodic systems: create a fake large periodic cell so
    # that graph construction matches the ASE calculator code path
    # (see _normalize_atoms in datasets/utils/converter.py).
    pos, cell, pbc = _normalize_nonperiodic_systems(
        pos=sim_state.positions,
        cell=sim_state.row_vector_cell,
        pbc=pbc,
        system_idx=sim_state.system_idx,
        n_systems=sim_state.n_systems,
        twobody_cutoff=twobody_cutoff,
        threebody_cutoff=threebody_cutoff,
    )

    return create_batch_graph_dict(
        pos=pos,
        cell=cell,
        atomic_numbers=sim_state.atomic_numbers,
        num_atoms=n_atoms_per_graph,
        twobody_cutoff=twobody_cutoff,
        threebody_cutoff=threebody_cutoff,
        pbc=pbc,
        max_num_neighbors_threshold=max_num_neighbors_threshold,
    )
