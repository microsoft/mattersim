"""GPU-accelerated batch graph converter for M3GNet.

Converts ASE Atoms to PyG Data objects with GPU-based neighbor search
and three-body index computation, avoiding CPU/GPU transfers after the
initial structural data is moved to the device.
"""

import os
import sys
from typing import Literal

import numpy as np
import torch
from ase import Atoms
from torch_geometric.data import Data
from tqdm import tqdm

from mattersim.datasets.utils.radius_graph_pbc import radius_graph_pbc_efficient
from mattersim.datasets.utils.threebody_indices_torch import compute_threebody_torch


class M3GNetData(Data):
    """Data subclass that tells PyG how to offset three_body_indices.

    ``three_body_indices`` contains pairs of *edge* indices (not node indices).
    PyG's default collation only auto-increments ``edge_index`` (by num_nodes).
    This override makes PyG increment ``three_body_indices`` by ``num_bonds``
    (= number of edges per graph) so that the batched tensor contains global
    edge indices ready for direct use in M3GNet — no manual offset needed.
    """

    def __inc__(self, key: str, value, *args, **kwargs):
        if key == "three_body_indices":
            return self.num_bonds
        return super().__inc__(key, value, *args, **kwargs)


def compute_threebody_indices_torch(
    edge_indices: torch.Tensor,
    distances: torch.Tensor,
    num_atoms: torch.Tensor,
    threebody_cutoff: float = 4.0,
):
    """Compute three-body indices from edge graph, filtering by cutoff.

    This is a wrapper around ``compute_threebody_torch`` that handles
    threebody cutoff filtering and index remapping.

    Args:
        edge_indices: [2, n_edges] edge index tensor
        distances: [n_edges] edge distances
        num_atoms: [n_structures] atoms per structure
        threebody_cutoff: cutoff radius for three-body interactions

    Returns:
        triple_bond_indices: [n_triples, 2] pairs of global edge indices
        n_triple_ij: [n_edges] triplets per edge
        n_triple_i: [total_atoms] triplets per atom
        n_triple_s: [n_structures] triplets per structure
    """
    num_edges = edge_indices.shape[1]
    total_num_atoms = num_atoms.sum().item()
    valid_edge_indices = None

    if num_edges > 0 and threebody_cutoff is not None:
        valid_three_body = distances <= threebody_cutoff
        ij_reverse_map = torch.where(valid_three_body)[0]
        original_index = torch.arange(num_edges, device=edge_indices.device)[
            valid_three_body
        ]
        valid_edge_indices = edge_indices[:, valid_three_body].transpose(0, 1)
    else:
        ij_reverse_map = None
        original_index = torch.arange(num_edges, device=edge_indices.device)

    if num_edges > 0 and valid_edge_indices is not None and valid_edge_indices.shape[0] > 0:
        (
            angle_indices,
            num_angles_per_edge,
            num_edges_per_atom,
            num_angles_per_structure,
        ) = compute_threebody_torch(
            valid_edge_indices,
            num_atoms,
        )
        if ij_reverse_map is not None:
            num_angles_per_edge_ = torch.zeros(
                (num_edges,), dtype=torch.long, device=edge_indices.device
            )
            num_angles_per_edge_[ij_reverse_map] = num_angles_per_edge
            num_angles_per_edge = num_angles_per_edge_
        angle_indices = original_index[angle_indices]
    else:
        angle_indices = torch.zeros(
            (0, 2), dtype=torch.long, device=edge_indices.device
        )
        if num_edges == 0:
            num_angles_per_edge = torch.zeros(
                (0,), dtype=torch.long, device=edge_indices.device
            )
        else:
            num_angles_per_edge = torch.zeros(
                (num_edges,), dtype=torch.long, device=edge_indices.device
            )
        num_edges_per_atom = torch.zeros(
            (total_num_atoms,), dtype=torch.long, device=edge_indices.device
        )
        num_angles_per_structure = torch.zeros(
            (num_atoms.shape[0],), dtype=torch.long, device=edge_indices.device
        )
    return (
        angle_indices,
        num_angles_per_edge,
        num_edges_per_atom,
        num_angles_per_structure,
    )


class BatchGraphConverter:
    """Convert a batch of ASE Atoms to M3GNet graphs on GPU.

    The converter:
    1. Normalizes structures on CPU (wrap positions)
    2. Moves positions/cell/atomic_numbers to GPU in one transfer
    3. Runs radius_graph_pbc_efficient on GPU for neighbor search
    4. Runs compute_threebody_indices_torch on GPU for three-body indices
    5. Returns list of M3GNetData objects (on GPU)

    Structures are processed in sub-batches (controlled by
    ``max_natoms_per_batch``) to avoid GPU OOM for large datasets.

    Args:
        model_type: Only "m3gnet" is supported.
        twobody_cutoff: Cutoff for two-body (edge) interactions in Angstrom.
        has_threebody: Whether to compute three-body indices.
        threebody_cutoff: Cutoff for three-body interactions in Angstrom.
        device: Target device. Defaults to CUDA if available.
    """

    def __init__(
        self,
        model_type: Literal["m3gnet"] = "m3gnet",
        twobody_cutoff: float = 5.0,
        has_threebody: bool = True,
        threebody_cutoff: float = 4.0,
        device: str | torch.device | None = None,
    ):
        self.model_type = model_type
        self.twobody_cutoff = twobody_cutoff
        self.threebody_cutoff = threebody_cutoff
        self.has_threebody = has_threebody
        if device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device
        if model_type != "m3gnet":
            raise NotImplementedError(
                "BatchGraphConverter only supports m3gnet model type"
            )

    def convert(
        self,
        atoms_list: list[Atoms],
        *,
        energy: list[float] | None = None,
        forces: list[np.ndarray] | None = None,
        stresses: list[np.ndarray] | None = None,
        max_natoms_per_batch: int = 8192,
    ) -> list[M3GNetData]:
        """Convert a list of ASE Atoms to M3GNetData objects on GPU.

        Args:
            atoms_list: Structures to convert.
            energy: Optional per-structure energies.
            forces: Optional per-structure forces.
            stresses: Optional per-structure stresses.
            max_natoms_per_batch: Max atoms per sub-batch (for GPU memory).

        Returns:
            List of M3GNetData objects on ``self.device``.
        """
        graphs: list[M3GNetData] = []
        pointer = 0
        pbar = tqdm(
            total=len(atoms_list),
            desc="Converting to graphs",
            disable=os.environ.get("DEBUG") not in ["1", "DEBUG"],
        )

        while pointer < len(atoms_list):
            # Accumulate a sub-batch
            natoms = torch.zeros((0,), dtype=torch.long, device=self.device)
            pos = torch.zeros((0, 3), dtype=torch.float32, device=self.device)
            cell = torch.zeros(
                (0, 3, 3), dtype=torch.float32, device=self.device
            )
            atomic_numbers = torch.zeros(
                (0,), dtype=torch.long, device=self.device
            )
            pbc = torch.zeros((0, 3), dtype=torch.bool, device=self.device)
            natoms_cumsum = 0
            num_graphs = 0

            while (
                pointer < len(atoms_list)
                and natoms_cumsum + len(atoms_list[pointer])
                <= max_natoms_per_batch
            ):
                atoms = atoms_list[pointer].copy()
                atoms.wrap()
                natoms = torch.cat(
                    (
                        natoms,
                        torch.tensor(
                            [len(atoms)],
                            dtype=torch.long,
                            device=self.device,
                        ),
                    ),
                )
                pos = torch.cat(
                    (
                        pos,
                        torch.tensor(
                            atoms.get_positions(),
                            dtype=torch.float32,
                            device=self.device,
                        ),
                    ),
                )
                cell = torch.cat(
                    (
                        cell,
                        torch.tensor(
                            np.array(atoms.cell),
                            dtype=torch.float32,
                            device=self.device,
                        ).unsqueeze(0),
                    ),
                )
                atomic_numbers = torch.cat(
                    (
                        atomic_numbers,
                        torch.tensor(
                            atoms.get_atomic_numbers(),
                            dtype=torch.long,
                            device=self.device,
                        ),
                    ),
                )
                pbc = torch.cat(
                    (
                        pbc,
                        torch.tensor(
                            np.array([atoms.pbc], dtype=bool),
                            dtype=torch.bool,
                            device=self.device,
                        ),
                    ),
                )
                natoms_cumsum += len(atoms)
                pointer += 1
                num_graphs += 1
                pbar.update(1)

            if num_graphs == 0:
                # Single structure exceeds max_natoms_per_batch; process it alone
                num_graphs = 1
                atoms = atoms_list[pointer].copy()
                atoms.wrap()
                natoms = torch.tensor(
                    [len(atoms)], dtype=torch.long, device=self.device
                )
                pos = torch.tensor(
                    atoms.get_positions(),
                    dtype=torch.float32,
                    device=self.device,
                )
                cell = torch.tensor(
                    np.array(atoms.cell),
                    dtype=torch.float32,
                    device=self.device,
                ).unsqueeze(0)
                atomic_numbers = torch.tensor(
                    atoms.get_atomic_numbers(),
                    dtype=torch.long,
                    device=self.device,
                )
                pbc = torch.tensor(
                    np.array([atoms.pbc], dtype=bool),
                    dtype=torch.bool,
                    device=self.device,
                )
                natoms_cumsum += len(atoms)
                pointer += 1
                pbar.update(1)

            # GPU: neighbor search
            edge_indices, offsets, num_edges, _, distances = (
                radius_graph_pbc_efficient(
                    pos=pos,
                    pbc=pbc,
                    cell=cell,
                    natoms=natoms,
                    radius=self.twobody_cutoff,
                    max_cell_images_per_dim=sys.maxsize,
                )
            )
            # Swap to MatterSim convention: (source, target)
            edge_indices = torch.cat(
                (edge_indices[1].unsqueeze(0), edge_indices[0].unsqueeze(0)),
                dim=0,
            )
            num_edges = num_edges.to(torch.long)

            # GPU: three-body indices
            if self.has_threebody:
                (
                    triple_bond_indices,
                    n_triple_ij,
                    n_triple_i,
                    n_triple_s,
                ) = compute_threebody_indices_torch(
                    edge_indices=edge_indices,
                    distances=distances,
                    num_atoms=natoms,
                    threebody_cutoff=self.threebody_cutoff,
                )

            # Split sub-batch into individual Data objects
            start_edge = 0
            start_atom = 0
            for i in range(num_graphs):
                n_atoms_i = natoms[i].item()
                n_edges_i = num_edges[i].item()
                graph = {}
                graph["num_atoms"] = n_atoms_i
                graph["num_nodes"] = n_atoms_i
                graph["atom_attr"] = (
                    atomic_numbers[start_atom : start_atom + n_atoms_i]
                    .unsqueeze(-1)
                    .to(torch.float32)
                )
                graph["atom_pos"] = pos[start_atom : start_atom + n_atoms_i]
                graph["cell"] = cell[i].unsqueeze(0)
                graph["num_bonds"] = n_edges_i
                graph["edge_index"] = (
                    edge_indices[:, start_edge : start_edge + n_edges_i]
                    - start_atom
                )
                graph["pbc_offsets"] = offsets[
                    start_edge : start_edge + n_edges_i
                ]
                if self.has_threebody:
                    n_triple_ij_i = n_triple_ij[
                        start_edge : start_edge + n_edges_i
                    ]
                    mask = (triple_bond_indices[:, 0] >= start_edge) & (
                        triple_bond_indices[:, 0] < start_edge + n_edges_i
                    )
                    triple_bond_indices_i = (
                        triple_bond_indices[mask, :] - start_edge
                    )
                    graph["three_body_indices"] = triple_bond_indices_i
                    graph["num_three_body"] = triple_bond_indices_i.shape[0]
                    graph["num_triple_ij"] = n_triple_ij_i.unsqueeze(-1)
                else:
                    graph["three_body_indices"] = None
                    graph["num_three_body"] = None
                    graph["num_triple_ij"] = None

                if (
                    energy is not None
                    and energy[pointer - num_graphs + i] is not None
                ):
                    graph["energy"] = torch.tensor(
                        [energy[pointer - num_graphs + i]],
                        dtype=torch.float32,
                        device=self.device,
                    )
                if (
                    forces is not None
                    and forces[pointer - num_graphs + i] is not None
                ):
                    graph["forces"] = torch.tensor(
                        forces[pointer - num_graphs + i],
                        dtype=torch.float32,
                        device=self.device,
                    )
                if (
                    stresses is not None
                    and stresses[pointer - num_graphs + i] is not None
                ):
                    graph["stress"] = torch.tensor(
                        stresses[pointer - num_graphs + i],
                        dtype=torch.float32,
                        device=self.device,
                    ).unsqueeze(0)

                graphs.append(M3GNetData(**graph).to(self.device))
                start_edge += n_edges_i
                start_atom += n_atoms_i

        pbar.close()
        return graphs
