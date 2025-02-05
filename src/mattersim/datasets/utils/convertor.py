# -*- coding: utf-8 -*-
import warnings
from typing import Optional, Tuple

import ase
import numpy as np
import torch
from ase import Atoms
from pymatgen.optimization.neighbors import find_points_in_spheres
from torch_geometric.data import Data

from .threebody_indices import compute_threebody as _compute_threebody

# Ensure the warning is only shown once
warnings.filterwarnings("once", category=UserWarning)

"""
Supported Properties:
    - "num_nodes"(set by default)  ## int
    - "num_edges"(set by default)  ## int
    - "num_atoms"                  ## int
    - "num_bonds"                  ## int
    - "atom_attr"                  ## tensor [num_atoms,atom_attr_dim=1]
    - "atom_pos"                   ## tensor [num_atoms,3]
    - "edge_length"                ## tensor [num_edges,1]
    - "edge_vector"                ## tensor [num_edges,3]
    - "edge_index"                 ## tensor [2,num_edges]
    - "three_body_indices"         ## tensor [num_three_body,2]
    - "num_three_body"              ## int
    - "num_triple_ij"              ## tensor [num_edges,1]
    - "num_triple_i"               ## tensor [num_atoms,1]
    - "num_triple_s"               ## tensor [1,1]
    - "theta_jik"                  ## tensor [num_three_body,1]
    - "triple_edge_length"         ## tensor [num_three_body,1]
    - "phi"                        ## tensor [num_three_body,1]
    - "energy"                     ## float
    - "forces"                     ## tensor [num_atoms,3]
    - "stress"                     ## tensor [3,3]
"""

"""
Computing various graph based operations (M3GNet)
"""


def compute_threebody_indices(
    bond_atom_indices: np.array,
    bond_length: np.array,
    n_atoms: int,
    atomic_number: np.array,
    threebody_cutoff: Optional[float] = None,
):
    """
    Given a graph without threebody indices, add the threebody indices
    according to a threebody cutoff radius
    Args:
        bond_atom_indices: np.array, [n_atoms, 2]
        bond_length: np.array, [n_atoms]
        n_atoms: int
        atomic_number: np.array, [n_atoms]
        threebody_cutoff: float, threebody cutoff radius

    Returns:
        triple_bond_indices, n_triple_ij, n_triple_i, n_triple_s

    """
    n_atoms = np.array(n_atoms).reshape(1)
    atomic_number = atomic_number.reshape(-1, 1)
    n_bond = bond_atom_indices.shape[0]
    if n_bond > 0 and threebody_cutoff is not None:
        valid_three_body = bond_length <= threebody_cutoff
        ij_reverse_map = np.where(valid_three_body)[0]
        original_index = np.arange(n_bond)[valid_three_body]
        bond_atom_indices = bond_atom_indices[valid_three_body, :]
    else:
        ij_reverse_map = None
        original_index = np.arange(n_bond)

    if bond_atom_indices.shape[0] > 0:
        bond_indices, n_triple_ij, n_triple_i, n_triple_s = _compute_threebody(
            np.ascontiguousarray(bond_atom_indices, dtype="int32"),
            np.array(n_atoms, dtype="int32"),
        )
        if ij_reverse_map is not None:
            n_triple_ij_ = np.zeros(shape=(n_bond,), dtype="int32")
            n_triple_ij_[ij_reverse_map] = n_triple_ij
            n_triple_ij = n_triple_ij_
        bond_indices = original_index[bond_indices]
        bond_indices = np.array(bond_indices, dtype="int32")
    else:
        bond_indices = np.reshape(np.array([], dtype="int32"), [-1, 2])
        if n_bond == 0:
            n_triple_ij = np.array([], dtype="int32")
        else:
            n_triple_ij = np.array([0] * n_bond, dtype="int32")
        n_triple_i = np.array([0] * len(atomic_number), dtype="int32")
        n_triple_s = np.array([0], dtype="int32")
    return bond_indices, n_triple_ij, n_triple_i, n_triple_s


def get_fixed_radius_bonding(
    structure: ase.Atoms,
    cutoff: float = 5.0,
    numerical_tol: float = 1e-8,
    pbc: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Get graph representations from structure within cutoff
    Args:
        structure (pymatgen Structure or molecule)
        cutoff (float): cutoff radius
        numerical_tol (float): numerical tolerance

    Returns:
        center_indices, neighbor_indices, images, distances
    """
    pbc_ = np.array(structure.pbc, dtype=int)

    lattice_matrix = np.ascontiguousarray(structure.cell[:], dtype=float)  # noqa: E501

    cart_coords = np.ascontiguousarray(
        np.array(structure.positions), dtype=float
    )  # noqa: E501
    r = float(cutoff)

    (
        center_indices,
        neighbor_indices,
        images,
        distances,
    ) = find_points_in_spheres(  # noqa: E501
        cart_coords,
        cart_coords,
        r=r,
        pbc=pbc_,
        lattice=lattice_matrix,
        tol=numerical_tol,
    )
    center_indices = center_indices.astype(np.int64)
    neighbor_indices = neighbor_indices.astype(np.int64)
    images = images.astype(np.int64)
    distances = distances.astype(float)
    exclude_self = (center_indices != neighbor_indices) | (
        distances > numerical_tol
    )  # noqa: E501
    return (
        center_indices[exclude_self],
        neighbor_indices[exclude_self],
        images[exclude_self],
        distances[exclude_self],
    )


class GraphConvertor:
    """
    Convert ase.Atoms to Graph
    """

    default_properties = ["num_nodes", "num_edges"]

    def __init__(
        self,
        model_type: str = "m3gnet",
        twobody_cutoff: float = 5.0,
        has_threebody: bool = True,
        threebody_cutoff: float = 4.0,
    ):
        self.model_type = model_type
        self.twobody_cutoff = twobody_cutoff
        self.threebody_cutoff = threebody_cutoff
        self.has_threebody = has_threebody

    def convert(
        self,
        atoms: Atoms,
        energy=None,
        forces=None,
        stress=None,
        pbc=True,
        **kwargs,
    ):
        """
        Convert the structure into graph
        Args:
            pbc: bool, whether to use periodic boundary condition, default True
        """
        # normalize the structure
        if isinstance(atoms, Atoms):
            pbc_ = np.array(atoms.pbc, dtype=int)
            if np.all(pbc_ < 0.01) or not pbc:
                min_x = np.min(atoms.positions[:, 0])
                min_y = np.min(atoms.positions[:, 1])
                min_z = np.min(atoms.positions[:, 2])
                max_x = np.max(atoms.positions[:, 0])
                max_y = np.max(atoms.positions[:, 1])
                max_z = np.max(atoms.positions[:, 2])      
                x_len = (max_x - min_x) + max(
                    self.twobody_cutoff, self.threebody_cutoff
                ) * 5
                y_len = (max_y - min_y) + max(
                    self.twobody_cutoff, self.threebody_cutoff
                ) * 5
                z_len = (max_z - min_z) + max(
                    self.twobody_cutoff, self.threebody_cutoff
                ) * 5
                max_len = max(x_len, y_len, z_len)
                x_len = y_len = z_len = max_len
                lattice_matrix = np.eye(3) * max_len
                pbc_ = np.array([1, 1, 1], dtype=int)
                warnings.warn(
                    "No PBC detected, using a large supercell with "
                    f"size {x_len}x{y_len}x{z_len} Angstrom**3",
                    UserWarning,
                )
               
                atoms.set_cell(lattice_matrix)
                atoms.set_pbc(pbc_)
            else:
                if np.all(abs(atoms.cell) < 1e-5):
                    raise ValueError("Cell vectors are too small")
        else:
            raise ValueError("structure type not supported")

        scaled_pos = atoms.get_scaled_positions()
        scaled_pos = np.mod(scaled_pos, 1)
        atoms.set_scaled_positions(scaled_pos)
        args = {}
        if self.model_type == "m3gnet":
            args["num_atoms"] = len(atoms)
            args["num_nodes"] = len(atoms)
            args["atom_attr"] = torch.FloatTensor(
                atoms.get_atomic_numbers()
            ).unsqueeze(  # noqa: E501
                -1
            )
            args["atom_pos"] = torch.FloatTensor(atoms.get_positions())
            args["cell"] = torch.FloatTensor(np.array(atoms.cell)).unsqueeze(0)
            (
                sent_index,
                receive_index,
                shift_vectors,
                distances,
            ) = get_fixed_radius_bonding(atoms, self.twobody_cutoff, pbc=pbc)
            args["num_bonds"] = len(sent_index)
            args["edge_index"] = torch.from_numpy(
                np.array([sent_index, receive_index])
            )  # noqa: E501
            args["pbc_offsets"] = torch.FloatTensor(shift_vectors)
            if self.has_threebody:
                (
                    triple_bond_index,
                    n_triple_ij,
                    n_triple_i,
                    n_triple_s,
                ) = compute_threebody_indices(
                    bond_atom_indices=args["edge_index"]
                    .numpy()
                    .transpose(1, 0),  # noqa: E501
                    bond_length=distances,
                    n_atoms=atoms.positions.shape[0],
                    atomic_number=atoms.get_atomic_numbers(),
                    threebody_cutoff=self.threebody_cutoff,
                )
                args["three_body_indices"] = torch.from_numpy(
                    triple_bond_index
                ).to(  # noqa: E501
                    torch.long
                )  # [num_three_body,2]
                args["num_three_body"] = args["three_body_indices"].shape[0]
                args["num_triple_ij"] = (
                    torch.from_numpy(n_triple_ij).to(torch.long).unsqueeze(-1)
                )
            else:
                args["three_body_indices"] = None
                args["num_three_body"] = None
                args["num_triple_ij"] = None
            if energy is not None:
                args["energy"] = torch.FloatTensor([energy])
            if forces is not None:
                args["forces"] = torch.FloatTensor(forces)
            if stress is not None:
                args["stress"] = torch.FloatTensor(stress).unsqueeze(0)
            return Data(**args)

        elif self.model_type == "graphormer":
            raise NotImplementedError
        else:
            raise NotImplementedError(
                "model type {} not implemented".format(self.model_type)
            )
