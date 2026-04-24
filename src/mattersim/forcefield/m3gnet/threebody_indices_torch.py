"""
Pure PyTorch implementation of three-body index construction.

This module provides a GPU-friendly alternative to the Cython-based
three-body index computation, avoiding CPU roundtrips and Python loops.
"""

import torch


def compute_threebody_torch(
    edge_indices: torch.Tensor, n_atoms: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Calculate the three body indices from pair atom indices of the structure(s).

    The implementation is not limited to a single structure, it can handle a
    batch of structures as long as the indices of atoms are properly offsetted
    in edge_indices. For example, if the first structure has 5 atoms, and the
    second structure has 4 atoms, then the edge_indices for the second structure
    should use atom indices starting from 5. This way, this is equivalent to
    treating the batch of structures as a single large structure, but some atoms
    are not bonded.

    Args:
        edge_indices (torch.Tensor): [n_edges, 2]
        n_atoms (torch.Tensor): per-structure atom counts [n_structures]

    Returns:
        triple_bond_indices: [n_triples, 2] pairs of edge indices forming angles
        n_triple_ij: [n_edges] number of angles each edge participates in
        n_triple_i: [n_atoms] number of angles centered on each atom
        n_triple_s: [n_structures] number of angles per structure
    """
    total_atoms = int(n_atoms.sum().item())
    n_structures = len(n_atoms)

    # 0. check if the edge_indices is sorted by the first column (central atom)
    if not edge_indices.shape[1] == 2:
        raise ValueError("edge_indices must have shape [n_edges, 2].")
    if not torch.all(edge_indices[:-1, 0] <= edge_indices[1:, 0]):
        raise ValueError(
            "edge_indices must be sorted by the first column (central atom)."
        )

    # 1. count the number of bonds associated with each atom
    n_bond_per_atom = torch.bincount(edge_indices[:, 0], minlength=total_atoms)

    # 2. compute the number of ordered angles around each central atom
    # n_triple_i = k * (k - 1) where k is n_bond_per_atom
    n_triple_i = n_bond_per_atom * (n_bond_per_atom - 1)

    # 3. compute the number of angles that a bond is part of
    # Each bond participates in (k - 1) angles
    n_triple_ij = n_bond_per_atom[edge_indices[:, 0]] - 1

    # 4. Compute three body indices
    counts = n_bond_per_atom

    # Only consider atoms with at least 2 bonds (k >= 2)
    valid_mask = counts >= 2
    valid_counts = counts[valid_mask]

    # Find the starting position of each atom's bond group in the sorted list
    valid_starts = (torch.cumsum(counts, dim=0) - counts)[valid_mask]

    # For each valid atom with k bonds, generate k * (k - 1) pairs directly,
    # avoiding the k^2 generation + filtering approach.
    n_triple_per_atom = valid_counts * (valid_counts - 1)
    total_triples = n_triple_per_atom.sum()

    if total_triples > 0:
        # Create group IDs, repeating each ID k*(k-1) times.
        group_ids = torch.repeat_interleave(
            torch.arange(len(valid_counts), device=edge_indices.device),
            n_triple_per_atom,
        )

        # Calculate the local index (0 to k*(k-1) - 1) for each pair
        cum_triples = torch.cumsum(n_triple_per_atom, dim=0)
        starts_triples = cum_triples - n_triple_per_atom
        local_idx = (
            torch.arange(total_triples, device=edge_indices.device)
            - starts_triples[group_ids]
        )

        # Decode local index directly into (u, v) without diagonals.
        # For k bonds, enumerate k*(k-1) pairs: u in [0,k), v in [0,k), v != u
        # Using: u = i // (k-1), v_tmp = i % (k-1), v = v_tmp + (v_tmp >= u)
        k_vec = valid_counts[group_ids]
        k_minus_1 = k_vec - 1
        u = local_idx // k_minus_1
        v_tmp = local_idx % k_minus_1
        v = v_tmp + (v_tmp >= u).long()

        # Map the local indices (u, v) back to the original bond indices.
        group_start_indices = valid_starts[group_ids]
        bond_u = group_start_indices + u
        bond_v = group_start_indices + v

        # Stack to form the final list of bond pairs.
        triple_bond_indices = torch.stack([bond_u, bond_v], dim=1)
    else:
        triple_bond_indices = torch.zeros(
            (0, 2), dtype=torch.long, device=edge_indices.device
        )

    # 5. compute the total number of angles per structure via scatter sum
    atom_to_structure = torch.repeat_interleave(
        torch.arange(n_structures, device=edge_indices.device),
        n_atoms,
    )
    n_triple_s = torch.zeros(
        n_structures, dtype=n_triple_i.dtype, device=edge_indices.device
    )
    n_triple_s.scatter_add_(0, atom_to_structure, n_triple_i)

    return triple_bond_indices, n_triple_ij, n_triple_i, n_triple_s
