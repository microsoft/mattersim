import torch


def compute_threebody_torch(edge_indices: torch.Tensor, n_atoms: int):
    """
    Calculate the three body indices from pair atom indices of the structure(s).

    The implementation is not limited to a single structure, it can handle a batch of structures
    as long as the indices of atoms are properly offsetted in edge_indices.
    For example, if the first structure has 5 atoms, and the second structure has 4 atoms,
    then the edge_indices for the second structure should use atom indices starting from 5.
    This way, this is equivalent to treating the batch of structures as a single large structure,
    but some atoms are not bonded.

    Args:
        edge_indices (torch.Tensor): [n_edges, 2]
        n_atoms (int): total number of atoms
    """
    # 0. check if the edge_indices is sorted by the first column (central atom)
    if not edge_indices.shape[1] == 2:
        raise ValueError("edge_indices must have shape [n_edges, 2].")
    if not torch.all(edge_indices[:-1, 0] <= edge_indices[1:, 0]):
        raise ValueError("edge_indices must be sorted by the first column (central atom).")

    # 1. count the number of bonds associated with each atom
    # shape: (n_atoms,)
    n_bond_per_atom = torch.bincount(edge_indices[:, 0], minlength=n_atoms)

    # 2. compute the number of ordered angles around each central atom
    # n_triple_i = k * (k - 1) where k is n_bond_per_atom
    n_triple_i = n_bond_per_atom * (n_bond_per_atom - 1)

    # 3. compute the number of angles that a bond is part of
    # Each bond participates in (k - 1) angles
    n_triple_ij = n_bond_per_atom[edge_indices[:, 0]] - 1

    # 4. Compute three body indices
    # Get the number of bonds for each atom (k).
    counts = n_bond_per_atom

    # We only consider atoms with at least 2 bonds (k >= 2), as they can form angles.
    valid_mask = counts >= 2
    valid_counts = counts[valid_mask]

    # Find the starting position of each atom's bond group in the sorted list.
    # Since input is sorted, the bonds for atom i are contiguous.
    # cumsum gives the end position, so we subtract counts to get the start.
    # We only keep starts for valid atoms.
    valid_starts = (torch.cumsum(counts, dim=0) - counts)[valid_mask]

    # For each valid atom with k bonds, we need to generate k * (k - 1) pairs of bonds.
    # We first generate all k * k pairs (including self-pairs) and then filter.

    # Calculate k^2 for each valid atom.
    sq_counts = valid_counts**2
    total_sq = sq_counts.sum()

    if total_sq > 0:
        # Create an array of group IDs, repeating each ID k^2 times.
        # This tells us which atom each generated pair belongs to.
        group_ids = torch.repeat_interleave(
            torch.arange(len(valid_counts), device=edge_indices.device), sq_counts
        )

        # Calculate the local index (0 to k^2 - 1) for each pair within its group.
        # We do this by subtracting the start index of the group from the global index.
        cum_sq = torch.cumsum(sq_counts, dim=0)
        starts_sq = cum_sq - sq_counts
        local_idx = torch.arange(total_sq, device=edge_indices.device) - starts_sq[group_ids]

        # Decode the local index into (u, v) coordinates, where u, v are in [0, k-1].
        # u corresponds to the first bond (outer loop), v to the second bond (inner loop).
        # This matches the order: for u in range(k): for v in range(k): ...
        k_vec = valid_counts[group_ids]
        u = local_idx // k_vec
        v = local_idx % k_vec

        # Filter out diagonal elements (u == v) because a bond cannot form an angle with itself.
        pair_mask = u != v
        u = u[pair_mask]
        v = v[pair_mask]
        group_ids = group_ids[pair_mask]

        # Map the local indices (u, v) back to the original bond indices.
        # We add the group's start offset in the sorted list to the local indices.
        group_start_indices = valid_starts[group_ids]
        bond_u = group_start_indices + u
        bond_v = group_start_indices + v

        # Stack to form the final list of bond pairs.
        triple_bond_indices = torch.stack([bond_u, bond_v], dim=1)
    else:
        triple_bond_indices = torch.zeros((0, 2), dtype=torch.long, device=edge_indices.device)

    # 5. compute the total number of angles in the structure
    n_triple_s = torch.sum(n_triple_i)

    return triple_bond_indices, n_triple_ij, n_triple_i, n_triple_s
