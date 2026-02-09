"""
Code modified from the OCP codebase: https://github.com/Open-Catalyst-Project/ocp
"""

import sys

import numpy as np
import torch
from torch_scatter import segment_coo, segment_csr


def radius_graph_pbc(
    pos: torch.Tensor,
    pbc: torch.Tensor | None,
    natoms: torch.Tensor,
    cell: torch.Tensor,
    radius: float,
    max_cell_images_per_dim: int = sys.maxsize,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Function computing the graph in periodic boundary conditions on a (batched) set of
    positions and cells.

    This function is copied from
    https://github.com/Open-Catalyst-Project/ocp/blob/main/ocpmodels/common/utils.py,
    commit 480eb9279ec4a5885981f1ee588c99dcb38838b5

    Args:
        pos (LongTensor): Atomic positions in cartesian coordinates
            :obj:`[n, 3]`
        pbc (BoolTensor): indicates periodic boundary conditions per structure.
            :obj:`[n_structures, 3]`
        natoms (IntTensor): number of atoms per structure. Has shape
            :obj:`[n_structures]`
        cell (Tensor): atomic cell. Has shape
            :obj:`[n_structures, 3, 3]`
        radius (float): cutoff radius distance

    Returns:
        edge_index (IntTensor): index of atoms in edges. Has shape
            :obj:`[n_edges, 2]`
        cell_offsets (IntTensor): cell displacement w.r.t. their original position of atoms in edges. Has shape
            :obj:`[n_edges, 3, 3]`
        num_neighbors_image (IntTensor): Number of neighbours per cell image.
            :obj:`[n_structures]`
        offsets (LongTensor): cartesian displacement w.r.t. their original position of atoms in edges. Has shape
            :obj:`[n_edges, 3, 3]`
        atom_distance (LongTensor): edge length. Has shape
            :obj:`[n_edges]`
    """
    device = pos.device
    batch_size = len(natoms)
    pbc_ = [False, False, False]

    if pbc is not None:
        pbc = torch.atleast_2d(pbc)
        for i in range(3):
            if not torch.any(pbc[:, i]).item():
                pbc_[i] = False
            elif torch.all(pbc[:, i]).item():
                pbc_[i] = True
            else:
                raise RuntimeError(
                    "Different structures in the batch have different PBC configurations. This is not currently supported."
                )

    natoms_squared = (natoms**2).long()

    # index offset between images
    index_offset = torch.cumsum(natoms, dim=0) - natoms

    index_offset_expand = torch.repeat_interleave(index_offset, natoms_squared)
    natoms_expand = torch.repeat_interleave(natoms, natoms_squared)

    # Compute a tensor containing sequences of numbers that range from 0 to num_atoms_per_image_squared for each image
    # that is used to compute indices for the pairs of atoms. This is a very convoluted way to implement
    # the following (but 10x faster since it removes the for loop)
    # for batch_idx in range(batch_size):
    #    batch_count = torch.cat([batch_count, torch.arange(num_atoms_per_image_squared[batch_idx], device=device)], dim=0)
    num_atom_pairs = torch.sum(natoms_squared)
    index_squared_offset = torch.cumsum(natoms_squared, dim=0) - natoms_squared
    index_squared_offset = torch.repeat_interleave(index_squared_offset, natoms_squared)
    atom_count_squared = torch.arange(num_atom_pairs, device=device) - index_squared_offset

    # Compute the indices for the pairs of atoms (using division and mod)
    # If the systems get too large this approach could run into numerical precision issues
    index1 = (
        torch.div(atom_count_squared, natoms_expand, rounding_mode="floor")
    ) + index_offset_expand
    index2 = (atom_count_squared % natoms_expand) + index_offset_expand
    # Get the positions for each atom
    pos1 = torch.index_select(pos, 0, index1)
    pos2 = torch.index_select(pos, 0, index2)

    # Calculate required number of unit cells in each direction.
    # Smallest distance between planes separated by a1 is
    # 1 / ||(a2 x a3) / V||_2, since a2 x a3 is the area of the plane.
    # Note that the unit cell volume V = a1 * (a2 x a3) and that
    # (a2 x a3) / V is also the reciprocal primitive vector
    # (crystallographer's definition).

    cross_a2a3 = torch.cross(cell[:, 1], cell[:, 2], dim=-1)
    cell_vol = torch.sum(cell[:, 0] * cross_a2a3, dim=-1, keepdim=True)

    if pbc_[0]:
        inv_min_dist_a1 = torch.norm(cross_a2a3 / cell_vol, p=2, dim=-1)
        rep_a1 = torch.ceil(radius * inv_min_dist_a1)
    else:
        rep_a1 = cell.new_zeros(1)

    if pbc_[1]:
        cross_a3a1 = torch.cross(cell[:, 2], cell[:, 0], dim=-1)
        inv_min_dist_a2 = torch.norm(cross_a3a1 / cell_vol, p=2, dim=-1)
        rep_a2 = torch.ceil(radius * inv_min_dist_a2)
    else:
        rep_a2 = cell.new_zeros(1)

    if pbc_[2]:
        cross_a1a2 = torch.cross(cell[:, 0], cell[:, 1], dim=-1)
        inv_min_dist_a3 = torch.norm(cross_a1a2 / cell_vol, p=2, dim=-1)
        rep_a3 = torch.ceil(radius * inv_min_dist_a3)
    else:
        rep_a3 = cell.new_zeros(1)

    # Take the max over all images for uniformity. This is essentially padding.
    # Note that this can significantly increase the number of computed distances
    # if the required repetitions are very different between images
    # (which they usually are). Changing this to sparse (scatter) operations
    # might be worth the effort if this function becomes a bottleneck.
    #
    # max_cell_images_per_dim limits the number of periodic
    # cell images that are considered per lattice vector dimension. This is
    # useful in case we encounter an extremely skewed or small lattice that
    # results in an explosion of the number of images considered.
    max_rep = [
        min(int(rep_a1.max()), max_cell_images_per_dim),
        min(int(rep_a2.max()), max_cell_images_per_dim),
        min(int(rep_a3.max()), max_cell_images_per_dim),
    ]

    # Tensor of unit cells
    offset_range1 = torch.arange(-max_rep[0], max_rep[0] + 1, device=device)
    offset_range2 = torch.arange(-max_rep[1], max_rep[1] + 1, device=device)
    offset_range3 = torch.arange(-max_rep[2], max_rep[2] + 1, device=device)
    cell_offsets = (
        torch.stack(
            torch.meshgrid(offset_range1, offset_range2, offset_range3, indexing="ij"), dim=-1
        )
        .reshape(-1, 3)
        .float()
    )
    # cell_offsets = torch.cartesian_prod(*cells_per_dim)
    num_cells = len(cell_offsets)
    cell_offsets_per_atom = cell_offsets.view(1, num_cells, 3).repeat(len(index2), 1, 1)
    cell_offsets = torch.transpose(cell_offsets, 0, 1)
    cell_offsets_batch = cell_offsets.view(1, 3, num_cells).expand(batch_size, -1, -1)

    # Compute the x, y, z positional offsets for each cell in each image
    data_cell = torch.transpose(cell, 1, 2)
    pbc_offsets = torch.bmm(data_cell, cell_offsets_batch)
    pbc_offsets_per_atom = torch.repeat_interleave(pbc_offsets, natoms_squared, dim=0)

    # Expand the positions and indices for the 9 cells
    pos1 = pos1.view(-1, 3, 1).expand(-1, -1, num_cells)
    pos2 = pos2.view(-1, 3, 1).expand(-1, -1, num_cells)
    index1 = index1.view(-1, 1).repeat(1, num_cells).view(-1)
    index2 = index2.view(-1, 1).repeat(1, num_cells).view(-1)
    # Add the PBC offsets for the second atom
    pos2 = pos2 + pbc_offsets_per_atom

    # Compute the squared distance between atoms
    atom_distance_squared = torch.sum((pos1 - pos2) ** 2, dim=1)
    atom_distance_squared = atom_distance_squared.view(-1)

    # Remove pairs that are too far apart
    mask_within_radius = torch.le(atom_distance_squared, radius * radius)
    # Remove pairs with the same atoms (distance = 0.0)
    # mask_not_same = torch.gt(atom_distance_squared, 0.0001)
    # mask = torch.logical_and(mask_within_radius, mask_not_same)
    mask_self_loop = (index1 == index2) & (atom_distance_squared <= 0.0001)
    mask = mask_within_radius & (~mask_self_loop)
    index1 = torch.masked_select(index1, mask)
    index2 = torch.masked_select(index2, mask)
    cell_offsets = torch.masked_select(
        cell_offsets_per_atom.view(-1, 3), mask.view(-1, 1).expand(-1, 3)
    )
    cell_offsets = cell_offsets.view(-1, 3)
    atom_distance_squared = torch.masked_select(atom_distance_squared, mask)

    # Compute num_neighbors_image (number of edges per structure in the batch)
    num_atoms = natoms.sum()
    ones = index1.new_ones(1).expand_as(index1)
    num_neighbors = segment_coo(ones, index1, dim_size=num_atoms)
    image_indptr = torch.zeros(natoms.shape[0] + 1, device=device, dtype=torch.long)
    image_indptr[1:] = torch.cumsum(natoms, dim=0)
    num_neighbors_image = segment_csr(num_neighbors, image_indptr)

    edge_index = torch.stack((index2, index1))
    # shifts = -torch.matmul(unit_cell, data.cell).view(-1, 3)

    cell_repeated = torch.repeat_interleave(cell, num_neighbors_image.long(), dim=0)
    offsets = -cell_offsets.float().view(-1, 1, 3).bmm(cell_repeated.float()).view(-1, 3)
    return (
        edge_index,
        cell_offsets,
        num_neighbors_image,
        offsets,
        torch.sqrt(atom_distance_squared),
    )

