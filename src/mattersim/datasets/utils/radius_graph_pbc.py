"""
Radius graph computation with periodic boundary conditions.

Provides GPU-accelerated neighbor search for atomistic systems with PBC
using a memory-efficient cdist-based approach. Some utility functions
are adapted from the OCP codebase.

Main API:
    radius_graph_pbc_efficient: Memory-efficient cdist-based implementation
"""

import dataclasses
import sys

import torch


@dataclasses.dataclass
class PBCRadiusGraphData:
    """Result of radius graph computation with periodic boundary conditions."""

    edge_index: torch.Tensor  # [2, num_edges], convention: [sender, receiver]
    cell_offsets: torch.Tensor  # [num_edges, 3]
    n_edges_per_graph: torch.Tensor  # [num_structures, ]
    edge_length: torch.Tensor | None  # [num_edges, ]


def _segment_coo(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    """Native replacement for torch_scatter.segment_coo (sum reduction).

    Equivalent to: out[i] = sum(src[j] for j where index[j] == i)
    """
    return torch.zeros(dim_size, dtype=src.dtype, device=src.device).scatter_add_(
        0, index, src
    )


def _segment_csr(src: torch.Tensor, indptr: torch.Tensor) -> torch.Tensor:
    """Native replacement for torch_scatter.segment_csr (sum reduction).

    For each segment i, sums src[indptr[i]:indptr[i+1]].
    """
    n_segments = indptr.shape[0] - 1
    result = torch.zeros(n_segments, dtype=src.dtype, device=src.device)
    for i in range(n_segments):
        result[i] = src[indptr[i] : indptr[i + 1]].sum()
    return result


def _wrap_positions(
    pos: torch.Tensor,
    cell: torch.Tensor,
    n_nodes_per_graph: torch.Tensor,
    pbc: list[bool],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Wrap positions into the unit cell. Based on ase.geometry.geometry.wrap."""
    if not any(pbc):
        return pos, torch.zeros_like(pos)

    shift_T = torch.zeros_like(pos).T

    cell = cell.repeat_interleave(n_nodes_per_graph, dim=0)
    cell_inv = torch.linalg.inv(cell)

    fractional = torch.bmm(pos.unsqueeze(-2), cell_inv)
    fractional = fractional.squeeze(-2)
    fractional_T = fractional.T

    for i, periodic in enumerate(pbc):
        if periodic:
            shift_T[i] = torch.floor(fractional_T[i])
            fractional_T[i] = fractional_T[i] - shift_T[i]

    pos_wrap = torch.bmm(fractional_T.T.unsqueeze(-2), cell)

    return pos_wrap.squeeze(-2), shift_T.T


@torch.no_grad()
def _get_radius_graph_pbc_cdist(
    pos: torch.Tensor,
    n_nodes_per_graph: torch.Tensor,
    pbc: torch.Tensor,
    cell: torch.Tensor,
    cutoff_radius: float,
    return_dist: bool = False,
    max_cell_images_per_dim: int = sys.maxsize,
) -> PBCRadiusGraphData:
    """Compute radius graph with PBC using cdist (memory-efficient).

    Returns edges with edge_index as [sender, receiver], sorted by receiver.
    """
    device = pos.device
    dtype = pos.dtype
    batch_size = len(n_nodes_per_graph)

    assert pbc.dim() == 2 and pbc.shape[1] == 3, "pbc tensor has the wrong shape"
    assert torch.all(pbc[0] == pbc), "PBCs must be equal across batch dimension"
    pbc_ = pbc[0].detach().cpu().numpy().tolist()

    # Calculate required number of unit cells in each direction.
    cross_a2a3 = torch.cross(cell[:, 1], cell[:, 2], dim=-1)
    cell_vol = torch.sum(cell[:, 0] * cross_a2a3, dim=-1, keepdim=True)

    if pbc_[0]:
        inv_min_dist_a1 = torch.norm(cross_a2a3 / cell_vol, p=2, dim=-1)
        rep_a1 = torch.ceil(cutoff_radius * inv_min_dist_a1)
    else:
        rep_a1 = cell.new_zeros(1)

    if pbc_[1]:
        cross_a3a1 = torch.cross(cell[:, 2], cell[:, 0], dim=-1)
        inv_min_dist_a2 = torch.norm(cross_a3a1 / cell_vol, p=2, dim=-1)
        rep_a2 = torch.ceil(cutoff_radius * inv_min_dist_a2)
    else:
        rep_a2 = cell.new_zeros(1)

    if pbc_[2]:
        cross_a1a2 = torch.cross(cell[:, 0], cell[:, 1], dim=-1)
        inv_min_dist_a3 = torch.norm(cross_a1a2 / cell_vol, p=2, dim=-1)
        rep_a3 = torch.ceil(cutoff_radius * inv_min_dist_a3)
    else:
        rep_a3 = cell.new_zeros(1)

    max_rep = [
        min(int(rep_a1.max()), max_cell_images_per_dim),
        min(int(rep_a2.max()), max_cell_images_per_dim),
        min(int(rep_a3.max()), max_cell_images_per_dim),
    ]

    cells_per_dim = [
        torch.arange(-rep, rep + 1, device=device, dtype=dtype) for rep in max_rep
    ]
    cell_offsets = torch.cartesian_prod(*cells_per_dim)
    n_cells = cell_offsets.shape[0]
    unit_cell_batch = (
        cell_offsets.view(1, n_cells, 3).expand(batch_size, -1, -1).contiguous()
    )

    # Compute positional offsets for each cell
    pbc_offsets = torch.bmm(unit_cell_batch, cell)
    pbc_offsets_per_atom = pbc_offsets.repeat_interleave(n_nodes_per_graph, dim=0)

    pos_orig_wrapped, shift_to_unwrap = _wrap_positions(
        pos, cell, n_nodes_per_graph, pbc_
    )
    pos_orig_wrapped = pos_orig_wrapped.view(-1, 1, 3)
    pos_pbc_shift = pos_orig_wrapped + pbc_offsets_per_atom

    @torch.no_grad()
    def dist_thresh(
        A: torch.Tensor, B: torch.Tensor, cutoff: float, _return_dist: bool = False
    ):
        D = torch.cdist(A, B)
        idx = torch.nonzero(D <= cutoff + 5e-6, as_tuple=False)
        if not _return_dist:
            return idx
        else:
            values = D[idx[:, 0], idx[:, 1]]
            return idx, values

    @torch.no_grad()
    def blockwise_dist_thresh(
        A: torch.Tensor,
        B: torch.Tensor,
        cutoff: float,
        block_size: int,
        _return_dist: bool = False,
    ):
        n = A.shape[0]
        m = B.shape[0]
        n_blocks = (n + block_size - 1) // block_size
        m_blocks = (m + block_size - 1) // block_size

        ret_idx, ret_val = [], []

        for i in range(n_blocks):
            for j in range(m_blocks):
                A_block = A[i * block_size : (i + 1) * block_size]
                B_block = B[j * block_size : (j + 1) * block_size]

                if not _return_dist:
                    idx = dist_thresh(A_block, B_block, cutoff, _return_dist=False)
                    idx += torch.tensor(
                        [i * block_size, j * block_size], device=device
                    ).view(1, 2)
                    ret_idx.append(idx)
                else:
                    idx, val = dist_thresh(
                        A_block, B_block, cutoff, _return_dist=True
                    )
                    idx += torch.tensor(
                        [i * block_size, j * block_size], device=device
                    ).view(1, 2)
                    ret_idx.append(idx)
                    ret_val.append(val)
        if not _return_dist:
            return ret_idx
        else:
            return ret_idx, ret_val

    @torch.no_grad()
    def compute_dist_one_graph(i: int, j: int, _return_dist: bool = False):
        A = pos_orig_wrapped[i:j].reshape(-1, 3).contiguous()
        B = pos_pbc_shift[i:j].reshape(-1, 3).contiguous()
        return blockwise_dist_thresh(A, B, cutoff_radius, 65536, _return_dist)

    graph_end = torch.cumsum(n_nodes_per_graph, dim=0)
    graph_begin = graph_end - n_nodes_per_graph

    compute_dist = [
        compute_dist_one_graph(i, j, _return_dist=return_dist)
        for (i, j) in zip(graph_begin, graph_end)
    ]

    if return_dist:
        idx_lst, dist_lst = map(list, zip(*compute_dist))
        dist = torch.concat(sum(dist_lst, start=[]))
    else:
        idx_lst = compute_dist
        dist = None

    def _compute_nr_edges(edges):
        return sum(map(len, edges))

    n_neighbors_image = torch.tensor(
        list(map(_compute_nr_edges, idx_lst)), device=device
    )
    index0 = torch.concat(sum(idx_lst, start=[]))
    ix = index0[:, 0]  # receiver in origin cell
    iy = torch.div(
        index0[:, 1], n_cells, rounding_mode="floor"
    )  # sender (may be in periodic image)
    graph_offset = torch.repeat_interleave(graph_begin, n_neighbors_image).view(
        1, -1
    )

    # edge_index is [sender, receiver]
    edge_index = torch.stack([iy, ix]) + graph_offset

    cell_offsets_index = index0[:, 1] % n_cells
    cell_offsets_result = cell_offsets[cell_offsets_index]
    cell_offsets_result += (
        shift_to_unwrap[edge_index[1]] - shift_to_unwrap[edge_index[0]]
    )

    # Remove self-loops
    if dist is not None:
        eps_self_loop = 0.01
        is_self_loop = (edge_index[0] == edge_index[1]) & (dist <= eps_self_loop)
        mask = ~is_self_loop
        edge_index = edge_index[:, mask]
        cell_offsets_result = cell_offsets_result[mask]
        dist = dist[mask]
        graph_id = torch.bucketize(edge_index[1], graph_end[:-1], right=True)
        n_neighbors_image = torch.bincount(graph_id, minlength=batch_size)

    # Sort by receiver index
    sort_idx = torch.argsort(edge_index[1])
    edge_index = edge_index[:, sort_idx]
    cell_offsets_result = cell_offsets_result[sort_idx]
    if dist is not None:
        dist = dist[sort_idx]

    return PBCRadiusGraphData(
        edge_index=edge_index,
        cell_offsets=cell_offsets_result,
        n_edges_per_graph=n_neighbors_image,
        edge_length=dist,
    )


def get_max_neighbors_mask(
    natoms: torch.Tensor,
    index: torch.Tensor,
    atom_distance_squared: torch.Tensor,
    max_num_neighbors_threshold: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Give a mask that filters out edges so that each atom has at most
    `max_num_neighbors_threshold` neighbors. Assumes `index` is sorted."""
    device = natoms.device
    num_atoms = natoms.sum()

    ones = index.new_ones(1).expand_as(index)
    num_neighbors = _segment_coo(ones, index, dim_size=int(num_atoms.item()))
    max_num_neighbors = num_neighbors.max()

    if max_num_neighbors_threshold <= 0:
        num_neighbors_thresholded = num_neighbors
    else:
        num_neighbors_thresholded = num_neighbors.clamp(
            max=max_num_neighbors_threshold
        )

    image_indptr = torch.zeros(
        natoms.shape[0] + 1, device=device, dtype=torch.long
    )
    image_indptr[1:] = torch.cumsum(natoms, dim=0)
    num_neighbors_image = _segment_csr(num_neighbors_thresholded, image_indptr)

    if (
        max_num_neighbors <= max_num_neighbors_threshold
        or max_num_neighbors_threshold <= 0
    ):
        mask_num_neighbors = torch.tensor(
            [True], dtype=torch.bool, device=device
        ).expand_as(index)
        return mask_num_neighbors, num_neighbors_image

    import numpy as np

    distance_sort = torch.full(
        [int((num_atoms * max_num_neighbors).long().item())],
        np.inf,
        device=device,
    )

    index_neighbor_offset = torch.cumsum(num_neighbors, dim=0) - num_neighbors
    index_neighbor_offset_expand = torch.repeat_interleave(
        index_neighbor_offset, num_neighbors
    )
    index_sort_map = (
        index * max_num_neighbors
        + torch.arange(len(index), device=device)
        - index_neighbor_offset_expand
    )
    distance_sort.index_copy_(0, index_sort_map, atom_distance_squared)
    distance_sort = distance_sort.view(num_atoms, max_num_neighbors)

    distance_sort, index_sort = torch.sort(distance_sort, dim=1)
    distance_sort = distance_sort[:, :max_num_neighbors_threshold]
    index_sort = index_sort[:, :max_num_neighbors_threshold]

    index_sort = index_sort + index_neighbor_offset.view(-1, 1).expand(
        -1, max_num_neighbors_threshold
    )
    mask_finite = torch.isfinite(distance_sort)
    index_sort = torch.masked_select(index_sort, mask_finite)

    mask_num_neighbors = torch.zeros(len(index), device=device, dtype=torch.bool)
    mask_num_neighbors.index_fill_(0, index_sort, torch.tensor(True))

    return mask_num_neighbors, num_neighbors_image


def radius_graph_pbc_efficient(
    pos: torch.Tensor,
    pbc: torch.Tensor | None,
    natoms: torch.Tensor,
    cell: torch.Tensor,
    radius: float,
    max_num_neighbors_threshold: int = -1,
    max_cell_images_per_dim: int = sys.maxsize,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the radius graph with PBC (memory-efficient cdist-based).

    Uses O(n²) memory per block instead of O(n² × num_cells), significantly
    more memory-efficient for large systems.

    Args:
        pos: Atomic positions [n, 3]
        pbc: PBC per structure [n_structures, 3] or None (defaults to all True)
        natoms: Atoms per structure [n_structures]
        cell: Unit cells [n_structures, 3, 3]
        radius: Cutoff radius
        max_num_neighbors_threshold: Max neighbors per atom (-1 = no limit)
        max_cell_images_per_dim: Max cell images per dimension

    Returns:
        edge_index [2, n_edges], cell_offsets [n_edges, 3],
        num_neighbors_image [n_structures], offsets [n_edges, 3],
        atom_distance [n_edges]
    """
    device = pos.device

    if pbc is None:
        pbc_tensor = torch.ones(len(natoms), 3, dtype=torch.bool, device=device)
    else:
        pbc_tensor = torch.atleast_2d(pbc)

    result = _get_radius_graph_pbc_cdist(
        pos=pos,
        n_nodes_per_graph=natoms,
        pbc=pbc_tensor,
        cell=cell,
        cutoff_radius=radius,
        return_dist=True,
        max_cell_images_per_dim=max_cell_images_per_dim,
    )

    edge_index = result.edge_index
    cell_offsets = result.cell_offsets
    num_neighbors_image = result.n_edges_per_graph
    atom_distance = result.edge_length

    # Apply max neighbors threshold
    if max_num_neighbors_threshold > 0:
        mask_num_neighbors, num_neighbors_image = get_max_neighbors_mask(
            natoms=natoms,
            index=edge_index[1],
            atom_distance_squared=atom_distance**2,
            max_num_neighbors_threshold=max_num_neighbors_threshold,
        )

        if not torch.all(mask_num_neighbors):
            edge_index = edge_index[:, mask_num_neighbors]
            cell_offsets = cell_offsets[mask_num_neighbors]
            atom_distance = atom_distance[mask_num_neighbors]

    # Compute cartesian offsets
    cell_repeated = torch.repeat_interleave(
        cell, num_neighbors_image.long(), dim=0
    )
    offsets = (
        -cell_offsets.float().view(-1, 1, 3).bmm(cell_repeated.float()).view(-1, 3)
    )

    return (
        edge_index,
        cell_offsets,
        num_neighbors_image,
        offsets,
        atom_distance,
    )
