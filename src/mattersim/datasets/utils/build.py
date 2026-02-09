from typing import Literal

import numpy as np
import torch
from ase import Atoms
from torch_geometric.loader import DataLoader as DataLoader_pyg
from tqdm import tqdm

from mattersim.datasets.utils.converter import GraphConverter, BatchGraphConverter


def build_dataloader(
    atoms: list[Atoms] | None = None,
    energies: list[float] | None = None,
    forces: list[np.ndarray] | None = None,
    stresses: list[np.ndarray] | None = None,
    cutoff: float = 5.0,
    threebody_cutoff: float = 4.0,
    batch_size: int = 64,
    model_type: Literal["m3gnet"] = "m3gnet",
    shuffle: bool = False,
    only_inference: bool = False,
    num_workers: int = 0,
    pin_memory: bool = False,
    batch_converter: bool = True,
    max_natoms_per_batch: int = 4096,
):
    """
    Build a dataloader given a list of atoms

    Args:
        atoms: list of Atoms objects
        energies: list of energies corresponding to the atoms
        forces: list of forces corresponding to the atoms
        stresses: list of stresses corresponding to the atoms
        cutoff: cutoff distance for graph construction
        threebody_cutoff: cutoff distance for three-body interactions
        batch_size: number of samples per batch
        model_type: type of model to use
        shuffle: whether to shuffle the data
        only_inference: whether to only perform inference
        num_workers: number of worker processes for data loading
        pin_memory: whether to pin memory
        batch_converter: whether to use batch converter
        max_natoms_per_batch: maximum number of atoms per batch in the batch converter, only used if batch_converter is True
            Do not confuse with batch_size, which is the number of samples per batch in the final dataloader.
            But, max_natoms_per_batch is used to control the number of atoms to construct the graph.
    """

    if not batch_converter:
        converter = GraphConverter(model_type, cutoff, True, threebody_cutoff)
    else:
        converter = BatchGraphConverter(
            model_type, 
            twobody_cutoff=cutoff, 
            has_threebody=True,
            threebody_cutoff=threebody_cutoff
        )

    preprocessed_data = []

    # sanity checks
    if not only_inference and np.any([x is None for x in [energies, forces, stresses]]):
        raise ValueError(
            "energies, forces, and stresses must be provided if only_inference is False"
        )

    if only_inference:
        length = len(atoms)
        if energies is None:
            energies = [None] * length
        if forces is None:
            forces = [None] * length
        if stresses is None:
            stresses = [None] * length
    else:
        assert (
            len(atoms) == len(energies) == len(forces) == len(stresses)
        ), "Length of atoms, energies, forces, and stresses must be the same"
    

    if model_type == "m3gnet":
        if not batch_converter:
            for graph, energy, force, stress in zip(
                atoms, energies, forces, stresses
            ):
                graph = converter.convert(
                    graph.copy(), energy, force, stress
                )
                if graph is not None:
                    preprocessed_data.append(graph)
        else:
            preprocessed_data = converter.convert(
                atoms,
                energy=energies,
                forces=forces,
                stresses=stresses,
                max_natoms_per_batch=max_natoms_per_batch,
            )
    else:
        raise NotImplementedError(f"model type not supported: {model_type}")


    return DataLoader_pyg(
        preprocessed_data,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )



def multiprocess_data(atoms: list[Atoms], number):
    converter = GraphConverter()
    result = []
    for graph in atoms:
        graph = converter.convert(
            graph,
            graph.get_potential_energy(),
            graph.get_forces(),
            graph.get_stress(voigt=False) * 160.2,
        )
        if graph is not None:
            result.append(graph)
    return result



def pad_1d_unsqueeze(x, padlen):
    x = x + 1  # pad id = 0
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen], dtype=x.dtype)
        new_x[:xlen] = x
        x = new_x
    return x.unsqueeze(0)


def pad_2d_unsqueeze(x, padlen):
    x = x + 1  # pad id = 0
    xlen, xdim = x.size()
    if xlen < padlen:
        new_x = x.new_zeros([padlen, xdim], dtype=x.dtype)
        new_x[:xlen, :] = x
        x = new_x
    return x.unsqueeze(0)


@torch.jit.script
def mask_after_k_persample(n_sample: int, n_len: int, persample_k: torch.Tensor):
    assert persample_k.shape[0] == n_sample
    assert persample_k.max() <= n_len
    device = persample_k.device
    mask = torch.zeros([n_sample, n_len + 1], device=device)
    mask[torch.arange(n_sample, device=device), persample_k] = 1
    mask = mask.cumsum(dim=1)[:, :-1]
    return mask.type(torch.bool)


def auto_cell(cell, cutoff=10.0):
    # find max value in x, y, z direction
    max_x = max(int(cutoff / torch.min(torch.abs(cell[:, 0, 0]))), 1)
    max_y = max(int(cutoff / torch.min(torch.abs(cell[:, 1, 1]))), 1)
    max_z = max(int(cutoff / torch.min(torch.abs(cell[:, 2, 2]))), 1)
    # loop
    cells = []
    for i in range(-max_x, max_x + 1):
        for j in range(-max_y, max_y + 1):
            for k in range(-max_z, max_z + 1):
                if i == 0 and j == 0 and k == 0:
                    continue
                cells.append([i, j, k])
    return cells


def cell_expand(pos, atoms, cell, cutoff=10.0):
    batch_size, max_num_atoms = pos.size()[:2]
    cells = auto_cell(cell, cutoff)
    cell_tensor = (
        torch.tensor(cells, device=pos.device)
        .to(cell.dtype)
        .unsqueeze(0)
        .expand(batch_size, -1, -1)
    )  # batch_size, n_cell, 3
    offset = torch.bmm(cell_tensor, cell)  # B x n_cell x 3
    expand_pos = pos.unsqueeze(1) + offset.unsqueeze(2)  # B x n_cell x T x 3
    expand_pos = expand_pos.view(batch_size, -1, 3)  # B x (n_cell x T) x 3
    expand_dist = torch.norm(
        pos.unsqueeze(2) - expand_pos.unsqueeze(1), p=2, dim=-1
    )  # B x T x (8 x T)
    expand_mask = expand_dist < cutoff  # B x T x (8 x T)
    expand_mask = torch.masked_fill(expand_mask, atoms.eq(0).unsqueeze(-1), False)
    expand_mask = (torch.sum(expand_mask, dim=1) > 0) & (
        ~(atoms.eq(0).repeat(1, len(cells)))
    )  # B x (8 x T)
    expand_len = torch.sum(expand_mask, dim=-1)
    max_expand_len = torch.max(expand_len)
    outcell_index = torch.zeros(
        [batch_size, max_expand_len], dtype=torch.long, device=pos.device
    )
    expand_pos_compressed = torch.zeros(
        [batch_size, max_expand_len, 3], dtype=pos.dtype, device=pos.device
    )
    outcell_all_index = torch.arange(
        max_num_atoms, dtype=torch.long, device=pos.device
    ).repeat(len(cells))
    for i in range(batch_size):
        outcell_index[i, : expand_len[i]] = outcell_all_index[expand_mask[i]]
        expand_pos_compressed[i, : expand_len[i], :] = expand_pos[i, expand_mask[i], :]
    return (
        expand_pos_compressed,
        expand_len,
        outcell_index,
        mask_after_k_persample(batch_size, max_expand_len, expand_len),
    )


def pad_spatial_pos_unsqueeze(x, padlen):
    x = x + 1
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen], dtype=x.dtype)
        new_x[:xlen, :xlen] = x
        x = new_x
    return x.unsqueeze(0)


@torch.jit.script
def convert_to_single_emb(x, offset: int = 512):
    feature_num = x.size(1) if len(x.size()) > 1 else 1
    feature_offset = 1 + torch.arange(0, feature_num * offset, offset, dtype=torch.long)
    x = x + feature_offset
    return x


class BatchedDataDataset(torch.utils.data.Dataset):
    # class BatchedDataDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        dataset,
        max_node=512,
        infer=False,
    ):
        super().__init__()
        self.dataset = dataset
        self.max_node = max_node

        self.infer = infer

    def __getitem__(self, index):
        item = self.dataset[int(index)]
        return item

    def __len__(self):
        return len(self.dataset)

    def collate(self, samples):
        return collator_ft(
            samples,
            max_node=self.max_node,
            use_pbc=True,
        )


def pad_pos_unsqueeze(x, padlen):
    xlen, xdim = x.size()
    if xlen < padlen:
        new_x = x.new_zeros([padlen, xdim], dtype=x.dtype)
        new_x[:xlen, :] = x
        x = new_x
    return x.unsqueeze(0)


def collator_ft(items, max_node=512, use_pbc=True):
    original_len = len(items)
    items = [item for item in items if item is not None and item.x.size(0) <= max_node]
    filtered_len = len(items)
    if filtered_len < original_len:
        pass
        # print("warning: molecules with atoms more than %d are filtered" % max_node)
    pos = None
    max_node_num = max(item.x.size(0) for item in items if item is not None)
    forces = None
    stress = None
    total_energy = None

    if hasattr(items[0], "pos") and items[0].pos is not None:
        poses = [item.pos - item.pos.mean(dim=0, keepdim=True) for item in items]
        # poses = [item.pos for item in items]
        pos = torch.cat([pad_pos_unsqueeze(i, max_node_num) for i in poses])
    if hasattr(items[0], "forces") and items[0].forces is not None:
        forcess = [item.forces for item in items]
        forces = torch.cat([pad_pos_unsqueeze(i, max_node_num) for i in forcess])
    if hasattr(items[0], "stress") and items[0].stress is not None:
        stress = torch.cat([item.stress.unsqueeze(0) for item in items], dim=0)
    if hasattr(items[0], "total_energy") and items[0].cell is not None:
        total_energy = torch.cat([item.total_energy for item in items])

    items = [
        (
            item.idx,
            item.x,
            item.y,
            (item.pbc if hasattr(item, "pbc") else torch.tensor([False, False, False]))
            if use_pbc
            else None,
            (item.cell if hasattr(item, "cell") else torch.zeros([3, 3]))
            if use_pbc
            else None,
            (int(item.num_atoms) if hasattr(item, "num_atoms") else item.x.size()[0]),
        )
        for item in items
    ]
    (
        idxs,
        xs,
        ys,
        pbcs,
        cells,
        natoms,
    ) = zip(*items)

    y = torch.cat(ys)
    x = torch.cat([pad_2d_unsqueeze(i, max_node_num) for i in xs])

    pbc = torch.cat([i.unsqueeze(0) for i in pbcs], dim=0) if use_pbc else None
    cell = torch.cat([i.unsqueeze(0) for i in cells], dim=0) if use_pbc else None
    natoms = torch.tensor(natoms) if use_pbc else None
    node_type_edge = None
    return dict(
        idx=torch.LongTensor(idxs),
        x=x,
        y=y,
        pos=pos + 1e-5,
        pbc=pbc,
        cell=cell,
        natoms=natoms,
        total_energy=total_energy,
        forces=forces,
        stress=stress,
        node_type_edge=node_type_edge,
    )
