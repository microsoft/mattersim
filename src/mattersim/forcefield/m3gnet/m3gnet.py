from functools import partial
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .modules import (  # noqa: F501
    MLP,
    GatedMLP,
    MainBlock,
    SmoothBesselBasis,
    SphericalBasisLayer,
)
from .modules.scatter import scatter_sum
from .scaling import AtomScaling


class M3Gnet(nn.Module):
    """
    M3Gnet
    """

    def __init__(
        self,
        num_blocks: int = 4,
        units: int = 128,
        max_l: int = 4,
        max_n: int = 4,
        cutoff: float = 5.0,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        max_z: int = 94,
        threebody_cutoff: float = 4.0,
        gradient_checkpointing: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.rbf = SmoothBesselBasis(r_max=cutoff, max_n=max_n)
        self.sbf = SphericalBasisLayer(max_n=max_n, max_l=max_l, cutoff=cutoff)
        self.edge_encoder = MLP(
            in_dim=max_n, out_dims=[units], activation="swish", use_bias=False
        )
        module_list = [
            MainBlock(max_n, max_l, cutoff, units, max_n, threebody_cutoff)
            for i in range(num_blocks)
        ]
        self.graph_conv = nn.ModuleList(module_list)
        self.final = GatedMLP(
            in_dim=units,
            out_dims=[units, units, 1],
            activation=["swish", "swish", None],
        )
        self.apply(self.init_weights)
        self.atom_embedding = MLP(
            in_dim=max_z + 1, out_dims=[units], activation=None, use_bias=False
        )
        self.atom_embedding.apply(self.init_weights_uniform)
        self.normalizer = AtomScaling(verbose=False, max_z=max_z, device=device)
        self.max_z = max_z
        self.device = device
        self.gradient_checkpointing = gradient_checkpointing
        self.model_args = {
            "num_blocks": num_blocks,
            "units": units,
            "max_l": max_l,
            "max_n": max_n,
            "cutoff": cutoff,
            "max_z": max_z,
            "threebody_cutoff": threebody_cutoff,
            "gradient_checkpointing": gradient_checkpointing,
        }

    def forward(
        self,
        input: Dict[str, torch.Tensor],
        dataset_idx: int = -1,
    ) -> torch.Tensor:
        # Exact data from input_dictionary
        pos = input["atom_pos"]
        cell = input["cell"]
        pbc_offsets = input["pbc_offsets"].float()
        atom_attr = input["atom_attr"]
        edge_index = input["edge_index"].long()
        three_body_indices = input["three_body_indices"].long()
        num_bonds = input["num_bonds"]
        num_triple_ij = input["num_triple_ij"]
        num_atoms = input["num_atoms"]
        num_graphs = input["num_graphs"]
        batch = input["batch"]

        # Use precomputed values if available, otherwise derive from shapes
        # (avoids device-to-host sync on MPS/CUDA)
        total_num_atoms = input.get("total_num_atoms", pos.shape[0])
        total_num_bonds = input.get("total_num_bonds", edge_index.shape[1])

        bond_index_bias = input.get("bond_index_bias", None)
        if bond_index_bias is None:
            cumsum = torch.cumsum(num_bonds, dim=0) - num_bonds
            bond_index_bias = torch.repeat_interleave(
                cumsum, input["num_three_body"], dim=0
            ).unsqueeze(-1)

        three_body_edge_map = input.get("three_body_edge_map", None)

        # -------------------------------------------------------------#
        three_body_indices = three_body_indices + bond_index_bias

        # === Refer to the implementation of M3GNet,        ===
        # === we should re-compute the following attributes ===
        # edge_length, edge_vector(optional), triple_edge_length, theta_jik
        edge_batch = batch[edge_index[0]]
        edge_vector = pos[edge_index[0]] - (
            pos[edge_index[1]]
            + torch.einsum("bi, bij->bj", pbc_offsets, cell[edge_batch])
        )
        edge_length = torch.linalg.norm(edge_vector, dim=1)
        vij = edge_vector[three_body_indices[:, 0].clone()]
        vik = edge_vector[three_body_indices[:, 1].clone()]
        rij = edge_length[three_body_indices[:, 0].clone()]
        rik = edge_length[three_body_indices[:, 1].clone()]
        cos_jik = torch.sum(vij * vik, dim=1) / (rij * rik)
        # eps = 1e-7 avoid nan in torch.acos function
        cos_jik = torch.clamp(cos_jik, min=-1.0 + 1e-7, max=1.0 - 1e-7)
        triple_edge_length = rik.view(-1)
        edge_length = edge_length.unsqueeze(-1)
        atomic_numbers = atom_attr.squeeze(1).long()

        # featurize
        atom_attr = self.atom_embedding(self.one_hot_atoms(atomic_numbers))
        edge_attr = self.rbf(edge_length.view(-1))
        edge_attr_zero = edge_attr  # e_ij^0
        edge_attr = self.edge_encoder(edge_attr)
        three_basis = self.sbf(triple_edge_length, torch.acos(cos_jik))

        # Main Loop - use gradient checkpointing if enabled to reduce memory
        # Note: checkpointing works in both training and eval mode
        # (for MD force computation)
        for idx, conv in enumerate(self.graph_conv):
            func = partial(
                conv,
                atom_attr=atom_attr,
                edge_attr=edge_attr,
                edge_attr_zero=edge_attr_zero,
                edge_index=edge_index,
                three_basis=three_basis,
                three_body_index=three_body_indices,
                edge_length=edge_length,
                num_edges=num_bonds,
                num_triple_ij=num_triple_ij,
                num_atoms=num_atoms,
                total_num_atoms=total_num_atoms,
                total_num_bonds=total_num_bonds,
                three_body_edge_map=three_body_edge_map,
            )
            if self.gradient_checkpointing:
                # use_reentrant=False is recommended and works with
                # native scatter operations
                atom_attr, edge_attr = checkpoint(
                    func,
                    use_reentrant=False,
                )
            else:
                atom_attr, edge_attr = func()

        energies_i = self.final(atom_attr).view(-1)  # [batch_size*num_atoms]
        energies_i = self.normalizer(energies_i, atomic_numbers)
        energies = scatter_sum(energies_i, batch, dim=0, dim_size=cell.shape[0])

        return energies  # [batch_size]

    def enable_gradient_checkpointing(self, enable: bool = True):
        """Enable or disable gradient checkpointing for memory-efficient
        training/inference.

        When enabled, intermediate activations are recomputed during backward
        pass instead of being stored, significantly reducing memory usage at
        the cost of up to 30% slower computation.
        """
        self.gradient_checkpointing = enable

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)

    def init_weights_uniform(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.uniform_(m.weight, a=-0.05, b=0.05)

    @torch.jit.export
    def one_hot_atoms(self, species):
        dtype = self.atom_embedding.mlp[0].linear.weight.dtype
        return F.one_hot(species, num_classes=self.max_z + 1).to(dtype)

    def print(self):
        from prettytable import PrettyTable

        table = PrettyTable(["Modules", "Parameters"])
        total_params = 0
        for name, parameter in self.named_parameters():
            if not parameter.requires_grad:
                continue
            params = parameter.numel()
            table.add_row([name, params])
            total_params += params
        print(table)
        print(f"Total Trainable Params: {total_params}")

    @torch.jit.export
    def set_normalizer(self, normalizer: AtomScaling):
        self.normalizer = normalizer

    def get_model_args(self):
        return self.model_args
