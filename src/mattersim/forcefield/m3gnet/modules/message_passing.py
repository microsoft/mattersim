# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch_runstats.scatter import scatter

from .layers import GatedMLP, LinearLayer, SigmoidLayer, SwishLayer


def polynomial(r: torch.Tensor, cutoff: float) -> torch.Tensor:
    """
    Polynomial cutoff function
    Args:
        r (tf.Tensor): radius distance tensor
        cutoff (float): cutoff distance
    Returns: polynomial cutoff functions
    """
    ratio = torch.div(r, cutoff)
    result = (
        1
        - 6 * torch.pow(ratio, 5)
        + 15 * torch.pow(ratio, 4)
        - 10 * torch.pow(ratio, 3)
    )
    return torch.clamp(result, min=0.0)


class ThreeDInteraction(nn.Module):
    def __init__(
        self,
        max_n,
        max_l,
        cutoff,
        units,
        spherecal_dim,
        threebody_cutoff,
    ):
        super().__init__()
        # self.sbf = SphericalBesselFunction(
        #            max_l=max_l, max_n=max_n, cutoff=cutoff, smooth=smooth)
        # self.shf = SphericalHarmonicsFunction(max_l=max_l, use_phi=use_phi)
        self.atom_mlp = SigmoidLayer(in_dim=units, out_dim=spherecal_dim)
        # Linyu have modified the self.edge_gate_mlp
        # by adding swish activation and use_bias=False
        self.edge_gate_mlp = GatedMLP(
            in_dim=spherecal_dim,
            out_dims=[units],
            activation="swish",
            use_bias=False,  # noqa: E501
        )
        self.cutoff = cutoff
        self.threebody_cutoff = threebody_cutoff

    def forward(
        self,
        edge_attr,
        three_basis,
        atom_attr,
        edge_index,
        three_body_index,
        edge_length,
        num_edges,
        num_triple_ij,
    ):
        atom_mask = (
            self.atom_mlp(atom_attr)[edge_index[0][three_body_index[:, 1]]]
            * polynomial(
                edge_length[three_body_index[:, 0]], self.threebody_cutoff  # noqa: E501
            )
            * polynomial(
                edge_length[three_body_index[:, 1]], self.threebody_cutoff  # noqa: E501
            )
        )
        three_basis = three_basis * atom_mask
        index_map = torch.arange(torch.sum(num_edges).item()).to(
            edge_length.device
        )  # noqa: E501
        index_map = torch.repeat_interleave(index_map, num_triple_ij).to(
            edge_length.device
        )
        e_ij_tuda = scatter(
            three_basis,
            index_map,
            dim=0,
            reduce="sum",
            dim_size=torch.sum(num_edges).item(),
        )
        edge_attr_prime = edge_attr + self.edge_gate_mlp(e_ij_tuda)
        return edge_attr_prime


class AtomLayer(nn.Module):
    """
    v_i'=v_i+sum(phi(v+i,v_j,e_ij',u)W*e_ij^0)
    """

    def __init__(
        self,
        atom_attr_dim,
        edge_attr_dim,
        spherecal_dim,
    ):
        super().__init__()
        self.gated_mlp = GatedMLP(
            in_dim=2 * atom_attr_dim + spherecal_dim,
            out_dims=[128, 64, atom_attr_dim],  # noqa: E501
        )  # [2*atom_attr_dim+edge_attr_prime_dim]  ->  [atom_attr_dim]
        self.edge_layer = LinearLayer(
            in_dim=edge_attr_dim, out_dim=1
        )  # [atom_attr_dim]  ->  [1]

    def forward(
        self,
        atom_attr,
        edge_attr,
        edge_index,
        edge_attr_prime,  # [sum(num_edges),edge_attr_dim]
        num_atoms,  # [batch_size]
    ):
        feat = torch.concat(
            [
                atom_attr[edge_index[0]],
                atom_attr[edge_index[1]],
                edge_attr_prime,
            ],  # noqa: E501
            dim=1,
        )
        atom_attr_prime = self.gated_mlp(feat) * self.edge_layer(edge_attr)
        atom_attr_prime = scatter(
            atom_attr_prime,
            edge_index[1],
            dim=0,
            dim_size=torch.sum(num_atoms).item(),  # noqa: E501
        )
        return atom_attr_prime + atom_attr


class EdgeLayer(nn.Module):
    """e_ij'=e_ij+phi(v_i,v_j,e_ij,u)W*e_ij^0"""

    def init(
        self,
        atom_attr_dim,
        edge_attr_dim,
        spherecal_dim,
    ):
        super().__init__()
        self.gated_mlp = GatedMLP(
            in_dim=2 * atom_attr_dim + spherecal_dim,
            out_dims=[128, 64, edge_attr_dim],  # noqa: E501
        )
        self.edge_layer = LinearLayer(in_dim=edge_attr_dim, out_dim=1)

    def forward(
        self,
        atom_attr,
        edge_attr,
        edge_index,
        edge_attr_prime,  # [sum(num_edges),edge_attr_dim]
    ):
        feat = torch.concat(
            [
                atom_attr[edge_index[0]],
                atom_attr[edge_index[1]],
                edge_attr_prime,
            ],  # noqa: E501
            dim=1,
        )
        edge_attr_prime = self.gated_mlp(feat) * self.edge_layer(edge_attr)
        return edge_attr_prime + edge_attr


class MainBlock(nn.Module):
    """
    MainBlock for Message Passing in M3GNet
    """

    def __init__(
        self,
        max_n,
        max_l,
        cutoff,
        units,
        spherical_dim,
        threebody_cutoff,
    ):
        super().__init__()
        self.gated_mlp_atom = GatedMLP(
            in_dim=2 * units + units,
            out_dims=[units, units],
            activation="swish",  # noqa: E501
        )  # [2*atom_attr_dim+edge_attr_prime_dim]  ->  [units]
        self.edge_layer_atom = SwishLayer(
            in_dim=spherical_dim, out_dim=units, bias=False  # noqa: E501
        )  # [spherecal_dim]  ->  [units]
        self.gated_mlp_edge = GatedMLP(
            in_dim=2 * units + units,
            out_dims=[units, units],
            activation="swish",  # noqa: E501
        )  # [2*atom_attr_dim+edge_attr_prime_dim]  ->  [units]
        self.edge_layer_edge = LinearLayer(
            in_dim=spherical_dim, out_dim=units, bias=False
        )  # [spherecal_dim]  ->  [units]
        self.three_body = ThreeDInteraction(
            max_n, max_l, cutoff, units, max_n * max_l, threebody_cutoff
        )

    def forward(
        self,
        atom_attr,
        edge_attr,
        edge_attr_zero,
        edge_index,
        three_basis,
        three_body_index,
        edge_length,
        num_edges,
        num_triple_ij,
        num_atoms,
    ):
        # threebody interaction
        edge_attr = self.three_body(
            edge_attr,
            three_basis,
            atom_attr,
            edge_index,
            three_body_index,
            edge_length,
            num_edges,
            num_triple_ij.view(-1),
        )

        # update bond feature
        feat = torch.concat(
            [atom_attr[edge_index[0]], atom_attr[edge_index[1]], edge_attr],
            dim=1,  # noqa: E501
        )
        edge_attr = edge_attr + self.gated_mlp_edge(
            feat
        ) * self.edge_layer_edge(  # noqa: E501
            edge_attr_zero
        )

        # update atom feature
        feat = torch.concat(
            [atom_attr[edge_index[0]], atom_attr[edge_index[1]], edge_attr],
            dim=1,  # noqa: E501
        )
        atom_attr_prime = self.gated_mlp_atom(feat) * self.edge_layer_atom(
            edge_attr_zero
        )
        atom_attr = atom_attr + scatter(  # noqa: E501
            atom_attr_prime,
            edge_index[0],
            dim=0,
            dim_size=torch.sum(num_atoms).item(),  # noqa: E501
        )

        return atom_attr, edge_attr
