# -*- coding: utf-8 -*-
"""
Ref:
    - https://github.com/mir-group/nequip
    - https://www.nature.com/articles/s41467-022-29939-5
"""

import math
from typing import Optional

import torch
from e3nn.math import soft_one_hot_linspace
from torch import nn

from mattersim.jit_compile_tools.jit import compile_mode


class e3nn_basias(nn.Module):
    def __init__(
        self,
        r_max: float,
        r_min: Optional[float] = None,
        e3nn_basis_name: str = "gaussian",
        num_basis: int = 8,
    ):
        super().__init__()
        self.r_max = r_max
        self.r_min = r_min if r_min is not None else 0.0
        self.e3nn_basis_name = e3nn_basis_name
        self.num_basis = num_basis

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return soft_one_hot_linspace(
            x,
            start=self.r_min,
            end=self.r_max,
            number=self.num_basis,
            basis=self.e3nn_basis_name,
            cutoff=True,
        )

    def _make_tracing_inputs(self, n: int):
        return [{"forward": (torch.randn(5, 1),)} for _ in range(n)]


class BesselBasis(nn.Module):
    def __init__(self, r_max, num_basis=8, trainable=True):
        r"""Radial Bessel Basis, as proposed in
            DimeNet: https://arxiv.org/abs/2003.03123

        Parameters
        ----------
        r_max : float
            Cutoff radius

        num_basis : int
            Number of Bessel Basis functions

        trainable : bool
            Train the :math:`n \pi` part or not.
        """
        super(BesselBasis, self).__init__()

        self.trainable = trainable
        self.num_basis = num_basis

        self.r_max = float(r_max)
        self.prefactor = 2.0 / self.r_max

        bessel_weights = (
            torch.linspace(start=1.0, end=num_basis, steps=num_basis) * math.pi
        )
        if self.trainable:
            self.bessel_weights = nn.Parameter(bessel_weights)
        else:
            self.register_buffer("bessel_weights", bessel_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluate Bessel Basis for input x.

        Parameters
        ----------
        x : torch.Tensor
            Input
        """
        numerator = torch.sin(
            self.bessel_weights * x.unsqueeze(-1) / self.r_max  # noqa: E501
        )

        return self.prefactor * (numerator / x.unsqueeze(-1))


@compile_mode("script")
class SmoothBesselBasis(nn.Module):
    def __init__(self, r_max, max_n=10):
        r"""Smooth Radial Bessel Basis, as proposed
            in DimeNet: https://arxiv.org/abs/2003.03123
            This is an orthogonal basis with first
            and second derivative at the cutoff
            equals to zero. The function was derived from
            the order 0 spherical Bessel function,
            and was expanded by the different zero roots
        Ref:
            https://arxiv.org/pdf/1907.02374.pdf
        Args:
            r_max: torch.Tensor distance tensor
            max_n: int, max number of basis, expanded by the zero roots
        Returns: expanded spherical harmonics with
                 derivatives smooth at boundary
        Parameters
        ----------
        """
        super(SmoothBesselBasis, self).__init__()
        self.max_n = max_n
        n = torch.arange(0, max_n).float()[None, :]
        PI = 3.1415926535897
        SQRT2 = 1.41421356237
        fnr = (
            (-1) ** n
            * SQRT2
            * PI
            / r_max**1.5
            * (n + 1)
            * (n + 2)
            / torch.sqrt(2 * n**2 + 6 * n + 5)
        )
        en = n**2 * (n + 2) ** 2 / (4 * (n + 1) ** 4 + 1)
        dn = [torch.tensor(1.0).float()]
        for i in range(1, max_n):
            dn.append(1 - en[0, i] / dn[-1])
        dn = torch.stack(dn)
        self.register_buffer("dn", dn)
        self.register_buffer("en", en)
        self.register_buffer("fnr_weights", fnr)
        self.register_buffer(
            "n_1_pi_cutoff",
            ((torch.arange(0, max_n).float() + 1) * PI / r_max).reshape(1, -1),
        )
        self.register_buffer(
            "n_2_pi_cutoff",
            ((torch.arange(0, max_n).float() + 2) * PI / r_max).reshape(1, -1),
        )
        self.register_buffer("r_max", torch.tensor(r_max))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluate Smooth Bessel Basis for input x.

        Parameters
        ----------
        x : torch.Tensor
            Input
        """
        x_1 = x.unsqueeze(-1) * self.n_1_pi_cutoff
        x_2 = x.unsqueeze(-1) * self.n_2_pi_cutoff
        fnr = self.fnr_weights * (torch.sin(x_1) / x_1 + torch.sin(x_2) / x_2)
        gn = [fnr[:, 0]]
        for i in range(1, self.max_n):
            gn.append(
                1
                / torch.sqrt(self.dn[i])
                * (
                    fnr[:, i]
                    + torch.sqrt(self.en[0, i] / self.dn[i - 1]) * gn[-1]  # noqa: E501
                )
            )
        return torch.transpose(torch.stack(gn), 1, 0)


# class GaussianBasis(nn.Module):
#     r_max: float

#     def __init__(self, r_max, r_min=0.0, num_basis=8, trainable=True):
#         super().__init__()

#         self.trainable = trainable
#         self.num_basis = num_basis

#         self.r_max = float(r_max)
#         self.r_min = float(r_min)

#         means = torch.linsspace(self.r_min, self.r_max, self.num_basis)
#         stds = torch.full(size=means.size, fill_value=means[1] - means[0])
#         if self.trainable:
#             self.means = nn.Parameter(means)
#             self.stds = nn.Parameter(stds)
#         else:
#             self.register_buffer("means", means)
#             self.register_buffer("stds", stds)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = (x[..., None] - self.means) / self.stds
#         x = x.square().mul(-0.5).exp() / self.stds  # sqrt(2 * pi)
