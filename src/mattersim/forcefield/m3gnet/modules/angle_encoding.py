# -*- coding: utf-8 -*-
# """
# Ref:
#     - https://github.com/akirasosa/pytorch-dimenet
#     - https://arxiv.org/abs/2003.03123
# """
#
import math

import torch
import torch.nn as nn


@torch.jit.script
def _spherical_harmonics(lmax: int, x: torch.Tensor) -> torch.Tensor:
    sh_0_0 = torch.ones_like(x) * 0.5 * math.sqrt(1.0 / math.pi)
    if lmax == 0:
        return torch.stack(
            [
                sh_0_0,
            ],
            dim=-1,
        )

    sh_1_1 = math.sqrt(3.0 / (4.0 * math.pi)) * x
    if lmax == 1:
        return torch.stack([sh_0_0, sh_1_1], dim=-1)

    sh_2_2 = math.sqrt(5.0 / (16.0 * math.pi)) * (3.0 * x**2 - 1.0)
    if lmax == 2:
        return torch.stack([sh_0_0, sh_1_1, sh_2_2], dim=-1)

    sh_3_3 = math.sqrt(7.0 / (16.0 * math.pi)) * x * (5.0 * x**2 - 3.0)
    if lmax == 3:
        return torch.stack([sh_0_0, sh_1_1, sh_2_2, sh_3_3], dim=-1)

    raise ValueError("lmax must be less than 8")


class SphericalBasisLayer(nn.Module):
    def __init__(self, max_n, max_l, cutoff):
        super(SphericalBasisLayer, self).__init__()

        assert max_l <= 4, "lmax must be less than 5"
        assert max_n <= 4, "max_n must be less than 5"

        self.max_n = max_n
        self.max_l = max_l
        self.cutoff = cutoff

        # retrieve formulas
        self.register_buffer(
            "factor", torch.sqrt(torch.tensor(2.0 / (self.cutoff**3)))
        )
        self.coef = torch.zeros(4, 9, 4)
        self.coef[0, 0, :] = torch.tensor(
            [
                3.14159274101257,
                6.28318548202515,
                9.42477798461914,
                12.5663709640503,
            ]  # noqa: E501
        )
        self.coef[1, :4, :] = torch.tensor(
            [
                [
                    -1.02446483277785,
                    -1.00834335996107,
                    -1.00419641763893,
                    -1.00252381898662,
                ],
                [
                    4.49340963363647,
                    7.7252516746521,
                    10.9041213989258,
                    14.0661935806274,
                ],  # noqa: E501
                [
                    0.22799275301076,
                    0.130525632358311,
                    0.092093290316619,
                    0.0712718627992818,
                ],
                [
                    4.49340963363647,
                    7.7252516746521,
                    10.9041213989258,
                    14.0661935806274,
                ],  # noqa: E501
            ]
        )
        self.coef[2, :6, :] = torch.tensor(
            [
                [
                    -1.04807944170731,
                    -1.01861796359391,
                    -1.01002272174988,
                    -1.00628955560036,
                ],
                [
                    5.76345920562744,
                    9.09501171112061,
                    12.322940826416,
                    15.5146026611328,
                ],  # noqa: E501
                [
                    0.545547077361439,
                    0.335992298618515,
                    0.245888396928293,
                    0.194582402961821,
                ],
                [
                    5.76345920562744,
                    9.09501171112061,
                    12.322940826416,
                    15.5146026611328,
                ],  # noqa: E501
                [
                    0.0946561878721665,
                    0.0369424811413594,
                    0.0199537107571916,
                    0.0125418876146463,
                ],
                [
                    5.76345920562744,
                    9.09501171112061,
                    12.322940826416,
                    15.5146026611328,
                ],  # noqa: E501
            ]
        )
        self.coef[3, :8, :] = torch.tensor(
            [
                [
                    1.06942831392075,
                    1.0292173312802,
                    1.01650804843248,
                    1.01069656069999,
                ],  # noqa: E501
                [
                    6.9879322052002,
                    10.4171180725098,
                    13.6980228424072,
                    16.9236221313477,
                ],  # noqa: E501
                [
                    0.918235852195231,
                    0.592803493701152,
                    0.445250264272671,
                    0.358326327374518,
                ],
                [
                    6.9879322052002,
                    10.4171180725098,
                    13.6980228424072,
                    16.9236221313477,
                ],  # noqa: E501
                [
                    0.328507713452024,
                    0.142266673367543,
                    0.0812617757677838,
                    0.0529328657590962,
                ],
                [
                    6.9879322052002,
                    10.4171180725098,
                    13.6980228424072,
                    16.9236221313477,
                ],  # noqa: E501
                [
                    0.0470107184508114,
                    0.0136570088173405,
                    0.0059323726279831,
                    0.00312775039221944,
                ],
                [
                    6.9879322052002,
                    10.4171180725098,
                    13.6980228424072,
                    16.9236221313477,
                ],  # noqa: E501
            ]
        )

    def forward(self, r, theta_val):
        r = r / self.cutoff
        # Denote empty lists for rbf and cbf
        rbfs = []

        for j in range(self.max_l):
            rbfs.append(torch.sin(self.coef[0, 0, j] * r) / r)

        if self.max_n > 1:
            for j in range(self.max_l):
                rbfs.append(
                    (
                        self.coef[1, 0, j]
                        * r
                        * torch.cos(self.coef[1, 1, j] * r)  # noqa: E501
                        + self.coef[1, 2, j]
                        * torch.sin(self.coef[1, 3, j] * r)  # noqa: E501
                    )
                    / r**2
                )

            if self.max_n > 2:
                for j in range(self.max_l):
                    rbfs.append(
                        (
                            self.coef[2, 0, j]
                            * (r**2)
                            * torch.sin(self.coef[2, 1, j] * r)
                            - self.coef[2, 2, j]
                            * r
                            * torch.cos(self.coef[2, 3, j] * r)  # noqa: E501
                            + self.coef[2, 4, j]
                            * torch.sin(self.coef[2, 5, j] * r)  # noqa: E501
                        )
                        / (r**3)
                    )

                if self.max_n > 3:
                    for j in range(self.max_l):
                        rbfs.append(
                            (
                                self.coef[3, 0, j]
                                * (r**3)
                                * torch.cos(self.coef[3, 1, j] * r)
                                - self.coef[3, 2, j]
                                * (r**2)
                                * torch.sin(self.coef[3, 3, j] * r)
                                - self.coef[3, 4, j]
                                * r
                                * torch.cos(self.coef[3, 5, j] * r)
                                + self.coef[3, 6, j]
                                * torch.sin(self.coef[3, 7, j] * r)  # noqa: E501
                            )
                            / r**4
                        )

        rbfs = torch.stack(rbfs, dim=-1)
        rbfs = rbfs * self.factor

        cbfs = _spherical_harmonics(self.max_l - 1, torch.cos(theta_val))
        cbfs = cbfs.repeat_interleave(self.max_n, dim=1)

        return rbfs * cbfs
