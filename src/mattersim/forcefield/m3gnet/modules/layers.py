# -*- coding: utf-8 -*-
from typing import Union

import torch.nn as nn


class LinearLayer(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        bias=True,
    ):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=bias)

    def forward(self, x):
        return self.linear(x)


class SigmoidLayer(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        bias=True,
    ):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=bias)
        self.sigmoid = nn.Sigmoid()

    def forward(
        self,
        x,
    ):
        return self.sigmoid(self.linear(x))


class SwishLayer(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        bias=True,
    ):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=bias)
        self.sigmoid = nn.Sigmoid()

    def forward(
        self,
        x,
    ):
        x = self.linear(x)
        return x * self.sigmoid(x)


class ReLULayer(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        bias=True,
    ):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=bias)
        self.relu = nn.ReLU()

    def forward(
        self,
        x,
    ):
        return self.relu(self.linear(x))


class GatedMLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dims: list,
        activation: Union[list[Union[str, None]], str] = "swish",
        use_bias: bool = True,
    ):
        super().__init__()
        input_dim = in_dim
        if isinstance(activation, str) or activation is None:
            activation = [activation] * len(out_dims)
        else:
            assert len(activation) == len(
                out_dims
            ), "activation and out_dims must have the same length"
        module_list_g = []
        for i in range(len(out_dims)):
            if activation[i] == "swish":
                module_list_g.append(  # noqa: E501
                    SwishLayer(input_dim, out_dims[i], bias=use_bias)
                )
            elif activation[i] == "sigmoid":
                module_list_g.append(
                    SigmoidLayer(input_dim, out_dims[i], bias=use_bias)
                )
            elif activation[i] is None:
                module_list_g.append(  # noqa: E501
                    LinearLayer(input_dim, out_dims[i], bias=use_bias)
                )
            input_dim = out_dims[i]
        module_list_sigma = []
        activation[-1] = "sigmoid"
        input_dim = in_dim
        for i in range(len(out_dims)):
            if activation[i] == "swish":
                module_list_sigma.append(
                    SwishLayer(input_dim, out_dims[i], bias=use_bias)
                )
            elif activation[i] == "sigmoid":
                module_list_sigma.append(
                    SigmoidLayer(input_dim, out_dims[i], bias=use_bias)
                )
            elif activation[i] is None:
                module_list_sigma.append(
                    LinearLayer(input_dim, out_dims[i], bias=use_bias)
                )
            else:
                raise NotImplementedError
            input_dim = out_dims[i]
        self.g = nn.Sequential(*module_list_g)
        self.sigma = nn.Sequential(*module_list_sigma)

    def forward(
        self,
        x,
    ):
        return self.g(x) * self.sigma(x)


class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dims: list,
        activation: Union[list[Union[str, None]], str, None] = "swish",
        use_bias: bool = True,
    ):
        super().__init__()
        input_dim = in_dim
        if isinstance(activation, str) or activation is None:
            activation = [activation] * len(out_dims)
        else:
            assert len(activation) == len(
                out_dims
            ), "activation and out_dims must have the same length"
        module_list = []
        for i in range(len(out_dims)):
            if activation[i] == "swish":
                module_list.append(
                    SwishLayer(input_dim, out_dims[i], bias=use_bias)  # noqa: E501
                )
            elif activation[i] == "sigmoid":
                module_list.append(
                    SigmoidLayer(input_dim, out_dims[i], bias=use_bias)  # noqa: E501
                )
            elif activation[i] is None:
                module_list.append(
                    LinearLayer(input_dim, out_dims[i], bias=use_bias)  # noqa: E501
                )
            else:
                raise NotImplementedError
            input_dim = out_dims[i]
        self.mlp = nn.Sequential(*module_list)

    def forward(
        self,
        x,
    ):
        return self.mlp(x)
