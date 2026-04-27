"""
Native PyTorch scatter operations compatible with gradient checkpointing.

This module provides scatter operations that don't use TorchScript, making them
compatible with torch.utils.checkpoint with use_reentrant=False.
"""

import torch


def scatter_sum(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = 0,
    dim_size: int | None = None,
) -> torch.Tensor:
    """Scatter sum operation using native PyTorch.

    Args:
        src: Source tensor to scatter from
        index: Index tensor specifying where to scatter
        dim: Dimension along which to scatter
        dim_size: Size of output tensor along dim. If None, inferred from index.

    Returns:
        Output tensor with scattered values summed
    """
    if dim_size is None:
        if index.numel() == 0:
            dim_size = 0
        else:
            dim_size = int(index.max().item()) + 1

    # Expand index to match src dimensions
    index_expanded = index
    for _ in range(src.dim() - index.dim()):
        index_expanded = index_expanded.unsqueeze(-1)
    index_expanded = index_expanded.expand_as(src)

    # Create output tensor and scatter
    out_shape = list(src.shape)
    out_shape[dim] = dim_size
    out = torch.zeros(out_shape, dtype=src.dtype, device=src.device)

    return out.scatter_add_(dim, index_expanded, src)
