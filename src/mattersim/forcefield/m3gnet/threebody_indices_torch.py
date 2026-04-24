"""
Pure PyTorch implementation of three-body index construction.

Canonical location is mattersim.datasets.utils.threebody_indices_torch.
This module re-exports for backward compatibility.
"""

from mattersim.datasets.utils.threebody_indices_torch import compute_threebody_torch

__all__ = ["compute_threebody_torch"]
