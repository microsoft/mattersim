"""Tests for batch_to_dict — verifies that tensor device placement works
correctly (related to GitHub issue #113).
"""

import pytest
import torch
from types import SimpleNamespace

from mattersim.forcefield.potential import batch_to_dict

TENSOR_KEYS = [
    "atom_pos",
    "cell",
    "pbc_offsets",
    "atom_attr",
    "edge_index",
    "three_body_indices",
    "num_three_body",
    "num_bonds",
    "num_triple_ij",
    "num_atoms",
    "num_graphs",
    "batch",
]


def _make_graph_batch(device="cpu"):
    """Create a minimal mock graph_batch with all required tensor fields."""
    return SimpleNamespace(
        atom_pos=torch.randn(4, 3, device=device),
        cell=torch.randn(1, 3, 3, device=device),
        pbc_offsets=torch.zeros(6, 3, device=device),
        atom_attr=torch.randn(4, 16, device=device),
        edge_index=torch.randint(0, 4, (2, 6), device=device),
        three_body_indices=torch.randint(0, 6, (3, 8), device=device),
        num_three_body=torch.tensor([8], device=device),
        num_bonds=torch.tensor([6], device=device),
        num_triple_ij=torch.tensor([3, 2, 1, 0, 0, 0], device=device),
        num_atoms=torch.tensor([4], device=device),
        num_graphs=1,  # scalar, not a tensor on the batch object
        batch=torch.zeros(4, dtype=torch.long, device=device),
    )


class TestBatchToDict:
    """Tests for the batch_to_dict helper function."""

    def test_all_tensors_on_target_device(self, available_device):
        """Every tensor in the returned dict must be on the requested device."""
        batch = _make_graph_batch("cpu")
        result = batch_to_dict(batch, device=available_device)

        for key in TENSOR_KEYS:
            assert key in result, f"Missing key: {key}"
            val = result[key]
            if isinstance(val, torch.Tensor):
                assert val.device.type == available_device, (
                    f"'{key}' on {val.device}, expected {available_device}"
                )

    def test_cross_device_move(self, available_device):
        """Tensors created on CPU must end up on the target device —
        the device parameter must not be silently ignored."""
        batch = _make_graph_batch("cpu")
        result = batch_to_dict(batch, device=available_device)

        for key in TENSOR_KEYS:
            val = result[key]
            if isinstance(val, torch.Tensor):
                assert val.device.type == available_device, (
                    f"'{key}' still on {val.device} instead of {available_device}"
                )

    def test_num_graphs_is_tensor_on_correct_device(self, available_device):
        """num_graphs (a plain int on the batch) must become a tensor
        on the target device."""
        batch = _make_graph_batch("cpu")
        result = batch_to_dict(batch, device=available_device)

        assert isinstance(result["num_graphs"], torch.Tensor)
        assert result["num_graphs"].item() == 1
        assert result["num_graphs"].device.type == available_device

    def test_returns_all_expected_keys(self):
        """The returned dict must contain exactly the expected keys."""
        batch = _make_graph_batch("cpu")
        result = batch_to_dict(batch, device="cpu")

        assert set(result.keys()) == set(TENSOR_KEYS)

    def test_tensor_values_preserved(self):
        """Moving to the same device must not alter tensor values."""
        batch = _make_graph_batch("cpu")
        result = batch_to_dict(batch, device="cpu")

        torch.testing.assert_close(result["atom_pos"], batch.atom_pos)
        torch.testing.assert_close(result["edge_index"], batch.edge_index)

    def test_unsupported_model_type_raises(self):
        """Non-m3gnet model types should raise NotImplementedError."""
        batch = _make_graph_batch("cpu")
        with pytest.raises(NotImplementedError):
            batch_to_dict(batch, model_type="graphormer", device="cpu")
        with pytest.raises(NotImplementedError):
            batch_to_dict(batch, model_type="unknown", device="cpu")
