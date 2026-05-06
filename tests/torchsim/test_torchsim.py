"""Tests for the mattersim.torchsim integration."""

from typing import Literal

import pytest
import torch
import torch_sim as ts

from mattersim.forcefield.potential import Potential
from mattersim.torchsim.torchsim_wrapper import TorchSimWrapper

requires_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)

DEVICE: Literal["cpu", "cuda"] = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Fixtures (torchsim-specific only; si_diamond_cubic and
# mattersim_potential_best_device come from conftest.py)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def torchsim_wrapper(mattersim_potential_best_device: Potential) -> TorchSimWrapper:
    """TorchSimWrapper around the shared potential."""
    return TorchSimWrapper(
        model=mattersim_potential_best_device,
        device=DEVICE,
        dtype=torch.float64,
    )


# ---------------------------------------------------------------------------
# Package import smoke test
# ---------------------------------------------------------------------------


def test_package_imports():
    """Verify that the public API is importable from the package root."""
    from mattersim.torchsim import TorchSimWrapper, get_torchsim_wrapper

    assert TorchSimWrapper is not None
    assert get_torchsim_wrapper is not None


# ---------------------------------------------------------------------------
# TorchSimWrapper tests
# ---------------------------------------------------------------------------


@pytest.mark.slow
@requires_gpu
class TestTorchSimWrapper:
    """Tests for the TorchSimWrapper model interface."""

    def test_wrapper_creation(self, torchsim_wrapper: TorchSimWrapper):
        assert torchsim_wrapper.two_body_cutoff > 0
        assert torchsim_wrapper.three_body_cutoff > 0
        assert "energy" in torchsim_wrapper.implemented_properties
        assert "forces" in torchsim_wrapper.implemented_properties
        assert "stress" in torchsim_wrapper.implemented_properties

    def test_wrapper_forward(
        self, torchsim_wrapper: TorchSimWrapper, si_diamond_cubic
    ):
        state = ts.initialize_state(
            [si_diamond_cubic], device=DEVICE, dtype=torch.float64
        )
        result = torchsim_wrapper(state)

        assert "energy" in result
        assert "forces" in result
        assert "stress" in result
        assert result["energy"].shape == (1,)
        assert result["forces"].shape == (len(si_diamond_cubic), 3)
        assert result["stress"].shape == (1, 3, 3)
