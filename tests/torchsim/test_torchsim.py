"""Tests for the mattersim.torchsim integration."""

from typing import Literal

import numpy as np
import pytest
import torch
import torch_sim as ts

from mattersim.forcefield.potential import Potential
from mattersim.torchsim.graph_construction import _normalize_nonperiodic_systems
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
# _normalize_nonperiodic_systems tests (lightweight, no model needed)
# ---------------------------------------------------------------------------


class TestNormalizeNonperiodicSystems:
    """Tests for the non-periodic normalization logic."""

    def test_periodic_system_unchanged(self, si_diamond_cubic):
        """Periodic systems should pass through without modification."""
        state = ts.initialize_state(
            [si_diamond_cubic], device="cpu", dtype=torch.float64
        )
        pbc = state.pbc.unsqueeze(0) if state.pbc.dim() == 1 else state.pbc
        cell = state.row_vector_cell
        pos = state.positions

        pos_out, cell_out, pbc_out = _normalize_nonperiodic_systems(
            pos=pos,
            cell=cell,
            pbc=pbc,
            system_idx=state.system_idx,
            n_systems=state.n_systems,
            twobody_cutoff=5.0,
            threebody_cutoff=4.0,
        )

        # Should return the same tensor objects (no clone) when all periodic
        assert pos_out is pos
        assert cell_out is cell
        assert pbc_out is pbc

    def test_nonperiodic_gets_fake_cell(self, water_molecule):
        """Non-periodic molecules should get a large fake periodic cell."""
        state = ts.initialize_state([water_molecule], device="cpu", dtype=torch.float64)
        pbc = state.pbc.unsqueeze(0) if state.pbc.dim() == 1 else state.pbc

        pos_out, cell_out, pbc_out = _normalize_nonperiodic_systems(
            pos=state.positions,
            cell=state.row_vector_cell,
            pbc=pbc,
            system_idx=state.system_idx,
            n_systems=state.n_systems,
            twobody_cutoff=5.0,
            threebody_cutoff=4.0,
        )

        # PBC should now be True in all directions
        assert pbc_out.all()

        # Cell should be a diagonal matrix with a large box length
        assert cell_out[0, 0, 0] > 20.0  # at least pad = 5*5 = 25 Å
        assert cell_out[0, 0, 0] == cell_out[0, 1, 1] == cell_out[0, 2, 2]
        assert cell_out[0, 0, 1] == 0.0  # off-diagonal should be zero

        # All positions should be inside the box [0, box_len)
        box_len = cell_out[0, 0, 0].item()
        assert (pos_out >= 0).all()
        assert (pos_out < box_len).all()

    def test_matches_ase_normalize_atoms(self, water_molecule):
        """Normalization should produce the same cell and positions as
        _normalize_atoms from the ASE code path."""
        from mattersim.datasets.utils.converter import _normalize_atoms

        twobody_cutoff = 5.0
        threebody_cutoff = 4.0

        # ASE path
        atoms_norm = _normalize_atoms(water_molecule, twobody_cutoff, threebody_cutoff)
        ase_cell = np.array(atoms_norm.cell)
        ase_pos = atoms_norm.positions

        # TorchSim path
        state = ts.initialize_state([water_molecule], device="cpu", dtype=torch.float64)
        pbc = state.pbc.unsqueeze(0) if state.pbc.dim() == 1 else state.pbc

        pos_out, cell_out, pbc_out = _normalize_nonperiodic_systems(
            pos=state.positions,
            cell=state.row_vector_cell,
            pbc=pbc,
            system_idx=state.system_idx,
            n_systems=state.n_systems,
            twobody_cutoff=twobody_cutoff,
            threebody_cutoff=threebody_cutoff,
        )

        np.testing.assert_allclose(
            cell_out[0].numpy(),
            ase_cell,
            atol=1e-10,
            err_msg="Cell matrices should match between TorchSim and ASE paths",
        )
        np.testing.assert_allclose(
            pos_out.numpy(),
            ase_pos,
            atol=1e-10,
            err_msg="Wrapped positions should match between TorchSim and ASE paths",
        )

    def test_original_tensors_not_mutated(self, water_molecule):
        """Normalization should not mutate the original SimState tensors."""
        state = ts.initialize_state([water_molecule], device="cpu", dtype=torch.float64)
        pbc = state.pbc.unsqueeze(0) if state.pbc.dim() == 1 else state.pbc

        orig_pos = state.positions.clone()
        orig_cell = state.row_vector_cell.clone()
        orig_pbc = pbc.clone()

        _normalize_nonperiodic_systems(
            pos=state.positions,
            cell=state.row_vector_cell,
            pbc=pbc,
            system_idx=state.system_idx,
            n_systems=state.n_systems,
            twobody_cutoff=5.0,
            threebody_cutoff=4.0,
        )

        torch.testing.assert_close(state.positions, orig_pos)
        torch.testing.assert_close(state.row_vector_cell, orig_cell)
        torch.testing.assert_close(pbc, orig_pbc)


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

    def test_wrapper_forward(self, torchsim_wrapper: TorchSimWrapper, si_diamond_cubic):
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

    def test_wrapper_molecule_consistency(
        self, torchsim_wrapper: TorchSimWrapper, water_molecule
    ):
        """TorchSimWrapper and MatterSimCalculator should agree on molecules.

        This is a regression test for GitHub issue #160.
        """
        from mattersim.forcefield.potential import MatterSimCalculator

        # ASE calculator path
        calc = MatterSimCalculator(
            potential=torchsim_wrapper.model,
            device=DEVICE,
            direct_graph=True,
        )
        water_molecule.calc = calc
        ase_energy = water_molecule.get_potential_energy()
        ase_forces = water_molecule.get_forces()

        # TorchSim wrapper path
        state = ts.initialize_state(
            [water_molecule], device=DEVICE, dtype=torch.float64
        )
        result = torchsim_wrapper(state)

        torch.testing.assert_close(
            result["energy"].item(),
            ase_energy,
            rtol=1e-5,
            atol=1e-5,
        )
        torch.testing.assert_close(
            result["forces"].cpu(),
            torch.tensor(ase_forces, dtype=result["forces"].dtype),
            rtol=1e-5,
            atol=1e-5,
        )
