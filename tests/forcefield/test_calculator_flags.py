"""Tests for MatterSimCalculator direct_graph / compile flag behavior."""

import numpy as np
import pytest

from mattersim.forcefield import MatterSimCalculator


class TestCalculatorDirectGraphFlag:
    """Tests that direct_graph and compile flags control graph path and
    torch.compile independently."""

    def test_default_flags(self, mattersim_calc_best_device):
        """By default both direct_graph and compile are off."""
        calc = mattersim_calc_best_device
        assert calc._use_direct_graph is False
        assert calc._compiled is False

    def test_direct_graph_only(self, mattersim_calc_best_device):
        """direct_graph=True enables direct graph path without torch.compile."""
        calc = MatterSimCalculator(
            direct_graph=True, device=mattersim_calc_best_device.device
        )
        assert calc._use_direct_graph is True
        assert calc._compiled is False

    def test_compile_implies_direct_graph(self, mattersim_calc_best_device):
        """compile=True should also enable the direct graph path."""
        calc = MatterSimCalculator(
            compile=True, device=mattersim_calc_best_device.device
        )
        assert calc._use_direct_graph is True
        assert calc._compiled is True

    def test_both_flags_enabled(self, mattersim_calc_best_device):
        """Setting both direct_graph=True and compile=True works."""
        calc = MatterSimCalculator(
            direct_graph=True, compile=True,
            device=mattersim_calc_best_device.device,
        )
        assert calc._use_direct_graph is True
        assert calc._compiled is True


class TestCalculatorDirectGraphResults:
    """Tests that direct_graph path produces the same results as the
    default legacy path."""

    def test_energy_matches_legacy(self, si_diamond, mattersim_calc_best_device):
        """Energy from direct_graph path must match the legacy path."""
        si_diamond.calc = mattersim_calc_best_device
        ref_energy = si_diamond.get_potential_energy()

        from ase.build import bulk
        atoms = bulk("Si", "diamond", a=5.43)
        calc_direct = MatterSimCalculator(
            direct_graph=True, device=mattersim_calc_best_device.device
        )
        atoms.calc = calc_direct
        energy_direct = atoms.get_potential_energy()

        np.testing.assert_allclose(energy_direct, ref_energy, atol=1e-4)

    def test_forces_match_legacy(self, si_diamond, mattersim_calc_best_device):
        """Forces from direct_graph path must match the legacy path."""
        si_diamond.calc = mattersim_calc_best_device
        ref_forces = si_diamond.get_forces()

        from ase.build import bulk
        atoms = bulk("Si", "diamond", a=5.43)
        calc_direct = MatterSimCalculator(
            direct_graph=True, device=mattersim_calc_best_device.device
        )
        atoms.calc = calc_direct
        forces_direct = atoms.get_forces()

        np.testing.assert_allclose(forces_direct, ref_forces, atol=1e-4)

    def test_stress_matches_legacy(self, si_diamond, mattersim_calc_best_device):
        """Stress from direct_graph path must match the legacy path."""
        si_diamond.calc = mattersim_calc_best_device
        ref_stress = si_diamond.get_stress()

        from ase.build import bulk
        atoms = bulk("Si", "diamond", a=5.43)
        calc_direct = MatterSimCalculator(
            direct_graph=True, device=mattersim_calc_best_device.device
        )
        atoms.calc = calc_direct
        stress_direct = atoms.get_stress()

        np.testing.assert_allclose(stress_direct, ref_stress, atol=1e-4)
