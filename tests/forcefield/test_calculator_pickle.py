"""Tests for MatterSimCalculator pickling (GitHub issue #83).

Previously, pickling MatterSimCalculator raised:
    TypeError: cannot pickle 'weakref.ReferenceType' object
due to torch_ema.ExponentialMovingAverage storing weakrefs to model
parameters in Potential.ema._params_refs.
"""

import copy
import pickle

import numpy as np
import pytest

from mattersim.forcefield import MatterSimCalculator


class TestMatterSimCalculatorPickle:
    """Tests for pickle and deepcopy support on MatterSimCalculator."""

    def test_pickle_dumps_succeeds(self, mattersim_calc_best_device):
        """pickle.dumps must not raise TypeError about weakrefs."""
        data = pickle.dumps(mattersim_calc_best_device)
        assert len(data) > 0

    def test_pickle_roundtrip_restores_calculator(self, mattersim_calc_best_device):
        """A pickled+unpickled calculator must still be functional."""
        data = pickle.dumps(mattersim_calc_best_device)
        restored = pickle.loads(data)
        assert isinstance(restored, MatterSimCalculator)
        assert restored.device == mattersim_calc_best_device.device

    def test_pickle_roundtrip_preserves_energy(
        self, si_diamond, mattersim_calc_best_device
    ):
        """Energy computed before and after pickle must match."""
        si_diamond.calc = mattersim_calc_best_device
        ref_energy = si_diamond.get_potential_energy()

        restored = pickle.loads(pickle.dumps(mattersim_calc_best_device))
        from ase.build import bulk

        atoms2 = bulk("Si", "diamond", a=5.43)
        atoms2.calc = restored
        energy2 = atoms2.get_potential_energy()

        np.testing.assert_allclose(energy2, ref_energy, atol=1e-5)

    def test_pickle_roundtrip_preserves_forces(
        self, si_diamond, mattersim_calc_best_device
    ):
        """Forces computed before and after pickle must match."""
        si_diamond.calc = mattersim_calc_best_device
        ref_forces = si_diamond.get_forces()

        restored = pickle.loads(pickle.dumps(mattersim_calc_best_device))
        from ase.build import bulk

        atoms2 = bulk("Si", "diamond", a=5.43)
        atoms2.calc = restored
        forces2 = atoms2.get_forces()

        np.testing.assert_allclose(forces2, ref_forces, atol=1e-5)

    def test_deepcopy_succeeds(self, mattersim_calc_best_device):
        """copy.deepcopy must work without errors."""
        calc_copy = copy.deepcopy(mattersim_calc_best_device)
        assert isinstance(calc_copy, MatterSimCalculator)

    def test_deepcopy_produces_independent_calculator(
        self, si_diamond, mattersim_calc_best_device
    ):
        """A deepcopied calculator must produce correct results independently."""
        si_diamond.calc = mattersim_calc_best_device
        ref_energy = si_diamond.get_potential_energy()

        calc_copy = copy.deepcopy(mattersim_calc_best_device)
        from ase.build import bulk

        atoms2 = bulk("Si", "diamond", a=5.43)
        atoms2.calc = calc_copy
        energy2 = atoms2.get_potential_energy()

        np.testing.assert_allclose(energy2, ref_energy, atol=1e-5)
