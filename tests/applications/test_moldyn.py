"""Tests for MolecularDynamics — verifies that all supported ensembles
can be initialized and run without errors.
"""

import pytest
from ase.calculators.lj import LennardJones

from mattersim.applications.moldyn import MolecularDynamics


class TestMolecularDynamics:
    """Tests for ensemble initialization and basic MD runs."""

    def test_nvt_nose_hoover_initializes(self, si_diamond_cubic):
        """NVT_NOSE_HOOVER ensemble must initialize without error."""
        si_diamond_cubic.calc = LennardJones()
        md = MolecularDynamics(
            atoms=si_diamond_cubic,
            ensemble="nvt_nose_hoover",
            temperature=300,
            timestep=1.0,
            logfile=None,
        )
        assert md.dyn is not None

    def test_nvt_nose_hoover_runs(self, si_diamond_cubic):
        """NVT_NOSE_HOOVER ensemble must run a few steps without error."""
        si_diamond_cubic.calc = LennardJones()
        md = MolecularDynamics(
            atoms=si_diamond_cubic,
            ensemble="nvt_nose_hoover",
            temperature=300,
            timestep=1.0,
            logfile=None,
        )
        md.run(n_steps=5)

    def test_nvt_berendsen_initializes(self, si_diamond_cubic):
        """NVT_BERENDSEN ensemble must initialize without error."""
        si_diamond_cubic.calc = LennardJones()
        md = MolecularDynamics(
            atoms=si_diamond_cubic,
            ensemble="nvt_berendsen",
            temperature=300,
            timestep=1.0,
            logfile=None,
        )
        assert md.dyn is not None

    def test_nvt_berendsen_runs(self, si_diamond_cubic):
        """NVT_BERENDSEN ensemble must run a few steps without error."""
        si_diamond_cubic.calc = LennardJones()
        md = MolecularDynamics(
            atoms=si_diamond_cubic,
            ensemble="nvt_berendsen",
            temperature=300,
            timestep=1.0,
            logfile=None,
        )
        md.run(n_steps=5)

    def test_unsupported_ensemble_raises(self, si_diamond_cubic):
        """An unsupported ensemble name must raise NotImplementedError."""
        si_diamond_cubic.calc = LennardJones()
        with pytest.raises(NotImplementedError):
            MolecularDynamics(
                atoms=si_diamond_cubic,
                ensemble="npt",
                temperature=300,
                timestep=1.0,
                logfile=None,
            )
