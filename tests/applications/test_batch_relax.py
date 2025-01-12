# -*- coding: utf-8 -*-
import unittest

import numpy as np
from ase import Atoms

from mattersim.applications.batch_relax import BatchRelaxer
from mattersim.forcefield.potential import Potential


class RelaxerTestCase(unittest.TestCase):
    def setUp(self):
        # Create an example structure for testing
        a = 3.567
        positions = [
            (0, 0, 0),
            (a / 4, a / 4, a / 4),
            (a / 2, a / 2, 0),
            (a / 2, 0, a / 2),
            (0, a / 2, a / 2),
            (a / 4, 3 * a / 4, 3 * a / 4),
            (3 * a / 4, a / 4, 3 * a / 4),
            (3 * a / 4, 3 * a / 4, a / 4),
        ]
        cell = [(a, 0, 0), (0, a, 0), (0, 0, a)]
        self.atoms_ideal = Atoms(
            "C8", positions=positions, cell=cell, pbc=True  # noqa: E501
        )

        # Create an example structure with displaced atoms for testing
        a = 3.567  # Angstroms
        positions = [
            (0, 0, 0),
            (a / 4, a / 4, a / 4),
            (a / 2, a / 2, 0),
            (a / 2, 0, a / 2),
            (0, a / 2, a / 2),
            (a / 4, 3 * a / 4, 3 * a / 4.01),  # displaced
            (3 * a / 4, a / 4.01, 3 * a / 4),  # displaced
            (3 * a / 4, 3 * a / 4, a / 4),
        ]
        cell = [(a, 0, 0), (0, a, 0), (0, 0, a)]
        self.atoms_displaced = Atoms(
            "C8", positions=positions, cell=cell, pbc=True  # noqa: E501
        )

        # Create an example structure with expanded cell for testing
        a = 3.567 * 1.2
        positions = [
            (0, 0, 0),
            (a / 4, a / 4, a / 4),
            (a / 2, a / 2, 0),
            (a / 2, 0, a / 2),
            (0, a / 2, a / 2),
            (a / 4, 3 * a / 4, 3 * a / 4),
            (3 * a / 4, a / 4, 3 * a / 4),
            (3 * a / 4, 3 * a / 4, a / 4),
        ]
        cell = [(a, 0, 0), (0, a, 0), (0, 0, a)]
        self.atoms_expanded = Atoms(
            "C8", positions=positions, cell=cell, pbc=True  # noqa: E501
        )
        # Create a batch of structures for testing
        self.atoms_batch = [self.atoms_ideal, self.atoms_displaced, self.atoms_expanded]

        self.potential = Potential.from_checkpoint()

    def test_default_batch_relaxer(self):
        relaxer = BatchRelaxer(self.potential, fmax=0.01, filter="EXPCELLFILTER")
        atoms_batch = self.atoms_batch.copy()
        relaxation_trajectories = relaxer.relax(atoms_batch)
        assert len(relaxation_trajectories) == len(atoms_batch)
        relaxed_ideal = relaxation_trajectories[0][-1]
        for trajectory in relaxation_trajectories.values():
            assert len(trajectory) > 0
            assert trajectory[-1].info["total_energy"] is not None
            assert trajectory[-1].arrays["forces"] is not None
            assert trajectory[-1].info["stress"] is not None
            assert np.allclose(trajectory[-1].get_positions(), relaxed_ideal.get_positions(), atol=0.01)
            assert np.allclose(trajectory[-1].get_cell(), relaxed_ideal.get_cell(), atol=0.01)


if __name__ == "__main__":
    unittest.main()
