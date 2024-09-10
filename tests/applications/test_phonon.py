# -*- coding: utf-8 -*-
import unittest

import numpy as np
from ase import Atoms
from ase.calculators.emt import EMT
from phonopy import Phonopy

from mattersim.applications.phonon import PhononWorkflow


class PhononTestCase(unittest.TestCase):
    def setUp(self):
        # Create an example structure for testing
        a = 1.786854996
        positions = [
            (1.78685500, 1.78685500, 1.78685500),
            (2.68028249, 2.68028249, 2.68028249),
        ]
        cell = [(0, a, a), (a, 0, a), (a, a, 0)]

        self.atoms = Atoms("C2", positions=positions, cell=cell, pbc=True)

        # Create an conventional cell for testing
        a2 = a * 2
        positions2 = [
            (0, 0, 0),
            (0, a2 / 2, a2 / 2),
            (a2 / 2, 0, a2 / 2),
            (a2 / 2, a2 / 2, 0),
            (a2 / 4, a2 / 4, a2 / 4),
            (a2 / 4, 3 * a2 / 4, 3 * a2 / 4),
            (3 * a2 / 4, a2 / 4, 3 * a2 / 4),
            (3 * a2 / 4, 3 * a2 / 4, a2 / 4),
        ]

        cell2 = [(a2, 0, 0), (0, a2, 0), (0, 0, a2)]

        self.atoms_conv = Atoms("C8", positions=positions2, cell=cell2, pbc=True)

        self.calculator = EMT()
        self.atoms.calc = self.calculator
        self.atoms_conv.calc = self.calculator

    def test_phonon(self):
        phononworkflow = PhononWorkflow(self.atoms, work_dir="/tmp/diamond")
        has_imaginary, phonon = phononworkflow.run()

        self.assertTrue(has_imaginary)
        self.assertIsInstance(phonon, Phonopy)

    def test_phonon_supercell(self):
        supercell_matrix = np.array([[4, 0, 0], [0, 4, 0], [0, 0, 4]])
        qpoints_mesh = np.array([12, 12, 12])
        phononworkflow = PhononWorkflow(
            self.atoms,
            work_dir="/tmp/diamond",
            supercell_matrix=supercell_matrix,
            qpoints_mesh=qpoints_mesh,
        )
        has_imaginary, phonon = phononworkflow.run()

        self.assertTrue(has_imaginary)
        self.assertIsInstance(phonon, Phonopy)

    def test_phonon_prim(self):
        phononworkflow = PhononWorkflow(
            self.atoms_conv, work_dir="/tmp/diamond_conv", find_prim=True
        )
        has_imaginary, phonon = phononworkflow.run()
        has_imaginary, phonon = phononworkflow.run()

        self.assertTrue(has_imaginary)
        self.assertIsInstance(phonon, Phonopy)


if __name__ == "__main__":
    unittest.main()
