# -*- coding: utf-8 -*-
import unittest

from ase import Atoms
from ase.calculators.emt import EMT
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from mattersim.applications.relax import Relaxer


class RelaxerTestCase(unittest.TestCase):
    def setUp(self):
        # Create an example structure with displaced atoms for testing
        a = 1.786854996  # Angstroms
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
        a = 1.786854996 * 1.2
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

        self.calculator = EMT()

    def test_default_relaxer(self):
        relaxer = Relaxer()
        atoms_displaced = self.atoms_displaced.copy()
        atoms_displaced.set_calculator(self.calculator)
        converged, relaxed_atoms = relaxer.relax(
            atoms_displaced, fmax=0.1, steps=500
        )  # noqa: E501
        self.assertTrue(converged)
        self.assertIsInstance(relaxed_atoms, Atoms)

    def test_relax_structures(self):
        atoms_list = [
            self.atoms_displaced.copy(),
            self.atoms_displaced.copy(),
            self.atoms_displaced.copy(),
        ]
        for atoms in atoms_list:
            atoms.set_calculator(self.calculator)

        converged_list, relaxed_atoms_list = Relaxer.relax_structures(
            atoms_list, fmax=0.1, steps=500
        )
        self.assertIsInstance(converged_list, list)
        for converged in converged_list:
            self.assertTrue(converged)

    def test_relax_structures_under_pressure(self):
        atoms_displaced = self.atoms_displaced.copy()
        atoms_displaced.set_calculator(self.calculator)
        init_volume = atoms_displaced.get_volume()
        print(f"Initial volume: {init_volume}")

        # First, relax under 0 pressure
        converged, relaxed_atoms = Relaxer.relax_structures(
            atoms_displaced,
            steps=500,
            fmax=0.1,
            filter="FrechetCellFilter",
            pressure_in_GPa=0.0,
        )
        intermediate_volume = relaxed_atoms.get_volume()
        print(f"Intermediate volume: {intermediate_volume}")
        self.assertTrue(converged)

        # Second, relax under 100 GPa
        converged, relaxed_atoms = Relaxer.relax_structures(
            relaxed_atoms,
            steps=500,
            fmax=0.1,
            filter="FrechetCellFilter",
            pressure_in_GPa=100.0,
        )
        final_volume = relaxed_atoms.get_volume()
        print(f"Final volume: {final_volume}")
        self.assertTrue(converged)
        self.assertLess(final_volume, intermediate_volume)
        print(f"Final cell: {relaxed_atoms.cell}")

    def test_relax_with_filter_and_constrained_symmetry(self):
        atoms_expanded = self.atoms_expanded.copy()
        atoms_expanded.set_calculator(self.calculator)
        init_volume = atoms_expanded.get_volume()
        print(f"Initial volume: {init_volume}")

        init_analyzer = SpacegroupAnalyzer(
            AseAtomsAdaptor.get_structure(self.atoms_expanded)
        )
        init_spacegroup = init_analyzer.get_space_group_number()

        # First, relax under 0 pressure
        converged, relaxed_atoms = Relaxer.relax_structures(
            atoms_expanded,
            steps=500,
            fmax=0.1,
            filter="FrechetCellFilter",
            pressure_in_GPa=0.0,
            constrain_symmetry=True,
        )
        intermediate_volume = relaxed_atoms.get_volume()
        print(f"Intermediate volume: {intermediate_volume}")
        self.assertTrue(converged)

        # Second, relax under 100 GPa
        converged, relaxed_atoms = Relaxer.relax_structures(
            relaxed_atoms,
            steps=500,
            fmax=0.1,
            filter="FrechetCellFilter",
            pressure_in_GPa=100.0,
            constrain_symmetry=True,
        )
        final_volume = relaxed_atoms.get_volume()
        print(f"Final volume: {final_volume}")
        self.assertTrue(converged)
        self.assertLess(final_volume, intermediate_volume)

        final_analyzer = SpacegroupAnalyzer(
            AseAtomsAdaptor.get_structure(relaxed_atoms)
        )
        final_spacegroup = final_analyzer.get_space_group_number()
        self.assertEqual(init_spacegroup, final_spacegroup)
        print(f"Final cell: {relaxed_atoms.cell}")
        cell_a = relaxed_atoms.cell[0, 0]
        cell_b = relaxed_atoms.cell[1, 1]
        cell_c = relaxed_atoms.cell[2, 2]
        self.assertAlmostEqual(cell_a, cell_b)
        self.assertAlmostEqual(cell_a, cell_c)


if __name__ == "__main__":
    unittest.main()
