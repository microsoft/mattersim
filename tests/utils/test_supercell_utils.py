# -*- coding: utf-8 -*-
import unittest

from ase.build import bulk

from mattersim.utils.supercell_utils import auto_grid_detection


class SupercellUtilsTestCase(unittest.TestCase):
    def test_auto_grid_detection_exact_cubes(self):
        # 1-atom primitive fcc Cu
        atoms = bulk("Cu", "fcc", a=3.6)
        # Perfect cube roots: 64 = 4^3, 216 = 6^3
        grid_64 = auto_grid_detection(
            atoms, max_atoms=64, is_santity_check=False, is_verbose=False
        )
        self.assertEqual(grid_64, (4, 4, 4))

        grid_216 = auto_grid_detection(
            atoms, max_atoms=216, is_santity_check=False, is_verbose=False
        )
        self.assertEqual(grid_216, (6, 6, 6))

    def test_auto_grid_detection_approximate_lengths(self):
        # Cell lengths within ratio_tolerance
        atoms = bulk("Cu", "fcc", a=3.6)
        atoms.cell[0] *= 1.05
        grid_64 = auto_grid_detection(
            atoms,
            max_atoms=64,
            ratio_tolerance=1.1,
            is_santity_check=False,
            is_verbose=False,
        )
        self.assertEqual(grid_64, (4, 4, 4))


if __name__ == "__main__":
    unittest.main()
