# -*- coding: utf-8 -*-
import numpy as np
import pytest
from ase import Atoms

from mattersim.applications.batch_relax import BatchRelaxer
from mattersim.forcefield.potential import Potential


def _make_test_atoms():
    """Return (ideal, displaced, expanded) diamond C8 structures."""
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
    atoms_ideal = Atoms("C8", positions=positions, cell=cell, pbc=True)

    a = 3.567
    positions_d = [
        (0, 0, 0),
        (a / 4, a / 4, a / 4),
        (a / 2, a / 2, 0),
        (a / 2, 0, a / 2),
        (0, a / 2, a / 2),
        (a / 4, 3 * a / 4, 3 * a / 4.01),  # displaced
        (3 * a / 4, a / 4.01, 3 * a / 4),  # displaced
        (3 * a / 4, 3 * a / 4, a / 4),
    ]
    atoms_displaced = Atoms("C8", positions=positions_d, cell=cell, pbc=True)

    a2 = 3.567 * 1.2
    positions_e = [
        (0, 0, 0),
        (a2 / 4, a2 / 4, a2 / 4),
        (a2 / 2, a2 / 2, 0),
        (a2 / 2, 0, a2 / 2),
        (0, a2 / 2, a2 / 2),
        (a2 / 4, 3 * a2 / 4, 3 * a2 / 4),
        (3 * a2 / 4, a2 / 4, 3 * a2 / 4),
        (3 * a2 / 4, 3 * a2 / 4, a2 / 4),
    ]
    cell_e = [(a2, 0, 0), (0, a2, 0), (0, 0, a2)]
    atoms_expanded = Atoms("C8", positions=positions_e, cell=cell_e, pbc=True)

    return atoms_ideal, atoms_displaced, atoms_expanded


def test_default_batch_relaxer(device):
    potential = Potential.from_checkpoint(device=device)
    atoms_ideal, atoms_displaced, atoms_expanded = _make_test_atoms()
    atoms_batch = [atoms_ideal, atoms_displaced, atoms_expanded]

    relaxer = BatchRelaxer(potential, fmax=0.01, filter="EXPCELLFILTER")
    relaxation_trajectories = relaxer.relax(atoms_batch)
    assert len(relaxation_trajectories) == len(atoms_batch)
    relaxed_ideal = relaxation_trajectories[0][-1]
    for trajectory in relaxation_trajectories.values():
        assert len(trajectory) > 0
        assert trajectory[-1].info["total_energy"] is not None
        assert trajectory[-1].arrays["forces"] is not None
        assert trajectory[-1].info["stress"] is not None
        assert np.allclose(
            trajectory[-1].get_positions(),
            relaxed_ideal.get_positions(),
            atol=0.01,
        )
        assert np.allclose(
            trajectory[-1].get_cell(),
            relaxed_ideal.get_cell(),
            atol=0.01,
        )

