# -*- coding: utf-8 -*-
import numpy as np
import pytest

from mattersim.applications.batch_relax import BatchRelaxer
from mattersim.forcefield.potential import Potential


def test_default_batch_relaxer(device, si_diamond_cubic, perturb):
    print(f"\n>>> Running test_default_batch_relaxer on device: {device}")
    potential = Potential.from_checkpoint(device=device)

    atoms_ideal = si_diamond_cubic
    atoms_displaced = perturb(si_diamond_cubic, displacement=0.05)
    atoms_expanded = perturb(si_diamond_cubic, strain=0.2)
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

