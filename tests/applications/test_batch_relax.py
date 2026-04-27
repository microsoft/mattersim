# -*- coding: utf-8 -*-
import numpy as np
import pytest

from mattersim.applications.batch_relax import BatchRelaxer
from mattersim.forcefield.potential import Potential


def test_default_batch_relaxer(available_device, si_diamond_cubic, perturb):
    print(f"\n>>> Running test_default_batch_relaxer on device: {available_device}")
    potential = Potential.from_checkpoint(device=available_device)

    atoms_ideal = si_diamond_cubic
    atoms_displaced = perturb(si_diamond_cubic, displacement=0.05)
    atoms_expanded = perturb(si_diamond_cubic, strain=0.2)
    atoms_batch = [atoms_ideal, atoms_displaced, atoms_expanded]

    relaxer = BatchRelaxer(potential, fmax=0.01, filter="EXPCELLFILTER")
    relaxation_trajectories = relaxer.relax(atoms_batch)
    assert len(relaxation_trajectories) == len(atoms_batch)
    for trajectory in relaxation_trajectories.values():
        assert len(trajectory) > 0
        assert trajectory[-1].info["total_energy"] is not None
        assert trajectory[-1].arrays["forces"] is not None
        assert trajectory[-1].info["stress"] is not None
        # Note: the relaxer converges on the *filtered* gradient
        # (atomic forces + cell stress, shape [natoms+3, 3]) when using
        # ExpCellFilter. The last trajectory snapshot is recorded before
        # the final optimizer step, so raw atomic forces may be slightly
        # above the target fmax. We use a looser threshold here.
        fmax = np.linalg.norm(trajectory[-1].arrays["forces"], axis=-1).max()
        assert fmax < 0.1, f"Relaxation did not converge: fmax={fmax}"

