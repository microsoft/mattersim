"""Shared pytest fixtures for mattersim tests."""

import numpy as np
import pytest
import torch
from ase import Atoms
from ase.build import bulk


def available_devices():
    """Return all available torch devices on this machine."""
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    if torch.backends.mps.is_available():
        devices.append("mps")
    return devices


def best_device():
    """Return the fastest available device."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@pytest.fixture(
    params=available_devices(),
    ids=lambda d: f"device={d}",
)
def available_device(request):
    """Yields each available device (cpu, cuda, mps)."""
    return request.param


@pytest.fixture()
def si_diamond():
    """Si diamond primitive cell (2 atoms, FCC lattice, periodic)."""
    return bulk("Si", "diamond", a=5.43)


@pytest.fixture()
def si_diamond_cubic():
    """Si diamond conventional cubic cell (8 atoms, periodic)."""
    return bulk("Si", "diamond", a=5.43, cubic=True)


@pytest.fixture()
def water_molecule():
    """A non-periodic water molecule (3 atoms, no PBC)."""
    return Atoms(
        "OH2",
        positions=[(0, 0, 0), (0.96, 0, 0), (-0.24, 0.93, 0)],
        pbc=False,
    )


@pytest.fixture()
def perturb():
    """Factory fixture that returns a function to perturb atomic structures.

    Usage:
        atoms_displaced = perturb(atoms, displacement=0.05)
        atoms_expanded = perturb(atoms, strain=0.2)
        atoms_both = perturb(atoms, strain=0.1, displacement=0.02)
    """

    def _perturb(atoms, strain=0.0, displacement=0.0, seed=42):
        """Return a copy of atoms with optional cell strain and position noise.

        Args:
            atoms: ASE Atoms object to perturb.
            strain: Fractional cell expansion (e.g. 0.2 for 20%).
            displacement: Standard deviation of Gaussian noise added to
                positions, in Angstrom.
            seed: Random seed for reproducibility.
        """
        copy = atoms.copy()
        if strain:
            copy.set_cell(copy.cell * (1 + strain), scale_atoms=True)
        if displacement:
            rng = np.random.default_rng(seed)
            copy.positions += rng.normal(scale=displacement, size=copy.positions.shape)
        return copy

    return _perturb


@pytest.fixture(scope="module")
def mattersim_calc_best_device():
    """MatterSim 1M calculator on the best available device."""
    from mattersim.forcefield import MatterSimCalculator

    return MatterSimCalculator(device=best_device())


@pytest.fixture(scope="module")
def mattersim_potential_best_device():
    """MatterSim 1M potential on the best available device (no training state)."""
    from mattersim.forcefield.potential import Potential

    return Potential.from_checkpoint(device=best_device(), load_training_state=False)
