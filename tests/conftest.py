"""Shared pytest fixtures for mattersim tests."""

import pytest
import torch
from ase import Atoms
from ase.build import bulk


def _available_devices():
    """Return all available torch devices on this machine."""
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    if torch.backends.mps.is_available():
        devices.append("mps")
    return devices


@pytest.fixture(
    params=_available_devices(),
    ids=lambda d: f"device={d}",
)
def device(request):
    """Yields each available device (cpu, cuda, mps)."""
    return request.param


@pytest.fixture()
def si_diamond():
    """Si diamond primitive cell (2 atoms, FCC lattice, periodic)."""
    return bulk("Si", "diamond", a=5.43)


@pytest.fixture()
def water_molecule():
    """A non-periodic water molecule (3 atoms, no PBC)."""
    return Atoms(
        "OH2",
        positions=[(0, 0, 0), (0.96, 0, 0), (-0.24, 0.93, 0)],
        pbc=False,
    )


@pytest.fixture()
def si_diamond_cubic():
    """Si diamond conventional cubic cell (8 atoms, periodic)."""
    return bulk("Si", "diamond", a=5.43, cubic=True)
