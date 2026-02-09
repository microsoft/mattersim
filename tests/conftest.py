"""
Pytest configuration and fixtures for mattersim tests.
"""
import pytest
import numpy as np
import torch
from ase.build import bulk, molecule
from pymatgen.core.structure import Structure
from pymatgen.core.lattice import Lattice


# =============================================================================
# GPU Detection
# =============================================================================

HAS_GPU = torch.cuda.is_available()

# Custom pytest marker for tests requiring GPU
requires_gpu = pytest.mark.skipif(not HAS_GPU, reason="No GPU available")


# =============================================================================
# Pytest Configuration
# =============================================================================


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers",
        "requires_gpu: mark test as requiring GPU (skip if no GPU available)",
    )


def pytest_collection_modifyitems(config, items):
    """Skip GPU tests if no GPU is available."""
    if HAS_GPU:
        # GPU is available, don't skip any tests
        return

    skip_gpu = pytest.mark.skip(reason="No GPU available")
    for item in items:
        if "requires_gpu" in item.keywords:
            item.add_marker(skip_gpu)


# =============================================================================
# Device Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def gpu_available():
    """Return whether GPU is available."""
    return HAS_GPU


@pytest.fixture
def gpu_device():
    """Return GPU device if available, otherwise skip test."""
    if not HAS_GPU:
        pytest.skip("No GPU available")
    return torch.device("cuda")


@pytest.fixture
def cpu_device():
    """Return CPU device."""
    return torch.device("cpu")


@pytest.fixture(params=["cpu"] + (["cuda"] if HAS_GPU else []))
def any_device(request):
    """Parametrized fixture that provides all available devices."""
    return torch.device(request.param)


# =============================================================================
# Structure Fixtures - ASE Atoms
# =============================================================================


@pytest.fixture
def si_atoms():
    """Silicon diamond structure (2 atoms)."""
    return bulk("Si", "diamond", a=5.43)


@pytest.fixture
def cu_atoms():
    """Copper FCC structure (1 atom)."""
    return bulk("Cu", "fcc", a=3.6)


@pytest.fixture
def nacl_atoms():
    """NaCl rocksalt structure (2 atoms)."""
    return bulk("NaCl", "rocksalt", a=5.64)


@pytest.fixture
def water_molecule():
    """Water molecule (non-periodic, 3 atoms)."""
    water = molecule("H2O")
    water.set_cell([10, 10, 10])
    water.set_pbc(False)
    return water


@pytest.fixture
def small_atoms():
    """Small structure - Si diamond (2 atoms)."""
    return bulk("Si", "diamond", a=5.43)


@pytest.fixture
def medium_atoms():
    """Medium structure - Si diamond 2x2x2 supercell (16 atoms)."""
    return bulk("Si", "diamond", a=5.43) * (2, 2, 2)


@pytest.fixture
def large_atoms():
    """Large structure - Si diamond 3x3x3 supercell (54 atoms)."""
    return bulk("Si", "diamond", a=5.43) * (3, 3, 3)


# =============================================================================
# Structure Fixtures - Pymatgen Structures
# =============================================================================


@pytest.fixture
def si_structure():
    """Silicon as pymatgen Structure (2 atoms)."""
    lattice = Lattice.cubic(5.43)
    return Structure(
        lattice,
        ["Si", "Si"],
        [[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]],
    )


@pytest.fixture
def cu_structure():
    """Copper FCC as pymatgen Structure (1 atom)."""
    lattice = Lattice.cubic(3.6)
    return Structure(
        lattice,
        ["Cu"],
        [[0.0, 0.0, 0.0]],
    )


@pytest.fixture
def nacl_structure():
    """NaCl rocksalt as pymatgen Structure (2 atoms)."""
    lattice = Lattice.cubic(5.64)
    return Structure(
        lattice,
        ["Na", "Cl"],
        [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
    )
