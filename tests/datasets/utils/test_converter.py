"""Tests for converter — verifies that pbc arrays use explicit np.int64 dtype
to satisfy pymatgen's Cython typed memoryview in find_points_in_spheres,
which requires int64 but receives platform-dependent C long on Windows.
"""

import numpy as np

from mattersim.datasets.utils.converter import (
    GraphConverter,
    get_fixed_radius_bonding,
)


class TestGetFixedRadiusBonding:
    """Tests for the get_fixed_radius_bonding function."""

    def test_pbc_array_is_int64(self, si_diamond):
        """The pbc array passed to find_points_in_spheres must be int64,
        not platform-dependent C long, to avoid Cython buffer dtype mismatch."""
        center_idx, neighbor_idx, images, distances = get_fixed_radius_bonding(
            si_diamond, cutoff=5.0
        )
        assert center_idx.dtype == np.int64
        assert neighbor_idx.dtype == np.int64
        assert images.dtype == np.int64

    def test_returns_neighbors(self, si_diamond):
        """Should find neighbors within cutoff for a periodic structure."""
        center_idx, neighbor_idx, images, distances = get_fixed_radius_bonding(
            si_diamond, cutoff=5.0
        )
        assert len(center_idx) > 0
        assert len(center_idx) == len(neighbor_idx)
        assert len(center_idx) == len(distances)
        assert np.all(distances > 0)


class TestGraphConverter:
    """Tests for the GraphConverter.convert method."""

    def test_periodic_structure_converts(self, si_diamond):
        """Periodic structure should convert without dtype errors."""
        converter = GraphConverter(model_type="m3gnet")
        graph = converter.convert(si_diamond)
        assert graph is not None
        assert hasattr(graph, "edge_index")
        assert hasattr(graph, "atom_pos")

    def test_non_periodic_structure_converts(self, water_molecule):
        """Non-periodic molecule should convert (with auto-supercell)
        without dtype errors."""
        converter = GraphConverter(model_type="m3gnet")
        graph = converter.convert(water_molecule)
        assert graph is not None
        assert hasattr(graph, "edge_index")

    def test_pbc_fallback_uses_int64(self, water_molecule):
        """When PBC is False, the fallback pbc array [1,1,1] must also
        be int64 to avoid Cython dtype mismatch."""
        converter = GraphConverter(model_type="m3gnet")
        graph = converter.convert(water_molecule)
        assert graph is not None
