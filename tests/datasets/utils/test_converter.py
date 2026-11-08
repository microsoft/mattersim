"""Tests for converter — verifies that pbc arrays use explicit np.int64 dtype
to satisfy pymatgen's Cython typed memoryview in find_points_in_spheres,
which requires int64 but receives platform-dependent C long on Windows.
"""

import numpy as np
import pytest
import torch
from ase import Atoms
from torch_geometric.data import Batch

from mattersim.datasets.utils.converter import (
    GraphConverter,
    M3GNetData,
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
        assert isinstance(graph, M3GNetData)
        assert hasattr(graph, "edge_index")
        assert hasattr(graph, "atom_pos")

    def test_three_body_indices_are_offset_when_batched(self, si_diamond):
        """PyG batches should offset three_body_indices by prior num_bonds."""
        converter = GraphConverter(model_type="m3gnet")
        graph0 = converter.convert(si_diamond)
        graph1 = converter.convert(si_diamond.copy())

        assert graph0.num_three_body > 0
        assert graph1.num_three_body > 0

        batch = Batch.from_data_list([graph0, graph1])
        first_graph_triples = int(graph0.num_three_body)
        second_graph_triples = int(graph1.num_three_body)
        batched_second_indices = batch.three_body_indices[
            first_graph_triples : first_graph_triples + second_graph_triples
        ]
        expected_second_indices = graph1.three_body_indices + graph0.num_bonds

        assert torch.equal(batched_second_indices, expected_second_indices)

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

    @pytest.mark.parametrize(
        "pbc",
        [
            [True, True, False],
            [True, False, False],
        ],
        ids=["slab_TTF", "wire_TFF"],
    )
    def test_mixed_pbc_wrapping_preserves_nonperiodic_coordinates(self, pbc):
        """Wrapping must not create bonds across nonperiodic cell boundaries."""
        nonperiodic_axis = pbc.index(False)
        positions = np.zeros((3, 3), dtype=np.float64)
        positions[:, 0] = 12.25
        positions[:, nonperiodic_axis] = [1.0, 1.5, 11.2]

        atoms = Atoms("Si3", positions=positions, cell=[10.0, 10.0, 10.0], pbc=pbc)
        converter = GraphConverter(
            model_type="m3gnet", twobody_cutoff=2.0, threebody_cutoff=2.0
        )
        graph = converter.convert(atoms)

        np.testing.assert_allclose(
            graph.atom_pos[:, nonperiodic_axis].detach().cpu().numpy(),
            [1.0, 1.5, 11.2],
            rtol=1e-5,
            atol=1e-5,
        )

        np.testing.assert_allclose(
            graph.atom_pos[:, 0].detach().cpu().numpy(),
            2.25,
            rtol=1e-5,
            atol=1e-5,
        )

        # Exact edge pairs: only (0, 1) and (1, 0) within 2 A cutoff
        edge_pairs = set(
            zip(graph.edge_index[0].tolist(), graph.edge_index[1].tolist())
        )
        assert edge_pairs == {(0, 1), (1, 0)}

        assert graph.num_three_body == 0
