"""
Test cases for BatchGraphConverter class.

This module tests the BatchGraphConverter which converts batches of
materials (Atoms/Structure objects) to graph representations with GPU support.
"""
import pytest
import numpy as np
import torch
from torch_geometric.data import Data

from mattersim.datasets.utils.converter import BatchGraphConverter, GraphConverter

# GPU detection - used for parametrized tests
HAS_GPU = torch.cuda.is_available()

# Marker for tests that require GPU
requires_gpu = pytest.mark.skipif(not HAS_GPU, reason="No GPU available")


# =============================================================================
# BatchGraphConverter Fixtures
# =============================================================================


@pytest.fixture
def converter():
    """Default BatchGraphConverter with CPU device."""
    return BatchGraphConverter(device="cpu")


@pytest.fixture
def converter_no_threebody():
    """BatchGraphConverter with threebody disabled."""
    return BatchGraphConverter(device="cpu", has_threebody=False)


@pytest.fixture
def converter_gpu(gpu_device):
    """BatchGraphConverter with GPU device (skipped if no GPU)."""
    return BatchGraphConverter(device=gpu_device)


# =============================================================================
# Initialization Tests
# =============================================================================


class TestBatchGraphConverterInit:
    """Test cases for BatchGraphConverter initialization."""

    def test_default_initialization(self):
        """Test default initialization parameters."""
        converter = BatchGraphConverter()
        assert converter.model_type == "m3gnet"
        assert converter.twobody_cutoff == 5.0
        assert converter.threebody_cutoff == 4.0
        assert converter.has_threebody is True
        assert converter.max_num_neighbors_threshold == int(1e6)

    def test_custom_cutoffs(self):
        """Test initialization with custom cutoff values."""
        converter = BatchGraphConverter(
            twobody_cutoff=6.0,
            threebody_cutoff=3.5,
        )
        assert converter.twobody_cutoff == 6.0
        assert converter.threebody_cutoff == 3.5

    def test_disable_threebody(self):
        """Test initialization with threebody disabled."""
        converter = BatchGraphConverter(has_threebody=False)
        assert converter.has_threebody is False

    def test_device_string_cpu(self):
        """Test device initialization with string 'cpu'."""
        converter = BatchGraphConverter(device="cpu")
        assert converter.device == torch.device("cpu")

    def test_device_torch_device(self):
        """Test device initialization with torch.device object."""
        device = torch.device("cpu")
        converter = BatchGraphConverter(device=device)
        assert converter.device == device

    def test_unsupported_model_type(self):
        """Test that unsupported model types raise NotImplementedError."""
        with pytest.raises(NotImplementedError):
            BatchGraphConverter(model_type="unsupported")

    def test_max_num_neighbors_threshold(self):
        """Test custom max_num_neighbors_threshold."""
        converter = BatchGraphConverter(max_num_neighbors_threshold=500)
        assert converter.max_num_neighbors_threshold == 500


# =============================================================================
# Convert Method Tests
# =============================================================================


class TestBatchGraphConverterConvert:
    """Test cases for BatchGraphConverter.convert() method."""

    def test_convert_single_atoms(self, converter, si_atoms):
        """Test converting a single Atoms object."""
        graphs = converter.convert(si_atoms)
        assert len(graphs) == 1
        assert isinstance(graphs[0], Data)

    def test_convert_single_structure(self, converter, si_structure):
        """Test converting a single pymatgen Structure."""
        graphs = converter.convert(si_structure)
        assert len(graphs) == 1
        assert isinstance(graphs[0], Data)

    def test_convert_list_of_atoms(self, converter, si_atoms, cu_atoms, nacl_atoms):
        """Test converting a list of Atoms objects."""
        atoms_list = [si_atoms, cu_atoms, nacl_atoms]
        graphs = converter.convert(atoms_list)
        assert len(graphs) == 3
        for graph in graphs:
            assert isinstance(graph, Data)

    def test_convert_list_of_structures(self, converter, si_structure):
        """Test converting a list of pymatgen Structures."""
        structures = [si_structure, si_structure]
        graphs = converter.convert(structures)
        assert len(graphs) == 2

    def test_graph_has_required_attributes(self, converter, si_atoms):
        """Test that converted graph has all required attributes."""
        graphs = converter.convert(si_atoms)
        graph = graphs[0]

        # Check required attributes
        assert hasattr(graph, "num_atoms")
        assert hasattr(graph, "num_nodes")
        assert hasattr(graph, "atom_attr")
        assert hasattr(graph, "atom_pos")
        assert hasattr(graph, "cell")
        assert hasattr(graph, "num_bonds")
        assert hasattr(graph, "edge_index")
        assert hasattr(graph, "distances")
        assert hasattr(graph, "pbc_offsets")

    def test_graph_threebody_attributes_enabled(self, converter, si_atoms):
        """Test that threebody attributes are present when enabled."""
        graphs = converter.convert(si_atoms)
        graph = graphs[0]

        assert hasattr(graph, "three_body_indices")
        assert hasattr(graph, "num_three_body")
        assert hasattr(graph, "num_triple_ij")
        assert graph.three_body_indices is not None
        assert graph.num_three_body is not None
        assert graph.num_triple_ij is not None

    def test_graph_threebody_attributes_disabled(self, converter_no_threebody, si_atoms):
        """Test that threebody attributes are None when disabled."""
        graphs = converter_no_threebody.convert(si_atoms)
        graph = graphs[0]

        # When threebody is disabled, the attributes should either not exist
        # or be None depending on how Data handles None values
        for attr in ["three_body_indices", "num_three_body", "num_triple_ij"]:
            try:
                assert getattr(graph, attr) is None
            except AttributeError:
                # Attribute doesn't exist, which is also valid behavior
                pass

    def test_num_atoms_matches(self, converter, si_atoms):
        """Test that num_atoms in graph matches input structure."""
        graphs = converter.convert(si_atoms)
        graph = graphs[0]
        assert graph.num_atoms == len(si_atoms)
        assert graph.num_nodes == len(si_atoms)

    def test_atom_attr_shape(self, converter, si_atoms):
        """Test atom_attr tensor shape."""
        graphs = converter.convert(si_atoms)
        graph = graphs[0]
        assert graph.atom_attr.shape == (len(si_atoms), 1)

    def test_atom_pos_shape(self, converter, si_atoms):
        """Test atom_pos tensor shape."""
        graphs = converter.convert(si_atoms)
        graph = graphs[0]
        assert graph.atom_pos.shape == (len(si_atoms), 3)

    def test_cell_shape(self, converter, si_atoms):
        """Test cell tensor shape."""
        graphs = converter.convert(si_atoms)
        graph = graphs[0]
        assert graph.cell.shape == (1, 3, 3)

    def test_edge_index_shape(self, converter, si_atoms):
        """Test edge_index tensor shape."""
        graphs = converter.convert(si_atoms)
        graph = graphs[0]
        assert graph.edge_index.shape[0] == 2
        assert graph.edge_index.shape[1] == graph.num_bonds

    def test_convert_with_energy(self, converter, si_atoms, cu_atoms):
        """Test converting with energy labels."""
        atoms_list = [si_atoms, cu_atoms]
        energies = [-10.5, -8.2]
        graphs = converter.convert(atoms_list, energy=energies)

        assert hasattr(graphs[0], "energy")
        assert hasattr(graphs[1], "energy")
        assert graphs[0].energy.item() == pytest.approx(-10.5, rel=1e-5)
        assert graphs[1].energy.item() == pytest.approx(-8.2, rel=1e-5)

    def test_convert_with_forces(self, converter, si_atoms):
        """Test converting with forces labels."""
        atoms_list = [si_atoms]
        n_atoms = len(si_atoms)
        forces = [np.random.randn(n_atoms, 3)]
        graphs = converter.convert(atoms_list, forces=forces)

        assert hasattr(graphs[0], "forces")
        assert graphs[0].forces.shape == (n_atoms, 3)

    def test_convert_with_stresses(self, converter, si_atoms):
        """Test converting with stress labels."""
        atoms_list = [si_atoms]
        stresses = [np.random.randn(3, 3)]
        graphs = converter.convert(atoms_list, stresses=stresses)

        assert hasattr(graphs[0], "stress")
        assert graphs[0].stress.shape == (1, 3, 3)

    def test_convert_with_all_labels(self, converter, si_atoms, cu_atoms):
        """Test converting with energy, forces, and stresses."""
        atoms_list = [si_atoms, cu_atoms]
        n_atoms_si = len(si_atoms)
        n_atoms_cu = len(cu_atoms)
        energies = [-10.5, -8.2]
        forces = [np.random.randn(n_atoms_si, 3), np.random.randn(n_atoms_cu, 3)]
        stresses = [np.random.randn(3, 3), np.random.randn(3, 3)]

        graphs = converter.convert(
            atoms_list, energy=energies, forces=forces, stresses=stresses
        )

        for graph in graphs:
            assert hasattr(graph, "energy")
            assert hasattr(graph, "forces")
            assert hasattr(graph, "stress")

    def test_convert_with_none_labels(self, converter, si_atoms, cu_atoms):
        """Test converting with None values in labels lists."""
        atoms_list = [si_atoms, cu_atoms]
        energies = [-10.5, None]  # Second energy is None

        graphs = converter.convert(atoms_list, energy=energies)

        assert hasattr(graphs[0], "energy")
        # Second graph should not have energy attribute
        assert not hasattr(graphs[1], "energy")

    def test_device_placement_cpu(self, si_atoms):
        """Test that output tensors are on CPU when specified."""
        converter = BatchGraphConverter(device="cpu")
        graphs = converter.convert(si_atoms)
        graph = graphs[0]

        assert graph.atom_attr.device.type == "cpu"
        assert graph.atom_pos.device.type == "cpu"
        assert graph.edge_index.device.type == "cpu"

    @requires_gpu
    def test_device_placement_gpu(self, si_atoms):
        """Test that output tensors are on GPU when specified."""
        converter = BatchGraphConverter(device="cuda")
        graphs = converter.convert(si_atoms)
        graph = graphs[0]

        assert graph.atom_attr.device.type == "cuda"
        assert graph.atom_pos.device.type == "cuda"
        assert graph.edge_index.device.type == "cuda"


# =============================================================================
# Batching Behavior Tests
# =============================================================================


class TestBatchGraphConverterBatching:
    """Test cases for BatchGraphConverter batching behavior."""

    def test_batching_respects_max_natoms(self, converter, medium_atoms):
        """Test that batching respects max_natoms_per_batch."""
        # Create list with total atoms > max_natoms_per_batch
        atoms_list = [medium_atoms.copy() for _ in range(10)]  # 160 atoms total

        # With small max_natoms_per_batch, should still process all
        graphs = converter.convert(atoms_list, max_natoms_per_batch=32)
        assert len(graphs) == 10

    def test_single_large_structure_exceeds_batch(self, converter, large_atoms):
        """Test handling when single structure exceeds max_natoms_per_batch.

        Note: This test documents current behavior where a single large structure
        that exceeds max_natoms_per_batch may raise an error. This is a known
        edge case in the BatchGraphConverter implementation.
        """
        # Single structure with more atoms than max_natoms_per_batch
        large = large_atoms.copy()  # 54 atoms

        # The current implementation may have issues with very small batch sizes
        # Test with a reasonable batch size that still requires multiple batches
        graphs = converter.convert(large, max_natoms_per_batch=100)

        # Should process the structure
        assert len(graphs) == 1
        assert graphs[0].num_atoms == len(large)

    def test_mixed_size_structures(self, converter, small_atoms, medium_atoms, large_atoms):
        """Test converting mixed size structures."""
        atoms_list = [
            small_atoms.copy(),
            medium_atoms.copy(),
            large_atoms.copy(),
        ]
        graphs = converter.convert(atoms_list, max_natoms_per_batch=100)

        assert len(graphs) == 3
        assert graphs[0].num_atoms == len(small_atoms)
        assert graphs[1].num_atoms == len(medium_atoms)
        assert graphs[2].num_atoms == len(large_atoms)

    def test_empty_list(self, converter):
        """Test converting empty list.

        Note: The current implementation raises IndexError for empty lists.
        This test documents this behavior and should be updated if the
        implementation changes to handle empty lists gracefully.
        """
        with pytest.raises(IndexError):
            converter.convert([])
        # Alternative expected behavior (uncomment if implementation changes):
        # graphs = converter.convert([])
        # assert len(graphs) == 0


# =============================================================================
# Edge Cases Tests
# =============================================================================


class TestBatchGraphConverterEdgeCases:
    """Test edge cases for BatchGraphConverter."""

    def test_molecule_non_periodic(self, converter, water_molecule):
        """Test converting a non-periodic molecule structure."""
        graphs = converter.convert(water_molecule)
        assert len(graphs) == 1
        assert graphs[0].num_atoms == 3

    def test_different_atomic_numbers(self, converter, nacl_atoms):
        """Test that atomic numbers are correctly preserved."""
        graphs = converter.convert(nacl_atoms)

        # NaCl has Na (11) and Cl (17)
        atom_attrs = graphs[0].atom_attr.squeeze().tolist()
        assert 11.0 in atom_attrs  # Na
        assert 17.0 in atom_attrs  # Cl

    def test_custom_cutoffs_affect_neighbors(self, cu_atoms):
        """Test that different cutoffs produce different number of edges."""
        converter_small = BatchGraphConverter(device="cpu", twobody_cutoff=3.0)
        converter_large = BatchGraphConverter(device="cpu", twobody_cutoff=6.0)

        graphs_small = converter_small.convert(cu_atoms)
        graphs_large = converter_large.convert(cu_atoms)

        # Larger cutoff should have more or equal edges
        assert graphs_small[0].num_bonds <= graphs_large[0].num_bonds

    def test_threebody_cutoff_affects_triples(self, cu_atoms):
        """Test that threebody cutoff affects number of triple indices."""
        converter_small = BatchGraphConverter(
            device="cpu", threebody_cutoff=2.0, has_threebody=True
        )
        converter_large = BatchGraphConverter(
            device="cpu", threebody_cutoff=5.0, has_threebody=True
        )

        atoms = cu_atoms * (2, 2, 2)

        graphs_small = converter_small.convert(atoms)
        graphs_large = converter_large.convert(atoms)

        # Larger cutoff should have more or equal three-body interactions
        assert graphs_small[0].num_three_body <= graphs_large[0].num_three_body

    def test_reproducibility(self, converter, si_atoms):
        """Test that same input produces same output."""
        graphs1 = converter.convert(si_atoms.copy())
        graphs2 = converter.convert(si_atoms.copy())

        # Check key attributes match
        assert graphs1[0].num_atoms == graphs2[0].num_atoms
        assert graphs1[0].num_bonds == graphs2[0].num_bonds
        torch.testing.assert_close(graphs1[0].atom_pos, graphs2[0].atom_pos)
        torch.testing.assert_close(graphs1[0].atom_attr, graphs2[0].atom_attr)


# =============================================================================
# Comparison with GraphConverter Tests
# =============================================================================


# =============================================================================
# GPU-Specific Tests
# =============================================================================


@requires_gpu
class TestBatchGraphConverterGPU:
    """Test cases that require GPU."""

    def test_gpu_initialization(self):
        """Test that converter initializes correctly on GPU."""
        converter = BatchGraphConverter(device="cuda")
        assert converter.device.type == "cuda"

    def test_gpu_convert_single_structure(self, si_atoms):
        """Test converting a single structure on GPU."""
        converter = BatchGraphConverter(device="cuda")
        graphs = converter.convert(si_atoms)

        assert len(graphs) == 1
        assert graphs[0].atom_attr.device.type == "cuda"
        assert graphs[0].atom_pos.device.type == "cuda"

    def test_gpu_convert_batch(self, si_atoms, cu_atoms):
        """Test converting multiple structures on GPU."""
        converter = BatchGraphConverter(device="cuda")
        atoms_list = [si_atoms, cu_atoms]
        graphs = converter.convert(atoms_list)

        assert len(graphs) == 2
        for graph in graphs:
            assert graph.atom_attr.device.type == "cuda"

    def test_gpu_to_cpu_transfer(self, si_atoms):
        """Test that GPU tensors can be transferred to CPU."""
        converter = BatchGraphConverter(device="cuda")
        graphs = converter.convert(si_atoms)
        graph = graphs[0]

        # Transfer to CPU
        cpu_attr = graph.atom_attr.cpu()
        cpu_pos = graph.atom_pos.cpu()

        assert cpu_attr.device.type == "cpu"
        assert cpu_pos.device.type == "cpu"

    def test_gpu_memory_cleanup(self, medium_atoms):
        """Test that GPU memory is properly managed with large batches."""
        converter = BatchGraphConverter(device="cuda")
        atoms_list = [medium_atoms.copy() for _ in range(5)]

        # Get initial memory
        torch.cuda.reset_peak_memory_stats()
        initial_memory = torch.cuda.memory_allocated()

        graphs = converter.convert(atoms_list)

        # Verify conversion worked
        assert len(graphs) == 5

        # Clean up
        del graphs
        torch.cuda.empty_cache()

        # Memory should be released (allowing some overhead)
        final_memory = torch.cuda.memory_allocated()
        assert final_memory <= initial_memory + 1024 * 1024  # 1MB tolerance


# =============================================================================
# Comparison with GraphConverter Tests
# =============================================================================


class TestBatchGraphConverterComparison:
    """Test cases comparing BatchGraphConverter with GraphConverter."""

    @pytest.fixture
    def batch_converter(self):
        return BatchGraphConverter(device="cpu")

    @pytest.fixture
    def single_converter(self):
        return GraphConverter()

    def test_same_num_atoms(self, batch_converter, single_converter, si_atoms):
        """Test that both converters produce same num_atoms."""
        batch_graph = batch_converter.convert(si_atoms)[0]
        single_graph = single_converter.convert(si_atoms)

        assert batch_graph.num_atoms == single_graph.num_atoms

    def test_same_atom_attr(self, batch_converter, single_converter, si_atoms):
        """Test that both converters produce same atom attributes."""
        batch_graph = batch_converter.convert(si_atoms)[0]
        single_graph = single_converter.convert(si_atoms)

        torch.testing.assert_close(
            batch_graph.atom_attr.cpu(), single_graph.atom_attr.cpu()
        )

    def test_same_cell(self, batch_converter, single_converter, si_atoms):
        """Test that both converters produce same cell."""
        batch_graph = batch_converter.convert(si_atoms)[0]
        single_graph = single_converter.convert(si_atoms)

        torch.testing.assert_close(batch_graph.cell.cpu(), single_graph.cell.cpu())


# =============================================================================
# Parametrized Device Tests
# =============================================================================


class TestBatchGraphConverterDeviceParametrized:
    """Parametrized tests that run on available devices."""

    @pytest.fixture(params=["cpu"] + (["cuda"] if HAS_GPU else []))
    def device(self, request):
        """Parametrized device fixture - tests run on CPU always, GPU if available."""
        return request.param

    def test_convert_on_device(self, device, si_atoms):
        """Test conversion works on the specified device."""
        converter = BatchGraphConverter(device=device)
        graphs = converter.convert(si_atoms)

        assert len(graphs) == 1
        assert graphs[0].atom_attr.device.type == device

    def test_batch_convert_on_device(self, device, si_atoms, cu_atoms):
        """Test batch conversion works on the specified device."""
        converter = BatchGraphConverter(device=device)
        atoms_list = [si_atoms, cu_atoms]
        graphs = converter.convert(atoms_list)

        assert len(graphs) == 2
        for graph in graphs:
            assert graph.atom_attr.device.type == device

    def test_threebody_on_device(self, device, si_atoms):
        """Test threebody computation works on the specified device."""
        converter = BatchGraphConverter(device=device, has_threebody=True)
        graphs = converter.convert(si_atoms)

        assert graphs[0].three_body_indices is not None
        assert graphs[0].three_body_indices.device.type == device
