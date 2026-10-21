"""Regression tests for the AOTI compilation path.

The bug these guard against: ``three_body_indices`` has shape ``[T, 2]`` where
``T`` is the number of angle (three-body) terms, which changes with the
topology on every MD step. An op that materialized ``T`` as a Python ``int``
during tracing baked the compile-time value into the AOTI artifact. At runtime
that crashed with a CUDA device-side assert for smaller cells and silently
returned wrong energies and forces for larger ones.
"""

import numpy as np
import pytest
import torch

from mattersim.datasets.utils.build import build_dataloader
from mattersim.forcefield.aoti_compile import (
    MATTERSIM_DYNAMIC_SHAPES,
    AOTISettings,
    M3GNetForAOTI,
    _get_example_inputs,
    _make_fx,
    _model_fingerprint,
)
from mattersim.forcefield.potential import batch_to_dict

requires_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)

# Number of three-body terms in the compile-time example
# (``_get_example_inputs``: two 8-atom Si-diamond-cubic cells).
EXAMPLE_NUM_TRIPLES = 3840
THREEBODY_INPUT_INDEX = 5


def _export(potential, device: str):
    """FX-trace and export the AOTI wrapper, mirroring ``compile_m3gnet_aoti``."""
    m3gnet = potential.model
    wrapper = M3GNetForAOTI(m3gnet, device=device, settings=AOTISettings(enabled=True))
    for p in wrapper.parameters():
        p.requires_grad_(False)
    wrapper.eval().to(device)

    example_inputs = _get_example_inputs(
        m3gnet.model_args["cutoff"],
        m3gnet.model_args["threebody_cutoff"],
        torch.device(device),
    )
    assert example_inputs[THREEBODY_INPUT_INDEX].shape[0] == EXAMPLE_NUM_TRIPLES
    fx_model = _make_fx(wrapper, example_inputs)
    return torch.export.export(
        fx_model, example_inputs, dynamic_shapes=MATTERSIM_DYNAMIC_SHAPES
    )


def _placeholder_shapes(exported_model):
    """Map each user input name to the shape of its placeholder."""
    by_name = {
        node.name: node
        for node in exported_model.graph.nodes
        if node.op == "placeholder"
    }
    return {
        name: tuple(by_name[name].meta["val"].shape)
        for name in exported_model.graph_signature.user_inputs
    }


def _is_dynamic_dim(dim) -> bool:
    """Whether a tensor dimension has at least one free symbol."""
    expr = getattr(getattr(dim, "node", None), "expr", None)
    return bool(getattr(expr, "free_symbols", set()))


@pytest.fixture(scope="module")
def exported_model_cpu(mattersim_potential_cpu):
    return _export(mattersim_potential_cpu, "cpu")


@pytest.fixture(scope="module")
def mattersim_potential_aoti():
    """MatterSim 1M potential with its model swapped for the AOTI artifact.

    Compiles on first use (or reuses the on-disk cache) and is therefore
    module-scoped. Loads its own ``Potential`` because ``_apply_aoti``
    replaces ``potential.model`` in place.
    """
    from mattersim.forcefield.potential import Potential
    from mattersim.torchsim.model_loading import _apply_aoti

    potential = Potential.from_checkpoint(device="cuda", load_training_state=False)
    _apply_aoti(potential, AOTISettings(enabled=True))
    return potential


class TestExportKeepsShapesDynamic:
    """Export-level tests: fast, no full AOTI compile, CPU is enough."""

    def test_threebody_dim_stays_symbolic(self, exported_model_cpu):
        """The three-body dimension must not be specialized to a constant.

        This is the direct regression for the root cause: a specialized ``T``
        makes the compiled artifact valid only for structures with exactly
        that many angle terms.
        """
        shapes = _placeholder_shapes(exported_model_cpu)
        three_body = shapes[
            exported_model_cpu.graph_signature.user_inputs[THREEBODY_INPUT_INDEX]
        ]

        assert _is_dynamic_dim(three_body[0]), (
            f"three_body_indices dim 0 was specialized to {three_body[0]}. "
            "the AOTI artifact would be wrong for any other three-body count"
        )
        # dim 1 is the (i, k) edge pair and is genuinely static
        assert three_body[1] == 2

    def test_all_variable_dims_stay_symbolic(self, exported_model_cpu):
        """Atom, edge, batch and three-body counts must all remain dynamic."""
        shapes = _placeholder_shapes(exported_model_cpu)

        specialized = {
            name: shape
            for name, shape in shapes.items()
            # every user input has at least one variable-length dim except the
            # scalar ``num_graphs``
            if shape and not any(_is_dynamic_dim(s) for s in shape)
        }
        assert not specialized, f"unexpectedly specialized inputs: {specialized}"


class TestSphericalHarmonicsScripting:
    """``_spherical_harmonics`` must stay traceable *and* scriptable.

    It is deliberately not decorated with ``@torch.jit.script``: the
    TorchScript interpreter emits ``aten::size``, which pins the three-body
    count during symbolic tracing. Scripting an enclosing module still works
    because TorchScript compiles called functions recursively.
    """

    def test_matches_scripted_version_bitwise(self):
        from mattersim.forcefield.m3gnet.modules.angle_encoding import (
            _spherical_harmonics,
        )

        scripted = torch.jit.script(_spherical_harmonics)
        x = torch.linspace(-1.0, 1.0, 257)
        for lmax in range(4):
            assert torch.equal(_spherical_harmonics(lmax, x), scripted(lmax, x))

    def test_enclosing_layer_is_still_scriptable(self):
        from mattersim.forcefield.m3gnet.modules.angle_encoding import (
            SphericalBasisLayer,
        )

        layer = SphericalBasisLayer(max_n=4, max_l=4, cutoff=5.0).eval()
        scripted = torch.jit.script(layer)
        r = torch.linspace(0.5, 4.5, 64)
        theta = torch.linspace(0.0, np.pi, 64)
        assert torch.equal(layer(r, theta), scripted(r, theta))


def test_model_fingerprint_tracks_weights_and_config(mattersim_potential_cpu):
    """Cached artifacts must identify both model weights and configuration."""
    model = mattersim_potential_cpu.model
    original = _model_fingerprint(model)
    parameter = next(model.parameters())
    original_value = parameter.view(-1)[0].clone()
    with torch.no_grad():
        parameter.view(-1)[0].add_(1)
    try:
        assert _model_fingerprint(model) != original
    finally:
        with torch.no_grad():
            parameter.view(-1)[0].copy_(original_value)

    assert _model_fingerprint(model) == original
    original_cutoff = model.model_args["cutoff"]
    model.model_args["cutoff"] = original_cutoff + 1
    try:
        assert _model_fingerprint(model) != original
    finally:
        model.model_args["cutoff"] = original_cutoff


@requires_gpu
class TestAOTIMatchesEager:
    """AOTI-vs-eager numeric parity across three-body counts.

    AOTI only runs on CUDA, so this is the layer that exercises the compiled
    artifact itself. Structures are chosen so that ``T`` straddles the
    compile-time example's 3840 in both directions.
    """

    @staticmethod
    def _dataloader(atoms, potential):
        return build_dataloader(
            [atoms],
            batch_size=1,
            model_type="m3gnet",
            shuffle=False,
            only_inference=True,
            cutoff=potential.model.model_args["cutoff"],
            threebody_cutoff=potential.model.model_args["threebody_cutoff"],
        )

    @pytest.mark.parametrize(
        "repeat,displacement",
        [
            ((1, 1, 1), 0.0),  # T = 1920  < 3840
            ((2, 2, 2), 0.0),  # T = 15360 > 3840
            ((2, 2, 2), 0.05),  # rattled: irregular topology, as in MD
        ],
        ids=["T_below_example", "T_above_example", "T_above_example_rattled"],
    )
    def test_energies_forces_stresses_match_eager(
        self,
        mattersim_potential_aoti,
        mattersim_potential_best_device,
        si_diamond_cubic,
        perturb,
        repeat,
        displacement,
    ):
        atoms = si_diamond_cubic.repeat(repeat)
        if displacement:
            atoms = perturb(atoms, displacement=displacement)

        eager = mattersim_potential_best_device
        aoti = mattersim_potential_aoti

        num_triples = int(
            next(iter(self._dataloader(atoms, eager))).three_body_indices.shape[0]
        )
        assert num_triples != EXAMPLE_NUM_TRIPLES, (
            "test structure must not have the compile-time three-body count, "
            "otherwise a specialized artifact would pass"
        )

        e_ref, f_ref, s_ref = eager.predict_properties(
            self._dataloader(atoms, eager),
            include_forces=True,
            include_stresses=True,
        )
        e_aoti, f_aoti, s_aoti = aoti.predict_properties(
            self._dataloader(atoms, aoti),
            include_forces=True,
            include_stresses=True,
        )

        assert abs(e_ref[0] - e_aoti[0]) / len(atoms) < 1e-4
        np.testing.assert_allclose(f_aoti[0], f_ref[0], atol=1e-3)
        np.testing.assert_allclose(s_aoti[0], s_ref[0], atol=1e-2)

    @pytest.mark.parametrize("cell_layout", ["contiguous", "transposed_view"])
    def test_non_contiguous_cell_matches_eager(
        self,
        mattersim_potential_aoti,
        mattersim_potential_best_device,
        si_diamond_cubic,
        perturb,
        cell_layout,
    ):
        """Regression: AOTI misread a non-contiguous ``cell`` input.

        TorchSim's cell filters store the column-vector cell, so the
        ``row_vector_cell`` handed to the model is a transposed view with
        strides ``(9, 1, 3)``. The artifact assumed the contiguous strides of
        its compile-time example and returned stresses off by up to thousands
        of GPa. A sheared (non-symmetric) cell is required: the transpose of
        a symmetric cell has the same values in either layout.
        """
        atoms = perturb(si_diamond_cubic, displacement=0.05)
        cell = atoms.cell.array.copy()
        cell[1, 0] += 0.8
        cell[2, 0] += 0.4
        cell[2, 1] += 0.6
        atoms.set_cell(cell, scale_atoms=True)

        results = {}
        for name, potential in (
            ("eager", mattersim_potential_best_device),
            ("aoti", mattersim_potential_aoti),
        ):
            graph_batch = next(iter(self._dataloader(atoms, potential)))
            input_dict = batch_to_dict(graph_batch, device="cuda")
            if name == "aoti" and cell_layout == "transposed_view":
                input_dict["cell"] = input_dict["cell"].mT.contiguous().mT
                assert not input_dict["cell"].is_contiguous()
            output = potential(input_dict, include_forces=True, include_stresses=True)
            results[name] = {
                key: output[key].detach().cpu().numpy()
                for key in ("total_energy", "forces", "stresses")
            }

        ref, aoti = results["eager"], results["aoti"]
        assert abs(ref["total_energy"][0] - aoti["total_energy"][0]) / len(atoms) < 1e-4
        np.testing.assert_allclose(aoti["forces"], ref["forces"], atol=1e-3)
        np.testing.assert_allclose(aoti["stresses"], ref["stresses"], atol=1e-2)
