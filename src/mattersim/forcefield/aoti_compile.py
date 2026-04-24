"""AOTI (Ahead-of-Time Inductor) compilation for MatterSim M3GNet models.

Compiles the M3GNet energy + force + stress computation into an optimized
.pt2 artifact using PyTorch AOTInductor. The compiled model is cached on disk
and reloaded on subsequent calls.

Reference: https://github.com/abhijeetgangan/aoti_mlip

Usage::

    from mattersim.forcefield.m3gnet.m3gnet import M3Gnet
    from mattersim.forcefield.aoti_compile import (
        AOTISettings, compile_m3gnet_aoti, load_aoti_model,
    )

    # Load your model
    m3gnet = M3Gnet(...)
    m3gnet.load_state_dict(...)
    m3gnet.eval()

    # Compile
    settings = AOTISettings(include_forces=True, include_stresses=True)
    pt2_path = compile_m3gnet_aoti(
        m3gnet, version="v1.0.0", device="cuda", settings=settings,
    )

    # Load the compiled model (drop-in replacement for M3Gnet as Potential.model)
    aoti_model = load_aoti_model(pt2_path, m3gnet.model_args, settings)
"""

import contextlib
import hashlib
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from ase.build import bulk
from ase.units import GPa
from torch.export.dynamic_shapes import Dim

from mattersim.datasets.utils.build import build_dataloader
from mattersim.forcefield.m3gnet.m3gnet import M3Gnet
from mattersim.forcefield.potential import batch_to_dict

logger = logging.getLogger(__name__)

_AOT_METADATA_KEY = "aot_inductor.metadata"


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class AOTISettings:
    """Configuration for AOTI compilation of MatterSim models.

    Controls which outputs are baked into the compiled artifact. Omitting an
    output at compile time avoids the corresponding computation entirely,
    unlike the eager model where the flag only controls whether the result is
    returned.

    Attributes:
        enabled: Whether to use AOTI compilation. Default True.
            Set to False to fall back to the eager model.
        include_forces: Compile forces (via autograd). Default True.
        include_stresses: Compile stresses (via strain perturbation).
            Requires ``include_forces=True``. Default True.
        force_recompile: Ignore cached artifacts and recompile. Default False.
    """

    enabled: bool = True
    include_forces: bool = True
    include_stresses: bool = True
    force_recompile: bool = False

    def __post_init__(self) -> None:
        if self.include_stresses and not self.include_forces:
            raise ValueError(
                "include_stresses=True requires include_forces=True "
                "(stress computation needs position gradients)."
            )
        if not self.enabled and (
            not self.include_forces
            or not self.include_stresses
            or self.force_recompile
        ):
            raise ValueError(
                "enabled=False disables AOTI entirely; other settings "
                "(include_forces, include_stresses, force_recompile) "
                "have no effect and should be left at their defaults."
            )

    @property
    def outputs_tag(self) -> str:
        """Short tag summarising compiled outputs, e.g. 'efs', 'ef', 'e'."""
        tag = "e"
        if self.include_forces:
            tag += "f"
        if self.include_stresses:
            tag += "s"
        return tag


# ---------------------------------------------------------------------------
# Dynamic shape specs for torch.export
# ---------------------------------------------------------------------------
_BATCH_DIM = Dim("batch_size", min=1)
_NODE_DIM = Dim("num_atoms", min=1)
_EDGE_DIM = Dim("num_edges", min=1)
_THREE_BODY_DIM = Dim("num_three_body", min=1)

MATTERSIM_DYNAMIC_SHAPES: tuple[dict[int, Dim], ...] = (
    {0: _NODE_DIM, 1: Dim.STATIC},  # atom_pos [N, 3]
    {0: _BATCH_DIM, 1: Dim.STATIC, 2: Dim.STATIC},  # cell [B, 3, 3]
    {0: _EDGE_DIM, 1: Dim.STATIC},  # pbc_offsets [E, 3]
    {0: _NODE_DIM, 1: Dim.STATIC},  # atom_attr [N, 1]
    {0: Dim.STATIC, 1: _EDGE_DIM},  # edge_index [2, E]
    {0: _THREE_BODY_DIM, 1: Dim.STATIC},  # three_body_indices [T, 2]
    {0: _BATCH_DIM},  # num_three_body [B]
    {0: _BATCH_DIM},  # num_bonds [B]
    {0: _EDGE_DIM, 1: Dim.STATIC},  # num_triple_ij [E, 1]
    {0: _BATCH_DIM},  # num_atoms [B]
    {},  # num_graphs (scalar)
    {0: _NODE_DIM},  # batch [N]
)


# ---------------------------------------------------------------------------
# Wrapper model for AOTI compilation
# ---------------------------------------------------------------------------
class M3GNetForAOTI(nn.Module):
    """Wraps M3Gnet to accept individual tensor args and compute forces/stresses.

    Required for torch.export which needs individual tensor arguments (not dicts).
    Force and stress computation are optional and controlled at compile time
    via :class:`AOTISettings`.
    """

    def __init__(
        self,
        m3gnet: M3Gnet,
        settings: AOTISettings,
        device: str = "cuda",
    ):
        super().__init__()
        self.m3gnet = m3gnet
        self._device = device
        self._settings = settings

    @property
    def _include_forces(self) -> bool:
        return self._settings.include_forces

    @property
    def _include_stresses(self) -> bool:
        return self._settings.include_stresses

    def forward(
        self,
        atom_pos: torch.Tensor,
        cell: torch.Tensor,
        pbc_offsets: torch.Tensor,
        atom_attr: torch.Tensor,
        edge_index: torch.Tensor,
        three_body_indices: torch.Tensor,
        num_three_body: torch.Tensor,
        num_bonds: torch.Tensor,
        num_triple_ij: torch.Tensor,
        num_atoms: torch.Tensor,
        num_graphs: torch.Tensor,
        batch: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if self._include_forces:
            atom_pos = atom_pos.requires_grad_(True)

        if self._include_stresses:
            strain = torch.zeros_like(cell, requires_grad=True)
            cell = torch.matmul(
                cell,
                (torch.eye(3, device=cell.device)[None, ...] + strain),
            )
            # Use batch index to expand strain to per-atom, avoiding an
            # unbacked symbol from repeat_interleave with tensor repeats.
            strain_augment = strain[batch]
            atom_pos = torch.einsum(
                "bi, bij -> bj",
                atom_pos,
                (torch.eye(3, device=cell.device)[None, ...] + strain_augment),
            )
            volume = torch.linalg.det(cell)

        # Precompute derived fields for the model
        cumsum = torch.cumsum(num_bonds, dim=0) - num_bonds
        bond_index_bias = torch.repeat_interleave(
            cumsum, num_three_body, dim=0
        ).unsqueeze(-1)
        index_map = torch.arange(edge_index.shape[1], device=num_triple_ij.device)
        three_body_edge_map = torch.repeat_interleave(
            index_map, num_triple_ij.view(-1)
        )

        input_dict = {
            "atom_pos": atom_pos,
            "cell": cell,
            "pbc_offsets": pbc_offsets,
            "atom_attr": atom_attr,
            "edge_index": edge_index,
            "three_body_indices": three_body_indices,
            "num_three_body": num_three_body,
            "num_bonds": num_bonds,
            "num_triple_ij": num_triple_ij,
            "num_atoms": num_atoms,
            "num_graphs": num_graphs,
            "batch": batch,
            "bond_index_bias": bond_index_bias,
            "three_body_edge_map": three_body_edge_map,
        }

        energies = self.m3gnet.forward(input_dict)
        result: dict[str, torch.Tensor] = {"total_energy": energies.detach()}

        if self._include_forces:
            grad_inputs = [atom_pos]
            if self._include_stresses:
                grad_inputs.append(strain)

            grad = torch.autograd.grad(
                outputs=[energies],
                inputs=grad_inputs,
                grad_outputs=[torch.ones_like(energies)],
                create_graph=False,
            )
            force_grad = grad[0]
            result["forces"] = torch.neg(force_grad).detach()

            if self._include_stresses:
                stress_grad = grad[1]
                stresses = 1 / volume[:, None, None] * stress_grad / GPa
                result["stresses"] = stresses.detach()

        return result


# ---------------------------------------------------------------------------
# FX tracing helpers
# ---------------------------------------------------------------------------
@contextlib.contextmanager
def _fx_duck_shape(enabled: bool):
    prev = torch.fx.experimental._config.use_duck_shape  # type: ignore[attr-defined]
    torch.fx.experimental._config.use_duck_shape = enabled  # type: ignore[attr-defined]
    try:
        yield
    finally:
        torch.fx.experimental._config.use_duck_shape = prev  # type: ignore[attr-defined]


def _make_fx(model: nn.Module, inputs: tuple):
    from torch.fx.experimental.proxy_tensor import make_fx

    with _fx_duck_shape(False):
        return make_fx(
            model,
            tracing_mode="symbolic",
            _allow_non_fake_inputs=True,
            _error_on_data_dependent_ops=True,
        )(*[i.clone() for i in inputs])


# ---------------------------------------------------------------------------
# Example input generation
# ---------------------------------------------------------------------------
def _get_example_inputs(
    cutoff: float,
    threebody_cutoff: float,
    device: torch.device,
) -> tuple[torch.Tensor, ...]:
    """Build example graph inputs from a 2-structure batch.

    A batch_size=2 example is used so that the batch dimension is not
    specialized to a constant during tracing, allowing dynamic batch sizes
    at runtime.
    """
    atoms = bulk("Si", "diamond", a=5.43, cubic=True)
    dataloader = build_dataloader(
        [atoms, atoms],
        batch_size=2,
        model_type="m3gnet",
        shuffle=False,
        only_inference=True,
        cutoff=cutoff,
        threebody_cutoff=threebody_cutoff,
    )
    graph_batch = next(iter(dataloader))
    input_dict = batch_to_dict(graph_batch, model_type="m3gnet", device=str(device))

    return (
        input_dict["atom_pos"],
        input_dict["cell"],
        input_dict["pbc_offsets"],
        input_dict["atom_attr"],
        input_dict["edge_index"],
        input_dict["three_body_indices"],
        input_dict["num_three_body"],
        input_dict["num_bonds"],
        input_dict["num_triple_ij"],
        input_dict["num_atoms"],
        input_dict["num_graphs"],
        input_dict["batch"],
    )


# ---------------------------------------------------------------------------
# Compilation
# ---------------------------------------------------------------------------
def _get_cache_path(
    version: str,
    device: str,
    settings: AOTISettings,
) -> str:
    """Deterministic cache path for a compiled model."""
    cache_dir = Path.home() / ".cache" / "mattersim" / "aoti"
    cache_dir.mkdir(parents=True, exist_ok=True)
    # Include torch version in hash so recompilation happens on upgrades
    outputs_tag = settings.outputs_tag
    key = f"{version}_{device}_{outputs_tag}_{torch.__version__}"
    h = hashlib.md5(key.encode()).hexdigest()[:12]
    return str(cache_dir / f"mattersim_{version}_{device}_{outputs_tag}_{h}.pt2")


def compile_m3gnet_aoti(
    m3gnet: M3Gnet,
    *,
    version: str,
    device: str = "cuda",
    settings: AOTISettings = AOTISettings(enabled=True),
) -> str:
    """Compile an M3Gnet model to an AOTI .pt2 package.

    Args:
        m3gnet: The M3Gnet model (already loaded with weights).
        version: MatterSim version string (used for cache key).
        device: Target device.
        settings: AOTI compilation settings controlling which outputs
            (forces, stresses) are baked into the artifact. Defaults to
            ``AOTISettings(enabled=True)`` (energy + forces + stresses).

    Returns:
        Path to the compiled .pt2 package.
    """
    cache_path = _get_cache_path(version, device, settings=settings)
    if os.path.exists(cache_path) and not settings.force_recompile:
        logger.info(f"AOTI compiled model found in cache: {cache_path}")
        return cache_path

    logger.info(f"Compiling M3Gnet to AOTI for device={device} ...")
    torch._dynamo.reset()

    model_args = m3gnet.model_args
    cutoff = model_args["cutoff"]
    threebody_cutoff = model_args["threebody_cutoff"]

    wrapper = M3GNetForAOTI(m3gnet, device=device, settings=settings)
    for p in wrapper.parameters():
        p.requires_grad_(False)
    wrapper.eval()
    wrapper.to(device)

    example_inputs = _get_example_inputs(
        cutoff, threebody_cutoff, torch.device(device)
    )

    # Validate eager model first
    logger.info("Validating eager model outputs...")
    results = wrapper(*example_inputs)
    logger.info(f"Eager model output keys: {list(results.keys())}")

    # FX trace → export → AOTI compile
    logger.info("FX tracing model...")
    fx_model = _make_fx(wrapper, example_inputs)

    logger.info("Exporting model...")
    exported_model = torch.export.export(
        fx_model,
        example_inputs,
        dynamic_shapes=MATTERSIM_DYNAMIC_SHAPES,
    )

    logger.info("AOTI compiling and packaging...")
    torch._inductor.aoti_compile_and_package(
        exported_model,
        package_path=cache_path,
        inductor_configs={
            _AOT_METADATA_KEY: {
                "cutoff": str(cutoff),
                "threebody_cutoff": str(threebody_cutoff),
                "version": version,
                "include_forces": str(settings.include_forces),
                "include_stresses": str(settings.include_stresses),
            },
        },
    )

    logger.info(f"AOTI compiled model saved to: {cache_path}")
    if torch.cuda.is_available() and "cuda" in str(device):
        torch.cuda.empty_cache()
    return cache_path


# ---------------------------------------------------------------------------
# AOTIModelWrapper: drop-in replacement for M3Gnet as Potential.model
# ---------------------------------------------------------------------------
class AOTIModelWrapper(nn.Module):
    """Wraps an AOTI-compiled model to be used as ``Potential.model``.

    Unlike the eager ``M3Gnet`` model that returns only energies (and relies on
    ``Potential.forward`` to compute forces/stresses via autograd), this wrapper
    returns a dict containing energy and (optionally) forces and stresses
    directly because the autograd computation is baked into the AOTI artifact.

    ``Potential.forward`` detects the ``_is_aoti`` flag and short-circuits the
    autograd path, returning the pre-computed results instead.

    Note:
        The outputs available depend on the :class:`AOTISettings` used at
        compile time.

    Attributes:
        model_args: Dict mirroring ``M3Gnet.model_args`` (cutoffs, etc.).
        include_forces: Whether the compiled artifact includes force
            computation.
        include_stresses: Whether the compiled artifact includes stress
            computation.
    """

    _is_aoti: bool = True

    def __init__(
        self,
        aoti_model,
        settings: AOTISettings,
        model_args: dict[str, Any],
    ):
        super().__init__()
        self._aoti_model = aoti_model
        self.model_args = model_args
        self._settings = settings

    @property
    def include_forces(self) -> bool:
        return self._settings.include_forces

    @property
    def include_stresses(self) -> bool:
        return self._settings.include_stresses

    def forward(
        self,
        input: dict[str, torch.Tensor],
        dataset_idx: int = -1,
    ) -> dict[str, torch.Tensor]:
        args = (
            input["atom_pos"],
            input["cell"],
            input["pbc_offsets"],
            input["atom_attr"],
            input["edge_index"],
            input["three_body_indices"],
            input["num_three_body"],
            input["num_bonds"],
            input["num_triple_ij"],
            input["num_atoms"],
            input["num_graphs"],
            input["batch"],
        )
        result = self._aoti_model(*args)
        output = {
            "energies": result["total_energy"],
            "total_energy": result["total_energy"],
        }
        if self.include_forces:
            output["forces"] = result["forces"]
        if self.include_stresses:
            output["stresses"] = result["stresses"]
        return output

    def enable_gradient_checkpointing(self, enable: bool = True) -> None:
        if enable:
            logger.warning(
                "Gradient checkpointing is not supported for "
                "AOTI-compiled models."
            )


def load_aoti_model(
    package_path: str,
    model_args: dict[str, Any],
    settings: AOTISettings,
) -> AOTIModelWrapper:
    """Load a compiled .pt2 model and wrap it as an ``AOTIModelWrapper``.

    Args:
        package_path: Path to the compiled .pt2 package.
        model_args: Dict mirroring ``M3Gnet.model_args`` (cutoffs, etc.).
        settings: AOTI settings that were used during compilation.

    Returns:
        An ``AOTIModelWrapper`` instance ready to use as ``Potential.model``.
    """
    logger.info(f"Loading AOTI compiled model from {package_path}")
    aoti_model = torch._inductor.aoti_load_package(package_path)
    return AOTIModelWrapper(
        aoti_model=aoti_model,
        model_args=model_args,
        settings=settings,
    )
