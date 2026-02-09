from typing import Literal

import numpy as np
import torch
from ase import Atoms
from torch_geometric.loader import DataLoader as DataLoader_pyg
from tqdm import tqdm

from mattersim.datasets.utils.converter import GraphConverter, BatchGraphConverter


def build_dataloader(
    atoms: list[Atoms] | None = None,
    energies: list[float] | None = None,
    forces: list[np.ndarray] | None = None,
    stresses: list[np.ndarray] | None = None,
    cutoff: float = 5.0,
    threebody_cutoff: float = 4.0,
    batch_size: int = 64,
    model_type: Literal["m3gnet"] = "m3gnet",
    shuffle: bool = False,
    only_inference: bool = False,
    num_workers: int = 0,
    pin_memory: bool = False,
    batch_converter: bool = True,
    max_natoms_per_batch: int = 4096,
):
    """
    Build a dataloader given a list of atoms

    Args:
        atoms: list of Atoms objects
        energies: list of energies corresponding to the atoms
        forces: list of forces corresponding to the atoms
        stresses: list of stresses corresponding to the atoms
        cutoff: cutoff distance for graph construction
        threebody_cutoff: cutoff distance for three-body interactions
        batch_size: number of samples per batch
        model_type: type of model to use
        shuffle: whether to shuffle the data
        only_inference: whether to only perform inference
        num_workers: number of worker processes for data loading
        pin_memory: whether to pin memory
        batch_converter: whether to use batch converter
        max_natoms_per_batch: maximum number of atoms per batch in the batch converter, only used if batch_converter is True
            Do not confuse with batch_size, which is the number of samples per batch in the final dataloader.
            But, max_natoms_per_batch is used to control the number of atoms to construct the graph.
    """

    if not batch_converter:
        converter = GraphConverter(model_type, cutoff, True, threebody_cutoff)
    else:
        converter = BatchGraphConverter(
            model_type, 
            twobody_cutoff=cutoff, 
            has_threebody=True,
            threebody_cutoff=threebody_cutoff
        )

    preprocessed_data = []

    # sanity checks
    if not only_inference and np.any([x is None for x in [energies, forces, stresses]]):
        raise ValueError(
            "energies, forces, and stresses must be provided if only_inference is False"
        )

    if only_inference:
        length = len(atoms)
        if energies is None:
            energies = [None] * length
        if forces is None:
            forces = [None] * length
        if stresses is None:
            stresses = [None] * length
    else:
        assert (
            len(atoms) == len(energies) == len(forces) == len(stresses)
        ), "Length of atoms, energies, forces, and stresses must be the same"
    

    if model_type == "m3gnet":
        if not batch_converter:
            for graph, energy, force, stress in zip(
                atoms, energies, forces, stresses
            ):
                graph = converter.convert(
                    graph.copy(), energy, force, stress
                )
                if graph is not None:
                    preprocessed_data.append(graph)
        else:
            preprocessed_data = converter.convert(
                atoms,
                energy=energies,
                forces=forces,
                stresses=stresses,
                max_natoms_per_batch=max_natoms_per_batch,
            )
    else:
        raise NotImplementedError(f"model type not supported: {model_type}")


    return DataLoader_pyg(
        preprocessed_data,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )