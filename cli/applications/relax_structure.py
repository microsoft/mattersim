# -*- coding: utf-8 -*-
import argparse
from typing import Iterable, List, Tuple, Union

from ase import Atoms
from ase.constraints import Filter
from ase.io import read as ase_read
from ase.optimize.optimize import Optimizer
from ase.units import GPa
from loguru import logger

from mattersim.applications.relax import Relaxer
from mattersim.forcefield import MatterSimCalculator


def relax_structures(
    atoms: Union[Atoms, Iterable[Atoms]],
    optimizer: Union[Optimizer, str] = "FIRE",
    filter: Union[Filter, str, None] = None,
    constrain_symmetry: bool = False,
    fix_axis: Union[bool, Iterable[bool]] = False,
    pressure_in_GPa: Union[float, None] = None,
    **kwargs,
) -> Union[Tuple[bool, Atoms], Tuple[List[bool], List[Atoms]]]:
    """
    Args:
        atoms: (Union[Atoms, Iterable[Atoms]]):
            The Atoms object or an iterable of Atoms objetcs to relax.
        optimizer (Union[Optimizer, str]): The optimizer to use.
        filter (Union[Filter, str, None]): The filter to use.
        constrain_symmetry (bool): Whether to constrain the symmetry.
        fix_axis (Union[bool, Iterable[bool]]): Whether to fix the axis.
        pressure_in_GPa (Union[float, None]): The pressure in GPa.
        **kwargs: Additional keyword arguments for the relax method.
    Returns:
        converged (Union[bool, List[bool]]):
            Whether the relaxation converged or a list of them
        Atoms (Union[Atoms, List[Atoms]]):
            The relaxed atoms object or a list of them
    """
    params_filter = {}

    if pressure_in_GPa:
        params_filter["scalar_pressure"] = (
            pressure_in_GPa * GPa
        )  # convert GPa to eV/Angstrom^3
        filter = "ExpCellFilter" if filter is None else filter
    elif filter:
        params_filter["scalar_pressure"] = 0.0

    relaxer = Relaxer(
        optimizer=optimizer,
        filter=filter,
        constrain_symmetry=constrain_symmetry,
        fix_axis=fix_axis,
    )

    if isinstance(atoms, (list, tuple)):
        relaxed_results = relaxed_results = [
            relaxer.relax(atom, params_filter=params_filter, **kwargs) for atom in atoms
        ]
        converged, relaxed_atoms = zip(*relaxed_results)
        return list(converged), list(relaxed_atoms)
    else:
        return relaxer.relax(atoms, params_filter=params_filter, **kwargs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predict single point properties for a list of atoms."
    )
    parser.add_argument(
        "--structure-file",
        type=str,
        nargs="+",
        help="Path to the atoms structure file(s).",
    )
    parser.add_argument(
        "--mattersim-model",
        type=str,
        choices=["mattersim-v1.0.0-1m", "mattersim-v1.0.0-5m"],
        default="mattersim-v1.0.0-1m",
        help="MatterSim model to use for prediction. Available models are: "
        "mattersim-v1.0.0-1m, mattersim-v1.0.0-5m",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to use for prediction. Default is cpu.",
    )
    parser.add_argument(
        "--pressure-in-GPa",
        type=float,
        default=None,
        help="Pressure in GPa to use for relaxation. Default is None.",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="FIRE",
        help="Optimizer to use for relaxation. Default is FIRE.",
    )
    parser.add_argument(
        "--filter",
        type=str,
        default=None,
        choices=["ExpCellFilter", "FrechetCellFilter"],
        help="Filter to use for relaxation. Default is None.",
    )
    parser.add_argument(
        "--constrain-symmetry",
        action="store_true",
        help="Whether to constrain the symmetry.",
    )
    parser.add_argument(
        "--fix-axis",
        type=bool,
        default=False,
        nargs="+",
        help="Whether to fix the axis. Default is False. "
        "If a list is provided, it sets which axis to fix.",
    )
    args = parser.parse_args()

    logger.info("Initializing MatterSim calculator.")
    calc = MatterSimCalculator(load_path=args.mattersim_model, device=args.device)

    logger.info(f"Reading atoms structures from {args.structure_file}")
    atoms_list = []
    for structure_file in args.structure_file:
        atoms_list += ase_read(structure_file, index=":")
    for atoms in atoms_list:
        atoms.calc = calc
    logger.info(f"Read {len(atoms_list)} atoms structures.")

    logger.info("Relaxing atoms structures.")
    relaxed_results = relax_structures(
        atoms_list,
        optimizer=args.optimizer,
        filter=args.filter,
        constrain_symmetry=args.constrain_symmetry,
        fix_axis=args.fix_axis,
        pressure_in_GPa=args.pressure_in_GPa,
    )
    logger.info("Relaxation completed.")

    for converged, relaxed_atoms in zip(*relaxed_results):
        logger.info(f"Relaxation converged: {converged}")
        logger.info(f"Relaxed atoms: {relaxed_atoms}")
