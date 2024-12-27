import argparse
import os
import uuid
from collections import defaultdict
from datetime import datetime
from typing import List, Union

# import numpy as np
import pandas as pd
from ase import Atoms
from ase.constraints import Filter
from ase.io import read as ase_read
from ase.optimize.optimize import Optimizer
from ase.units import GPa
from loguru import logger
from pymatgen.io.ase import AseAtomsAdaptor
from tqdm import tqdm

# from mattersim.applications.phonon import PhononWorkflow
from mattersim.applications.relax import Relaxer
from mattersim.forcefield import MatterSimCalculator

__all__ = ["singlepoint", "phonon", "relax", "moldyn"]


def singlepoint(
    atoms_list: List[Atoms],
    *,
    mattersim_model: str,
    device: str = "cpu",
    work_dir: str = str(uuid.uuid4()),
    save_csv: str = "results.csv.gz",
    **kwargs,
) -> dict:
    """
    Predict single point properties for a list of atoms.

    """
    logger.info("Predicting single point properties.")
    predicted_properties = defaultdict(list)
    for atoms in tqdm(
        atoms_list, total=len(atoms_list), desc="Predicting single point properties"
    ):
        predicted_properties["structure"].append(AseAtomsAdaptor.get_structure(atoms))
        predicted_properties["energy"].append(atoms.get_potential_energy())
        predicted_properties["energy_per_atom"].append(
            atoms.get_potential_energy() / len(atoms)
        )
        predicted_properties["forces"].append(atoms.get_forces())
        predicted_properties["stress"].append(atoms.get_stress(voigt=False))
        predicted_properties["stress_GPa"].append(atoms.get_stress(voigt=False) / GPa)

    if not os.path.exists(work_dir):
        os.makedirs(work_dir)

    logger.info(f"Saving the results to {os.path.join(work_dir, save_csv)}")

    df = pd.DataFrame(predicted_properties)
    df.to_csv(os.path.join(work_dir, save_csv), index=False)
    return predicted_properties


def singlepoint_cli(args: argparse.Namespace) -> dict:
    """
    CLI wrapper for singlepoint function.

    Args:
        args (argparse.Namespace): Command line arguments.

    Returns:
        dict: Dictionary containing the predicted properties.

    """
    atoms_list = parse_atoms_list(
        args.structure_file, args.mattersim_model, args.device
    )
    return singlepoint(atoms_list, **vars(args))


def relax(
    atoms_list: List[Atoms],
    *,
    optimizer: Union[str, Optimizer] = "FIRE",
    filter: Union[str, Filter, None] = None,
    constrain_symmetry: bool = False,
    fix_axis: Union[bool, List[bool]] = False,
    pressure_in_GPa: float = None,
    fmax: float = 0.01,
    steps: int = 500,
    work_dir: str = str(uuid.uuid4()),
    save_csv: str = "results.csv.gz",
    **kwargs,
) -> dict:
    """
    Relax a list of atoms structures.

    Args:
        atoms_list (List[Atoms]): List of ASE Atoms objects.
        optimizer (Union[str, Optimizer]): The optimizer to use. Default is "FIRE".
        filter (Union[str, Filter, None]): The filter to use.
        constrain_symmetry (bool): Whether to constrain symmetry. Default is False.
        fix_axis (Union[bool, List[bool]]): Whether to fix the axis. Default is False.
        pressure_in_GPa (float): Pressure in GPa to use for relaxation.
        fmax (float): Maximum force tolerance for relaxation. Default is 0.01.
        steps (int): Maximum number of steps for relaxation. Default is 500.
        work_dir (str): Working directory for the calculations.
            Default is a UUID with timestamp.
        save_csv (str): Save the results to a CSV file. Default is `results.csv.gz`.

    Returns:
        pd.DataFrame: DataFrame containing the relaxed results.
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

    relaxed_results = defaultdict(list)
    for atoms in tqdm(atoms_list, total=len(atoms_list), desc="Relaxing structures"):
        converged, relaxed_atoms = relaxer.relax(
            atoms,
            params_filter=params_filter,
            fmax=fmax,
            steps=steps,
        )
        relaxed_results["converged"].append(converged)
        relaxed_results["structure"].append(
            AseAtomsAdaptor.get_structure(relaxed_atoms).to_json()
        )
        relaxed_results["energy"].append(relaxed_atoms.get_potential_energy())
        relaxed_results["energy_per_atom"].append(
            relaxed_atoms.get_potential_energy() / len(relaxed_atoms)
        )
        relaxed_results["forces"].append(relaxed_atoms.get_forces())
        relaxed_results["stress"].append(relaxed_atoms.get_stress(voigt=False))
        relaxed_results["stress_GPa"].append(
            relaxed_atoms.get_stress(voigt=False) / GPa
        )

        logger.info(f"Relaxed structure: {relaxed_atoms}")

    if not os.path.exists(work_dir):
        os.makedirs(work_dir)

    logger.info(f"Saving the results to {os.path.join(work_dir, save_csv)}")
    df = pd.DataFrame(relaxed_results)
    df.to_csv(os.path.join(work_dir, save_csv), index=False)
    return relaxed_results


def relax_cli(args: argparse.Namespace) -> dict:
    """
    CLI wrapper for relax function.

    Args:
        args (argparse.Namespace): Command line arguments.

    Returns:
        dict: Dictionary containing the relaxed results
    """
    atoms_list = parse_atoms_list(
        args.structure_file, args.mattersim_model, args.device
    )
    return relax(atoms_list, **vars(args))


def phonon():
    pass


def moldyn():
    pass


def add_common_args(parser: argparse.ArgumentParser):
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
        help="MatterSim model to use.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to use for prediction. Default is cpu.",
    )
    parser.add_argument(
        "--work-dir",
        type=str,
        default=datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + "-" + str(uuid.uuid4()),
        help="Working directory for the calculations. "
        "Defaults to a UUID with timestamp when not set.",
    )
    parser.add_argument(
        "--save-csv",
        type=str,
        default="results.csv.gz",
        help="Save the results to a CSV file. "
        "Defaults to `results.csv.gz` when not set.",
    )


def parse_atoms_list(
    structure_file_list: Union[str, List[str]],
    mattersim_model: str,
    device: str = "cpu",
) -> List[Atoms]:
    if isinstance(structure_file_list, str):
        structure_file_list = [structure_file_list]

    calc = MatterSimCalculator(load_path=mattersim_model, device=device)
    atoms_list = []
    for structure_file in structure_file_list:
        atoms_list += ase_read(structure_file, index=":")
    for atoms in atoms_list:
        atoms.calc = calc
    return atoms_list


def main():
    argparser = argparse.ArgumentParser(description="CLI for MatterSim.")
    subparsers = argparser.add_subparsers(
        title="Subcommands",
        description="Valid subcommands",
        help="Available subcommands",
    )

    # Sub-command for single-point prediction
    singlepoint_parser = subparsers.add_parser(
        "singlepoint", help="Predict single point properties for a list of atoms."
    )
    add_common_args(singlepoint_parser)
    singlepoint_parser.set_defaults(func=singlepoint_cli)

    # Sub-command for relax
    relax_parser = subparsers.add_parser(
        "relax", help="Relax a list of atoms structures."
    )
    add_common_args(relax_parser)
    relax_parser.add_argument(
        "--optimizer",
        type=str,
        default="FIRE",
        help="The optimizer to use. Default is FIRE.",
    )
    relax_parser.add_argument(
        "--filter",
        type=str,
        default=None,
        help="The filter to use.",
    )
    relax_parser.add_argument(
        "--constrain-symmetry",
        action="store_true",
        help="Constrain symmetry.",
    )
    relax_parser.add_argument(
        "--fix-axis",
        type=bool,
        default=False,
        nargs="+",
        help="Fix the axis.",
    )
    relax_parser.add_argument(
        "--pressure-in-GPa",
        type=float,
        default=None,
        help="Pressure in GPa to use for relaxation.",
    )
    relax_parser.add_argument(
        "--fmax",
        type=float,
        default=0.01,
        help="Maximum force tolerance for relaxation.",
    )
    relax_parser.add_argument(
        "--steps",
        type=int,
        default=500,
        help="Maximum number of steps for relaxation.",
    )
    relax_parser.set_defaults(func=relax_cli)

    # Parse arguments
    args = argparser.parse_args()
    print(args)

    # Call the function associated with the sub-command
    if hasattr(args, "func"):
        args.func(args)
    else:
        argparser.print_help()


if __name__ == "__main__":
    main()
