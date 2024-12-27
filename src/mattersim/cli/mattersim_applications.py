import argparse
import os
import uuid
from collections import defaultdict
from typing import List, Union

# import numpy as np
import pandas as pd
from ase import Atoms

# from ase.constraints import Filter
from ase.io import read as ase_read

# from ase.optimize.optimize import Optimizer
from ase.units import GPa
from loguru import logger
from pymatgen.io.ase import AseAtomsAdaptor
from tqdm import tqdm

# from mattersim.applications.phonon import PhononWorkflow
# from mattersim.applications.relax import Relaxer
from mattersim.forcefield import MatterSimCalculator

__all__ = ["singlepoint", "phonon", "relax", "moldyn"]


def singlepoint(
    structure_file: Union[str, List[str]],
    mattersim_model: str,
    device: str = "cpu",
    work_dir: str = str(uuid.uuid4()),
    save_csv: str = "results.csv.gz",
):
    """
    Predict single point properties for a list of atoms.

    """
    atoms_list = parse_atoms_list(structure_file, mattersim_model, device)
    logger.info(f"Predicting single point properties for {len(atoms_list)} structures.")

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


def singlepoint_cli(args: argparse.Namespace):
    singlepoint(
        args.structure_file,
        args.mattersim_model,
        args.device,
        args.work_dir,
        args.save_csv,
    )


def relax(args: argparse.Namespace):
    pass


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
        default=str(uuid.uuid4()),
        help="Working directory for the calculations. Defaults to a UUID when not set.",
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
