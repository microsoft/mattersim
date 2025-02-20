import argparse
import uuid
from datetime import datetime
from typing import List, Union

from ase import Atoms
from ase.io import read as ase_read
from loguru import logger

from mattersim.cli.applications.moldyn import moldyn
from mattersim.cli.applications.phonon import phonon
from mattersim.cli.applications.relax import relax
from mattersim.cli.applications.singlepoint import singlepoint
from mattersim.forcefield import MatterSimCalculator


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
    singlepoint_args = {
        k: v
        for k, v in vars(args).items()
        if k not in ["structure_file", "mattersim_model", "device"]
    }
    return singlepoint(atoms_list, **singlepoint_args)


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
    relax_args = {
        k: v
        for k, v in vars(args).items()
        if k not in ["structure_file", "mattersim_model", "device"]
    }
    return relax(atoms_list, **relax_args)


def phonon_cli(args: argparse.Namespace) -> dict:
    """
    CLI wrapper for phonon function.

    Args:
        args (argparse.Namespace): Command line arguments.

    Returns:
        dict: Dictionary containing the phonon properties.
    """
    atoms_list = parse_atoms_list(
        args.structure_file, args.mattersim_model, args.device
    )
    phonon_args = {
        k: v
        for k, v in vars(args).items()
        if k not in ["structure_file", "mattersim_model", "device"]
    }
    return phonon(atoms_list, **phonon_args)


def moldyn_cli(args: argparse.Namespace) -> dict:
    """
    CLI wrapper for moldyn function.

    Args:
        args (argparse.Namespace): Command line arguments.

    Returns:
        dict: Dictionary containing the molecular dynamics properties.
    """
    atoms_list = parse_atoms_list(
        args.structure_file, args.mattersim_model, args.device
    )
    if len(atoms_list) > 1:
        logger.error("Molecular dynamics may take too long for multiple structures.")

    moldyn_args = {
        k: v
        for k, v in vars(args).items()
        if k not in ["structure_file", "mattersim_model", "device"]
    }
    return moldyn(atoms_list, **moldyn_args)


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


def add_relax_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--optimizer",
        type=str,
        default="FIRE",
        help="The optimizer to use. Default is FIRE.",
    )
    parser.add_argument(
        "--filter",
        type=str,
        default=None,
        help="The filter to use.",
    )
    parser.add_argument(
        "--constrain-symmetry",
        action="store_true",
        help="Constrain symmetry.",
    )
    parser.add_argument(
        "--fix-axis",
        type=bool,
        default=False,
        nargs="+",
        help="Fix the axis.",
    )
    parser.add_argument(
        "--pressure-in-GPa",
        type=float,
        default=None,
        help="Pressure in GPa to use for relaxation.",
    )
    parser.add_argument(
        "--fmax",
        type=float,
        default=0.01,
        help="Maximum force tolerance for relaxation.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=500,
        help="Maximum number of steps for relaxation.",
    )


def add_phonon_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--find-prim",
        action="store_true",
        help="If find the primitive cell and use it to calculate phonon.",
    )
    parser.add_argument(
        "--amplitude",
        type=float,
        default=0.01,
        help="Magnitude of the finite difference to displace in "
        "force constant calculation, in Angstrom.",
    )
    parser.add_argument(
        "--supercell-matrix",
        type=int,
        nargs=3,
        default=None,
        help="Supercell matrix for construct supercell, must be a list of 3 integers.",
    )
    parser.add_argument(
        "--qpoints-mesh",
        type=int,
        nargs=3,
        default=None,
        help="Qpoint mesh for IBZ integral, must be a list of 3 integers.",
    )
    parser.add_argument(
        "--max-atoms",
        type=int,
        default=None,
        help="Maximum atoms number limitation for the supercell generation.",
    )
    parser.add_argument(
        "--enable-relax",
        action="store_true",
        help="Whether to relax the structure before predicting phonon properties.",
    )


def add_moldyn_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--temperature",
        type=float,
        default=300,
        help="Temperature in Kelvin.",
    )
    parser.add_argument(
        "--timestep",
        type=float,
        default=1,
        help="Timestep in femtoseconds.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=1000,
        help="Number of steps for the molecular dynamics simulation.",
    )
    parser.add_argument(
        "--ensemble",
        type=str,
        choices=["nvt_berendsen", "nvt_nose_hoover"],
        default="nvt_nose_hoover",
        help="Simulation ensemble to use.",
    )
    parser.add_argument(
        "--logfile",
        type=str,
        default="md.log",
        help="Logfile to write the output to. Default is stdout.",
    )
    parser.add_argument(
        "--loginterval",
        type=int,
        default=10,
        help="Log interval for writing the output.",
    )
    parser.add_argument(
        "--trajectory",
        type=str,
        default="md.traj",
        help="Path to the trajectory file.",
    )
    parser.add_argument(
        "--taut",
        type=float,
        default=None,
        help="Taut in femtoseconds. It is used for the Nose-Hoover thermostat.",
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
    add_relax_args(relax_parser)
    relax_parser.set_defaults(func=relax_cli)

    # Sub-command for phonon
    phonon_parser = subparsers.add_parser(
        "phonon",
        help="Predict phonon properties for a list of structures.",
    )
    add_common_args(phonon_parser)
    add_relax_args(phonon_parser)
    add_phonon_args(phonon_parser)
    phonon_parser.set_defaults(func=phonon_cli)

    # Sub-command for molecular dynamics
    moldyn_parser = subparsers.add_parser(
        "moldyn",
        help="Perform molecular dynamics simulation for a list of structures.",
    )
    add_common_args(moldyn_parser)
    add_moldyn_args(moldyn_parser)
    moldyn_parser.set_defaults(func=moldyn_cli)

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
