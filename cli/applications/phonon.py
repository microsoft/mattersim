"""
Relax one or a list of structures using MatterSim.
"""
import argparse

import numpy as np
from ase import Atoms
from ase.io import read as ase_read
from loguru import logger
from tqdm import tqdm

from mattersim.applications.phonon import PhononWorkflow
from mattersim.forcefield import MatterSimCalculator


def predict_phonon(
    atoms_list: list[Atoms],
    find_prim: bool = False,
    work_dir: str = None,
    amplitude: float = 0.01,
    supercell_matrix: np.ndarray = None,
    qpoints_mesh: np.ndarray = None,
    max_atoms: int = None,
):
    """
    Predict phonon properties for a list of atoms.

    Args:
        atoms_list (list[Atoms]): List of ASE Atoms objects.
        find_prim (bool, optional): If find the primitive cell and use it
            to calculate phonon. Default to False.
        work_dir (str, optional): workplace path to contain phonon result.
            Defaults to data + chemical_symbols + 'phonon'
        amplitude (float, optional): Magnitude of the finite difference to
            displace in force constant calculation, in Angstrom. Defaults
            to 0.01 Angstrom.
        supercell_matrix (nd.array, optional): Supercell matrix for constr
            -uct supercell, priority over than max_atoms. Defaults to None.
        qpoints_mesh (nd.array, optional): Qpoint mesh for IBZ integral,
            priority over than max_atoms. Defaults to None.
        max_atoms (int, optional): Maximum atoms number limitation for the
            supercell generation. If not set, will automatic generate super
            -cell based on symmetry. Defaults to None.
    """
    has_imaginary_list = []
    pred_phonon_list = []
    for atoms in tqdm(
        atoms_list, total=len(atoms_list), desc="Predicting phonon properties"
    ):
        ph = PhononWorkflow(
            atoms=atoms,
            find_prim=find_prim,
            work_dir=work_dir,
            amplitude=amplitude,
            supercell_matrix=supercell_matrix,
            qpoints_mesh=qpoints_mesh,
            max_atoms=max_atoms,
        )
        has_imaginary, phonon = ph.run()
        has_imaginary_list.append(has_imaginary)
        pred_phonon_list.append(phonon)

    return has_imaginary_list, pred_phonon_list


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        description="Predict phonon properties for one or a list of structures."
    )
    argparser.add_argument(
        "--structure-file",
        type=str,
        nargs="+",
        help="Path to the atoms structure file(s).",
    )
    argparser.add_argument(
        "--find-prim",
        action="store_true",
        help="If find the primitive cell and use it to calculate phonon.",
    )
    argparser.add_argument(
        "--work-dir",
        type=str,
        default=None,
        help="Workplace path to contain phonon result.",
    )
    argparser.add_argument(
        "--amplitude",
        type=float,
        default=0.01,
        help="Magnitude of the finite difference to displace in "
        "force constant calculation, in Angstrom.",
    )
    argparser.add_argument(
        "--supercell-matrix",
        type=int,
        nargs=3,
        default=None,
        help="Supercell matrix for construct supercell, must be a list of 3 integers.",
    )
    argparser.add_argument(
        "--qpoints-mesh",
        type=int,
        nargs=3,
        default=None,
        help="Qpoint mesh for IBZ integral, must be a list of 3 integers.",
    )
    argparser.add_argument(
        "--max-atoms",
        type=int,
        default=None,
        help="Maximum atoms number limitation for the supercell generation. "
        "If not set, will automatic generate supercell based on symmetry.",
    )
    argparser.add_argument(
        "--mattersim-model",
        type=str,
        choices=["mattersim-v1.0.0-1m", "mattersim-v1.0.0-5m"],
        default="mattersim-v1.0.0-1m",
        help="MatterSim model to use for prediction. Available models are: "
        "mattersim-v1.0.0-1m, mattersim-v1.0.0-5m",
    )
    argparser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to use for prediction. Default is cpu.",
    )

    args = argparser.parse_args()

    logger.warning(
        "This script predicts phonon properties for a list of atoms. "
        "Please note this script assumes that the structures have already been relaxed."
    )

    logger.info("Initializing MatterSim calculator.")
    calc = MatterSimCalculator(load_path=args.mattersim_model, device=args.device)

    logger.info(f"Reading atoms structures from {args.structure_file}")
    atoms_list = []
    for structure_file in args.structure_file:
        atoms_list += ase_read(structure_file, index=":")
    for atoms in atoms_list:
        atoms.calc = calc
    logger.info(f"Read {len(atoms_list)} atoms structures.")

    pred_phonon_list = predict_phonon(
        atoms_list=atoms_list,
        find_prim=args.find_prim,
        work_dir=args.work_dir,
        amplitude=args.amplitude,
        supercell_matrix=args.supercell_matrix,
        qpoints_mesh=args.qpoints_mesh,
        max_atoms=args.max_atoms,
    )
