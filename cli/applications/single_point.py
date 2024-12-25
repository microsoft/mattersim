# -*- coding: utf-8 -*-
"""
This script is used to predict single point properties for a list of atoms.
The predicted properties include energy, energy per atom, forces, and stress.
The energies are predicted in eV, forces in eV/Angstrom, and stress in eV/Angstrom^3.
"""
import argparse

from ase import Atoms
from ase.io import read as ase_read
from loguru import logger
from tqdm import tqdm

from mattersim.forcefield import MatterSimCalculator


def predict_single_point_properties(
    atoms_list: list[Atoms],
) -> dict:
    """
    Predict single point properties for a list of atoms.

    Args:
        atoms_list (list[Atoms]): List of ASE Atoms objects.
    """
    pred_energy_list = []
    pred_energy_per_atom_list = []
    pred_forces_list = []
    pred_stress_list = []
    for atoms in tqdm(
        atoms_list, total=len(atoms_list), desc="Predicting single point properties"
    ):
        (pred_energy, pred_forces, pred_stress) = (
            atoms.get_potential_energy(),
            atoms.get_forces(),
            atoms.get_stress(voigt=False),
        )
        pred_energy_list.append(pred_energy)
        pred_energy_per_atom_list.append(pred_energy / len(atoms))
        pred_forces_list.append(pred_forces)
        pred_stress_list.append(pred_stress)

    return {
        "energy": pred_energy_list,
        "energy_per_atom": pred_energy_per_atom_list,
        "forces": pred_forces_list,
        "stress": pred_stress_list,
    }


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

    pred_results = predict_single_point_properties(atoms_list)

    for key, value in pred_results.items():
        logger.info(f"Predicted {key}: {value}")
    logger.info("Prediction finished.")
