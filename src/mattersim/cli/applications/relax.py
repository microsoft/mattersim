import os
import uuid
from collections import defaultdict
from typing import List, Union

import pandas as pd
from ase import Atoms
from ase.constraints import Filter
from ase.optimize.optimize import Optimizer
from ase.units import GPa
from loguru import logger
from pymatgen.io.ase import AseAtomsAdaptor
from tqdm import tqdm

from mattersim.applications.relax import Relaxer


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
    df.to_csv(os.path.join(work_dir, save_csv), index=False, mode="a")
    return relaxed_results
