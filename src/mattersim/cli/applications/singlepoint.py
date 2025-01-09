import os
import uuid
from collections import defaultdict
from typing import List

import pandas as pd
from ase import Atoms
from ase.units import GPa
from loguru import logger
from pymatgen.io.ase import AseAtomsAdaptor
from tqdm import tqdm


def singlepoint(
    atoms_list: List[Atoms],
    *,
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
        predicted_properties["structure"].append(AseAtomsAdaptor.get_structure(atoms).as_dict())
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
    df.to_csv(os.path.join(work_dir, save_csv), index=False, mode="a")
    return predicted_properties
