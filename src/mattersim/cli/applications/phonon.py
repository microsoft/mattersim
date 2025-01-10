import os
import uuid
from collections import defaultdict
from typing import List

import numpy as np
import pandas as pd
import yaml
from ase import Atoms
from loguru import logger
from pymatgen.core.structure import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from tqdm import tqdm

from mattersim.applications.phonon import PhononWorkflow
from mattersim.cli.applications.relax import relax


def phonon(
    atoms_list: List[Atoms],
    *,
    find_prim: bool = False,
    work_dir: str = str(uuid.uuid4()),
    save_csv: str = "results.csv.gz",
    amplitude: float = 0.01,
    supercell_matrix: np.ndarray = None,
    qpoints_mesh: np.ndarray = None,
    max_atoms: int = None,
    enable_relax: bool = False,
    **kwargs,
) -> dict:
    """
    Predict phonon properties for a list of atoms.

    Args:
        atoms_list (List[Atoms]): List of ASE Atoms objects.
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
        enable_relax (bool, optional): Whether to relax the structure before
            predicting phonon properties. Defaults to False.
    """
    phonon_results = defaultdict(list)

    for atoms in tqdm(
        atoms_list, total=len(atoms_list), desc="Predicting phonon properties"
    ):
        if enable_relax:
            relaxed_results = relax(
                [atoms],
                constrain_symmetry=True,
                work_dir=work_dir,
                save_csv=save_csv.replace(".csv", "_relax.csv"),
            )
            structure = Structure.from_str(relaxed_results["structure"][0], fmt="json")
            _atoms = AseAtomsAdaptor.get_atoms(structure)
            _atoms.calc = atoms.calc
            atoms = _atoms
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
        phonon_results["has_imaginary"].append(has_imaginary)
        # phonon_results["phonon"].append(phonon)
        phonon_results["phonon_band_plot"].append(
            os.path.join(os.path.abspath(work_dir), f"{atoms.symbols}_phonon_band.png")
        )
        phonon_results["phonon_dos_plot"].append(
            os.path.join(os.path.abspath(work_dir), f"{atoms.symbols}_phonon_dos.png")
        )
        os.rename(
            os.path.join(os.path.abspath(work_dir), "band.yaml"),
            os.path.join(os.path.abspath(work_dir), f"{atoms.symbols}_band.yaml"),
        )
        os.rename(
            os.path.join(os.path.abspath(work_dir), "phonopy_params.yaml"),
            os.path.join(
                os.path.abspath(work_dir), f"{atoms.symbols}_phonopy_params.yaml"
            ),
        )
        os.rename(
            os.path.join(os.path.abspath(work_dir), "total_dos.dat"),
            os.path.join(os.path.abspath(work_dir), f"{atoms.symbols}_total_dos.dat"),
        )
        phonon_results["phonon_band"].append(
            yaml.safe_load(
                open(
                    os.path.join(
                        os.path.abspath(work_dir), f"{atoms.symbols}_band.yaml"
                    ),
                    "r",
                )
            )
        )
        phonon_results["phonopy_params"].append(
            yaml.safe_load(
                open(
                    os.path.join(
                        os.path.abspath(work_dir),
                        f"{atoms.symbols}_phonopy_params.yaml",
                    ),
                    "r",
                )
            )
        )
        phonon_results["total_dos"].append(
            np.loadtxt(
                os.path.join(
                    os.path.abspath(work_dir), f"{atoms.symbols}_total_dos.dat"
                ),
                comments="#",
            )
        )

    if not os.path.exists(work_dir):
        os.makedirs(work_dir)

    logger.info(f"Saving the results to {os.path.join(work_dir, save_csv)}")
    df = pd.DataFrame(phonon_results)
    df.to_csv(
        os.path.join(work_dir, save_csv.replace(".csv", "_phonon.csv")),
        index=False,
        mode="a",
    )
    return phonon_results
