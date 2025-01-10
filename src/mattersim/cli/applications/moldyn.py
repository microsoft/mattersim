import os
import re
import uuid
from collections import defaultdict
from typing import List

import pandas as pd
from ase import Atoms
from ase.io import read
from loguru import logger
from pymatgen.io.ase import AseAtomsAdaptor

from mattersim.applications.moldyn import MolecularDynamics


def moldyn(
    atoms_list: List[Atoms],
    *,
    temperature: float = 300,
    timestep: float = 1,
    steps: int = 1000,
    ensemble: str = "nvt_nose_hoover",
    logfile: str = "-",
    loginterval: int = 10,
    trajectory: str = None,
    taut: float = None,
    work_dir: str = str(uuid.uuid4()),
    save_csv: str = "results.csv.gz",
    **kwargs,
) -> dict:
    if len(atoms_list) != 1:
        raise ValueError("molecular dynamics workflow currently only supports one structure")

    moldyn_results = defaultdict(list)

    for atoms in atoms_list:
        # check if the atoms object has non-zero values in the lower triangle
        # of the cell. If so, the cell will be rotated and permuted to upper
        # triangular form. This is to avoid numerical issues in the MD simulation.
        print(atoms.cell.array)
        if any(atoms.cell.array[2, 0:2]) or atoms.cell.array[1, 0] != 0:
            logger.warning(
                "The lower triangle of the cell is not zero. "
                "The cell will be rotated and permuted to upper triangular form."
            )

            # The following code is from the PR
            # https://gitlab.com/ase/ase/-/merge_requests/3277.
            # It will be removed once the PR is merged.
            # This part of the codes rotates the cell and permutes the axes
            # such that the cell will be in upper triangular form.

            from ase.build import make_supercell

            _calc = atoms.calc
            logger.info(f"Initial cell: {atoms.cell.array}")

            atoms.set_cell(atoms.cell.standard_form()[0], scale_atoms=True)

            # Permute a and c axes
            atoms = make_supercell(atoms, [[0, 0, 1], [0, 1, 0], [1, 0, 0]])

            atoms.rotate(90, "y", rotate_cell=True)

            # set the lower triangle of the cell to be exactly zero
            # to avoid numerical issues
            atoms.cell.array[1, 0] = 0
            atoms.cell.array[2, 0] = 0
            atoms.cell.array[2, 1] = 0

            logger.info(f"Cell after rotation/permutation: {atoms.cell.array}")
            atoms.calc = _calc

        if not os.path.exists(work_dir):
            os.makedirs(work_dir)

        if os.path.exists(os.path.join(work_dir, logfile)):
            os.remove(os.path.join(work_dir, logfile))

        if os.path.exists(os.path.join(work_dir, trajectory)):
            os.remove(os.path.join(work_dir, trajectory))

        md = MolecularDynamics(
            atoms,
            ensemble=ensemble,
            temperature=temperature,
            timestep=timestep,
            loginterval=loginterval,
            logfile=os.path.join(work_dir, logfile),
            trajectory=os.path.join(work_dir, trajectory),
            taut=taut,
        )
        md.run(steps)

        # parse the logfile

        # Read the file into a pandas DataFrame
        df = pd.read_csv(
            os.path.join(work_dir, logfile),
            sep="\\s+",
        )
        df.columns = list(
            map(lambda x: re.sub(r"\[.*?\]", "", x).strip().lower(), df.columns)
        )
        traj = read(os.path.join(work_dir, trajectory), index=":")
        print(df.shape)
        print(len(traj))
        structure_list = [AseAtomsAdaptor.get_structure(atoms) for atoms in traj]

        # Add the structure list to the DataFrame
        df["structure"] = [structure.to_json() for structure in structure_list]

        # Print the DataFrame
        print(df)

        # Save the DataFrame to a CSV file
        df.to_csv(os.path.join(work_dir, save_csv))

        moldyn_results = df

    return moldyn_results
