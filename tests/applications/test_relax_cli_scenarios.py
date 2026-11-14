"""Pressure-controlled relaxation through the CLI application and saved results."""

import numpy as np
import pandas as pd
import pytest
from ase import Atoms
from ase.calculators.lj import LennardJones
from ase.filters import ExpCellFilter, FrechetCellFilter
from ase.optimize import BFGS
from ase.units import GPa
from pymatgen.core import Structure

from mattersim.cli.applications.relax import relax


@pytest.mark.parametrize(
    "pressure,filter_name,fix_axis,expected_filter",
    [
        (None, None, False, None),
        (0.0, None, False, ExpCellFilter),
        (1.0, None, False, ExpCellFilter),
        (-1.0, None, False, ExpCellFilter),
        (None, "ExpCellFilter", False, ExpCellFilter),
        (0.0, "ExpCellFilter", False, ExpCellFilter),
        (0.0, "FrechetCellFilter", False, FrechetCellFilter),
        (1.0, "FrechetCellFilter", False, FrechetCellFilter),
        (0.0, None, True, ExpCellFilter),
        (0.0, None, [True, False, True, True, True, True], ExpCellFilter),
    ],
)
def test_pressure_relaxation_scenarios(
    pressure, filter_name, fix_axis, expected_filter, tmp_path
):
    atoms = Atoms(
        "Ar2", positions=[[0, 0, 0], [1, 0.2, 0]], cell=[10, 10, 10], pbc=True
    )
    atoms.calc = LennardJones()
    reference = atoms.copy()
    reference.calc = LennardJones()
    if expected_filter is None:
        optimizable = reference
    else:
        mask = (
            [not fix_axis] * 6
            if isinstance(fix_axis, bool)
            else [not x for x in fix_axis]
        )
        optimizable = expected_filter(
            reference,
            mask=mask,
            scalar_pressure=0.0 if pressure is None else pressure * GPa,
        )
    BFGS(optimizable, logfile=None).run(fmax=0.01, steps=1)

    results = relax(
        [atoms],
        optimizer="BFGS",
        filter=filter_name,
        fix_axis=fix_axis,
        pressure_in_GPa=pressure,
        steps=1,
        fmax=0.01,
        work_dir=str(tmp_path),
    )
    persisted = pd.read_csv(tmp_path / "results.csv.gz")
    structures = [
        Structure.from_str(results["structure"][0], fmt="json"),
        Structure.from_str(persisted["structure"].iloc[0], fmt="json"),
    ]
    for structure in structures:
        if not np.allclose(structure.lattice.matrix, reference.cell.array, atol=1e-12):
            pytest.fail(
                "Returned or persisted cell differs from the pressure-controlled ASE run"
            )
        if not np.allclose(structure.cart_coords, reference.positions, atol=1e-12):
            pytest.fail("Returned or persisted positions differ from the ASE run")
    if not np.allclose(atoms.cell.array, reference.cell.array, atol=1e-12):
        pytest.fail("The input structure did not receive the relaxed cell")
    for energy in (results["energy"][0], persisted["energy"].iloc[0]):
        if not np.isclose(energy, reference.get_potential_energy(), atol=1e-12):
            pytest.fail("Returned or persisted energy differs from the ASE run")
