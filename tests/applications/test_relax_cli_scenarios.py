"""Pressure-controlled relaxation through the CLI application and saved results."""

import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from ase import Atoms
from ase.calculators.lj import LennardJones
from ase.filters import ExpCellFilter, FrechetCellFilter
from ase.io import write
from ase.optimize import BFGS
from ase.units import GPa
from pymatgen.core import Structure

from mattersim.cli.applications.relax import relax
from mattersim.forcefield import MatterSimCalculator


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
            pytest.fail("Returned or persisted cell differs from the ASE reference")
        if not np.allclose(structure.cart_coords, reference.positions, atol=1e-12):
            pytest.fail("Returned or persisted positions differ from the ASE run")
    if not np.allclose(atoms.cell.array, reference.cell.array, atol=1e-12):
        pytest.fail("The input structure did not receive the relaxed cell")
    for energy in (results["energy"][0], persisted["energy"].iloc[0]):
        if not np.isclose(energy, reference.get_potential_energy(), atol=1e-12):
            pytest.fail("Returned or persisted energy differs from the ASE run")


@pytest.mark.parametrize(
    "axis_values,free_axes",
    [
        (["True"], [False] * 6),
        (["fAlSe"], [True] * 6),
        (
            ["False", "True", "True", "True", "True", "True"],
            [True, False, False, False, False, False],
        ),
    ],
)
def test_zero_pressure_axis_cli(axis_values, free_axes, si_diamond_cubic, tmp_path):
    atoms = si_diamond_cubic.copy()
    atoms.set_cell(atoms.cell * 1.05, scale_atoms=True)
    atoms.positions[0] += [0.02, -0.01, 0.03]
    structure_file = tmp_path / "silicon.xyz"
    write(structure_file, atoms)
    atoms.calc = MatterSimCalculator(device="cpu")
    BFGS(ExpCellFilter(atoms, mask=free_axes, scalar_pressure=0.0), logfile=None).run(
        fmax=0.0001, steps=1
    )

    output = tmp_path / "relaxed"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "mattersim.cli.mattersim_app",
            "relax",
            "--structure-file",
            str(structure_file),
            "--pressure-in-GPa",
            "0",
            "--fix-axis",
            *axis_values,
            "--optimizer",
            "BFGS",
            "--steps",
            "1",
            "--fmax",
            "0.0001",
            "--work-dir",
            str(output),
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode:
        pytest.fail(result.stdout + result.stderr)
    saved = pd.read_csv(output / "results.csv.gz")
    actual = Structure.from_str(saved["structure"].iloc[0], fmt="json")
    if not np.allclose(actual.lattice.matrix, atoms.cell.array, atol=1e-7, rtol=0):
        pytest.fail("CLI axis selection changed the expected relaxed cell")
    if not np.allclose(actual.cart_coords, atoms.positions, atol=1e-7, rtol=0):
        pytest.fail("CLI axis selection changed the expected relaxed positions")
    if not np.isclose(
        saved["energy"].iloc[0], atoms.get_potential_energy(), atol=1e-5, rtol=0
    ):
        pytest.fail("CLI axis selection changed the expected saved energy")


@pytest.mark.parametrize("command", ["relax", "phonon"])
@pytest.mark.parametrize("axis_values", [["invalid"], ["True", "False"]])
def test_invalid_axis_cli_leaves_output_untouched(command, axis_values, tmp_path):
    output = tmp_path / "output"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "mattersim.cli.mattersim_app",
            command,
            "--structure-file",
            str(tmp_path / "missing.xyz"),
            "--fix-axis",
            *axis_values,
            "--work-dir",
            str(output),
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 2 or "--fix-axis" not in result.stderr:
        pytest.fail("Invalid axis selection was not rejected at the CLI boundary")
    if "Expected one or six True/False values." not in result.stderr:
        pytest.fail("Invalid axis selection did not explain its accepted values")
    if output.exists():
        pytest.fail("Invalid axis selection created output")
