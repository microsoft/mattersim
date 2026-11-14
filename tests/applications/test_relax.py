"""Tests for the Relaxer class."""

import csv
import gzip

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.lj import LennardJones
from ase.optimize import BFGS, FIRE
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from mattersim.applications.relax import Relaxer
from mattersim.cli.applications.relax import relax as run_relaxation


@pytest.mark.parametrize("optimizer", [FIRE, BFGS])
@pytest.mark.parametrize(
    "distance,steps,fmax,expected",
    [
        (2 ** (1 / 6), 0, 0.01, True),
        (1.0, 1, 5.0, True),
        (1.0, 1, 0.01, False),
        (1.0, 0, 0.01, False),
        (2 ** (1 / 6), 5, 0.01, True),
    ],
    ids=[
        "initial-convergence", "final-step", "step-limit", "no-steps", "early-convergence"
    ],
)
def test_relaxation_termination(optimizer, distance, steps, fmax, expected):
    atoms = Atoms("Ar2", positions=[[0, 0, 0], [distance, 0, 0]])
    atoms.calc = LennardJones()
    control = atoms.copy()
    control.calc = LennardJones()
    optimizer(control, logfile=None).run(fmax=fmax, steps=steps)

    converged, relaxed = Relaxer(
        optimizer=optimizer.__name__, constrain_symmetry=False
    ).relax(atoms, fmax=fmax, steps=steps, verbose=False)
    if converged is not expected:
        pytest.fail(f"Expected convergence={expected}, received {converged!r}")
    if not np.array_equal(relaxed.positions, control.positions):
        pytest.fail("Relaxation changed the optimizer's final positions")


def test_final_step_convergence_reaches_csv(tmp_path):
    atoms = Atoms(
        "Ar2", positions=[[0, 0, 0], [1.0, 0, 0]], cell=[10, 10, 10], pbc=True
    )
    atoms.calc = LennardJones()
    results = run_relaxation(
        [atoms],
        optimizer="BFGS",
        constrain_symmetry=False,
        steps=1,
        fmax=5.0,
        work_dir=str(tmp_path),
    )
    with gzip.open(tmp_path / "results.csv.gz", "rt") as stream:
        rows = list(csv.DictReader(stream))
    if results["converged"] != [True] or [row["converged"] for row in rows] != ["True"]:
        pytest.fail("Final-step convergence did not reach the returned results and CSV")


class TestRelaxer:
    """Fast tests for Relaxer using LennardJones on all available devices."""

    def test_default_relaxer(self, si_diamond_cubic, perturb, available_device):
        atoms = perturb(si_diamond_cubic, displacement=0.01)
        atoms.calc = LennardJones()
        relaxer = Relaxer()
        converged, relaxed_atoms = relaxer.relax(
            atoms, fmax=0.1, steps=10, verbose=False
        )
        assert isinstance(relaxed_atoms, Atoms)

    def test_relax_structures(self, si_diamond_cubic, perturb, available_device):
        atoms_list = []
        for _ in range(3):
            a = perturb(si_diamond_cubic, displacement=0.01)
            a.calc = LennardJones()
            atoms_list.append(a)

        converged_list, relaxed_atoms_list = Relaxer.relax_structures(
            atoms_list, fmax=0.1, steps=10
        )
        assert isinstance(converged_list, list)

    def test_relax_under_pressure(self, si_diamond_cubic, perturb, available_device):
        atoms = perturb(si_diamond_cubic, displacement=0.01)
        atoms.calc = LennardJones()

        converged, relaxed_atoms = Relaxer.relax_structures(
            atoms,
            steps=10,
            fmax=0.1,
            filter="FrechetCellFilter",
            pressure_in_GPa=0.0,
        )
        assert isinstance(relaxed_atoms, Atoms)

    def test_relax_with_constrained_symmetry(
        self, si_diamond_cubic, perturb, available_device
    ):
        atoms = perturb(si_diamond_cubic, strain=0.2)
        atoms.calc = LennardJones()

        init_analyzer = SpacegroupAnalyzer(
            AseAtomsAdaptor.get_structure(atoms)
        )
        init_spacegroup = init_analyzer.get_space_group_number()

        converged, relaxed_atoms = Relaxer.relax_structures(
            atoms,
            steps=50,
            fmax=0.1,
            filter="FrechetCellFilter",
            pressure_in_GPa=0.0,
            constrain_symmetry=True,
        )
        assert isinstance(relaxed_atoms, Atoms)

        final_analyzer = SpacegroupAnalyzer(
            AseAtomsAdaptor.get_structure(relaxed_atoms)
        )
        assert final_analyzer.get_space_group_number() == init_spacegroup


class TestRelaxerWithMatterSim:
    """Strict relaxation test with MatterSim on a perturbed Si structure.

    Reference values computed with MatterSim v1.0.0-1M on Si diamond primitive
    cell (a=5.43 Å) perturbed with displacement=0.05 (seed=42), relaxed to
    fmax=0.01.

    Run with: pytest -m slow
    """

    REF_ENERGY_PER_ATOM = -5.4125  # eV/atom
    REF_FMAX = 0.01
    REF_STRESS_DIAG = -0.01089  # xx ≈ yy ≈ zz for cubic Si

    def test_relax_perturbed_si(self, si_diamond, perturb, available_device):
        """Relax a perturbed Si primitive cell and check energy/forces/stress."""
        from mattersim.forcefield import MatterSimCalculator

        atoms = perturb(si_diamond, displacement=0.05)
        atoms.calc = MatterSimCalculator(device=available_device)

        relaxer = Relaxer()
        converged, relaxed = relaxer.relax(
            atoms, fmax=0.01, steps=200, verbose=False
        )

        assert converged

        # Energy per atom
        energy_per_atom = relaxed.get_potential_energy() / len(relaxed)
        np.testing.assert_allclose(
            energy_per_atom, self.REF_ENERGY_PER_ATOM, rtol=1e-3
        )

        # Forces should be below fmax
        forces = relaxed.get_forces()
        assert np.max(np.linalg.norm(forces, axis=1)) < self.REF_FMAX

        # Stress: diagonal components should match reference
        stress = relaxed.get_stress()
        np.testing.assert_allclose(stress[0], self.REF_STRESS_DIAG, rtol=0.01)
        np.testing.assert_allclose(stress[1], self.REF_STRESS_DIAG, rtol=0.01)
        np.testing.assert_allclose(stress[2], self.REF_STRESS_DIAG, rtol=0.01)

        # Off-diagonal stress should be near zero
        assert np.all(np.abs(stress[3:]) < 1e-3)


class TestRelaxerVerbose:
    """Tests for the verbose parameter on Relaxer.relax() (issue #59)."""

    def test_verbose_true_prints_output(self, si_diamond, capsys):
        si_diamond.calc = LennardJones()
        relaxer = Relaxer()
        relaxer.relax(si_diamond, fmax=0.1, steps=5, verbose=True)

        captured = capsys.readouterr()
        assert len(captured.out) > 0

    def test_verbose_false_suppresses_output(self, si_diamond, capsys):
        si_diamond.calc = LennardJones()
        relaxer = Relaxer()
        relaxer.relax(si_diamond, fmax=0.1, steps=5, verbose=False)

        captured = capsys.readouterr()
        assert captured.out == ""

    def test_verbose_default_is_true(self, si_diamond, capsys):
        """Default behavior should print output (backward compatible)."""
        si_diamond.calc = LennardJones()
        relaxer = Relaxer()
        relaxer.relax(si_diamond, fmax=0.1, steps=5)

        captured = capsys.readouterr()
        assert len(captured.out) > 0

    def test_verbose_false_still_relaxes(self, si_diamond):
        """Suppressing output must not affect the relaxation result."""
        si_diamond.calc = LennardJones()
        relaxer = Relaxer()
        converged, relaxed = relaxer.relax(
            si_diamond, fmax=0.1, steps=50, verbose=False
        )
        assert isinstance(converged, bool)
        assert relaxed is not None
