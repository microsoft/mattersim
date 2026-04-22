"""Tests for the BTE (Boltzmann Transport Equation) thermal conductivity
workflow using phono3py.
"""

import numpy as np
import pytest

from mattersim.applications.bte import BTEWorkflow, BTEWorkflowError


class TestBTEWorkflowInit:
    """Tests for BTEWorkflow initialization and parameter validation."""

    def test_init_with_required_params(self, si_diamond, mattersim_calc):
        si_diamond.calc = mattersim_calc
        workflow = BTEWorkflow(
            atoms=si_diamond,
            supercell_matrix=np.array([2, 2, 2]),
            qpoints_mesh=np.array([2, 2, 2]),
        )
        assert workflow.method == "RTA"
        assert workflow.tmin == 50
        assert workflow.tmax == 500
        assert workflow.tstep == 50
        assert np.array_equal(
            workflow.supercell_matrix, np.diag([2, 2, 2])
        )

    def test_init_supercell_matrix_shape_3x3(self, si_diamond, mattersim_calc):
        si_diamond.calc = mattersim_calc
        sc = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 2]])
        workflow = BTEWorkflow(
            atoms=si_diamond,
            supercell_matrix=sc,
            qpoints_mesh=np.array([2, 2, 2]),
        )
        assert np.array_equal(workflow.supercell_matrix, sc)

    def test_init_lbte_method(self, si_diamond, mattersim_calc):
        si_diamond.calc = mattersim_calc
        workflow = BTEWorkflow(
            atoms=si_diamond,
            supercell_matrix=np.array([2, 2, 2]),
            qpoints_mesh=np.array([2, 2, 2]),
            method="LBTE",
        )
        assert workflow.method == "LBTE"

    def test_init_unsupported_method_raises(self, si_diamond, mattersim_calc):
        si_diamond.calc = mattersim_calc
        with pytest.raises(AssertionError):
            BTEWorkflow(
                atoms=si_diamond,
                supercell_matrix=np.array([2, 2, 2]),
                qpoints_mesh=np.array([2, 2, 2]),
                method="invalid",
            )

    def test_init_no_calculator_raises(self, si_diamond):
        with pytest.raises(AssertionError):
            BTEWorkflow(
                atoms=si_diamond,
                supercell_matrix=np.array([2, 2, 2]),
                qpoints_mesh=np.array([2, 2, 2]),
            )

    def test_init_custom_temperature_range(self, si_diamond, mattersim_calc):
        si_diamond.calc = mattersim_calc
        workflow = BTEWorkflow(
            atoms=si_diamond,
            supercell_matrix=np.array([2, 2, 2]),
            qpoints_mesh=np.array([2, 2, 2]),
            tmin=100,
            tmax=500,
            tstep=200,
        )
        assert workflow.tmin == 100
        assert workflow.tmax == 500
        assert workflow.tstep == 200


class TestBTEWorkflowRun:
    """Integration tests for the full BTE workflow using MatterSim.

    Reference values computed with MatterSim v1.0.0-1M on Si diamond primitive
    cell (a=5.43 Å), 2x2x2 supercell, 2x2x2 q-mesh, RTA method.
    """

    # Reference force constant norms — deterministic across devices/runs.
    REF_FC2_NORM = 109.146
    REF_FC3_NORM = 580.46

    # Reference thermal conductivity for Si (W/m·K), diagonal components.
    # The RTA solver in phono3py has inherent numerical variance (~5%)
    # across runs even on CPU, so we use a loose tolerance for kappa.
    REF_KAPPA_100K = 16.2   # xx = yy = zz at 100 K
    REF_KAPPA_200K = 43.1   # xx = yy = zz at 200 K
    REF_KAPPA_300K = 40.5   # xx = yy = zz at 300 K

    def test_rta_workflow_produces_valid_results(
        self, si_diamond, device, tmp_path
    ):
        """Full RTA workflow should produce FC2/FC3 and correct kappa for Si."""
        from mattersim.forcefield import MatterSimCalculator

        si_diamond.calc = MatterSimCalculator(device=device)
        workflow = BTEWorkflow(
            atoms=si_diamond,
            work_dir=str(tmp_path / f"bte_rta_{device}"),
            supercell_matrix=np.array([2, 2, 2]),
            qpoints_mesh=np.array([2, 2, 2]),
            tmin=100,
            tmax=400,
            tstep=100,
            method="RTA",
        )
        work_dir, ph3 = workflow.run()

        # Force constants shapes
        assert ph3.fc2.shape == (16, 16, 3, 3)
        assert ph3.fc3.shape == (16, 16, 16, 3, 3, 3)

        # Force constants norms — tight tolerance, these are deterministic
        np.testing.assert_allclose(
            np.linalg.norm(ph3.fc2), self.REF_FC2_NORM, rtol=1e-3
        )
        np.testing.assert_allclose(
            np.linalg.norm(ph3.fc3), self.REF_FC3_NORM, rtol=1e-3
        )

        # Thermal conductivity: positive and isotropic for cubic Si
        kappa = ph3.thermal_conductivity.kappa  # shape (1, n_temps, 6)
        assert kappa.shape == (1, 3, 6)

        # T=100K
        kxx_100, kyy_100, kzz_100 = kappa[0, 0, 0], kappa[0, 0, 1], kappa[0, 0, 2]
        np.testing.assert_allclose(kxx_100, self.REF_KAPPA_100K, rtol=0.15)
        np.testing.assert_allclose(kxx_100, kyy_100, rtol=1e-4)
        np.testing.assert_allclose(kxx_100, kzz_100, rtol=1e-4)

        # T=200K
        kxx_200, kyy_200, kzz_200 = kappa[0, 1, 0], kappa[0, 1, 1], kappa[0, 1, 2]
        np.testing.assert_allclose(kxx_200, self.REF_KAPPA_200K, rtol=0.15)
        np.testing.assert_allclose(kxx_200, kyy_200, rtol=1e-4)
        np.testing.assert_allclose(kxx_200, kzz_200, rtol=1e-4)

        # T=300K
        kxx_300, kyy_300, kzz_300 = kappa[0, 2, 0], kappa[0, 2, 1], kappa[0, 2, 2]
        np.testing.assert_allclose(kxx_300, self.REF_KAPPA_300K, rtol=0.15)
        np.testing.assert_allclose(kxx_300, kyy_300, rtol=1e-4)
        np.testing.assert_allclose(kxx_300, kzz_300, rtol=1e-4)

        # Off-diagonal components should be near zero
        assert np.all(np.abs(kappa[0, :, 3:]) < 0.1)

    def test_workflow_creates_output_dir(
        self, si_diamond, mattersim_calc, tmp_path
    ):
        """Workflow should create the work directory."""
        si_diamond.calc = mattersim_calc
        out_dir = str(tmp_path / "new_dir")
        workflow = BTEWorkflow(
            atoms=si_diamond,
            work_dir=out_dir,
            supercell_matrix=np.array([2, 2, 2]),
            qpoints_mesh=np.array([2, 2, 2]),
            tmin=100,
            tmax=200,
            tstep=100,
            method="RTA",
        )
        workflow.run()
        assert (tmp_path / "new_dir").exists()


@pytest.mark.slow
class TestBTEWorkflowRunStrict:
    """Strict integration tests with larger supercell and denser q-mesh.

    Reference values computed with MatterSim v1.0.0-1M on Si diamond primitive
    cell (a=5.43 Å), 4x4x4 supercell, 16x16x16 q-mesh, RTA method. These
    settings produce well-converged, stable results.

    Run with: pytest -m slow
    """

    REF_FC2_NORM = 308.570
    REF_FC3_NORM = 1647.52

    REF_KAPPA_100K = 919.6
    REF_KAPPA_200K = 228.4
    REF_KAPPA_300K = 130.7

    def test_rta_strict(self, si_diamond, mattersim_calc, tmp_path):
        """Strict BTE test with 4x4x4 supercell and 16x16x16 q-mesh.
        Requires CUDA or MPS — skipped on CPU-only machines."""
        if str(mattersim_calc.device) == "cpu":
            pytest.skip("No accelerator (CUDA/MPS) available")

        si_diamond.calc = mattersim_calc
        workflow = BTEWorkflow(
            atoms=si_diamond,
            work_dir=str(tmp_path / "bte_strict"),
            supercell_matrix=np.array([4, 4, 4]),
            qpoints_mesh=np.array([16, 16, 16]),
            tmin=100,
            tmax=400,
            tstep=100,
            method="RTA",
        )
        work_dir, ph3 = workflow.run()

        # Force constants shapes
        assert ph3.fc2.shape == (128, 128, 3, 3)
        assert ph3.fc3.shape == (128, 128, 128, 3, 3, 3)

        # Force constants norms — tight tolerance
        np.testing.assert_allclose(
            np.linalg.norm(ph3.fc2), self.REF_FC2_NORM, rtol=1e-3
        )
        np.testing.assert_allclose(
            np.linalg.norm(ph3.fc3), self.REF_FC3_NORM, rtol=1e-3
        )

        kappa = ph3.thermal_conductivity.kappa
        assert kappa.shape == (1, 3, 6)

        # T=100K
        kxx_100, kyy_100, kzz_100 = kappa[0, 0, 0], kappa[0, 0, 1], kappa[0, 0, 2]
        np.testing.assert_allclose(kxx_100, self.REF_KAPPA_100K, rtol=0.05)
        np.testing.assert_allclose(kxx_100, kyy_100, rtol=1e-4)
        np.testing.assert_allclose(kxx_100, kzz_100, rtol=1e-4)

        # T=200K
        kxx_200, kyy_200, kzz_200 = kappa[0, 1, 0], kappa[0, 1, 1], kappa[0, 1, 2]
        np.testing.assert_allclose(kxx_200, self.REF_KAPPA_200K, rtol=0.05)
        np.testing.assert_allclose(kxx_200, kyy_200, rtol=1e-4)
        np.testing.assert_allclose(kxx_200, kzz_200, rtol=1e-4)

        # T=300K
        kxx_300, kyy_300, kzz_300 = kappa[0, 2, 0], kappa[0, 2, 1], kappa[0, 2, 2]
        np.testing.assert_allclose(kxx_300, self.REF_KAPPA_300K, rtol=0.05)
        np.testing.assert_allclose(kxx_300, kyy_300, rtol=1e-4)
        np.testing.assert_allclose(kxx_300, kzz_300, rtol=1e-4)

        # Off-diagonal components should be near zero
        assert np.all(np.abs(kappa[0, :, 3:]) < 0.1)


class TestBTEStaticMethods:
    """Tests for static helper methods."""

    def test_compute_per_mode_kappa_isotropic(self):
        """Isotropic mode kappa should be the mean of xx, yy, zz."""
        n_bands, n_qpoints = 6, 4
        mode_kappa = np.random.default_rng(42).random((n_bands, n_qpoints, 6))
        weight = np.ones(n_qpoints)

        result = BTEWorkflow.compute_per_mode_kappa(
            mode_kappa, weight, is_isotropic=True
        )
        expected = (
            mode_kappa[:, :, 0] + mode_kappa[:, :, 1] + mode_kappa[:, :, 2]
        ) / (3.0 * weight.sum())
        np.testing.assert_allclose(result, expected.flatten(order="C"))

    def test_compute_per_mode_kappa_anisotropic(self):
        """Anisotropic mode kappa should return three separate components."""
        n_bands, n_qpoints = 6, 4
        mode_kappa = np.random.default_rng(42).random((n_bands, n_qpoints, 6))
        weight = np.ones(n_qpoints)

        kxx, kyy, kzz = BTEWorkflow.compute_per_mode_kappa(
            mode_kappa, weight, is_isotropic=False
        )
        np.testing.assert_allclose(kxx, mode_kappa[:, :, 0] / weight.sum())
        np.testing.assert_allclose(kyy, mode_kappa[:, :, 1] / weight.sum())
        np.testing.assert_allclose(kzz, mode_kappa[:, :, 2] / weight.sum())

    def test_compute_mean_free_path_norm(self):
        """MFP norm should be sqrt((vx*tau)^2 + (vy*tau)^2 + (vz*tau)^2)."""
        n_bands, n_qpoints = 3, 2
        gv = np.ones((n_bands, n_qpoints, 3)) * 10.0
        lifetime = np.ones(n_bands * n_qpoints) * 0.5

        mfp = BTEWorkflow.compute_mean_free_path(gv, lifetime, is_return_norm=True)
        expected_component = 10.0 * 0.5
        expected_norm = np.sqrt(3 * expected_component**2)
        np.testing.assert_allclose(mfp, expected_norm, atol=1e-10)

    def test_compute_mean_free_path_components(self):
        """MFP components should be v_i * tau for each direction."""
        n_bands, n_qpoints = 3, 2
        gv = np.zeros((n_bands, n_qpoints, 3))
        gv[:, :, 0] = 1.0
        gv[:, :, 1] = 2.0
        gv[:, :, 2] = 3.0
        lifetime = np.ones(n_bands * n_qpoints) * 0.5

        mfp_matrix = BTEWorkflow.compute_mean_free_path(
            gv, lifetime, is_return_norm=False
        )
        assert mfp_matrix.shape == (n_bands * n_qpoints, 3)
        np.testing.assert_allclose(mfp_matrix[:, 0], 0.5)
        np.testing.assert_allclose(mfp_matrix[:, 1], 1.0)
        np.testing.assert_allclose(mfp_matrix[:, 2], 1.5)
