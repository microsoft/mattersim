"""Tests for the mattersim.torchsim integration."""

import os
from tempfile import TemporaryDirectory
from typing import Any, Literal

import pytest
import torch
import torch_sim as ts

from mattersim.forcefield.potential import Potential
from mattersim.torchsim.batch_relax import TorchSimBatchRelaxer
from mattersim.torchsim.md import TorchSimBatchMD
from mattersim.torchsim.settings import IntegratorSettings, OptimizerSettings
from mattersim.torchsim.settings_base import INTEGRATOR_PARAMS
from mattersim.torchsim.torchsim_wrapper import TorchSimWrapper
from mattersim.torchsim.trajectory_loader import MDTrajectoryLoader

requires_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)

DEVICE: Literal["cpu", "cuda"] = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32

RELAXATION_CHECK_KEYS = [
    "positions",
    "cell",
    "energy",
    "forces",
    "stress",
    "atomic_numbers",
]
MD_CHECK_KEYS = [
    "positions",
    "cell",
    "energy",
    "forces",
    "momenta",
    "atomic_numbers",
]
STRUCTURE_CHECK_KEYS = ["positions", "cell", "atomic_numbers"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def create_relaxer_settings(max_steps: int = 1, **kwargs) -> OptimizerSettings:
    """Create standard relaxer settings with overridable defaults."""
    defaults: dict[str, Any] = {
        "name": "fire",
        "max_steps": max_steps,
        "device": DEVICE,
        "autobatcher": False,
        "save_checkpoint_every": 1,
    }
    defaults.update(kwargs)
    return OptimizerSettings(**defaults)


def create_md_settings(
    num_steps: int = 1, integrator_name: str = "nve", **kwargs
) -> IntegratorSettings:
    """Create standard MD settings with overridable defaults."""
    defaults: dict[str, Any] = {
        "name": integrator_name,
        "temperature_K": 10.0,
        "timestep_ps": 1e-3,
        "num_steps": num_steps,
        "device": DEVICE,
        "save_checkpoint_every": 1,
    }
    defaults.update(kwargs)
    return IntegratorSettings(**defaults)


def assert_states_equal(state1, state2, check_keys: list[str]) -> None:
    """Assert that two states have equal attributes for specified keys."""
    for key in check_keys:
        arr1 = getattr(state1, key)
        arr2 = getattr(state2, key)
        torch.testing.assert_close(arr1, arr2)


def assert_trajectories_equal(traj1, traj2, check_keys: list[str]) -> None:
    """Assert that two trajectories are equal."""
    assert traj1.n_systems == traj2.n_systems
    assert_states_equal(traj1, traj2, check_keys)


def verify_trajectory_frames(
    filenames: list[str] | None,
    expected_frames: int,
    device=DEVICE,
    dtype=DTYPE,
) -> None:
    """Verify that trajectory files have the expected number of frames."""
    assert filenames is not None
    for filename in filenames:
        loader = MDTrajectoryLoader(filename, device=device, dtype=dtype)
        assert loader.num_frames == expected_frames


# ---------------------------------------------------------------------------
# Fixtures (torchsim-specific only; si_diamond_cubic and
# mattersim_potential_best_device come from conftest.py)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def torchsim_wrapper(mattersim_potential_best_device: Potential) -> TorchSimWrapper:
    """TorchSimWrapper around the shared potential."""
    return TorchSimWrapper(
        model=mattersim_potential_best_device,
        device=DEVICE,
        dtype=torch.float64,
    )


# ---------------------------------------------------------------------------
# Settings validation tests (no model needed)
# ---------------------------------------------------------------------------


class TestSettingsValidation:
    """Tests for settings dataclass validation logic."""

    def test_optimizer_settings_creation(self):
        s = OptimizerSettings(name="fire", max_steps=100, device=DEVICE)
        assert s.name == "fire"
        assert s.max_steps == 100
        assert s.cell_filter == ts.CellFilter.frechet
        assert s.optimizer == ts.Optimizer.fire

    def test_integrator_settings_creation(self):
        s = IntegratorSettings(
            name="nvt_langevin",
            temperature_K=300.0,
            num_steps=100,
            gamma=1.0,
            device=DEVICE,
        )
        assert s.name == "nvt_langevin"
        assert s.temperature_K == 300.0
        assert s.integrator == ts.Integrator.nvt_langevin
        assert s.simulation_time_ps == pytest.approx(0.1)

    def test_npt_requires_pressure(self):
        with pytest.raises(ValueError, match="Pressure must be specified"):
            IntegratorSettings(
                name="npt_langevin",
                temperature_K=300.0,
                num_steps=100,
                device=DEVICE,
            )

    def test_invalid_parameter_rejected(self):
        with pytest.raises(ValueError, match="not valid for"):
            IntegratorSettings(
                name="nve",
                temperature_K=300.0,
                num_steps=100,
                gamma=1.0,
                device=DEVICE,
            )

    def test_checkpoint_every_must_divide_steps(self):
        with pytest.raises(ValueError, match="must be a multiple"):
            OptimizerSettings(
                name="fire", max_steps=7, save_checkpoint_every=3
            )

    def test_with_per_system_temperatures_scalar(self):
        settings = IntegratorSettings.with_per_system_temperatures(
            temperatures_K=300.0,
            name="nvt_langevin",
            num_steps=100,
            save_checkpoint_every=100,
            device=DEVICE,
        )
        assert isinstance(settings, IntegratorSettings)
        assert settings.temperature_K == 300.0

    def test_with_per_system_temperatures_1d_schedule(self):
        schedule = torch.tensor([300.0, 350.0, 400.0])
        settings = IntegratorSettings.with_per_system_temperatures(
            temperatures_K=schedule,
            name="nvt_langevin",
            num_steps=3,
            save_checkpoint_every=1,
            device=DEVICE,
        )
        assert isinstance(settings, IntegratorSettings)
        assert settings.temperature_K == schedule.tolist()

    def test_with_per_system_temperatures_2d_schedules(self):
        schedules = torch.tensor(
            [
                [300.0, 350.0, 400.0],
                [400.0, 450.0, 500.0],
                [500.0, 550.0, 600.0],
            ]
        )
        settings_list = IntegratorSettings.with_per_system_temperatures(
            temperatures_K=schedules,
            name="nvt_langevin",
            num_steps=3,
            save_checkpoint_every=1,
            device=DEVICE,
        )
        assert isinstance(settings_list, list)
        assert len(settings_list) == 3
        for i, settings in enumerate(settings_list):
            assert isinstance(settings, IntegratorSettings)
            assert settings.temperature_K == schedules[i].tolist()

    def test_with_per_system_temperatures_validates_schedule_length(self):
        with pytest.raises(
            ValueError, match="Temperature schedule length.*must match"
        ):
            IntegratorSettings.with_per_system_temperatures(
                temperatures_K=torch.tensor([300.0, 350.0, 400.0]),
                name="nvt_langevin",
                num_steps=5,
                save_checkpoint_every=1,
                device=DEVICE,
            )

    def test_from_thermostat_barostat_settings(self):
        schedules = torch.tensor(
            [
                [300.0, 350.0],
                [400.0, 450.0],
            ]
        )
        settings_list = IntegratorSettings.from_thermostat_barostat_settings(
            name="nvt_langevin",
            temperature_K=schedules,
            num_steps=2,
            save_checkpoint_every=1,
            thermostat_setting_name="default",
            device=DEVICE,
        )
        assert isinstance(settings_list, list)
        assert len(settings_list) == 2
        for i, settings in enumerate(settings_list):
            assert settings.temperature_K == schedules[i].tolist()
            assert settings.gamma is not None


# ---------------------------------------------------------------------------
# TorchSimWrapper tests
# ---------------------------------------------------------------------------


@pytest.mark.slow
@requires_gpu
class TestTorchSimWrapper:
    """Tests for the TorchSimWrapper model interface."""

    def test_wrapper_creation(self, torchsim_wrapper: TorchSimWrapper):
        assert torchsim_wrapper.two_body_cutoff > 0
        assert torchsim_wrapper.three_body_cutoff > 0
        assert "energy" in torchsim_wrapper.implemented_properties
        assert "forces" in torchsim_wrapper.implemented_properties
        assert "stress" in torchsim_wrapper.implemented_properties

    def test_wrapper_forward(
        self, torchsim_wrapper: TorchSimWrapper, si_diamond_cubic
    ):
        state = ts.initialize_state(
            [si_diamond_cubic], device=DEVICE, dtype=torch.float64
        )
        result = torchsim_wrapper(state)

        assert "energy" in result
        assert "forces" in result
        assert "stress" in result
        assert result["energy"].shape == (1,)
        assert result["forces"].shape == (len(si_diamond_cubic), 3)
        assert result["stress"].shape == (1, 3, 3)


# ---------------------------------------------------------------------------
# Batch relaxation tests
# ---------------------------------------------------------------------------


@pytest.mark.slow
@requires_gpu
class TestBatchRelaxation:
    """Tests for TorchSimBatchRelaxer."""

    def test_batch_relaxation(
        self, si_diamond_cubic, mattersim_potential_best_device: Potential
    ):
        settings = create_relaxer_settings(max_steps=1)

        with TemporaryDirectory() as tmpdir:
            relaxer = TorchSimBatchRelaxer.from_structures(
                structures=[si_diamond_cubic, si_diamond_cubic],
                settings=settings,
                save_folder=tmpdir,
                potential=mattersim_potential_best_device,
            )
            assert not relaxer.has_trajectories
            with pytest.raises(
                ValueError, match="No trajectory files found"
            ):
                _ = relaxer.trajectories

            relaxations, converged = relaxer.run()
            assert relaxations.n_systems == 2
            assert relaxer.filenames is not None
            assert all(
                os.path.exists(f) for f in relaxer.filenames
            )
            assert relaxer.has_trajectories

            for traj in relaxer.trajectories:
                assert traj.n_systems == settings.max_steps + 1

            relaxer2 = TorchSimBatchRelaxer.from_trajectory_folder(
                trajectory_folder=tmpdir,
                potential=mattersim_potential_best_device,
            )
            assert relaxer.metadata == relaxer2.metadata
            for t1, t2 in zip(
                relaxer.trajectories, relaxer2.trajectories
            ):
                assert_trajectories_equal(t1, t2, RELAXATION_CHECK_KEYS)

    def test_batch_relaxation_in_memory(
        self, si_diamond_cubic, mattersim_potential_best_device: Potential
    ):
        """Test relaxation without saving trajectories (in-memory mode)."""
        settings = create_relaxer_settings(max_steps=1)

        relaxer = TorchSimBatchRelaxer.from_structures(
            structures=[si_diamond_cubic, si_diamond_cubic],
            settings=settings,
            save_folder=None,
            potential=mattersim_potential_best_device,
        )
        result = relaxer.relax()
        assert len(result) == 2
        for ix, traj in result.items():
            assert len(traj) == 2
            assert traj[-1].info["converged"] is not None

    def test_batch_relaxation_from_trajectory(
        self, si_diamond_cubic, mattersim_potential_best_device: Potential
    ):
        """Test creating relaxer from existing trajectories."""
        settings = create_relaxer_settings(max_steps=1)

        with TemporaryDirectory() as tmpdir:
            relaxer = TorchSimBatchRelaxer.from_structures(
                structures=[si_diamond_cubic, si_diamond_cubic],
                settings=settings,
                save_folder=tmpdir,
                potential=mattersim_potential_best_device,
            )
            _ = relaxer.run()

            relaxer2 = TorchSimBatchRelaxer.from_trajectory_folder(
                trajectory_folder=tmpdir,
                potential=mattersim_potential_best_device,
            )
            assert relaxer2.input_structures.n_systems == 2
            assert_states_equal(
                relaxer.original_structures,
                relaxer2.original_structures,
                STRUCTURE_CHECK_KEYS,
            )

    def test_empty_trajectory_folder_raises(
        self, mattersim_potential_best_device: Potential
    ):
        with TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="No trajectory files found"):
                TorchSimBatchRelaxer.from_trajectory_folder(
                    trajectory_folder=tmpdir,
                    potential=mattersim_potential_best_device,
                )

    def test_from_structures_validates_settings_list_length(
        self, si_diamond_cubic, mattersim_potential_best_device: Potential
    ):
        """Settings list length must match number of structures."""
        structures = [si_diamond_cubic] * 3
        settings_list = [
            create_relaxer_settings(max_steps=1),
            create_relaxer_settings(max_steps=1),
        ]

        with TemporaryDirectory() as tmpdir:
            with pytest.raises(
                ValueError, match="Number of settings.*must match"
            ):
                TorchSimBatchRelaxer.from_structures(
                    structures=structures,
                    settings=settings_list,
                    save_folder=tmpdir,
                    potential=mattersim_potential_best_device,
                )


# ---------------------------------------------------------------------------
# Batch MD tests
# ---------------------------------------------------------------------------


@pytest.mark.slow
@requires_gpu
class TestBatchMD:
    """Tests for TorchSimBatchMD."""

    @pytest.mark.parametrize(
        "integrator_name",
        [
            name
            for name in [
                "nve",
                "nvt_nose_hoover",
                "npt_nose_hoover",
                "npt_langevin",
                "nvt_langevin",
            ]
            if name in ts.Integrator.__members__
        ],
    )
    def test_batch_md(
        self,
        si_diamond_cubic,
        integrator_name: str,
        mattersim_potential_best_device: Potential,
    ):
        integrator = ts.Integrator[integrator_name]
        all_params = set(
            v for params in INTEGRATOR_PARAMS.values() for v in params
        )
        allowed_params = INTEGRATOR_PARAMS[integrator.name]
        invalid_params = all_params - allowed_params
        maybe_pressure_arg = {}
        if "pressure_bar" in allowed_params:
            maybe_pressure_arg["pressure_bar"] = 1.0

        # Invalid parameters raise errors
        for invalid_param in invalid_params:
            with pytest.raises(ValueError, match="not valid for"):
                create_md_settings(
                    integrator_name=integrator.name,
                    **{invalid_param: 1.0},
                    **maybe_pressure_arg,
                )

        # All allowed parameters work
        all_allowed = {param: 1.0 for param in allowed_params}
        _ = create_md_settings(
            integrator_name=integrator.name, **all_allowed
        )

        # MD simulation runs and produces trajectories
        settings = create_md_settings(
            integrator_name=integrator.name, **maybe_pressure_arg
        )
        with TemporaryDirectory() as tmpdir:
            md = TorchSimBatchMD.from_structures(
                structures=[si_diamond_cubic, si_diamond_cubic],
                settings=settings,
                save_folder=tmpdir,
                potential=mattersim_potential_best_device,
            )
            assert not md.has_trajectories

            final_states = md.run()
            assert final_states.n_systems == 2
            assert md.filenames is not None
            assert all(os.path.exists(f) for f in md.filenames)
            assert md.has_trajectories

            for traj in md.trajectories:
                assert traj.n_systems == settings.num_steps + 1

            md2 = TorchSimBatchMD.from_trajectory_folder(
                trajectory_folder=tmpdir,
                potential=mattersim_potential_best_device,
            )
            assert md.metadata == md2.metadata
            for t1, t2 in zip(md.trajectories, md2.trajectories):
                assert_trajectories_equal(t1, t2, MD_CHECK_KEYS)

    def test_batch_md_from_trajectory(
        self, si_diamond_cubic, mattersim_potential_best_device: Potential
    ):
        """Test creating MD runner from existing trajectories."""
        settings = create_md_settings(num_steps=2)

        with TemporaryDirectory() as tmpdir:
            md = TorchSimBatchMD.from_structures(
                structures=[si_diamond_cubic, si_diamond_cubic],
                settings=settings,
                save_folder=tmpdir,
                potential=mattersim_potential_best_device,
            )
            _ = md.run()

            md2 = TorchSimBatchMD.from_trajectory_folder(
                trajectory_folder=tmpdir,
                potential=mattersim_potential_best_device,
            )
            assert md2.input_structures.n_systems == 2
            assert_states_equal(
                md.original_structures,
                md2.original_structures,
                STRUCTURE_CHECK_KEYS,
            )

    def test_md_continuation_adjusts_steps(
        self, si_diamond_cubic, mattersim_potential_best_device: Potential
    ):
        """Continuing MD from trajectories adjusts remaining steps."""
        max_steps = 2
        settings = create_md_settings(num_steps=max_steps)

        with TemporaryDirectory() as tmpdir:
            md = TorchSimBatchMD.from_structures(
                structures=[si_diamond_cubic, si_diamond_cubic],
                settings=settings,
                save_folder=tmpdir,
                potential=mattersim_potential_best_device,
            )
            assert not md.is_continuing_from_trajectories
            assert md.get_steps_already_completed() == [0, 0]
            _ = md.run()

            verify_trajectory_frames(md.filenames, max_steps + 1)

            md2 = TorchSimBatchMD.from_trajectory_filenames(
                filenames=md.filenames,
                potential=mattersim_potential_best_device,
            )
            assert md2.is_continuing_from_trajectories
            assert md2.get_steps_already_completed() == [
                max_steps,
                max_steps,
            ]
            remaining = md2._prepare_continuation()
            assert remaining == 0
            verify_trajectory_frames(md2.filenames, max_steps + 1)

    def test_md_continuation_with_partial_steps(
        self, si_diamond_cubic, mattersim_potential_best_device: Potential
    ):
        """MD continuation when fewer steps completed than max."""
        max_steps = 2
        initial_steps = 1

        with TemporaryDirectory() as tmpdir:
            md = TorchSimBatchMD.from_structures(
                structures=[si_diamond_cubic, si_diamond_cubic],
                settings=create_md_settings(num_steps=initial_steps),
                save_folder=tmpdir,
                potential=mattersim_potential_best_device,
            )
            _ = md.run()
            verify_trajectory_frames(md.filenames, initial_steps + 1)

            md2 = TorchSimBatchMD.from_trajectory_filenames(
                filenames=md.filenames,
                potential=mattersim_potential_best_device,
            )
            new_settings = create_md_settings(num_steps=max_steps)
            md2._per_system_settings = [new_settings] * 2

            assert md2.is_continuing_from_trajectories
            assert md2.get_steps_already_completed() == [
                initial_steps,
                initial_steps,
            ]
            _ = md2.run()
            verify_trajectory_frames(md2.filenames, max_steps + 1)

    def test_per_system_temperatures_integration(
        self, si_diamond_cubic, mattersim_potential_best_device: Potential
    ):
        """Per-system temperature schedules are stored and recoverable."""
        num_steps = 3
        schedules = torch.tensor(
            [
                [300.0, 310.0, 320.0],
                [400.0, 410.0, 420.0],
            ]
        )

        settings_list = IntegratorSettings.with_per_system_temperatures(
            temperatures_K=schedules,
            name="nvt_langevin",
            num_steps=num_steps,
            timestep_ps=1e-3,
            device=DEVICE,
            save_checkpoint_every=1,
            autobatcher=False,
        )
        assert isinstance(settings_list, list)

        with TemporaryDirectory() as tmpdir:
            md = TorchSimBatchMD.from_structures(
                structures=[si_diamond_cubic, si_diamond_cubic],
                settings=settings_list,
                save_folder=tmpdir,
                potential=mattersim_potential_best_device,
            )

            assert (
                md.get_settings_for_system(0).temperature_K
                == schedules[0].tolist()
            )
            assert (
                md.get_settings_for_system(1).temperature_K
                == schedules[1].tolist()
            )
            _ = md.run()

            assert md.filenames is not None
            loader0 = MDTrajectoryLoader(
                md.filenames[0], device=DEVICE, dtype=DTYPE
            )
            loader1 = MDTrajectoryLoader(
                md.filenames[1], device=DEVICE, dtype=DTYPE
            )
            assert loader0.temperature_schedule == schedules[0].tolist()
            assert loader1.temperature_schedule == schedules[1].tolist()

    def test_from_structures_validates_settings_list_length(
        self, si_diamond_cubic, mattersim_potential_best_device: Potential
    ):
        """Settings list length must match number of structures."""
        structures = [si_diamond_cubic] * 3
        settings_list = [
            create_md_settings(num_steps=1),
            create_md_settings(num_steps=1),
        ]

        with TemporaryDirectory() as tmpdir:
            with pytest.raises(
                ValueError, match="Number of settings.*must match"
            ):
                TorchSimBatchMD.from_structures(
                    structures=structures,
                    settings=settings_list,
                    save_folder=tmpdir,
                    potential=mattersim_potential_best_device,
                )


# ---------------------------------------------------------------------------
# Trajectory loader tests
# ---------------------------------------------------------------------------


class TestTrajectoryLoader:
    """Tests for trajectory loader functionality."""

    def test_remote_path_rejected(self):
        """Non-local paths are rejected with a clear error."""
        with pytest.raises(ValueError, match="Remote storage paths"):
            MDTrajectoryLoader("blob-storage://account/container/file.h5md")

    @pytest.mark.slow
    @requires_gpu
    def test_convenience_constructors(
        self, si_diamond_cubic, mattersim_potential_best_device: Potential
    ):
        """from_structures, from_trajectory_filenames, from_trajectory_folder
        all produce consistent results."""
        settings = create_relaxer_settings(max_steps=1)

        with TemporaryDirectory() as tmpdir:
            relaxer1 = TorchSimBatchRelaxer.from_structures(
                structures=[si_diamond_cubic, si_diamond_cubic],
                settings=settings,
                save_folder=tmpdir,
                potential=mattersim_potential_best_device,
            )
            assert relaxer1.filenames is not None
            assert len(relaxer1.filenames) == 2
            _ = relaxer1.run()

            relaxer2 = TorchSimBatchRelaxer.from_trajectory_filenames(
                filenames=relaxer1.filenames,
                potential=mattersim_potential_best_device,
            )
            assert relaxer2.input_structures.n_systems == 2

            relaxer3 = TorchSimBatchRelaxer.from_trajectory_folder(
                trajectory_folder=tmpdir,
                potential=mattersim_potential_best_device,
            )
            assert relaxer3.input_structures.n_systems == 2

            assert_states_equal(
                relaxer1.original_structures,
                relaxer2.original_structures,
                STRUCTURE_CHECK_KEYS,
            )
            assert_states_equal(
                relaxer1.original_structures,
                relaxer3.original_structures,
                STRUCTURE_CHECK_KEYS,
            )
