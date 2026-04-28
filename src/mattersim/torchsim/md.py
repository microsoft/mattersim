"""Batch molecular dynamics simulation using TorchSim integrators."""

import logging

import torch
import torch_sim as ts

from mattersim.torchsim.base import TorchSimBatchRunner
from mattersim.torchsim.settings import IntegratorSettings
from mattersim.torchsim.trajectory_loader import MDTrajectoryLoader

LOG = logging.getLogger(__name__)


class TorchSimBatchMD(TorchSimBatchRunner[IntegratorSettings]):
    """Batch molecular dynamics simulation using TorchSim integrators."""

    settings_class = IntegratorSettings
    trajectory_prefix = "md_traj"
    trajectory_loader_class = MDTrajectoryLoader

    @property
    def integrator(self) -> ts.Integrator:
        return ts.Integrator[self.settings.name]

    def _get_system_temperature(self, system_idx: int) -> torch.Tensor:
        """Get the temperature for a specific system as a tensor."""
        settings = self.get_settings_for_system(system_idx)
        return torch.tensor(
            settings.temperature_K,
            dtype=self.input_structures.dtype,
            device=self.device,
        )

    @property
    def temperature_tensor(self) -> torch.Tensor:
        """Build the temperature tensor for ts.integrate.

        Returns:
            torch.Tensor with shape:
            - (): 0D scalar if all systems have the same scalar temperature
            - (n_steps,): 1D if all systems have the same schedule
            - (n_steps, n_systems): 2D if systems have different temperatures
        """
        n_systems = self.input_structures.n_systems
        temperatures = [
            self._get_system_temperature(i) for i in range(n_systems)
        ]

        first = temperatures[0]
        if all(torch.equal(t, first) for t in temperatures):
            return first

        n_steps = self.settings.num_steps
        normalized = []
        for i, temp in enumerate(temperatures):
            if temp.ndim == 0:
                normalized.append(temp.expand(n_steps))
            elif temp.shape[0] != n_steps:
                raise ValueError(
                    f"Temperature schedule length ({temp.shape[0]}) "
                    f"must match num_steps ({n_steps})"
                )
            else:
                normalized.append(temp)

        return torch.stack(normalized, dim=1)

    def _prepare_continuation(self) -> int:
        """Prepare for continuing from existing trajectories.

        Returns:
            Number of remaining steps to run.
        """
        steps_already_completed = self.get_steps_already_completed()
        min_steps_already_completed = min(steps_already_completed)

        if len(set(steps_already_completed)) > 1:
            LOG.warning(
                "Continuing from existing trajectories with different "
                f"numbers of steps completed ({steps_already_completed}). "
                "Will truncate all to the minimum number of completed steps "
                f"({min_steps_already_completed})."
            )
            for filename in self.filenames:
                loader = self.trajectory_loader_class(filename)
                with loader.get_trajectory(mode="a") as traj:
                    traj.truncate_to_step(min_steps_already_completed)

        max_steps = self.settings.num_steps
        remaining_steps = max(0, max_steps - min_steps_already_completed)

        if min_steps_already_completed > 0:
            LOG.info(
                f"Continuing from existing trajectory: "
                f"{min_steps_already_completed} steps already completed, "
                f"running {remaining_steps} more steps "
                f"(total: {max_steps})."
            )

        return remaining_steps

    def run(self) -> ts.state.SimState:
        """Run the MD simulation and return the final state.

        If continuing from existing trajectories, adjusts the number of
        steps to run so that the total equals the maximum in settings.
        """
        input_state = self.input_structures
        init_kwargs = self.settings.init_kwargs

        integrator_kwargs = self.settings.step_kwargs.copy()

        temperature_tensor = self.temperature_tensor
        integrator_kwargs["temperature"] = temperature_tensor

        if self.is_continuing_from_trajectories:
            remaining_steps = self._prepare_continuation()
            if remaining_steps == 0:
                LOG.warning(
                    "No remaining steps to run; returning final states "
                    "from existing trajectories."
                )
                return self.trajectory_loader_class.load_states_from_files(
                    filenames=self.filenames,
                    device=input_state.device,
                    dtype=input_state.dtype,
                    frame=-1,
                )
            if temperature_tensor.ndim >= 1:
                integrator_kwargs["temperature"] = temperature_tensor[
                    -remaining_steps:
                ]
            integrator_kwargs["n_steps"] = remaining_steps

        if hasattr(input_state, "momenta"):
            init_kwargs["momenta"] = input_state.momenta

        trajectory_reporter = None
        if self.filenames:
            trajectory_reporter_dict = dict(
                filenames=self.filenames,
                state_frequency=self.settings.save_checkpoint_every,
                metadata=self.metadata,
                trajectory_kwargs=dict(mode="a"),
                state_kwargs=dict(save_velocities=True, save_forces=True),
            )
            _properties = ["kinetic_energy", "potential_energy", "temperature"]
            if self.integrator.name == "npt_langevin":
                _properties.append("stress")
            trajectory_reporter = ts.runners._configure_reporter(
                trajectory_reporter_dict,
                properties=_properties,
                prop_frequency=self.settings.save_checkpoint_every,
            )

            self._write_temperature_to_trajectories(trajectory_reporter)

        LOG.info("Starting MD simulation...")
        md_state = ts.integrate(
            system=input_state,
            model=self.potential,
            integrator=self.integrator,
            init_kwargs=init_kwargs,
            autobatcher=self.settings.get_autobatcher(self.potential),
            trajectory_reporter=trajectory_reporter,
            pbar=True,
            **integrator_kwargs,
        )

        LOG.info("MD simulation completed.")
        return md_state

    def _write_temperature_to_trajectories(
        self, trajectory_reporter: ts.TrajectoryReporter
    ) -> None:
        """Write temperature as a global array to each trajectory file.

        Each trajectory file receives its corresponding system's temperature.
        Skips writing if temperature is already present (continuation case).
        """
        for i, traj in enumerate(trajectory_reporter.trajectories):
            if "temperature_schedule_K" in traj.array_registry:
                continue

            temp_np = self._get_system_temperature(i).cpu().numpy()
            temp_array = (
                temp_np.reshape(1, -1)
                if temp_np.ndim > 0
                else temp_np.reshape(1, 1)
            )
            traj.write_global_array("temperature_schedule_K", temp_array)
