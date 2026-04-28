"""Batch structure relaxation using TorchSim optimizers."""

import logging
from typing import Callable

import torch
import torch_sim as ts
from ase import Atoms
from torch_sim.constraints import FixSymmetry
from torch_sim.optimizers.state import OptimState

from mattersim.torchsim.base import TorchSimBatchRunner
from mattersim.torchsim.settings import OptimizerSettings
from mattersim.torchsim.trajectory_loader import RelaxationTrajectoryLoader

LOG = logging.getLogger(__name__)


class TorchSimBatchRelaxer(TorchSimBatchRunner[OptimizerSettings]):
    """Batch structure relaxation using TorchSim optimizers."""

    settings_class = OptimizerSettings
    trajectory_prefix = "relax_traj"
    trajectory_loader_class = RelaxationTrajectoryLoader

    @property
    def optimizer(self) -> ts.Optimizer:
        return ts.Optimizer[self.settings.name]

    @property
    def convergence_fn(
        self,
    ) -> Callable[[ts.state.SimState, torch.Tensor | None], torch.Tensor]:
        force_convergence = ts.generate_force_convergence_fn(
            self.settings.fmax, include_cell_forces=True
        )

        def _nan_aware_convergence(
            state: ts.state.SimState, last_energy: torch.Tensor | None
        ) -> torch.Tensor:
            converged = force_convergence(state, last_energy)
            return converged | torch.isnan(state.energy)

        return _nan_aware_convergence

    def run(self) -> tuple[ts.state.SimState, torch.Tensor]:
        """Run batch relaxation.

        Returns:
            Tuple of (relaxed_state, converged) where converged is a
            per-system boolean tensor.

        After calling this method, ``self.diverged`` contains a per-system
        boolean tensor indicating which systems produced non-finite energy.
        """
        trajectory_reporter = None
        state = self.input_structures

        if self.settings.constrain_symmetry:
            constraint = FixSymmetry.from_state(state)
            state.constraints = constraint
        if self.filenames:
            trajectory_reporter_dict = dict(
                filenames=self.filenames,
                state_frequency=self.settings.save_checkpoint_every,
                metadata=self.metadata,
                trajectory_kwargs=dict(mode="a"),
                state_kwargs=dict(save_velocities=True, save_forces=True),
                prop_calculators=self.settings.prop_calculators,
            )
            trajectory_reporter = ts.runners._configure_reporter(
                trajectory_reporter_dict,
                prop_frequency=self.settings.save_checkpoint_every,
            )

        relaxed_state = ts.optimize(
            system=state,
            model=self.potential,
            optimizer=self.optimizer,
            autobatcher=self.settings.get_autobatcher(self.potential),
            max_steps=self.settings.max_steps,
            convergence_fn=self.convergence_fn,
            init_kwargs=self.settings.init_kwargs,
            trajectory_reporter=trajectory_reporter,
            steps_between_swaps=self.settings.steps_between_swaps,
            pbar=True,
        )
        self.diverged = torch.isnan(relaxed_state.energy)
        converged = self.convergence_fn(relaxed_state, None) & ~self.diverged
        return relaxed_state, converged

    def relax(self) -> dict[int, list[Atoms]]:
        """Run batch relaxation and return per-system trajectories as ASE Atoms.

        The final frame of each trajectory has ``atoms.info["converged"]``
        set to ``True``/``False``.

        Returns:
            Mapping from system index to a list of Atoms forming a trajectory.
        """
        if not self.filenames:
            return self._relax_in_memory()
        _, converged = self.run()
        result = {
            ix: self._optim_state_to_atoms(traj)
            for ix, traj in enumerate(self.trajectories)
        }
        for ix, traj in result.items():
            traj[-1].info["converged"] = bool(converged[ix])
            if self._original_atoms_info is not None:
                protected_keys = {
                    "total_energy",
                    "stress",
                    "forces",
                    "converged",
                }
                for key, value in self._original_atoms_info[ix].items():
                    if (
                        key not in protected_keys
                        and key not in traj[-1].info
                    ):
                        traj[-1].info[key] = value
        return result

    def _relax_in_memory(self) -> dict[int, list[Atoms]]:
        """Run relaxation without trajectory files."""
        initial_state = self._evaluate_batched(self.input_structures)
        initial_atoms = self._optim_state_to_atoms(initial_state)

        relaxed_state, converged = self.run()
        relaxed_atoms = self._optim_state_to_atoms(relaxed_state)

        for ix, atoms in enumerate(relaxed_atoms):
            atoms.info["converged"] = bool(converged[ix])

        return {
            ix: [initial_atoms[ix], relaxed_atoms[ix]]
            for ix in range(len(initial_atoms))
        }

    @staticmethod
    def _optim_state_to_atoms(state: OptimState) -> list[Atoms]:
        """Convert an OptimState to Atoms with energy/forces/stress."""
        atoms_list = state.to_atoms()
        force_splits = torch.split(
            state.forces.detach().cpu(),
            state.n_atoms_per_system.tolist(),
        )
        for ix, atoms in enumerate(atoms_list):
            atoms.arrays["forces"] = force_splits[ix].numpy()
            atoms.info["total_energy"] = (
                state.energy[ix].detach().cpu().item()
            )
            atoms.info["stress"] = state.stress[ix].detach().cpu().numpy()
        return atoms_list

    def _evaluate_batched(
        self, state: ts.SimState, max_atoms: int = 512
    ) -> OptimState:
        """Evaluate model on state, chunking to avoid OOM."""
        if state.n_atoms <= max_atoms:
            chunks = [self.potential(state)]
        else:
            batch_size = max(
                1, max_atoms * state.n_systems // state.n_atoms
            )
            chunks = [
                self.potential(state[i : i + batch_size])
                for i in range(0, state.n_systems, batch_size)
            ]
        energy = torch.cat([c["energy"] for c in chunks])
        forces = torch.cat([c["forces"] for c in chunks])
        stress = torch.cat([c["stress"] for c in chunks])
        return OptimState.from_state(
            state, energy=energy, forces=forces, stress=stress
        )
