"""Utilities for loading TorchSim trajectories."""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

import torch
import torch_sim as ts
from torch_sim.integrators.md import MDState
from torch_sim.optimizers.state import OptimState
from torch_sim.trajectory import TorchSimTrajectory


@dataclass(kw_only=True)
class MDStateWithStress(MDState):
    """MDState extended with stress tensor."""

    stress: torch.Tensor
    _system_attributes = MDState._system_attributes | {"stress"}  # noqa: SLF001


LOG = logging.getLogger(__name__)


class BaseTrajectoryLoader(ABC):
    """Base class for loading states from TorchSim trajectory files."""

    def __init__(
        self,
        trajectory_path: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        dtype: torch.dtype = torch.get_default_dtype(),
    ):
        """Initialize the trajectory loader.

        Args:
            trajectory_path: Path to the trajectory file (local only).
            device: Device to load tensors to.
            dtype: Data type for tensors.
        """
        if trajectory_path.startswith(("blob-storage://", "s3://", "gs://")):
            raise ValueError(
                f"Remote storage paths are not supported in the public release. "
                f"Got: {trajectory_path!r}. Please use local file paths."
            )
        self.trajectory_path = trajectory_path
        self.device = device
        self.dtype = dtype

    def get_trajectory(
        self, mode: Literal["r", "w", "a"] = "r"
    ) -> TorchSimTrajectory:
        """Open the trajectory file.

        Args:
            mode: File mode ('r' for read, 'w' for write, 'a' for append).

        Returns:
            Opened TorchSimTrajectory object.
        """
        return TorchSimTrajectory(self.trajectory_path, mode=mode)

    @property
    def metadata(self) -> dict:
        """Load metadata from the trajectory file."""
        with self.get_trajectory(mode="r") as traj:
            return traj.metadata

    @property
    def num_frames(self) -> int:
        """Get the number of frames in the trajectory."""
        return len(self.steps)

    @property
    def steps(self) -> list[int]:
        """Get the simulation steps corresponding to each frame."""
        with self.get_trajectory(mode="r") as traj:
            return traj.get_steps("positions").tolist()

    def _load_array_as_tensor(
        self,
        traj: TorchSimTrajectory,
        array_name: str,
        frame_from: int,
        frame_to: int,
    ) -> torch.Tensor:
        """Load an array from trajectory and convert to torch tensor.

        Args:
            traj: Open trajectory object.
            array_name: Name of the array to load.
            frame_from: Starting frame index.
            frame_to: Ending frame index.

        Returns:
            Tensor with the array data, converted to the target dtype.
        """
        data_shape = getattr(traj._file.root.data, array_name).shape  # noqa: SLF001
        if data_shape[0] == 1:
            frame_from, frame_to = 0, 1

        array = traj.get_array(array_name, frame_from, frame_to)
        tensor = torch.from_numpy(array).to(dtype=self.dtype)

        return tensor

    def _load_pbc(self, traj: TorchSimTrajectory) -> torch.Tensor:
        """Load periodic boundary conditions from trajectory.

        Args:
            traj: Open trajectory object.

        Returns:
            Boolean tensor indicating PBC along each axis.
        """
        pbc_array = traj.get_array("pbc", 0, 3)
        return torch.from_numpy(pbc_array).to(dtype=torch.bool)

    def _load_base_states(
        self,
        traj: TorchSimTrajectory,
        frame_from: int,
        frame_to: int,
    ) -> ts.state.SimState:
        """Load base SimState with positions, cell, atomic numbers, masses, pbc.

        Args:
            traj: Open trajectory object.
            frame_from: Starting frame index.
            frame_to: Ending frame index (exclusive).

        Returns:
            SimState with basic structural information.
        """
        num_frames = frame_to - frame_from

        positions = self._load_array_as_tensor(
            traj, "positions", frame_from, frame_to
        ).reshape(-1, 3)
        cell = self._load_array_as_tensor(
            traj, "cell", frame_from, frame_to
        ).reshape(-1, 3, 3)
        atomic_numbers = self._load_array_as_tensor(
            traj, "atomic_numbers", frame_from, frame_to
        ).flatten()
        num_atoms = atomic_numbers.shape[0]
        masses = self._load_array_as_tensor(
            traj, "masses", frame_from, frame_to
        ).flatten()
        pbc = self._load_pbc(traj)

        return ts.state.SimState(
            positions=positions,
            cell=cell,
            atomic_numbers=atomic_numbers.to(torch.long).repeat(num_frames),
            masses=masses.repeat(num_frames),
            pbc=pbc.to(torch.bool),
            system_idx=torch.repeat_interleave(
                torch.arange(num_frames, dtype=torch.long),
                num_atoms,
            ),
        )

    def load_state(self, frame: int = -1) -> ts.state.SimState:
        """Load a single state from trajectory file.

        Args:
            frame: Frame index to load (default: -1 for last frame).

        Returns:
            State for the specified frame.
        """
        if frame < 0:
            frame = self.num_frames + frame
        return self.load_states(frame_from=frame, frame_to=frame + 1)

    def load_states(
        self, frame_from: int = 0, frame_to: int | None = None
    ) -> ts.state.SimState:
        """Load states from trajectory file for a range of frames.

        Args:
            frame_from: Starting frame index to load (default: 0).
            frame_to: Ending frame index (exclusive). None for last frame.

        Returns:
            Concatenated states for all frames in the range.
        """
        with self.get_trajectory(mode="r") as traj:
            num_frames = len(traj.get_steps("positions"))
            frame_from = frame_from if frame_from >= 0 else num_frames + frame_from
            frame_to = frame_to or num_frames
            frame_to = frame_to if frame_to >= 0 else num_frames + frame_to
            assert 0 <= frame_from < num_frames, "frame_from index out of bounds"
            assert 0 < frame_to <= num_frames, "frame_to index out of bounds"
            assert frame_from < frame_to, "frame_from must be less than frame_to"

            base_state = self._load_base_states(traj, frame_from, frame_to)
            state = self._create_state(traj, base_state, frame_from, frame_to)

        return state.to(device=self.device, dtype=self.dtype)

    @abstractmethod
    def _create_state(
        self,
        traj: TorchSimTrajectory,
        base_state: ts.state.SimState,
        frame_from: int,
        frame_to: int,
    ) -> ts.state.SimState:
        """Create state by loading type-specific properties.

        Args:
            traj: Open trajectory object.
            base_state: Base SimState with positions, cell, etc.
            frame_from: Starting frame index.
            frame_to: Ending frame index (exclusive).

        Returns:
            State with full information (MDState or OptimState).
        """
        pass

    @property
    def frames(self) -> ts.state.SimState:
        """Load all frames from trajectory and concatenate them."""
        return self.load_states()

    @classmethod
    def load_states_from_files(
        cls,
        filenames: list[str],
        device: str,
        dtype: torch.dtype = torch.get_default_dtype(),
        frame: int = -1,
    ) -> ts.state.SimState:
        """Load states from multiple trajectory files and concatenate.

        Args:
            filenames: List of trajectory file paths.
            device: Device to load tensors to.
            dtype: Data type for tensors.
            frame: Frame index to load (default: -1 for last frame).

        Returns:
            Concatenated state containing structures from all systems.
        """
        if not filenames:
            raise ValueError(
                "No trajectory files found. Cannot load structures "
                "from empty trajectory list."
            )
        states = []
        for filepath in filenames:
            loader = cls(filepath, device=device, dtype=dtype)
            state = loader.load_state(frame=frame)
            states.append(state)
        return ts.concatenate_states(states)


class MDTrajectoryLoader(BaseTrajectoryLoader):
    """Loader for MD trajectory files.

    Loads MDState objects with velocities, forces, and energy.
    """

    def _create_state(
        self,
        traj: TorchSimTrajectory,
        base_state: ts.state.SimState,
        frame_from: int,
        frame_to: int,
    ) -> MDState | MDStateWithStress:
        """Create MDState by loading MD-specific properties.

        Args:
            traj: Open trajectory object.
            base_state: Base SimState with positions, cell, etc.
            frame_from: Starting frame index.
            frame_to: Ending frame index (exclusive).

        Returns:
            MDState with full dynamics information.
        """
        velocities = self._load_array_as_tensor(
            traj, "velocities", frame_from, frame_to
        ).reshape_as(base_state.positions)
        forces = self._load_array_as_tensor(
            traj, "forces", frame_from, frame_to
        ).reshape_as(base_state.positions)
        potential_energy = self._load_array_as_tensor(
            traj, "potential_energy", frame_from, frame_to
        ).flatten()

        momenta = velocities * base_state.masses.unsqueeze(-1)

        if "stress" in traj.array_registry:
            stress = self._load_array_as_tensor(
                traj, "stress", frame_from, frame_to
            ).reshape_as(base_state.cell)
            return MDStateWithStress.from_state(
                base_state,
                momenta=momenta,
                forces=forces,
                energy=potential_energy,
                stress=stress,
            )
        return MDState.from_state(
            base_state,
            momenta=momenta,
            forces=forces,
            energy=potential_energy,
        )

    @property
    def temperature_schedule(self) -> float | list[float] | None:
        """Load the temperature from the trajectory file if it exists.

        Returns:
            - float: If a scalar temperature was stored
            - list[float]: If a temperature schedule was stored
            - None: If no temperature data exists in the trajectory
        """
        with self.get_trajectory(mode="r") as traj:
            if "temperature_schedule_K" not in traj.array_registry:
                try:
                    return self.metadata["settings"]["temperature_K"]
                except (KeyError, TypeError):
                    return None
            temp_array = traj.get_array("temperature_schedule_K", 0, 1)
            squeezed = temp_array.squeeze(0)
            if squeezed.shape[0] == 1:
                return float(squeezed[0])
            return squeezed.tolist()


class RelaxationTrajectoryLoader(BaseTrajectoryLoader):
    """Loader for relaxation trajectory files.

    Loads OptimState objects with forces, energy, and stress.
    """

    def _create_state(
        self,
        traj: TorchSimTrajectory,
        base_state: ts.state.SimState,
        frame_from: int,
        frame_to: int,
    ) -> OptimState:
        """Create OptimState by loading optimization-specific properties.

        Args:
            traj: Open trajectory object.
            base_state: Base SimState with positions, cell, etc.
            frame_from: Starting frame index.
            frame_to: Ending frame index (exclusive).

        Returns:
            OptimState with forces, energy, and stress.
        """
        forces = self._load_array_as_tensor(
            traj, "forces", frame_from, frame_to
        ).reshape_as(base_state.positions)
        energy = self._load_array_as_tensor(
            traj, "potential_energy", frame_from, frame_to
        ).flatten()
        stress = self._load_array_as_tensor(
            traj, "stress", frame_from, frame_to
        ).reshape_as(base_state.cell)

        return OptimState.from_state(
            base_state, forces=forces, energy=energy, stress=stress
        )
