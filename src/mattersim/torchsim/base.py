"""Abstract base class for TorchSim batch runners."""

import glob as glob_module
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import asdict
from typing import Generic, TypeVar

import torch_sim as ts
from ase import Atoms
from torch_sim.typing import StateLike
from typing_extensions import Self

from mattersim.torchsim.model_loading import (
    TorchSimPotentialLike,
    get_torchsim_wrapper,
)
from mattersim.torchsim.settings import (
    DTYPE,
    IntegratorSettings,
    OptimizerSettings,
)
from mattersim.torchsim.torchsim_wrapper import TorchSimWrapper
from mattersim.torchsim.trajectory_loader import BaseTrajectoryLoader

LOG = logging.getLogger(__name__)

SettingsT = TypeVar("SettingsT", bound=OptimizerSettings | IntegratorSettings)

unit_system = ts.units.UnitSystem.metal


class TorchSimBatchRunner(ABC, Generic[SettingsT]):
    """Abstract base class for batch processing of structures with TorchSim.

    Provides common infrastructure for relaxation and MD simulations.

    Use the class methods to construct instances:

    - ``from_structures()``: Start new simulations from structures
    - ``from_trajectory_filenames()``: Resume from existing trajectory files
    - ``from_trajectory_folder()``: Resume from a folder of trajectory files
    """

    # Subclasses must define these class variables
    settings_class: type[SettingsT]
    trajectory_prefix: str
    trajectory_loader_class: type[BaseTrajectoryLoader]

    def __init__(
        self,
        settings: SettingsT | list[SettingsT],
        input_structures: ts.state.SimState,
        filenames: list[str],
        potential: TorchSimWrapper,
        original_atoms_info: list[dict] | None = None,
    ):
        """Internal constructor. Use class methods instead.

        Args:
            settings: Optimizer or integrator settings. Can be a single
                settings object or a list (one per system).
            input_structures: Pre-initialized input structures.
            filenames: List of trajectory file paths.
            potential: TorchSimWrapper wrapping the potential model.
            original_atoms_info: Optional list of original atoms.info dicts
                to restore in trajectories.
        """
        n_systems = input_structures.n_systems

        if isinstance(settings, list):
            if len(settings) != n_systems:
                raise ValueError(
                    f"Number of settings ({len(settings)}) must match "
                    f"number of systems ({n_systems})"
                )
            self._per_system_settings = settings
        else:
            self._per_system_settings = [settings] * n_systems

        self.filenames = filenames
        self.input_structures = input_structures
        self.potential = potential
        self._original_atoms_info = original_atoms_info

    @property
    def settings(self) -> SettingsT:
        """Get the settings for the batch (first system's settings)."""
        return self._per_system_settings[0]

    def get_settings_for_system(self, system_idx: int) -> SettingsT:
        """Get settings for a specific system."""
        return self._per_system_settings[system_idx]

    @property
    def device(self) -> str:
        return self.settings.device

    @property
    def has_trajectories(self) -> bool:
        """Check if trajectory files exist for all systems."""
        if not self.filenames:
            return False
        return all(os.path.exists(f) for f in self.filenames)

    @property
    def is_continuing_from_trajectories(self) -> bool:
        """Detect if we're continuing from existing trajectories.

        Returns True if trajectory files exist and contain frames beyond
        the initial state.
        """
        if not self.has_trajectories:
            return False
        loader = self.trajectory_loader_class(self.filenames[0])
        return loader.num_frames > 1

    def get_steps_already_completed(self) -> list[int]:
        """Get the number of steps already completed in existing trajectories.

        Returns:
            List of steps completed per system (0 if not continuing).
        """
        if not self.is_continuing_from_trajectories:
            return [0 for _ in range(self.input_structures.n_systems)]

        num_steps = []
        for filename in self.filenames:
            num_steps.append(
                max(self.trajectory_loader_class(filename).steps)
            )
        return num_steps

    @classmethod
    def from_structures(
        cls,
        structures: StateLike,
        settings: SettingsT | list[SettingsT],
        save_folder: str | None = None,
        potential: TorchSimPotentialLike | None = None,
        override: bool = False,
        gradient_checkpointing: bool = False,
        sanitize_nan: bool = False,
        max_neighbors: int = 0,
    ) -> Self:
        """Create a runner instance from structures for new simulations.

        Args:
            structures: Input structures to process.
            settings: Optimizer or integrator settings.
            save_folder: Folder for trajectory files. If None, results are
                kept in memory only.
            potential: Optional potential model.
            override: Whether to override existing trajectory files.
            gradient_checkpointing: Enable gradient checkpointing.
            sanitize_nan: Replace non-finite model outputs with NaN.
            max_neighbors: Maximum neighbors per atom. 0 = no limit.

        Returns:
            Instance of the runner class.
        """
        device = (
            settings.device if not isinstance(settings, list) else settings[0].device
        )

        original_atoms_info = (
            [s.info.copy() for s in structures]
            if isinstance(structures, list)
            and structures
            and isinstance(structures[0], Atoms)
            else None
        )
        input_structures = ts.initialize_state(structures, device, DTYPE)
        if (
            isinstance(settings, list)
            and len(settings) != input_structures.n_systems
        ):
            raise ValueError(
                f"Number of settings ({len(settings)}) must match number of "
                f"systems ({input_structures.n_systems})"
            )

        if save_folder is not None:
            filenames = [
                f"{save_folder}/{cls.trajectory_prefix}_{i}.h5md"
                for i in range(input_structures.n_systems)
            ]
            for filepath in filenames:
                if os.path.exists(filepath):
                    if override:
                        LOG.warning(
                            f"Overriding existing trajectory file: {filepath}"
                        )
                        os.remove(filepath)
        else:
            filenames = []

        wrapper = get_torchsim_wrapper(
            potential,
            device=device,
            gradient_checkpointing=gradient_checkpointing,
            sanitize_nan=sanitize_nan,
            max_neighbors=max_neighbors,
        )
        return cls(
            settings=settings,
            filenames=filenames,
            potential=wrapper,
            input_structures=input_structures,
            original_atoms_info=original_atoms_info,
        )

    @classmethod
    def from_trajectory_filenames(
        cls,
        filenames: list[str],
        potential: TorchSimPotentialLike | None = None,
        gradient_checkpointing: bool = False,
    ) -> Self:
        """Create a runner instance from existing trajectory files.

        Args:
            filenames: List of trajectory file paths.
            potential: Optional potential to use.
            gradient_checkpointing: Enable gradient checkpointing.

        Returns:
            Instance of the runner class.
        """
        if not filenames:
            raise ValueError(
                "At least one trajectory file must be provided."
            )

        per_system_settings = []
        loaded_potential = None

        for filename in filenames:
            loader = cls.trajectory_loader_class(filename)
            metadata = loader.metadata
            settings_dict = metadata["settings"].copy()

            if loaded_potential is None:
                loaded_potential = metadata["potential"]

            if settings_dict.get("temperature_K") is None and hasattr(
                cls.trajectory_loader_class, "temperature_schedule"
            ):
                temp = loader.temperature_schedule  # type: ignore[attr-defined]
                if temp is None:
                    raise ValueError(
                        "Temperature marked as stored separately but "
                        f"not found in: {filename}"
                    )
                settings_dict["temperature_K"] = temp

            per_system_settings.append(cls.settings_class(**settings_dict))

        potential = potential or loaded_potential

        first_settings = per_system_settings[0]
        input_structures = cls.trajectory_loader_class.load_states_from_files(
            filenames=filenames,
            device=first_settings.device,
            dtype=DTYPE,
            frame=-1,
        )

        wrapper = get_torchsim_wrapper(
            potential,
            device=first_settings.device,
            gradient_checkpointing=gradient_checkpointing,
        )
        return cls(
            settings=per_system_settings,
            filenames=filenames,
            potential=wrapper,
            input_structures=input_structures,
        )

    @classmethod
    def from_trajectory_folder(
        cls,
        trajectory_folder: str,
        potential: TorchSimPotentialLike | None = None,
        gradient_checkpointing: bool = False,
    ) -> Self:
        """Create a runner instance from a folder containing trajectory files.

        Args:
            trajectory_folder: Path to folder containing trajectory files.
            potential: Optional potential to use.
            gradient_checkpointing: Enable gradient checkpointing.

        Returns:
            Instance of the runner class.
        """
        pattern = f"{trajectory_folder}/{cls.trajectory_prefix}_*.h5md"
        filenames = sorted(glob_module.glob(pattern))

        if not filenames:
            raise ValueError(
                f"No trajectory files found in {trajectory_folder} "
                f"matching pattern {cls.trajectory_prefix}_*.h5md"
            )

        return cls.from_trajectory_filenames(
            filenames=filenames,
            potential=potential,
            gradient_checkpointing=gradient_checkpointing,
        )

    @property
    def metadata(self) -> dict:
        """Generate metadata dict for trajectory files.

        Uses the first system's settings. Temperature is excluded from
        metadata and stored separately as a tensor in the trajectory files.
        """
        settings_dict = asdict(self.settings)
        if "temperature_K" in settings_dict:
            settings_dict["temperature_K"] = None
        return {
            "potential": self.potential.model.version,
            "settings": settings_dict,
        }

    @property
    def original_structures(self) -> ts.state.SimState:
        """Get the original structures (frame 0) from trajectories or input."""
        if not self.has_trajectories:
            return self.input_structures
        states = []
        for filepath in self.filenames:
            loader = self.trajectory_loader_class(
                filepath, device=self.device, dtype=DTYPE
            )
            states.append(loader.load_state(frame=0))
        return ts.concatenate_states(states)

    @property
    def trajectories(self) -> list[ts.state.SimState]:
        """Load all frames from trajectory files.

        Returns:
            List of states, one per system, each containing all frames.
        """
        if not self.has_trajectories:
            raise ValueError(
                f"No trajectory files found. Please run "
                f"{self.__class__.__name__} first and save trajectories."
            )
        states = []
        for filepath in self.filenames:
            loader = self.trajectory_loader_class(
                filepath, device=self.device, dtype=DTYPE
            )
            states.append(loader.frames)
        return states

    @abstractmethod
    def run(self) -> ts.state.SimState:
        """Run the simulation.

        Returns:
            Final state after simulation.
        """
        pass
