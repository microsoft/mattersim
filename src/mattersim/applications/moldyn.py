# -*- coding: utf-8 -*-
from typing import Union

from ase import Atoms, units
from ase.io import Trajectory
from ase.md.npt import NPT
from ase.md.nvtberendsen import NVTBerendsen
from ase.md.velocitydistribution import (  # noqa: E501
    MaxwellBoltzmannDistribution,
    Stationary,
)


class MolecularDynamics:
    """
    This class is used for Molecular Dynamics.
    """

    SUPPORTED_ENSEMBLE = ["NVT_BERENDSEN", "NVT_NOSE_HOOVER"]

    def __init__(
        self,
        atoms: Atoms,
        ensemble: str = "nvt_nose_hoover",
        temperature: float = 300,
        timestep: float = 1.0,
        taut: Union[float, None] = None,
        trajectory: Union[str, Trajectory, None] = None,
        logfile: Union[str, None] = "-",
        loginterval: int = 10,
        append_trajectory: bool = False,
    ):
        """
        Args:
            atoms (Union[Atoms, Structure]): ASE atoms object contains
                structure information and calculator.
            ensemble (str, optional): Simulation ensemble chosen. Defaults
                to nvt_nose_hoover'
            temperature (float, optional): Simulation temperature, in Kelvin.
                Defaults to 300 K.
            timestep (float, optional): The simulation time step, in fs. Defaults
                to 1 fs.
            taut (float, optional): Characteristic timescale of the thermostat,
                in fs. If is None, automatically set it to 1000 * timestep.
            trajectory (Union[str, Trajectory], optional): Attach trajectory
                object. If trajectory is a string a Trajectory will be constructed.
                Defaults to None, which means for no trajectory.
            logfile (str, optional): If logfile is a string, a file with that name
                will be opened. Defaults to '-', which means output to stdout.
            loginterval (int, optional): Only write a log line for every loginterval
                time steps. Defaults to 10.
            append_trajectory (bool, optional): If False the trajectory file to be
                overwritten each time the dynamics is restarted from scratch. If True,
                the new structures are appended to the trajectory file instead.

        """
        assert atoms.calc is not None, (
            "Molecular Dynamics simulation only accept "
            "ase atoms with an attached calculator"
        )
        if ensemble.upper() not in self.SUPPORTED_ENSEMBLE:
            raise NotImplementedError(  # noqa: E501
                f"Ensemble {ensemble} is not yet supported."
            )

        self.atoms = atoms

        self.ensemble = ensemble.upper()
        self._temperature = temperature
        self.timestep = timestep

        if taut is None:
            taut = 1000 * timestep * units.fs
        self.taut = taut

        self._trajectory = trajectory
        self.logfile = logfile
        self.loginterval = loginterval
        self.append_trajectory = append_trajectory

        self._initialize_dynamics()

    def _initialize_dynamics(self):
        """
        Initialize the Dynamic ensemble class.
        """
        MaxwellBoltzmannDistribution(
            self.atoms, temperature_K=self._temperature, force_temp=True
        )
        Stationary(self.atoms)

        if self.ensemble == "NVT_BERENDSEN":  # noqa: E501
            self.dyn = NVTBerendsen(
                self.atoms,
                timestep=self.timestep * units.fs,
                temperature_K=self._temperature,
                taut=self.taut,
                trajectory=self._trajectory,
                logfile=self.logfile,
                loginterval=self.loginterval,
                append_trajectory=self.append_trajectory,
            )
        elif self.ensemble == "NVT_NOSE_HOOVER":
            self.dyn = NPT(
                self.atoms,
                timestep=self.timestep * units.fs,
                temperature_K=self._temperature,
                ttime=self.taut,
                pfactor=None,
                trajectory=self._trajectory,
                logfile=self.logfile,
                loginterval=self.loginterval,
                append_trajectory=self.append_trajectory,
            )
        else:
            raise NotImplementedError(  # noqa: E501
                f"Ensemble {self.ensemble} is not yet supported."
            )

    def run(self, n_steps: int = 1):
        """
        Run Molecular Dynamic simulation.

        Args:
            n_steps (int, optional): Number of steps to simulations. Defaults to 1.
        """
        self.dyn.run(n_steps)

    @property
    def temperature(self):
        return self._temperature

    @temperature.setter
    def temperature(self, temperature: float):
        self._temperature = temperature
        self._initialize_dynamics()

    @property
    def trajectory(self):
        return self._trajectory

    @trajectory.setter
    def trajectory(self, trajectory: Union[str, Trajectory, None]):
        self._trajectory = trajectory
        self._initialize_dynamics()
