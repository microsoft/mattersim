# -*- coding: utf-8 -*-
import logging
import sys
from typing import Dict, List, Union

from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.filters import ExpCellFilter
from ase.optimize import BFGS, FIRE
from ase.optimize.optimize import Optimizer
from tqdm import tqdm

from mattersim.datasets.utils.build import build_dataloader
from mattersim.forcefield.potential import Potential

FORMAT = "[%(levelname)s] [%(name)s]: %(message)s"
logging.basicConfig(
    level=logging.INFO,
    format=FORMAT,
    # use stdout to flush tqdm out periodically
    stream=sys.stdout,
)
LOG = logging.getLogger(__name__)


class DummyBatchCalculator(Calculator):
    def __init__(self):
        super().__init__()

    def calculate(self, atoms=None, properties=None, system_changes=None):
        pass

    def get_potential_energy(self, atoms=None):
        return atoms.info["total_energy"]

    def get_forces(self, atoms=None):
        return atoms.arrays["forces"]

    def get_stress(self, atoms=None):
        return atoms.info["stress"]


class BatchRelaxer(object):
    """Relaxer is a class for structural relaxation with fixed volume."""

    SUPPORTED_OPTIMIZERS = {"BFGS": BFGS, "FIRE": FIRE}

    def __init__(
        self,
        potential: Potential,
        optimizer: Union[Optimizer, str] = "FIRE",
        fmax: float = 0.05,
        max_natoms_per_batch: int = 512,
        ):
        self.potential = potential
        self.device = potential.device
        self.optimizer = (
            self.SUPPORTED_OPTIMIZERS[optimizer.upper()]
            if isinstance(optimizer, str)
            else optimizer
        )
        self.fmax = fmax
        self.max_natoms_per_batch = max_natoms_per_batch
        self.optimizer_instances: List[Optimizer] = []
        self.is_active_instance: List[bool] = []
        self.finished = False
        self.total_converged = 0
        self.trajectories: Dict[int, List[Atoms]] = {}

    def insert(self, atoms: Atoms):
        atoms.set_calculator(DummyBatchCalculator())
        optimizer_instance = self.optimizer(
            ExpCellFilter(atoms),
        )
        optimizer_instance.fmax = self.fmax
        self.optimizer_instances.append(optimizer_instance)
        self.is_active_instance.append(True)

    def step_batch(self):
        atoms_list = []
        for idx, fire in enumerate(self.optimizer_instances):
            if self.is_active_instance[idx]:
                atoms_list.append(fire.atoms.atoms)

        # Note: we use a batch size of len(atoms_list) because we only want to run one batch at a time
        dataloader = build_dataloader(atoms_list, batch_size=len(atoms_list), only_inference=True)
        # in case we get a CUDA error inside the try/except, we can't get the number of atoms
        # from CUDA anymore, so we need to get it before copying to CUDA.
        energy_batch, forces_batch, stress_batch = self.potential.predict_properties(
            dataloader, include_forces=True, include_stresses=True
        )

        counter = 0
        self.finished = True
        for idx, fire in enumerate(self.optimizer_instances):
            if self.is_active_instance[idx]:
                # Set the properties so the dummy calculator can return them within the optimizer step
                fire.atoms.atoms.info["total_energy"] = energy_batch[counter]
                fire.atoms.atoms.arrays["forces"] = forces_batch[counter]
                fire.atoms.atoms.info["stress"] = stress_batch[counter]
                try:
                    self.trajectories[fire.atoms.atoms.info["structure_index"]].append(fire.atoms.atoms.copy())
                except KeyError:
                    self.trajectories[fire.atoms.atoms.info["structure_index"]] = [fire.atoms.atoms.copy()]

                fire.step()
                if fire.converged():
                    self.is_active_instance[idx] = False
                    self.total_converged += 1
                    if self.total_converged % 100 == 0:
                        LOG.info(f"Relaxed {self.total_converged} structures.")
                else:
                    self.finished = False
                counter += 1

        # remove inactive instances
        self.optimizer_instances = [
            fire for fire, active in zip(self.optimizer_instances, self.is_active_instance) if active
        ]
        self.is_active_instance = [True] * len(self.optimizer_instances)


    def relax(
        self,
        atoms_list: List[Atoms],
    ) -> Dict[int, List[Atoms]]:
        self.trajectories = {}
        self.tqdmcounter = tqdm(total=len(atoms_list), file=sys.stdout)
        pointer = 0
        atoms_list_ = []
        for i in range(len(atoms_list)):
            atoms_list_.append(atoms_list[i].copy())
            atoms_list_[i].info["structure_index"] = i

        while (
            pointer < len(atoms_list) or not self.finished
        ):  # While there are unfinished instances or atoms left to insert
            while pointer < len(atoms_list) and (
                sum([len(atoms.atoms) - 3 for atoms in self.optimizer_instances])
                + len(atoms_list[pointer])
                <= self.max_natoms_per_batch
            ):  # While there are enough n_atoms slots in the batch and we have not reached the end of the list.
                # The -3 is to account for the 3 degrees of freedom in the expcell that are not atoms but are counted in the len(atoms)
                self.insert(atoms_list_[pointer])  #  Insert new structure to fire instances
                self.tqdmcounter.update(1)
                pointer += 1
            self.step_batch()
        self.tqdmcounter.close()

        return self.trajectories