# -*- coding: utf-8 -*-
import csv
import datetime
import os
from typing import Iterable, Optional, Union

import numpy as np
import yaml
from phonopy import Phonopy, PhonopyQHA
from tqdm import tqdm
from yaml import CLoader as Loader

from mattersim.applications.phonon import PhononWorkflow
from mattersim.utils.supercell_utils import get_supercell_parameters


class PhonopyQHAWorkflow(PhononWorkflow):
    """
    This class is used to calculate the thermal
    conductivity of a material using phono3py.
    """

    def __init__(
        self,
        lower_deformation: int = -5,
        upper_deformation: int = 5,
        deformation_step: int = 1,
        work_dir: Optional[str] = None,
        qha_pressures: Optional[Iterable[float]] = None,
        tmin: float = 0.0,
        tmax: float = 1000.0,
        tstep: float = 10.0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.lower_deformation = lower_deformation
        self.upper_deformation = upper_deformation
        self.deformation_step = deformation_step
        self.tmin = tmin
        self.tmax = tmax
        self.tstep = tstep
        current_dir = os.getcwd()
        if work_dir is not None:
            self.work_dir = os.path.join(current_dir, work_dir)
        else:
            current_datetime = datetime.datetime.now()
            formatted_datetime = current_datetime.strftime("%Y-%m-%d-%H-%M")
            self.work_dir = os.path.join(current_dir, f"{formatted_datetime}-qha")
        self.qha_pressures = qha_pressures if qha_pressures is not None else [0.0]
        self.entropy: Optional[np.ndarray] = None
        self.cv: Optional[np.ndarray] = None
        self.free_energy: Optional[np.ndarray] = None
        self.displaced_energies: Optional[np.ndarray] = None
        self.volumes: Optional[np.ndarray] = None
        self.temperatures = np.arange(self.tmin, self.tmax, self.tstep)
        self.atom_nums = self.atoms.get_number_of_atoms()

    def compute_thermal_properties(self, phonon: Phonopy, frac: Union[int, np.int64]):
        phonon.run_thermal_properties(
            t_min=self.tmin, t_max=self.tmax - self.tstep, t_step=self.tstep
        )
        phonon.write_yaml_thermal_properties(
            filename=os.path.join(self.work_dir, f"thermal_properties-{frac}.yaml")
        )

        phonon.plot_thermal_properties().savefig(
            os.path.join(self.work_dir, f"thermal_properties-{frac}.png"), dpi=600
        )

    def read_thermal_properties(self):
        if self.volumes is None or self.displaced_energies is None:
            e_v_file = os.path.join(self.work_dir, "e-v.dat")
            try:
                volumes, displaced_energies = np.loadtxt(e_v_file, unpack=True)
            except RuntimeError:
                raise RuntimeError(f"Failed to read {e_v_file}")
            self.volumes = np.array(volumes)
            self.displaced_energies = np.array(displaced_energies)

        cv, entropy, free_energy = [], [], []
        for frac in range(
            self.lower_deformation,
            self.upper_deformation + self.deformation_step,
            self.deformation_step,
        ):
            yaml_file = os.path.join(self.work_dir, f"thermal_properties-{frac}.yaml")
            thermal_properties = yaml.load(open(yaml_file), Loader=Loader)[
                "thermal_properties"
            ]
            cv.append([v["heat_capacity"] for v in thermal_properties])
            entropy.append([v["entropy"] for v in thermal_properties])
            free_energy.append([v["free_energy"] for v in thermal_properties])
        self.cv = np.array(cv, dtype=float)
        self.entropy = np.array(entropy)
        self.free_energy = np.array(free_energy, dtype=float)

    def compute_qha(self):
        with open(os.path.join(self.work_dir, "e-v.dat"), "w") as f:
            for v, e in zip(self.volumes, self.displaced_energies):
                f.write(f"{v} {e}\n")

        self.read_thermal_properties()

        for i, pressure in tqdm(enumerate(self.qha_pressures)):
            task_dir = f"task_{i + 1:02d}_pressure_{pressure:.2f}"
            task_cwd = os.path.join(self.work_dir, task_dir)
            os.makedirs(task_cwd, exist_ok=True)
            qha = PhonopyQHA(
                free_energy=self.free_energy.T,
                volumes=self.volumes,
                temperatures=self.temperatures,
                t_max=self.tmax,
                cv=self.cv.T,
                entropy=self.entropy.T,
                electronic_energies=self.displaced_energies,
                pressure=pressure,
            )
            qha.plot_pdf_gibbs_temperature()
            qha.write_thermal_expansion(
                filename=os.path.join(task_cwd, "thermal_expansion.dat")
            )
            qha.write_heat_capacity_P_numerical(
                filename=os.path.join(task_cwd, "heat_capacity_P_numerical.dat")
            )
            qha.write_heat_capacity_P_polyfit(
                filename=os.path.join(task_cwd, "heat_capacity_P_polyfit.dat")
            )
            qha.plot_thermal_expansion().savefig(
                os.path.join(task_cwd, "thermal_expansion.png"), dpi=600
            )
            gibbs_energies = qha.get_gibbs_temperature()
            total_gibbs_file = os.path.join(
                task_cwd, f"gibbs-temperature.dat-pressure-{pressure:.2f}"
            )
            with open(total_gibbs_file, "w") as f:
                for temp, gibbs in zip(self.temperatures, gibbs_energies):
                    f.write(f"{temp} {gibbs}\n")

            gibbs_file_per_atom = os.path.join(
                task_cwd, f"gibbs_per_atom-temperature.dat-pressure-{pressure:.2f}.csv"
            )
            fieldnames = [str(temp) for temp in self.temperatures]
            with open(gibbs_file_per_atom, "w", newline="") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                data_row = {
                    str(temp): gibbs / self.atom_nums
                    for temp, gibbs in zip(self.temperatures, gibbs_energies)
                }
                writer.writerow(data_row)

    def run(self):
        print(f"Working directory: {self.work_dir}")
        os.makedirs(self.work_dir, exist_ok=True)
        os.chdir(self.work_dir)

        nrep_second, k_points_mesh = get_supercell_parameters(
            self.atoms, self.supercell_matrix, self.qpoints_mesh, self.max_atoms
        )
        deformations = np.arange(
            self.lower_deformation,
            self.upper_deformation + self.deformation_step,
            self.deformation_step,
        )
        displaced_energies, volumes = [], []
        for frac in tqdm(deformations):
            self.save_fc2 = (
                True
                if frac == (self.lower_deformation + self.upper_deformation) // 2
                else False
            )
            # deep copy of atoms object
            atoms = self.atoms.copy()
            atoms.calc = self.atoms.calc
            # apply deformation
            atoms.set_cell(atoms.get_cell() * (1 + frac * 0.01), scale_atoms=True)
            # calculate energy and volume
            displaced_energies.append(atoms.get_potential_energy())
            volumes.append(atoms.get_volume())
            # compute force constants
            phonon = self.compute_force_constants(atoms, nrep_second)
            phonon_work_dir = os.path.join(self.work_dir, f"phonon-{frac}")
            os.makedirs(phonon_work_dir, exist_ok=True)
            os.chdir(phonon_work_dir)
            self.compute_phonon_spectrum_dos(atoms, phonon, k_points_mesh)
            os.chdir(self.work_dir)
            self.compute_thermal_properties(phonon, frac)
        self.displaced_energies = np.array(displaced_energies)
        self.volumes = np.array(volumes)
        self.compute_qha()


if __name__ == "__main__":
    mp_id_list = ["mp-1195001","mp-602"]
    from mattersim.utils.atoms_utils import AtomsAdaptor
    from mattersim.forcefield.potential import DeepCalculator, Potential
    model_path = "/blob/model/m3gnet.pth"
    potential = Potential.load(load_path=model_path, device="cuda:0")
    calculator = DeepCalculator(potential, stress_weight=1 / 160.21766208)
    for mp_id in mp_id_list:
        atoms = AtomsAdaptor.from_mp_id(mp_id)
        work_dir = f"/tmp/qha/{mp_id}"
        atoms.calc = calculator
        relaxed_atoms = atoms.copy()
        relaxed_atoms.calc = calculator
        from mattersim.applications.relax import Relaxer

        relaxer = Relaxer()
        converged, relaxed_atoms = relaxer.relax_structures(
            relaxed_atoms, constrain_symmetry=True, filter="ExpCellFilter"
        )
        qha = PhonopyQHAWorkflow(
            atoms=relaxed_atoms,
            qpoints_mesh=None,
            tmin=0,
            tmax=1000,
            tstep=10,
            lower_deformation=-5,
            upper_deformation=5,
            deformation_step=1,
            work_dir=work_dir,
        )
        qha.run()

