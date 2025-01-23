# -*- coding: utf-8 -*-
import datetime
import os
from typing import Iterable, Union

import numpy as np
from ase import Atoms
from phonopy import Phonopy
from tqdm import tqdm

from mattersim.utils.phonon_utils import (
    get_primitive_cell,
    to_ase_atoms,
    to_phonopy_atoms,
)
from mattersim.utils.supercell_utils import get_supercell_parameters


class PhononWorkflow(object):
    """
    This class is used to calculate the phonon dispersion relationship of
    material using phonopy
    """

    def __init__(
        self,
        atoms: Atoms,
        find_prim: bool = False,
        work_dir: str = None,
        amplitude: float = 0.01,
        supercell_matrix: np.ndarray = None,
        qpoints_mesh: np.ndarray = None,
        max_atoms: int = None,
        calc_spec: bool = True,
    ):
        """_summary

        Args:
            atoms (Atoms): ASE atoms object contains structure information and
                calculator.
            find_prim (bool, optional): If find the primitive cell and use it
                to calculate phonon. Default to False.
            work_dir (str, optional): workplace path to contain phonon result.
                Defaults to data + chemical_symbols + 'phonon'
            amplitude (float, optional): Magnitude of the finite difference to
                displace in force constant calculation, in Angstrom. Defaults
                to 0.01 Angstrom.
            supercell_matrix (nd.array, optional): Supercell matrix for constr
                -uct supercell, priority over than max_atoms. Defaults to None.
            qpoints_mesh (nd.array, optional): Qpoint mesh for IBZ integral,
                priority over than max_atoms. Defaults to None.
            max_atoms (int, optional): Maximum atoms number limitation for the
                supercell generation. If not set, will automatic generate super
                -cell based on symmetry. Defaults to None.
            calc_spec (bool, optional): If calculate the spectrum and check 
                imaginary frequencies. Default to True.
        """
        assert (
            atoms.calc is not None
        ), "PhononWorkflow only accepts ase atoms with an attached calculator"
        if find_prim:
            self.atoms = get_primitive_cell(atoms)
            self.atoms.calc = atoms.calc
        else:
            self.atoms = atoms
        if work_dir is not None:
            self.work_dir = work_dir
        else:
            current_datetime = datetime.datetime.now()
            formatted_datetime = current_datetime.strftime("%Y-%m-%d-%H-%M")
            self.work_dir = (
                f"{formatted_datetime}-{atoms.get_chemical_formula()}-phonon"
            )
        self.amplitude = amplitude
        if supercell_matrix is not None:
            if supercell_matrix.shape == (3, 3):
                self.supercell_matrix = supercell_matrix
            elif supercell_matrix.shape == (3,):
                self.supercell_matrix = np.diag(supercell_matrix)
            else:
                assert (
                    False
                ), "Supercell matrix must be an array (3,1) or a matrix (3,3)."
        else:
            self.supercell_matrix = supercell_matrix

        if qpoints_mesh is not None:
            assert qpoints_mesh.shape == (3,), "Qpoints mesh must be an array (3,1)."
            self.qpoints_mesh = qpoints_mesh
        else:
            self.qpoints_mesh = qpoints_mesh

        self.max_atoms = max_atoms
        self.calc_spec = calc_spec

    def compute_force_constants(self, atoms: Atoms, nrep_second: np.ndarray):
        """
        Calculate force constants

        Args:
            atoms (Atoms): ASE atoms object to provide lattice informations.
            nrep_second (np.ndarray): Supercell size used for 2nd force
                constant calculations.
        """
        print(f"Supercell matrix for 2nd force constants : \n{nrep_second}")
        # Generate phonopy object
        phonon = Phonopy(
            to_phonopy_atoms(atoms),
            supercell_matrix=nrep_second,
            primitive_matrix="auto",
            log_level=2,
        )

        # Generate displacements
        phonon.generate_displacements(distance=self.amplitude)

        # Compute force constants
        second_scs = phonon.supercells_with_displacements
        second_force_sets = []
        print("\n")
        print("Inferring forces for displaced atoms and computing fcs ...")
        for disp_second in tqdm(second_scs):
            pa_second = to_ase_atoms(disp_second)
            pa_second.calc = self.atoms.calc
            second_force_sets.append(pa_second.get_forces())

        phonon.forces = np.array(second_force_sets)
        phonon.produce_force_constants()
        phonon.symmetrize_force_constants()

        return phonon

    @staticmethod
    def compute_phonon_spectrum_dos(
        atoms: Atoms, phonon: Phonopy, k_point_mesh: Union[int, Iterable[int]]
    ):
        """
        Calculate phonon spectrum and DOS based on force constant matrix in
        phonon object

        Args:
            atoms (Atoms): ASE atoms object to provide lattice information
            phonon (Phonopy): Phonopy object which contains force constants matrix
            k_point_mesh (Union[int, Iterable[int]]): The qpoints number in First
                Brillouin Zone in three directions for DOS calculation.
        """
        print(f"Qpoints mesh for Brillouin Zone integration : {k_point_mesh}")
        phonon.run_mesh(k_point_mesh)
        print(
            "Dispersion relations using phonopy for "
            + str(atoms.symbols)
            + " ..."
            + "\n"
        )

        # plot phonon spectrum
        phonon.auto_band_structure(plot=True, write_yaml=True, with_eigenvectors=True).savefig(
            f"{str(atoms.symbols)}_phonon_band.png", dpi=300
        )
        phonon.auto_total_dos(plot=True, write_dat=True).savefig(
            f"{str(atoms.symbols)}_phonon_dos.png", dpi=300
        )

        # Save additional files
        phonon.save(settings={"force_constants": True})

    @staticmethod
    def check_imaginary_freq(phonon: Phonopy):
        """
        Check whether phonon has imaginary frequency.

        Args:
            phonon (Phonopy): Phonopy object which contains phonon spectrum frequency.
        """
        band_dict = phonon.get_band_structure_dict()
        frequencies = np.concatenate(
            [np.array(freq).flatten() for freq in band_dict["frequencies"]], axis=None
        )
        has_imaginary = False
        if np.all(np.array(frequencies) >= -0.299):
            pass
        else:
            print("Warning! Imaginary frequencies found!")
            has_imaginary = True

        return has_imaginary

    def run(self):
        """
        The entrypoint to start the workflow.
        """
        current_path = os.path.abspath(".")
        try:
            # check folder exists
            if not os.path.exists(self.work_dir):
                os.makedirs(self.work_dir)

            os.chdir(self.work_dir)

            try:
                # Generate supercell parameters based on optimized structures
                nrep_second, k_point_mesh = get_supercell_parameters(
                    self.atoms, self.supercell_matrix, self.qpoints_mesh, self.max_atoms
                )
            except Exception as e:
                print("Error whille generating supercell parameters:", e)
                raise

            try:
                # Calculate 2nd force constants
                phonon = self.compute_force_constants(self.atoms, nrep_second)
            except Exception as e:
                print("Error while computing force constants:", e)
                raise

            if self.calc_spec:
                try:
                    # Calculate phonon spectrum
                    self.compute_phonon_spectrum_dos(self.atoms, phonon, k_point_mesh)
                    # check whether has imaginary frequency
                    has_imaginary = self.check_imaginary_freq(phonon)
                except Exception as e:
                    print("Error while computing phonon spectrum and dos:", e)
                    raise
            else:
                has_imaginary = 'Not calculated, set calc_spec True'
                phonon.save(settings={"force_constants": True})

        except Exception as e:
            print("An error occurred during the Phonon workflow:", e)
            raise

        finally:
            os.chdir(current_path)

        return has_imaginary, phonon
