# -*- coding: utf-8 -*-
import datetime
import os
from glob import glob
from typing import Iterable, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from ase import Atoms
from phono3py import Phono3py
from phono3py.file_IO import write_fc2_to_hdf5, write_fc3_to_hdf5
from tqdm import tqdm

# from mattersim.utils.kpoints_utils import get_kpoints_mesh
from mattersim.utils.phonon_utils import (
    get_primitive_cell,
    to_ase_atoms,
    to_phonopy_atoms,
)
from mattersim.utils.supercell_utils import get_supercell_parameters


class BTEWorkflowError(Exception):
    "Custom exception class for errors in the bte workflow"
    pass


class BTEWorkflow(object):
    """
    This class is used to calculate thermal conductivity of materials using phono3py
    """

    SUPPORTED_BTE_METHOD = ["RTA", "LBTE"]

    def __init__(
        self,
        atoms: Atoms,
        find_prim: bool = False,
        work_dir: str = None,
        is_symmetry: bool = True,
        symprec: float = 1e-5,
        is_mesh_symmetry: bool = True,
        amplitude: float = 0.01,
        supercell_matrix: np.ndarray = None,
        supercell_matrix_phonon: np.ndarray = None,
        qpoints_mesh: np.array = None,
        qspacing: float = None,
        qdensity: float = None,
        max_atoms: int = None,
        nac_params: dict = None,
        method: str = "RTA",
        is_isotope: bool = False,
        tmin: float = 50,
        tmax: float = 500,
        tstep: float = 50,
        save_fcs: bool = True,
    ):
        """
        Args:
            atoms (Atoms): ASE atoms object contains structure information and
                calculator.
            work_dir (str, optional): workplace path to contain phonon result.
                Defaults to data + chemical_symbols + 'bte'.
            is_symmetry (bool, optional): Whether use symmetry. Default to True.
            symprec (float, optional): Symmetry precision. Defaults to 1e-5.
            is_mesh_symmetry (bool, optional): Whether use symmetry in reciprocal
                space. Default to True.
            find_prim (bool, optional): If find the primitive cell and use it
                to calculate phonon. Default to False.
            amplitude (float, optional): Magnitude of the finite difference to
                displace in force constant calculation, in Angstrom. Defaults
                to 0.01 Angstrom.
            max_atoms (int, optional): Maximum atoms number limitation for the
                supercell generation. if not ser, will automatic generate supe
                -rcell based on symmetry.
            nac_param (dict, optional): A dictionary which contains born effective
                charge, dielectric matrix and unit conversion factor. Defaults to
                None. The structure is as this: {'born': born, 'factor': factor,
                'dielectric': epsilon}.
            supercell_matrix (np.ndarray, optional): Supercell matrix for constr
                -uct supercell to calculate 3rd fcs, prioriry over than max_atoms.
                if supercell_matrix_phonon not given, it will also be used for
                calcuating 2nd fcs.
            supercell_matrix_phonon (np.ndarray, optional): Supercell matrix for
                calculating 2nd fcs. If not given but supercell_matrix is assigned,
                it will be set equal to supercell_matrix.
            qpoints_mesh (nd.array, optional): Number of qpoints in three lattice
                directions. Defaults to None.
            qspacing (float, optional): Qponts spacing in reciprocal space axis,
                in 2 * pi/Angstrom. Default to None.
            qdensity (float, optional): Qpoints density in reciprocal space axis,
                the relation to qspacing is : qdensity = 1 / qspacing, this
                variable is mainly for phonopy and phono3py setting.
            method (str, optional): The method to solve BTE equation. Either
                rta or lbte. Defaults to rta.
            is_isotope (bool, optional): With or without isotope scattering.
                Defaults to False.
            tmin (float, optional): Minimum temperature for solver BTE equation
                , in Kelvin. Defatuls to 50 K.
            tmax (float, optional): Maximum temperature for solver BTE equation
                , in Kelvin. Defaults to 500 K.
            tstep (float, optional): Temperature interval, the temperature will
                increase from tmin to tmax, each step is tstep, in Kelvin.
                Defaults to 50 K.
            save_fcs (bool, optional): Save second and third force constants.
                Defaults to True.
        """
        assert (
            atoms.calc is not None
        ), "BTEWorkflow only accepts ase atoms with an attached calculator"
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
            self.work_dir = f"{formatted_datetime}-{atoms.get_chemical_formula()}-bte"

        self.amplitude = amplitude
        self.is_symmetry = is_symmetry
        self.is_mesh_symmetry = is_mesh_symmetry
        self.symprec = symprec

        if supercell_matrix is not None:
            if supercell_matrix.shape == (3, 3):
                self.supercell_matrix = supercell_matrix
            elif supercell_matrix.shape == (3,):
                self.supercell_matrix = np.diag(supercell_matrix)
            else:
                assert (
                    False
                ), "supercell_matrix must be an array (3,1) or a matrix (3,3)."
        else:
            self.supercell_matrix = supercell_matrix

        if supercell_matrix_phonon is not None:
            if supercell_matrix is None:
                assert False, (
                    "Must provide supercell_matrix if supercell_matrix" " is assigned"
                )
            else:
                if supercell_matrix_phonon.shape == (3, 3):
                    self.supercell_matrix_phonon = supercell_matrix_phonon
                elif supercell_matrix.shape == (3,):
                    self.supercell_matrix_phonon = np.diag(supercell_matrix_phonon)
                else:
                    assert False, (
                        "supercell_matrix_phonon must be an array (3,1) or "
                        "a matrix (3,3)."
                    )
        else:
            self.supercell_matrix_phonon = supercell_matrix

        if qpoints_mesh is not None:
            assert qpoints_mesh.shape == (3,), "Qpoints mesh must be an array (3,1)."
            self.qpoints_mesh = qpoints_mesh
        else:
            self.qpoints_mesh = qpoints_mesh

        self.qspacing = qspacing
        self.qdensity = qdensity

        self.max_atoms = max_atoms

        self.nac_params = nac_params

        assert (
            method.upper() in self.SUPPORTED_BTE_METHOD
        ), "BTEWorkflow.method only support 'RTA' or 'LBTE'."
        self.method = method.upper()
        self.is_isotope = is_isotope
        self.tmin = tmin
        self.tmax = tmax
        self.tstep = tstep

        self.save_fcs = save_fcs

    def compute_force_constants(
        self,
        atoms: Atoms,
        nrep_second: np.ndarray,
        nrep_third: np.ndarray,
    ):
        """
        Calculate second and third order force constants

        Args:
            atoms (Atoms): ASE atoms object to provide lattice informations.
            nrep_second (np.ndarray): Supcercell size used for 2nd force
                constants calculations
            nrep_third (np.ndarray): Supcercell size used for 3rd force
                constants calculations
        """
        print(f"Supercell matrix for 2nd force constants : \n{nrep_second}")
        print(f"Supercell matrix for 3rd force constants : \n{nrep_third}")
        # Generage phono3py object
        phonon3 = Phono3py(
            to_phonopy_atoms(atoms),
            supercell_matrix=nrep_third,
            primitive_matrix="auto",
            phonon_supercell_matrix=nrep_second,
            is_symmetry=self.is_symmetry,
            symprec=self.symprec,
            is_mesh_symmetry=self.is_mesh_symmetry,
            log_level=2,
        )
        # phonon3.nac_params = self.nac_params,

        # Generate displacements
        phonon3.generate_displacements(distance=self.amplitude)

        # Compute 2nd force constants
        second_scs = phonon3.phonon_supercells_with_displacements
        second_force_sets = []
        print("\n")
        print("Inferring forces for displaced atoms and computing second order fcs ...")
        for disp_second in tqdm(second_scs):
            pa_second = to_ase_atoms(disp_second)
            pa_second.calc = self.atoms.calc
            second_force_sets.append(pa_second.get_forces())

        phonon3.phonon_forces = second_force_sets
        phonon3.produce_fc2(symmetrize_fc2=True)

        # Compute 3rd force constants
        third_scs = phonon3.supercells_with_displacements
        third_force_sets = []
        print("\n")
        print("Inferring forces for displaced atoms and computing third order fcs ...")
        for disp_third in tqdm(third_scs):
            pa_third = to_ase_atoms(disp_third)
            pa_third.calc = self.atoms.calc
            third_force_sets.append(pa_third.get_forces())

        phonon3.forces = third_force_sets
        phonon3.produce_fc3(symmetrize_fc3r=True)

        # Save to file
        if self.save_fcs:
            write_fc2_to_hdf5(
                phonon3.fc2,
                p2s_map=phonon3.phonon_primitive.p2s_map,
                physical_unit="eV/angstrom^2",
            )
            write_fc3_to_hdf5(phonon3.fc3, p2s_map=phonon3.phonon_primitive.p2s_map)

        phonon3.save(settings={"force_sets": True, "force_constants": self.save_fcs})

        return phonon3

    def compute_kappa_bte(
        self, phonon3: Phono3py, k_point_mesh: Union[int, Iterable[int], float]
    ):
        """
        Calculate thermal conductivity based on BTE (Boltzmann Transport Equation)

        Args:
            phonon3 (Phono3py): Phono3py object which contains 3rd force
                constants matrix
            k_point_mesh (Union[int, Iterable[int], float]): The qpoints number
                in First Brillouin Zone in three directions for integration.
        """
        # Set temperature points
        temperatures = np.arange(self.tmin, self.tmax, self.tstep)

        # Assign qpoints mesh for ph-ph interaction
        print(f"Qpoints mesh (density) for ph-ph interaction : {k_point_mesh}")
        phonon3.mesh_numbers = k_point_mesh

        # Calculate ph-ph interaction
        phonon3.init_phph_interaction()

        # Run BTE calculation
        if self.method == "RTA":
            phonon3.run_thermal_conductivity(
                temperatures=temperatures, is_isotope=self.is_isotope, write_kappa=True
            )
        elif self.method == "LBTE":
            phonon3.run_thermal_conductivity(
                temperatures=temperatures,
                is_LBTE=True,
                is_isotope=self.is_isotope,
                write_kappa=True,
            )

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
                nrep_third, k_points_mesh = get_supercell_parameters(
                    self.atoms, self.supercell_matrix, self.qpoints_mesh, self.max_atoms
                )
            except Exception as e:
                raise BTEWorkflowError(
                    "Error while generating supercell parameters:"
                ) from e

            try:
                if self.supercell_matrix_phonon is not None:
                    nrep_second = self.supercell_matrix_phonon
                else:
                    nrep_second = nrep_third

                # Calculate 3rd force constants
                ph3 = self.compute_force_constants(self.atoms, nrep_second, nrep_third)
            except Exception as e:
                raise BTEWorkflowError("Error while computing force constants:") from e

            try:
                # Calculate thermal conductivity
                self.compute_kappa_bte(ph3, k_points_mesh)
            except Exception as e:
                raise BTEWorkflowError(
                    "Error while computing thermal conductivity:"
                ) from e

        except Exception as e:
            raise BTEWorkflowError("An error occurred during the BTE workflow:") from e

        finally:
            os.chdir(current_path)

        return self.work_dir, ph3

    @staticmethod
    def summarize_hdf5(filename: str):
        """
        Summarizes the thermal properties data stored in an hdf5 file.
        Including temperatures, thermal conductivity, mode_kappa, phonon
        frequencies, group velocities, phonon lifetime.

        Args:
            filename (str): File name of hdf5 file contains data.

        Returns:
        tuple: A tuple containing the following elements extracted and
            processed from the HDF5 file:
        - temperatures (numpy.ndarray): An array of temperatures at
            which properties are calculated.
        - kappa (numpy.ndarray): An array of the thermal conductivity
            tensor at different temperatures.
        - frequencies (numpy.ndarray): A flattened array of phonon
            frequencies.
        - group_velocity (numpy.ndarray): An array of group velocities
            for phonons.
        - lifetimes (numpy.ndarray): An array of phonon lifetimes calculated
            from the imaginary part of the phonon self-energy.
        - mode_kappa (numpy.ndarray): An array of mode-resolved thermal
            conductivities.
        - weight (numpy.ndarray): An array of weighting factors for phonon modes.
        """
        import h5py

        try:
            f = h5py.File(filename, "r")
        except Exception as e:
            raise BTEWorkflowError(f"Can not open file: {filename}") from e

        # Read data
        temperatures = f["temperature"][:]
        n_temperatures = len(temperatures)
        kappa = f["kappa"][:]
        mode_kappa = f["mode_kappa"][:]
        weight = f["weight"][:]
        frequencies = f["frequency"][:].flatten(order="C")
        group_velocity = f["group_velocity"][:]
        gamma = f["gamma"][:].reshape(n_temperatures, -1)

        # Calculate phonon lifetime
        lifetimes = np.where(gamma > 0, 1.0 / (2.0 * 2.0 * np.pi * gamma), 0)

        f.close()

        return (
            temperatures,
            kappa,
            frequencies,
            group_velocity,
            lifetimes,
            mode_kappa,
            weight,
        )

    @staticmethod
    def compute_per_mode_kappa(
        mode_kappa_at_T: np.ndarray, weight: np.ndarray, is_isotropic: bool = True
    ):
        """
        Compute the per mode thermal conductivity at a specific temperature,
        either as isotropic averages or for individual directions.

        Args:
            mode_kappa_at_T (np.ndarray): A 3D array contains the model thermal
                conductivities for each direction at a specific temperature, in
                W/(m·K). The dimensions of the array are expected to be [n_bands,
                n_qpoints, n_directions]
            weight (np.ndarray): A 1D array contains the weight factors for each
                modes. The sum of weights is used for normalization.
            is_isotropic (bool, optional): Flag indicating whether the thermal
                conductivity is averaged. Defaults to True.
        """

        # Compute mode kappa along each direction
        kappa_xx = mode_kappa_at_T[:, :, 0] / weight.sum()
        kappa_yy = mode_kappa_at_T[:, :, 1] / weight.sum()
        kappa_zz = mode_kappa_at_T[:, :, 2] / weight.sum()

        # Compute averages base on isotropic average
        if is_isotropic:
            kappa_average_at_T = (kappa_xx + kappa_yy + kappa_zz) / 3.0
            kappa_average_at_T_1d = kappa_average_at_T.flatten(order="C")
            return kappa_average_at_T_1d
        else:
            return kappa_xx, kappa_yy, kappa_zz

    @staticmethod
    def compute_mean_free_path(
        group_velocity_matrix: np.ndarray,
        life_time_arr: np.ndarray,
        is_return_norm: bool = True,
    ):
        """
        Calculate the mean free path for phonons given the group velocities
        and lifetime.

        Args:
            group_velocity_matrix (np.ndarray): A 3D array of shape (nbands,
                nqpoints, n_directions) contains the group velocity of phonons.
            life_time_arr (np.ndarray): A 1D array (nmodes) contains phonon
                lifetimes.
            is_return_norm (bool, optional): Flag indicate whether to return
                the norm of the mean free path vectors of the individual
                components for each direcion. Defaults to True.
        """
        # Calculate mean free path along each direction
        mean_free_path_matrix_xx = life_time_arr * group_velocity_matrix[
            :, :, 0
        ].flatten(order="C")
        mean_free_path_matrix_yy = life_time_arr * group_velocity_matrix[
            :, :, 1
        ].flatten(order="C")
        mean_free_path_matrix_zz = life_time_arr * group_velocity_matrix[
            :, :, 2
        ].flatten(order="C")

        # Stack the mean free path components for all directions
        mean_free_path_matrix = np.vstack(
            [
                mean_free_path_matrix_xx,
                mean_free_path_matrix_yy,
                mean_free_path_matrix_zz,
            ]
        ).T

        # Return the norm of the mean free path vector or the components
        if is_return_norm:
            # Calculate the norm of the mean free path vector
            return np.linalg.norm(mean_free_path_matrix, axis=-1)
        else:
            # Return the mean free path components in each direction
            return mean_free_path_matrix

    @staticmethod
    def set_fig_properties(ax_list):
        """
        Sets the tick properties of the axes provided in the list.
        For each Matplotlib Axes object in the provided list, this
        function configures the appearance of the major and minor
        ticks as well as their direction, ensuring a consistent look
        and feel across multiple plots. Ticks are set to be inside
        the plot area on both the left/right and top/bottom axes.

        Parameters:
        ax_list (list): A list of Matplotlib Axes objects whose tick
            properties are to be set.
        tl = 6
        tw = 2
        tlm = 4
        """

        tl = 6
        tw = 2
        tlm = 4

        for ax in ax_list:
            ax.tick_params(which="major", length=tl, width=tw)
            ax.tick_params(which="minor", length=tlm, width=tw)
            ax.tick_params(
                which="both", axis="both", direction="in", right=True, top=True
            )

    @staticmethod
    def get_kappa(
        work_dir: str = None, filename: str = None, temperature_index: int = None
    ):
        """
        Read the kappa-*.hdf5 file, then output thermal conductivity
        and plot figures.

        Args:
            work_dir (str, optional): Work directory which contains .hdf5
                files. If None, set to current path.
            filename (str, optional): File name contains phono3py output
                results. Defaults to None, which means the function will
                automatically find the one file prefix is kappa and
                suffix is .hdf5.
            temperature_index (int, optional): temperature index indate
                which temperature you want to extract. If None, will
                automatically find a temperature nearest to 300K.
        """

        current_path = os.getcwd()

        if not work_dir:
            work_dir = os.getcwd()

        os.chdir(work_dir)

        try:
            # Find hdf5 file in the current directory
            if filename is not None:
                hdf5_files = filename
            else:
                hdf5_files = glob("kappa-*.hdf5")
                if len(hdf5_files) > 0:
                    hdf5_path = hdf5_files[0]
                    if len(hdf5_files) > 1:
                        print("You have more than one file named kappa-*.hdf5")
                        print(f"Total is {len(hdf5_files)}\n")
                        print(f"Only handle the first {hdf5_path}")
                else:
                    raise ValueError(
                        f"No kappa-*.hdf5 file found! Current path {os.getcwd()}"
                    )

            # Define format of plotting
            aw = 2
            fs = 12
            font = {"size": fs}
            matplotlib.rc("font", **font)
            matplotlib.rc("axes", linewidth=aw)

            # Configure Matplotlib to use a LaTeX-like style without LaTeX
            plt.rcParams["text.usetex"] = False
            plt.rcParams["font.family"] = "serif"
            plt.rcParams["mathtext.fontset"] = "cm"

            # Access data of BTE calculations
            (
                temperatures,
                kappas,
                frequencies,
                group_velocity,
                lifetime,
                mode_kappa,
                weight,
            ) = BTEWorkflow.summarize_hdf5(hdf5_path)

            kappa_data_file = open("kappa_summarize.dat", "w")
            kappa_data_file.write(
                "----------- Thermal conductivity (W/m-k) with tetrahedron method"
                " -----------\n"
            )
            kappa_data_file.write(
                "#  T(K)        xx         yy         zz         yz         xz"
                "         xy        \n"
            )

            for temperature, kappa in zip(temperatures, kappas):
                kappa_data_file.write(f"{temperature:7.1f}")
                for kappa1 in kappa:
                    kappa_data_file.write(f"{kappa1:10.3f} ")
                kappa_data_file.write("\n")

            # Find the temperature index
            if temperature_index is not None:
                pass
            else:
                temperature_index = min(
                    range(len(temperatures)), key=lambda i: abs(temperatures[i] - 300)
                )

            # Compute per mode kappa at temperature index
            per_mode_kappa = BTEWorkflow.compute_per_mode_kappa(
                mode_kappa[temperature_index, :, :, :], weight, is_isotropic=True
            )
            kappa_data_file.write(
                "Thermal conductivity of this material at"
                f" {temperatures[temperature_index]} K is : {per_mode_kappa.sum()}."
            )
            kappa_data_file.close()

            argsort_indices_by_freq = np.argsort(frequencies)
            kappa_cum_wrt_freq = np.cumsum(per_mode_kappa[argsort_indices_by_freq])

            # Compute Mean Free Path at temperature index
            mfp = BTEWorkflow.compute_mean_free_path(
                group_velocity, lifetime[temperature_index, :], is_return_norm=True
            )
            argsort_indices_by_lambda = np.argsort(mfp)
            kappa_cum_wrt_lambda = np.cumsum(per_mode_kappa[argsort_indices_by_lambda])

            group_velocity_norm = (
                np.linalg.norm(group_velocity.reshape(-1, 3), axis=1) / 10.0
            )

            # Plot norm of group velocity, lifetime (or scattering rate) and
            # mean free path as a function of frequency and perform 1-to-1
            # comparison
            plt.figure(figsize=(19.2, 4.8))
            plt.subplot(1, 3, 1)
            BTEWorkflow.set_fig_properties([plt.gca()])
            plt.scatter(
                frequencies,
                group_velocity_norm,
                s=10,
                facecolor="w",
                edgecolor="b",
                marker="^",
                label="$|v|_{Mattersim}$",
            )
            plt.xlabel(r"$\omega$ (THz)", fontsize=24)
            plt.ylabel(r"${v} \ (\frac{km}{s})$", fontsize=24)
            plt.legend(loc="best", fontsize=18)

            plt.subplot(1, 3, 2)
            BTEWorkflow.set_fig_properties([plt.gca()])
            plt.scatter(
                frequencies,
                lifetime[temperature_index, :],
                s=10,
                facecolor="w",
                edgecolors="b",
                marker="s",
                label=r"$\tau_{MatterSim}$ ",
            )
            plt.yscale("log")
            plt.xlabel(r"$\omega$ (THz)", fontsize=24)
            plt.ylabel(r"$\tau \ (ps)$", fontsize=24)
            plt.legend(loc="best", fontsize=18)

            plt.subplot(1, 3, 3)
            BTEWorkflow.set_fig_properties([plt.gca()])
            plt.scatter(
                frequencies,
                mfp / 10.0,
                s=10,
                facecolor="w",
                edgecolor="b",
                marker="<",
                label=r"$\lambda_{MatterSim}$",
            )
            plt.xlabel(r"$\omega$ (THz)", fontsize=24)
            plt.ylabel(r"$\lambda \ (nm)$", fontsize=24)
            plt.legend(loc="best", fontsize=18)
            plt.yscale("log")
            plt.subplots_adjust(wspace=0.3)

            # save the figure
            plt.savefig("gv_lifetime_MFP.png", dpi=300)

            # Plot cumulative kappa versus frequencies, lambda and plot
            # kappa per mode versus frequencies
            plt.figure(figsize=(19.2, 4.8))
            plt.subplot(1, 3, 1)
            BTEWorkflow.set_fig_properties([plt.gca()])
            plt.plot(
                frequencies[argsort_indices_by_freq],
                kappa_cum_wrt_freq,
                "b-",
                lw=2,
                label=r"$\kappa_{MatterSim, cumulative, \omega}$",
            )
            plt.xlabel(r"$\omega$ (THz)", fontsize=24)
            plt.ylabel(
                r"$\kappa_{cumulative, \omega} \ (W \  m^{-1}K^{-1})$", fontsize=24
            )
            plt.legend(loc="best", frameon=True, fontsize=18)

            plt.subplot(1, 3, 2)
            BTEWorkflow.set_fig_properties([plt.gca()])
            # add the x scale for best figure plot
            mfp_rescale = []
            kappa_cum_lambda_rescale = []
            for i, j in zip(mfp[argsort_indices_by_lambda], kappa_cum_wrt_lambda):
                if j > 1e-10 and j != kappa_cum_wrt_lambda[-1]:
                    mfp_rescale.append(i)
                    kappa_cum_lambda_rescale.append(j)

            mfp_rescale.insert(0, mfp_rescale[0] * 1e-1)
            mfp_rescale.insert(len(mfp_rescale), mfp_rescale[-1] * 1e1)
            kappa_cum_lambda_rescale.insert(0, 0)
            kappa_cum_lambda_rescale.insert(
                len(kappa_cum_lambda_rescale), kappa_cum_lambda_rescale[-1]
            )

            plt.plot(
                mfp_rescale,
                kappa_cum_lambda_rescale,
                "b-",
                lw=2,
                label=r"$\kappa_{MatterSim, cumulative, \lambda}$",
            )
            plt.ylabel(
                r"$\kappa_{cumulative, \lambda} \ (W \  m^{-1}K^{-1})$", fontsize=24
            )
            plt.xlabel(r"$\lambda \ (nm)$", fontsize=24)
            plt.xscale("log")
            plt.legend(loc="best", frameon=True, fontsize=18)

            plt.subplot(1, 3, 3)
            BTEWorkflow.set_fig_properties([plt.gca()])
            plt.scatter(
                frequencies,
                per_mode_kappa,
                s=10,
                facecolor="w",
                edgecolor="b",
                marker="o",
                label=r"$\kappa_{MatterSim, per \ mode}$",
            )
            plt.xlabel(r"$\omega$ (THz)", fontsize=24)
            plt.ylabel(r"$\kappa_{per \ mode} \ (W \  m^{-1}K^{-1})$", fontsize=24)
            plt.legend(loc="best", frameon=True, fontsize=18)
            plt.subplots_adjust(wspace=0.35)

            # save the figure
            plt.savefig("cumulatvie_kappa.png", dpi=300)

        except Exception as e:
            raise BTEWorkflowError("An error occurred during postprocess:") from e

        finally:
            os.chdir(current_path)

