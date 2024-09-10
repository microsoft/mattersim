# -*- coding: utf-8 -*-
import numpy as np
from ase import Atoms
from ase.spacegroup.symmetrize import check_symmetry


def auto_grid_detection(
    atom: Atoms,
    max_atoms: int,
    ratio_tolerance: float = 1.1,
    is_santity_check: bool = True,
    is_verbose: bool = True,
):
    """
    This function automates the detection of grid for a given atomic structure
    and max_atoms. If lattice vectors lenght in three direction is same or the
    difference is smaller than 0.1, the supercell vector element will has the
    same value in three direction, the vaule is (max_atoms/atoms)^(1/3). Else
    the supercell vector element will be set proportionally to make the three
    supercell lattice vector length as same as possible.

    Args:
        atom (Atoms): ASE atoms object to provide lattice informations.
        max_atoms: (int): Maximum atom number limitation for supercell.
        ratio_tolerance (float, optional): The tolerance for the ratio of the
            lengths of the lattice vectors. Defaults to 1.1.
        is_santity_check (bool, optional): If True, performs a sanity check to
            ensure symmetry is preserved after replications. Defaults to True.
        is_verbose (bool, optional): If True, prints detailed information about
            the atomic structure and the replication process. Defaults to True.
    """
    # Get the cell length
    lattice_vector_lengths = atom.cell.cellpar()[:3]

    # Base case, absolute the same length
    if (
        lattice_vector_lengths[0]
        == lattice_vector_lengths[1]
        == lattice_vector_lengths[2]
    ):
        number_of_replicas = int(np.round(max_atoms / (len(atom))) ** (1 / 3))
        number_of_replicas = max(number_of_replicas, 1)
        max_replication = (
            number_of_replicas,
            number_of_replicas,
            number_of_replicas,
        )

    # Case 1: Non-symmetry lengths within tolerance
    else:
        lattice_vector_lengths_argsort_indices = np.argsort(lattice_vector_lengths)[
            ::-1
        ]
        sorted_lattice_vector_lengths = lattice_vector_lengths[
            lattice_vector_lengths_argsort_indices
        ]
        ratios = [
            sorted_lattice_vector_lengths[0] / sorted_lattice_vector_lengths[1],
            sorted_lattice_vector_lengths[0] / sorted_lattice_vector_lengths[2],
        ]

        # Variation in cell length with in the tolerance we still consider it
        # as an N-N-N replications
        if ratios[0] <= ratio_tolerance and ratios[1] <= ratio_tolerance:
            number_of_replicas = int(np.round(max_atoms / (len(atom))) ** (1 / 3))
            number_of_replicas = max(number_of_replicas, 1)
            max_replication = (
                number_of_replicas,
                number_of_replicas,
                number_of_replicas,
            )

        # Case 2: Non-symmetry lengths beyond tolerance
        else:
            # Compute the replica along the most asymmetric direction
            asymmetric_replica = int(
                (max_atoms / len(atom) / np.prod(ratios)) ** (1 / 3)
            )
            asymmetric_replica = max(asymmetric_replica, 1)

            # Recover replica on the other two direction based on ratios
            replica_r0 = max(int(np.round(asymmetric_replica * ratios[0])), 1)
            replica_r1 = max(int(np.round(asymmetric_replica * ratios[1])), 1)
            indices_to_recover_lattice_vector_order = np.argsort(
                lattice_vector_lengths_argsort_indices
            )
            max_replication_arr = np.array(
                [asymmetric_replica, replica_r0, replica_r1]
            )[indices_to_recover_lattice_vector_order]
            max_replication = tuple(max_replication_arr)

    # Broad cast unit cell infomraiton
    if is_verbose:
        print("System:", atom)
        print("Number of atoms in the unit cell: ", len(atom))
        print("Lattice vector and angles: ", atom.cell.cellpar())
        print(
            "Space group: ",
            check_symmetry(atom, 1e-3, verbose=False)["international"],
        )

    if is_santity_check:
        symmetry_of_unit_cell = check_symmetry(atom, 1e-3, verbose=False)[
            "international"
        ]
        symmetry_of_replicated_supercell = check_symmetry(
            atom.copy().repeat(max_replication), 1e-3, verbose=False
        )["international"]
        if symmetry_of_unit_cell == symmetry_of_replicated_supercell:
            print(
                "symmetry is preserved after replications, safely return "
                "replication combination !\n"
            )
            return max_replication
        else:
            print(
                "Symmetry is lose after replications. No possible replications"
                " can be found !\n"
            )
            return (1, 1, 1)

    # Check if max_replication is still the initial value
    if max_replication == (1, 1, 1):
        print("No possible replications. Returning unit cell.")
        return (1, 1, 1)
    else:
        return max_replication


def get_supercell_parameters(
    atom: Atoms,
    supercell_matrix: np.ndarray = None,
    qpoints_mesh: np.ndarray = None,
    max_atoms: int = None,
):
    """
    Based on symmetry to get supercell setting parameters.
    First, setting the maximum atoms number limitation for supercell. If max_atoms
    is None, will automatic setting it, else use user assigned. If the lattice
    parameters in three direction is same or approximately same, the max_atoms will
    be set small, e.g. 216 or 300; else some direction needed expand larger than
    others, so the max_atoms also need more, e.g. 450. Then, based on max_atoms,
    call auto_grid_dection function to obtain supercell matrix diagonal elements.
    Finally setting the k_point_mesh used to integrate Brillouin Zone. Cause kpoints
    in inverse space, smaller real space means the inverse space is larger, will
    need more kpoints.

    Args:
        atom (Atoms): ASE atoms object to provide lattice information.
        supercell_matrix (nd.ndarray, optional): Supercell matrix for construct
            supercell, prior than max_atoms.
        qpoints_mesh (np.ndarray, optional): Qpoints mesh for IBZ integral, prio
            over than max_atoms.
        max_atoms (int, optional): If not None, will use user setting maximum
            atoms number limitation for generate supercell, else automatic set.
            Defaults to None.
    """
    if supercell_matrix is not None:
        nrep_second = np.diag(supercell_matrix)
        if nrep_second[0] == nrep_second[1] == nrep_second[2]:
            k_point_mesh = 6 * np.array(nrep_second)
        else:
            k_point_mesh = 3 * np.array(nrep_second)

        if qpoints_mesh is not None:
            k_point_mesh = qpoints_mesh

        return supercell_matrix, k_point_mesh

    lattice_vector_lengths = atom.cell.cellpar()[:3]
    lattice_vector_lengths_argsort_indices = np.argsort(lattice_vector_lengths)[::-1]
    sorted_lattice_vector_lengths = lattice_vector_lengths[
        lattice_vector_lengths_argsort_indices
    ]
    ratios = [
        sorted_lattice_vector_lengths[0] / sorted_lattice_vector_lengths[1],
        sorted_lattice_vector_lengths[0] / sorted_lattice_vector_lengths[2],
    ]
    if max_atoms:
        pass
    elif (
        check_symmetry(atom, 1e-3, verbose=False)["international"] == "Fd-3m"
        or check_symmetry(atom, 1e-3, verbose=False)["international"] == "Fm-3m"
        or check_symmetry(atom, 1e-3, verbose=False)["international"] == "F-43m"
    ):
        max_atoms = 216
    elif check_symmetry(atom, 1e-3, verbose=False)["international"] == "P6_3mc":
        max_atoms = 450
    elif ratios[0] <= 1.1 and ratios[1] <= 1.1:
        max_atoms = 300
    else:
        max_atoms = 300

    nrep_second = auto_grid_detection(atom, max_atoms, is_verbose=False)

    if nrep_second[0] == nrep_second[1] == nrep_second[2]:
        k_point_mesh = 6 * np.array(nrep_second)
    else:
        k_point_mesh = 3 * np.array(nrep_second)

    return nrep_second, k_point_mesh
