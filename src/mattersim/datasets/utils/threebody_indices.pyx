# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: cdivision=True
# cython: profile=True
# cython: language_level=3
# distutils: language = c
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION

cimport numpy as np
import numpy as np
from libc.stdlib cimport free, malloc
from libc.string cimport memset


def compute_threebody(const int[:, ::1] bond_atom_indices,
    const int[::1] n_atoms):
    """
    Calculate the three body indices from pair atom indices
    Args:
        bond_atom_indices (np.ndarray): pair atom indices
        n_atoms (int): number of atoms
    Returns:
        triple_bond_indices (np.ndarray): bond indices that form three-body
        py_n_triple_ij (np.ndarray): number of three-body angles for each bond
        py_n_triple_i (np.ndarray): number of three-body angles each atom
        py_n_triple_s (np.ndarray): number of three-body angles for each
            structure
    """
    cdef int i, j, k
    cdef int n_bond = bond_atom_indices.shape[0]
    cdef int n_atom = 0
    cdef int n_struct = n_atoms.shape[0]
    for i in range(n_struct):
        n_atom += n_atoms[i]

    cdef int* n_bond_per_atom = <int *> malloc(n_atom * sizeof(int))
    memset(n_bond_per_atom, 0, n_atom * sizeof(int))

    for i in range(n_bond):
        n_bond_per_atom[bond_atom_indices[i, 0]] += 1

    cdef int* n_triple_i = <int *> malloc(n_atom * sizeof(int))
    cdef int* n_triple_ij = <int *> malloc(n_bond * sizeof(int))
    cdef int* n_triple_s = <int *> malloc(n_struct * sizeof(int))

    memset(n_triple_s, 0, n_struct * sizeof(int))

    cdef int n_triple = 0
    cdef int n_triple_temp
    cdef int start = 0

    for i in range(n_atom):
        n_triple_temp = n_bond_per_atom[i] * (n_bond_per_atom[i] - 1)
        for j in range(n_bond_per_atom[i]):
            n_triple_ij[start + j] = n_bond_per_atom[i] - 1
        n_triple += n_triple_temp
        n_triple_i[i] = n_triple_temp
        start += n_bond_per_atom[i]

    cdef np.ndarray triple_bond_indices = np.empty(shape=(n_triple, 2),
                                             dtype=np.int32)

    start = 0
    cdef int index = 0
    for i in range(n_atom):
        for j in range(n_bond_per_atom[i]):
            for k in range(n_bond_per_atom[i]):
                if j != k:
                    triple_bond_indices[index, 0] = start + j
                    triple_bond_indices[index, 1] = start + k
                    index += 1
        start += n_bond_per_atom[i]

    start = 0
    cdef int end = start
    cdef int n_atom_temp
    for i in range(n_struct):
        end += n_atoms[i]
        for j in range(start, end):
            n_triple_s[i] += n_triple_i[j]
        start = end
    py_n_triple_ij = np.array(<int[:n_bond]>n_triple_ij)
    py_n_triple_i = np.array(<int[:n_atom]>n_triple_i)
    py_n_triple_s = np.array(<int[:n_struct]>n_triple_s)

    free(n_triple_ij)
    free(n_triple_i)
    free(n_triple_s)
    free(n_bond_per_atom)
    return triple_bond_indices, py_n_triple_ij, py_n_triple_i, py_n_triple_s
