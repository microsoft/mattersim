# -*- coding: utf-8 -*-
from ase import Atoms
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms


def get_primitive_cell(atoms: Atoms):
    """
    Get primitive cell from ASE atoms object
    Args:
        atoms (Atoms): ASE atoms object to provide lattice information
    """
    phonopy_atoms = Phonopy(
        to_phonopy_atoms(atoms), primitive_matrix="auto", log_level=2
    )
    primitive = phonopy_atoms.primitive
    atoms = to_ase_atoms(primitive)
    return atoms


def to_phonopy_atoms(atoms: Atoms):
    """
    Transform ASE atoms object to Phonopy object
    Args:
        atoms (Atoms): ASE atoms object to provide lattice informations.
    """
    phonopy_atoms = PhonopyAtoms(
        symbols=atoms.get_chemical_symbols(),
        cell=atoms.get_cell(),
        masses=atoms.get_masses(),
        positions=atoms.get_positions(),
    )
    return phonopy_atoms


def to_ase_atoms(phonopy_atoms):
    """
    Transform Phonopy object to ASE atoms object
    Args:
        phonopy_atoms (Phonopy): Phonopy object to provide lattice informations.
    """
    atoms = Atoms(
        symbols=phonopy_atoms.symbols,
        cell=phonopy_atoms.cell,
        masses=phonopy_atoms.masses,
        positions=phonopy_atoms.positions,
        pbc=True,
    )
    return atoms
