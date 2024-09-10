# -*- coding: utf-8 -*-
import os

from ase import Atoms
from ase.io import read as ase_read
from mp_api.client import MPRester
from pymatgen.core.structure import Structure
from pymatgen.io.ase import AseAtomsAdaptor


class AtomsAdaptor(object):
    """
    This class is used to read different structures type
    and transform it to ASE Atoms object.
    """

    def __init__(self):
        pass

    @classmethod
    def from_ase_atoms(cls, atoms: Atoms):
        """
        Get Atoms from Atoms.

        Args:
            atoms (Atoms): ASE Atoms object.
        """
        if not isinstance(atoms, Atoms):
            raise TypeError("Input must be ASE Atoms object.")
        return atoms

    @classmethod
    def from_pymatgen_structure(cls, structure: Structure):
        """
        Get Atoms from Structure.

        Args:
            structure (Structure): pymatgen Structure object.
        """
        if not isinstance(structure, Structure):
            raise TypeError("Input must be pymatgen Structure object")
        return AseAtomsAdaptor.get_atoms(structure, msonable=False)

    @classmethod
    def from_mp_id(cls, mp_id: str, api_key: str = None):
        """
        Get Atoms from mp-id.

        mp_id (str): mp_id for materials.
        api_key (str, optional): api_key to access Material Projects.
            If not provided, try to extract it from environment variables.
        """
        mp_api_key = api_key or os.getenv("MP_API_KEY")
        if not mp_api_key:
            raise ValueError(
                "An MP API key is required to fetch data from"
                " Materials Project, but was not found in the"
                " environment variables or provided."
            )
        with MPRester(mp_api_key) as m:
            structure = m.get_structure_by_material_id(mp_id)
            return AseAtomsAdaptor.get_atoms(structure, msonable=False)

    @classmethod
    def from_file(cls, filename: str, format: str = None):
        """
        Get Atoms from file.

        filename (str): file name which contains structures.
        format (str, optional): file format. If None, will automately
            guess.
        """
        if not os.path.exists(filename):
            raise FileNotFoundError(f"File {filename} not found.")

        if format:
            atoms_list = ase_read(filename, index=":", format=format)
        else:
            try:
                atoms_list = ase_read(filename, index=":")
            except Exception as e:
                raise ValueError(f"Can not automately guess the file format: {e}")

        return atoms_list
