# -*- coding: utf-8 -*-
import os
import random

from ase.io import write

from mattersim.utils.atoms_utils import AtomsAdaptor

vasp_files = [
    "work/data/H/vasp/vasprun.xml",
    "work/data/H/vasp_2/vasprun.xml",
    "work/data/H/vasp_3/vasprun.xml",
    "work/data/H/vasp_4/vasprun.xml",
    "work/data/H/vasp_5/vasprun.xml",
    "work/data/H/vasp_6/vasprun.xml",
    "work/data/H/vasp_7/vasprun.xml",
    "work/data/H/vasp_8/vasprun.xml",
    "work/data/H/vasp_9/vasprun.xml",
    "work/data/H/vasp_10/vasprun.xml",
]
train_ratio = 0.8
validation_ratio = 0.1
test_ratio = 0.1

save_dir = "./xyz_files"
os.makedirs(save_dir, exist_ok=True)


def main():
    atoms_train = []
    atoms_validation = []
    atoms_test = []

    random.seed(42)

    for vasp_file in vasp_files:
        atoms_list = AtomsAdaptor.from_file(filename=vasp_file)
        random.shuffle(atoms_list)
        num_atoms = len(atoms_list)
        num_train = int(num_atoms * train_ratio)
        num_validation = int(num_atoms * validation_ratio)

        atoms_train.extend(atoms_list[:num_train])
        atoms_validation.extend(atoms_list[num_train : num_train + num_validation])
        atoms_test.extend(atoms_list[num_train + num_validation :])

    print(
        f"Total number of atoms: {len(atoms_train) + len(atoms_validation) + len(atoms_test)}"  # noqa: E501
    )

    print(f"Number of atoms in the training set: {len(atoms_train)}")
    print(f"Number of atoms in the validation set: {len(atoms_validation)}")
    print(f"Number of atoms in the test set: {len(atoms_test)}")

    # Save the training, validation, and test datasets to xyz files

    write(f"{save_dir}/train.xyz", atoms_train)
    write(f"{save_dir}/valid.xyz", atoms_validation)
    write(f"{save_dir}/test.xyz", atoms_test)


if __name__ == "__main__":
    main()
