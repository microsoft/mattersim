# -*- coding: utf-8 -*-
import argparse
import os
import random

from ase.io import write

from mattersim.utils.atoms_utils import AtomsAdaptor


def main(args):
    vasp_files = []
    for root, dirs, files in os.walk(args.data_path):
        for file in files:
            if file.endswith(".xml"):
                vasp_files.append(os.path.join(root, file))

    train_ratio = args.train_ratio
    validation_ratio = args.validation_ratio

    save_dir = args.save_path
    os.makedirs(save_dir, exist_ok=True)

    atoms_train = []
    atoms_validation = []
    atoms_test = []

    random.seed(args.seed)

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

    write(f"{save_dir}/train.xyz", atoms_train, format="extxyz")
    write(f"{save_dir}/valid.xyz", atoms_validation, format="extxyz")
    write(f"{save_dir}/test.xyz", atoms_test, format="extxyz")


if __name__ == "__main__":
    # Some important arguments
    parser = argparse.ArgumentParser()

    # path parameters
    parser.add_argument("--data_path", type=str, default=None, help="vasprun data path")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="train ratio")
    parser.add_argument(
        "--validation_ratio", type=float, default=0.1, help="validation ratio"
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="./xyz_files",
        help="path to save the xyz files",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    args = parser.parse_args()
    main(args)
