# -*- coding: utf-8 -*-
"""
Atomic scaling module. Used for predicting extensive properties.
"""

from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn
from ase import Atoms
from torch_runstats.scatter import scatter_mean

from mattersim.datasets.utils.regressor import solver

DATA_INDEX = {
    "total_energy": 0,
    "forces": 2,
    "per_atom_energy": 1,
    "per_species_energy": 0,
}


class AtomScaling(nn.Module):
    """
    Atomic extensive property rescaling module
    """

    def __init__(
        self,
        atoms: list[Atoms] = None,
        total_energy: list[float] = None,
        forces: list[np.ndarray] = None,
        atomic_numbers: list[np.ndarray] = None,
        num_atoms: list[float] = None,
        max_z: int = 94,
        scale_key: str = None,
        shift_key: str = None,
        init_scale: Union[torch.Tensor, float] = None,
        init_shift: Union[torch.Tensor, float] = None,
        trainable_scale: bool = False,
        trainable_shift: bool = False,
        verbose: bool = False,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        **kwargs,
    ):
        """
        Args:
            forces: a list of atomic forces (np.ndarray) in each graph
            max_z: (int) maximum atomic number
                - if scale_key or shift_key is specified,
                  max_z should be equal to the maximum atomic_number.
            scale_key: valid options are:
                - total_energy_std
                - per_atom_energy_std
                - per_species_energy_std
                - forces_rms
                - per_species_forces_rms (default)
            shift_key: valid options are:
                - total_energy_mean
                - per_atom_energy_mean
                - per_species_energy_mean :
                  default option is gaussian regression (NequIP)
                - per_species_energy_mean_linear_reg :
                  an alternative choice is linear regression (M3GNet)
            init_scale (torch.Tensor or float)
            init_shift (torch.Tensor or float)
        """
        super().__init__()

        self.max_z = max_z
        self.device = device

        # === Data preprocessing ===
        if scale_key or shift_key:
            total_energy = torch.from_numpy(np.array(total_energy))
            forces = (
                torch.from_numpy(np.concatenate(forces, axis=0))
                if forces is not None
                else None
            )
            if atomic_numbers is None:
                atomic_numbers = [atom.get_atomic_numbers() for atom in atoms]
            atomic_numbers = (
                torch.from_numpy(np.concatenate(atomic_numbers, axis=0))
                .squeeze(-1)
                .long()
            )  # (num_atoms,)
            # assert max_z == atomic_numbers.max().item(),
            # "max_z should be equal to the maximum atomic_number"
            if num_atoms is None:
                num_atoms = [  # noqa: E501
                    atom.positions.shape[0] for atom in atoms
                ]  # (N_GRAPHS, )
            num_atoms = torch.from_numpy(np.array(num_atoms))
            per_atom_energy = total_energy / num_atoms
            data_list = [total_energy, per_atom_energy, forces]

            assert (
                num_atoms.size()[0] == total_energy.size()[0]
            ), "num_atoms and total_energy should have the same size, "
            f"but got {num_atoms.size()[0]} and {total_energy.size()[0]}"
            if forces is not None:
                assert (
                    forces.size()[0] == atomic_numbers.size()[0]
                ), "forces and atomic_numbers should have the same length, "
                f"but got {forces.size()[0]} and {atomic_numbers.size()[0]}"

            # === Calculate the scaling factors ===
            if (
                scale_key == "per_species_energy_std"
                and shift_key == "per_species_energy_mean"
                and init_shift is None
                and init_scale is None
            ):
                # Using gaussian regression two times
                # to get the shift and scale is potentially unstable
                init_shift, init_scale = self.get_gaussian_statistics(
                    atomic_numbers, num_atoms, total_energy
                )
            else:
                if shift_key and init_shift is None:
                    init_shift = self.get_statistics(
                        shift_key, max_z, data_list, atomic_numbers, num_atoms
                    )
                if scale_key and init_scale is None:
                    init_scale = self.get_statistics(
                        scale_key, max_z, data_list, atomic_numbers, num_atoms
                    )

        # === initial values are given ===
        if init_scale is None:
            init_scale = torch.ones(max_z + 1)
        elif isinstance(init_scale, float):
            init_scale = torch.tensor(init_scale).repeat(max_z + 1)
        else:
            assert init_scale.size()[0] == max_z + 1

        if init_shift is None:
            init_shift = torch.zeros(max_z + 1)
        elif isinstance(init_shift, float):
            init_shift = torch.tensor(init_shift).repeat(max_z + 1)
        else:
            assert init_shift.size()[0] == max_z + 1

        init_shift = init_shift.float()
        init_scale = init_scale.float()
        if trainable_scale is True:
            self.scale = torch.nn.Parameter(init_scale)
        else:
            self.register_buffer("scale", init_scale)

        if trainable_shift is True:
            self.shift = torch.nn.Parameter(init_shift)
        else:
            self.register_buffer("shift", init_shift)

        if verbose is True:
            print("Current scale: ", init_scale)
            print("Current shift: ", init_shift)

        self.to(device)

    def transform(
        self, atomic_energies: torch.Tensor, atomic_numbers: torch.Tensor
    ) -> torch.Tensor:
        """
        Take the origin values from model and get the transformed values
        """
        curr_shift = self.shift[atomic_numbers]
        curr_scale = self.scale[atomic_numbers]
        normalized_energies = curr_scale * atomic_energies + curr_shift
        return normalized_energies

    def inverse_transform(
        self, atomic_energies: torch.Tensor, atomic_numbers: torch.Tensor
    ) -> torch.Tensor:
        """
        Take the transformed values and get the original values
        """
        curr_shift = self.shift[atomic_numbers]
        curr_scale = self.scale[atomic_numbers]
        unnormalized_energies = (atomic_energies - curr_shift) / curr_scale
        return unnormalized_energies

    def forward(
        self, atomic_energies: torch.Tensor, atomic_numbers: torch.Tensor
    ) -> torch.Tensor:
        """
        Atomic_energies and atomic_numbers should have the same size
        """
        return self.transform(atomic_energies, atomic_numbers)

    def get_statistics(
        self, key, max_z, data_list, atomic_numbers, num_atoms
    ) -> torch.Tensor:
        """
        Valid key:
            scale_key: valid options are:
                - total_energy_mean
                - per_atom_energy_mean
                - per_species_energy_mean
                - per_species_energy_mean_linear_reg :
                  an alternative choice is linear regression
            shift_key: valid options are:
                - total_energy_std
                - per_atom_energy_std
                - per_species_energy_std
                - forces_rms
                - per_species_forces_rms
        """
        data = None
        for data_key in DATA_INDEX:
            if data_key in key:
                data = data_list[DATA_INDEX[data_key]]
        assert data is not None

        statistics = None
        if "mean" in key:
            if "per_species" in key:
                n_atoms = torch.repeat_interleave(repeats=num_atoms)
                if "linear_reg" in key:
                    features = bincount(
                        atomic_numbers, n_atoms, minlength=self.max_z + 1
                    ).numpy()
                    # print(features[0], features.shape)
                    data = data.numpy()
                    assert features.ndim == 2  # [batch, n_type]
                    features = features[
                        (features > 0).any(axis=1)
                    ]  # deal with non-contiguous batch indexes
                    statistics = np.linalg.pinv(features.T.dot(features)).dot(
                        features.T.dot(data)
                    )
                    statistics = torch.from_numpy(statistics)
                else:
                    N = bincount(
                        atomic_numbers,
                        num_atoms,
                        minlength=self.max_z + 1,  # noqa: E501
                    )
                    assert N.ndim == 2  # [batch, n_type]
                    # deal with non-contiguous batch indexes
                    N = N[(N > 0).any(dim=1)]
                    N = N.type(torch.get_default_dtype())
                    statistics, _ = solver(
                        N, data, regressor="NormalizedGaussianProcess"
                    )
            else:
                statistics = torch.mean(data).item()
        elif "std" in key:
            if "per_species" in key:
                print(
                    "Warning: calculating per_species_energy_std for "
                    "full periodic table systems is risky, please use "
                    "per_species_forces_rms instead."
                )
                n_atoms = torch.repeat_interleave(repeats=num_atoms)
                N = bincount(atomic_numbers, n_atoms, minlength=self.max_z + 1)
                assert N.ndim == 2  # [batch, n_type]
                # deal with non-contiguous batch indexes
                N = N[(N > 0).any(dim=1)]
                N = N.type(torch.get_default_dtype())
                _, statistics = solver(  # noqa: E501
                    N, data, regressor="NormalizedGaussianProcess"
                )
            else:
                statistics = torch.std(data).item()
        elif "rms" in key:
            if "per_species" in key:
                square = scatter_mean(
                    data.square(), atomic_numbers, dim=0, dim_size=max_z + 1
                )
                statistics = square.mean(axis=-1)
            else:
                statistics = torch.sqrt(torch.mean(data.square())).item()

        if isinstance(statistics, torch.Tensor) is not True:
            statistics = torch.tensor(statistics).repeat(max_z + 1)

        assert statistics.size()[0] == max_z + 1

        return statistics

    def get_gaussian_statistics(
        self,
        atomic_numbers: torch.Tensor,
        num_atoms: torch.Tensor,
        total_energy: torch.Tensor,
    ):
        """
        Get the gaussian process mean and variance
        """
        n_atoms = torch.repeat_interleave(repeats=num_atoms)
        N = bincount(atomic_numbers, n_atoms, minlength=self.max_z + 1)
        assert N.ndim == 2  # [batch, n_type]
        N = N[(N > 0).any(dim=1)]  # deal with non-contiguous batch indexes
        N = N.type(torch.get_default_dtype())
        mean, std = solver(  # noqa: E501
            N, total_energy, regressor="NormalizedGaussianProcess"
        )
        assert mean.size()[0] == self.max_z + 1
        assert std.size()[0] == self.max_z + 1
        return mean, std


def bincount(
    input: torch.Tensor,
    batch: Optional[torch.Tensor] = None,
    minlength: int = 0,  # noqa: E501
):
    assert input.ndim == 1
    if batch is None:
        return torch.bincount(input, minlength=minlength)
    else:
        assert batch.shape == input.shape

        length = input.max().item() + 1
        if minlength == 0:
            minlength = length
        if length > minlength:
            raise ValueError(
                f"minlength {minlength} too small for input "
                f"with integers up to and including {length}"  # noqa: E501
            )

        # Flatten indexes
        # Make each "class" in input into a per-input class.
        input_ = input + batch * minlength

        num_batch = batch.max() + 1

        return torch.bincount(input_, minlength=minlength * num_batch).reshape(
            num_batch, minlength
        )
