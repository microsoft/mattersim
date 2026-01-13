# -*- coding: utf-8 -*-
import warnings
from typing import Iterable, List, Tuple, Union

from ase import Atoms
from ase.constraints import Filter, FixSymmetry
from ase.filters import ExpCellFilter, FrechetCellFilter
from ase.optimize import BFGS, FIRE
from ase.optimize.optimize import Optimizer
from ase.units import GPa
from deprecated import deprecated


class Relaxer(object):
    """Relaxer is a class for structural relaxation with fixed volume."""

    SUPPORTED_OPTIMIZERS = {"BFGS": BFGS, "FIRE": FIRE}
    SUPPORTED_FILTERS = {
        "EXPCELLFILTER": ExpCellFilter,
        "FRECHETCELLFILTER": FrechetCellFilter,
    }

    def __init__(
        self,
        optimizer: Union[Optimizer, str] = "FIRE",
        filter: Union[Filter, str, None] = None,
        constrain_symmetry: bool = True,
        fix_axis: Union[bool, Iterable[bool]] = False,
    ) -> None:
        """
        Args:
            optimizer (Union[Optimizer, str]): The optimizer to use.
            filter (Union[Filter, str, None]): The filter to use.
            constrain_symmetry (bool): Whether to constrain the symmetry.
            fix_axis (Union[bool, Iterable[bool]]): Whether to fix the axis.
        """
        self.optimizer = (
            self.SUPPORTED_OPTIMIZERS[optimizer.upper()]
            if isinstance(optimizer, str)
            else optimizer
        )
        self.relax_cell = filter is not None
        if filter is not None:
            self.filter = (
                self.SUPPORTED_FILTERS[filter.upper()]
                if isinstance(filter, str)
                else filter
            )
        self.constrain_symmetry = constrain_symmetry
        self.fix_axis = fix_axis

    def relax(
        self,
        atoms: Atoms,
        steps: int = 500,
        fmax: float = 0.01,
        params_filter: dict = {},
        **kwargs,
    ) -> Tuple[bool, Atoms]:
        """
        Relax the atoms object.

        Args:
            atoms (Atoms): The atoms object to relax.
            steps (int): The maximum number of steps to take.
            fmax (float): The maximum force allowed.
            params_filter (dict): The parameters for the filter.
            kwargs: Additional keyword arguments for the optimizer.
        """

        if atoms.calc is None:
            raise ValueError("Atoms object must have a calculator.")

        if self.constrain_symmetry:
            atoms.set_constraint(FixSymmetry(atoms))

        if self.relax_cell:
            # Set the mask for the fixed axis
            if isinstance(self.fix_axis, bool):
                mask = [not self.fix_axis for i in range(6)]
            else:
                assert (
                    len(self.fix_axis) == 6
                ), "The length of fix_axis list not equal 6."
                mask = [not elem for elem in self.fix_axis]

            # check if the scalar_pressure is provided
            if (
                "scalar_pressure" in params_filter
                and params_filter["scalar_pressure"] > 1
            ):
                warnings.warn(
                    "The scalar_pressure used in ExpCellFilter assumes "
                    "eV/A^3 unit and 1 eV/A^3 is already 160 GPa. "
                    "Please make sure you have converted your pressure "
                    "from GPa to eV/A^3 by dividing by 160.21766208."
                )
            ecf = self.filter(atoms, mask=mask, **params_filter)
        else:
            ecf = atoms
        optimizer = self.optimizer(ecf, **kwargs)
        optimizer.run(fmax=fmax, steps=steps)

        converged = optimizer.get_number_of_steps() < steps

        if self.constrain_symmetry:
            atoms.set_constraint(None)

        return converged, atoms

    @classmethod
    @deprecated(reason="Use cli/applications/relax_structure.py instead.")
    def relax_structures(
        cls,
        atoms: Union[Atoms, Iterable[Atoms]],
        optimizer: Union[Optimizer, str] = "FIRE",
        filter: Union[Filter, str, None] = None,
        constrain_symmetry: bool = False,
        fix_axis: Union[bool, Iterable[bool]] = False,
        pressure_in_GPa: Union[float, None] = None,
        **kwargs,
    ) -> Union[Tuple[bool, Atoms], Tuple[List[bool], List[Atoms]]]:
        """
        Args:
            atoms: (Union[Atoms, Iterable[Atoms]]):
                The Atoms object or an iterable of Atoms objects to relax.
            optimizer (Union[Optimizer, str]): The optimizer to use.
            filter (Union[Filter, str, None]): The filter to use.
            constrain_symmetry (bool): Whether to constrain the symmetry.
            fix_axis (Union[bool, Iterable[bool]]): Whether to fix the axis.
            **kwargs: Additional keyword arguments for the relax method.
        Returns:
            converged (Union[bool, List[bool]]):
                Whether the relaxation converged or a list of them
            Atoms (Union[Atoms, List[Atoms]]):
                The relaxed atoms object or a list of them
        """
        params_filter = {}

        if filter is None and pressure_in_GPa is None:
            pass
        elif filter is None and pressure_in_GPa is not None:
            filter = "ExpCellFilter"
            params_filter["scalar_pressure"] = (
                pressure_in_GPa * GPa
            )  # GPa = 1 / 160.21766208
        elif filter is not None and pressure_in_GPa is None:
            params_filter["scalar_pressure"] = 0.0
        else:
            params_filter["scalar_pressure"] = (
                pressure_in_GPa * GPa
            )  # GPa = / 160.21766208

        relaxer = Relaxer(
            optimizer=optimizer,
            filter=filter,
            constrain_symmetry=constrain_symmetry,
            fix_axis=fix_axis,
        )

        if isinstance(atoms, (list, tuple)):
            relaxed_results = relaxed_results = [
                relaxer.relax(atom, params_filter=params_filter, **kwargs)
                for atom in atoms
            ]
            converged, relaxed_atoms = zip(*relaxed_results)
            return list(converged), list(relaxed_atoms)
        else:
            return relaxer.relax(atoms, params_filter=params_filter, **kwargs)
