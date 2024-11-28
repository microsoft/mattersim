from typing import Dict, List, Optional

import torch
import torch.distributed
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.constraints import full_3x3_to_voigt_6_stress
from ase.units import GPa

from mattersim.datasets.utils.build import build_dataloader
from mattersim.forcefield.potential import Potential,batch_to_dict

class DeepCalculator(Calculator):
    """
    Deep calculator based on ase Calculator
    """

    implemented_properties = ["energy", "free_energy", "forces", "stress"]

    def __init__(
        self,
        potential: Potential = None,
        args_dict: dict = {},
        compute_stress: bool = True,
        stress_weight: float = GPa,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        **kwargs,
    ):
        """
        Args:
            potential (Potential): m3gnet.models.Potential
            compute_stress (bool): whether to calculate the stress
            stress_weight (float): the stress weight.
            **kwargs:
        """
        super().__init__(**kwargs)
        if potential is None:
            self.potential = Potential()
        else:
            self.potential = potential
        self.compute_stress = compute_stress
        self.stress_weight = stress_weight
        self.args_dict = args_dict
        self.device = device

    def load(
        load_path: str = None,
        *,
        model_name: str = "m3gnet",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        args: Dict = None,
        load_training_state: bool = True,
        args_dict: dict = {},
        compute_stress: bool = True,
        stress_weight: float = GPa,
        **kwargs,
    ):
        potential = Potential.load(
            load_path=load_path,
            model_name=model_name,
            device=device,
            args=args,
            load_training_state=load_training_state,
        )
        return DeepCalculator(
            potential=potential,
            args_dict=args_dict,
            compute_stress=compute_stress,
            stress_weight=stress_weight,
            device=device,
            **kwargs,
        )
        
    def calculate(
        self,
        atoms: Optional[Atoms] = None,
        properties: Optional[list] = None,
        system_changes: Optional[list] = None,
    ):
        """
        Args:
            atoms (ase.Atoms): ase Atoms object
            properties (list): list of properties to calculate
            system_changes (list): monitor which properties of atoms were
                changed for new calculation. If not, the previous calculation
                results will be loaded.
        Returns:
        """

        all_changes = [
            "positions",
            "numbers",
            "cell",
            "pbc",
            "initial_charges",
            "initial_magmoms",
        ]

        properties = properties or ["energy"]
        system_changes = system_changes or all_changes
        super().calculate(
            atoms=atoms, properties=properties, system_changes=system_changes
        )

        self.args_dict["batch_size"] = 1
        self.args_dict["only_inference"] = 1
        dataloader = build_dataloader(
            [atoms], model_type=self.potential.model_name, **self.args_dict
        )
        for graph_batch in dataloader:
            # Resemble input dictionary
            if (
                self.potential.model_name == "graphormer"
                or self.potential.model_name == "geomformer"
            ):
                raise NotImplementedError
            else:
                graph_batch = graph_batch.to(self.device)
                input = batch_to_dict(graph_batch)

            result = self.potential.forward(
                input, include_forces=True, include_stresses=self.compute_stress
            )
            if (
                self.potential.model_name == "graphormer"
                or self.potential.model_name == "geomformer"
            ):
                raise NotImplementedError
            else:
                self.results.update(
                    energy=result["total_energy"].detach().cpu().numpy()[0],
                    free_energy=result["total_energy"].detach().cpu().numpy()[0],
                    forces=result["forces"].detach().cpu().numpy(),
                )
            if self.compute_stress:
                self.results.update(
                    stress=self.stress_weight
                    * full_3x3_to_voigt_6_stress(
                        result["stresses"].detach().cpu().numpy()[0]
                    )
                )
