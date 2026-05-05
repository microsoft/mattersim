"""
Schema builders for MatterSim application results.

Classes here convert raw ASE Atoms outputs into structured documents
compatible with the atomate2 / Materials Project ecosystem, while
preserving MatterSim-specific metadata not captured by the base schemas.
"""

import json
from typing import Literal, Optional

import numpy as np
from ase import Atoms
from atomate2.ase.schemas import AseResult
from atomate2.forcefields.schemas import ForceFieldTaskDocument
from atomate2.forcefields.utils import MLFF
from monty.json import MontyEncoder
from pydantic import BaseModel, Field
from pymatgen.core.trajectory import Trajectory as PmgTrajectory
from pymatgen.io.ase import AseAtomsAdaptor

_adaptor = AseAtomsAdaptor()

#: Allowed MatterSim model checkpoint identifiers.
MatterSimCheckpoint = Literal["mattersim-v1.0.0-1m", "mattersim-v1.0.0-5m"]


class MatterSimRelaxMetadata(BaseModel):
    """All parameters needed to reproduce a MatterSim relaxation run.

    Centralises both MatterSim-specific fields (checkpoint, device, filter,
    pressure) and the relaxation settings that atomate2 stores in its nested
    InputDoc (fmax, steps, relax_cell, constrain_symmetry). The duplication
    with InputDoc is intentional: this flat record is self-contained and lets
    downstream users read every run parameter from a single location without
    navigating the atomate2 schema hierarchy.
    """

    model_checkpoint: MatterSimCheckpoint = Field(
        description="Model checkpoint identifier."
    )
    device: str = Field(
        description="Compute device used, e.g. 'cuda' or 'cpu'."
    )
    optimizer: str = Field(
        description="ASE optimizer name, e.g. 'FIRE' or 'BFGS'."
    )
    filter: Optional[str] = Field(
        default=None,
        description="Cell filter class name ('FrechetCellFilter', 'ExpCellFilter'), "
                    "or None for fixed-cell relaxation.",
    )
    pressure: float = Field(
        default=0.0,
        description="Target external pressure applied during relaxation, "
                    "in the unit given by pressure_unit.",
    )
    pressure_unit: str = Field(
        default="GPa",
        description="Unit of the pressure field: 'GPa', 'kbar', or 'eV/A^3'.",
    )
    fmax: float = Field(
        description="Force convergence criterion (eV/Å)."
    )
    steps: int = Field(
        description="Maximum number of optimisation steps allowed."
    )
    relax_cell: bool = Field(
        description="Whether cell vectors were relaxed (i.e. a cell filter was applied)."
    )
    constrain_symmetry: bool = Field(
        description="Whether space-group symmetry was constrained via FixSymmetry."
    )


class MatterSimRelaxTaskDocument(ForceFieldTaskDocument):
    """Atomate2-compatible relaxation task document with MatterSim metadata.

    Inherits all fields from ForceFieldTaskDocument (energy, forces, stress,
    ionic steps, input/output structures, convergence status, etc.) and adds
    a ``mattersim_metadata`` field that fully describes the run parameters.
    """

    mattersim_metadata: MatterSimRelaxMetadata = Field(
        description="Complete MatterSim relaxation parameters for reproducibility."
    )

    @classmethod
    def from_relax(
        cls,
        initial_atoms: Atoms,
        relaxed_atoms: Atoms,
        initial_energy: float,
        converged: bool,
        elapsed: float,
        model_checkpoint: MatterSimCheckpoint,
        device: str,
        optimizer: str,
        fmax: float = 0.01,
        steps: int = 500,
        relax_cell: bool = False,
        constrain_symmetry: bool = False,
        filter_name: Optional[str] = None,
        pressure: float = 0.0,
        pressure_unit: str = "GPa",
    ) -> "MatterSimRelaxTaskDocument":
        """Build a MatterSimRelaxTaskDocument from a completed relaxation.

        Args:
            initial_atoms: Structure before relaxation (no calculator required;
                used only for the initial trajectory frame).
            relaxed_atoms: Structure after relaxation, as returned by
                ``Relaxer.relax()``.
            initial_energy: Potential energy of the initial structure (eV),
                obtained by calling ``atoms.get_potential_energy()`` before
                relaxation starts.
            converged: Whether the relaxation converged (from ``Relaxer.relax()``).
            elapsed: Wall-clock time of the relaxation in seconds.
            model_checkpoint: Model checkpoint identifier.
            device: Compute device used ('cuda' or 'cpu').
            optimizer: ASE optimizer name ('FIRE' or 'BFGS').
            fmax: Force convergence criterion (eV/Å).
            steps: Maximum number of optimisation steps allowed.
            relax_cell: Whether cell vectors were relaxed.
            constrain_symmetry: Whether space-group symmetry was constrained.
            filter_name: Cell filter class name, or None for fixed-cell relaxation.
            pressure: Target pressure value (default 0.0).
            pressure_unit: Unit of ``pressure`` ('GPa', 'kbar', or 'eV/A^3').

        Returns:
            MatterSimRelaxTaskDocument with full atomate2-compatible fields plus
            a self-contained ``mattersim_metadata`` block.
        """
        is_periodic = all(relaxed_atoms.pbc)

        initial_energy = float(initial_energy)
        final_energy = float(relaxed_atoms.get_potential_energy())
        final_forces = relaxed_atoms.get_forces()
        is_force_converged = bool(np.all(np.linalg.norm(final_forces, axis=1) < fmax))

        if is_periodic:
            # Build 2-frame trajectory so that forces and stress propagate into
            # OutputDoc. "stress" is excluded from ionic_step_data to avoid a
            # None-conversion crash on the initial frame (which has no stress);
            # the final stress is still captured in output.stress by the schema.
            traj = PmgTrajectory.from_structures(
                [
                    _adaptor.get_structure(initial_atoms),
                    _adaptor.get_structure(relaxed_atoms),
                ],
                frame_properties=[
                    {"energy": initial_energy},
                    {
                        "energy": final_energy,
                        "forces": final_forces.tolist(),
                        "stress": relaxed_atoms.get_stress().tolist(),  # eV/Å³ Voigt
                    },
                ],
                constant_lattice=not relax_cell,
            )
            final_pmg = _adaptor.get_structure(relaxed_atoms)
        else:
            traj = None
            final_pmg = _adaptor.get_molecule(relaxed_atoms)

        ase_result = AseResult(
            final_mol_or_struct=final_pmg,
            final_energy=final_energy,
            trajectory=traj,
            converged=converged,
            is_force_converged=is_force_converged,
            energy_downhill=final_energy < initial_energy,
            elapsed_time=elapsed,
        )

        ff_doc = ForceFieldTaskDocument.from_ase_compatible_result(
            ase_calculator_name="MatterSim",
            result=ase_result,
            steps=steps,
            calculator_meta=MLFF.MatterSim,
            relax_kwargs={"fmax": fmax},
            fix_symmetry=constrain_symmetry,
            forcefield_name="MatterSim",
            ionic_step_data=("energy", "forces", "mol_or_struct"),
            relax_cell=is_periodic and relax_cell,
            relax_shape=False,
        )

        metadata = MatterSimRelaxMetadata(
            model_checkpoint=model_checkpoint,
            device=device,
            optimizer=optimizer,
            filter=filter_name,
            pressure=pressure,
            pressure_unit=pressure_unit,
            fmax=fmax,
            steps=steps,
            relax_cell=relax_cell,
            constrain_symmetry=constrain_symmetry,
        )

        # model_construct bypasses re-validation so pymatgen objects in ff_doc
        # are carried over as-is without a lossy serialize/deserialize round-trip.
        return cls.model_construct(
            _fields_set=ff_doc.model_fields_set | {"mattersim_metadata"},
            mattersim_metadata=metadata,
            **{field: getattr(ff_doc, field) for field in ForceFieldTaskDocument.model_fields},
        )

    def to_json(self, indent: int = 2) -> str:
        """Serialize this document to a JSON string.

        Uses MontyEncoder to handle pymatgen objects (Structure, Molecule,
        etc.) that pydantic cannot serialize natively.

        Args:
            indent: JSON indentation level.

        Returns:
            JSON string.
        """
        return json.dumps(self.model_dump(), cls=MontyEncoder, indent=indent)
