"""
Reference implementation for batch relaxation (> 50 structures).

Use this path only when there are many structures and per-structure
filter/symmetry customisation is not needed. For typical jobs (≤ 50
structures), use relax_template.py instead.
"""

import torch

from mattersim.applications.batch_relax import BatchRelaxer
from mattersim.forcefield import Potential

# =============================================================================
# Step 1 — Build or load structures (same as relax_template.py Step 1)
# =============================================================================

atoms_list = []  # <-- replace with actual structures

# =============================================================================
# Step 2 — Load potential and configure batch relaxer
# =============================================================================

device = "cuda" if torch.cuda.is_available() else "cpu"

potential = Potential.from_checkpoint(device=device)
batch_relaxer = BatchRelaxer(
    potential=potential,
    optimizer="FIRE",           # or "BFGS"
    filter=None,                # None, "FrechetCellFilter", or "ExpCellFilter"
    fmax=0.05,                  # eV/Å — BatchRelaxer default is coarser
    max_natoms_per_batch=512,   # tune for GPU memory
    max_n_steps=1_000_000,
)

# =============================================================================
# Step 3 — Run batch relaxation
# =============================================================================

trajectories = batch_relaxer.relax(atoms_list)
# trajectories: {structure_index: [Atoms snapshots...]}
# The final snapshot in each list is the relaxed structure.

for idx, snapshots in trajectories.items():
    relaxed = snapshots[-1]
    print(f"Structure {idx}: {relaxed.get_chemical_formula()}, "
          f"E = {relaxed.info.get('total_energy', 'N/A')}")
