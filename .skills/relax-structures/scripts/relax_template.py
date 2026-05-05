"""
Reference implementation for the relax-structures skill.

AI agents should adapt this script to the user's specific request, modifying
the structure construction, model selection, and relaxation parameters as needed.

This script is NOT meant to be run directly — see examples/relax_bulk_si.py for
a runnable example. This file serves as the canonical template that agents copy
and adapt.

Output format: one ForceFieldTaskDocument JSON per structure, compatible with
the atomate2 / Materials Project ecosystem.
"""

import os
import time
from datetime import datetime, timezone

import numpy as np
import torch
from ase.io import read as ase_read
from ase.io import write as ase_write

from mattersim.applications.relax import Relaxer
from mattersim.applications.schemas import build_relax_task_doc
from mattersim.forcefield import MatterSimCalculator


# =============================================================================
# Step 1 — Build or load structures
# =============================================================================
# Adapt this section to the user's request. Examples:
#
#   from ase.build import bulk
#   atoms_list = [bulk("Si", "diamond", a=5.43, cubic=True)]
#
#   atoms_list = list(ase_read("structures.cif", index=":"))
#
#   atoms_list = []
#   for path in ["file1.cif", "file2.poscar"]:
#       atoms_list.extend(ase_read(path, index=":"))

atoms_list = []  # <-- replace with actual structures

# =============================================================================
# Step 2 — Load calculator
# =============================================================================
# Adapt model_name and device to the user's request.

model_name = "mattersim-v1.0.0-1m"  # or "mattersim-v1.0.0-5m"
device = "cuda" if torch.cuda.is_available() else "cpu"

calc = MatterSimCalculator(load_path=model_name, device=device)
for atoms in atoms_list:
    atoms.calc = calc

# =============================================================================
# Step 3 — Configure and run relaxation
# =============================================================================
# Adapt parameters to the user's request.

optimizer = "FIRE"              # or "BFGS"
fmax = 0.01                     # eV/Å
steps = 500                     # max steps
filter_name = None              # None, "FrechetCellFilter", or "ExpCellFilter"
pressure = None                 # external target pressure (implies cell relaxation)
pressure_unit = "GPa"           # "GPa", "kbar", or "eV/A^3"
constrain_symmetry = False      # True to preserve space-group symmetry

# --- Convert pressure to eV/Å³ (the unit ASE cell filters expect) ---
# 1 eV/Å³ = 160.2177 GPa = 1602.177 kbar
params_filter = {}
if pressure is not None:
    from ase.units import GPa as _GPa
    if pressure_unit == "GPa":
        scalar_pressure = pressure * _GPa
    elif pressure_unit == "kbar":
        scalar_pressure = pressure * _GPa / 10.0
    elif pressure_unit == "eV/A^3":
        scalar_pressure = pressure
    else:
        raise ValueError(f"Unknown pressure_unit: {pressure_unit!r}. "
                         f"Use 'GPa', 'kbar', or 'eV/A^3'.")
    params_filter["scalar_pressure"] = scalar_pressure
    if filter_name is None:
        filter_name = "FrechetCellFilter"
elif filter_name is not None:
    params_filter["scalar_pressure"] = 0.0

relaxer = Relaxer(
    optimizer=optimizer,
    filter=filter_name,
    constrain_symmetry=constrain_symmetry,
)

results = []
for i, atoms in enumerate(atoms_list):
    is_periodic = all(atoms.pbc)

    # Disable cell filter for non-periodic structures
    if not is_periodic:
        eff_relaxer = Relaxer(
            optimizer=optimizer, filter=None,
            constrain_symmetry=constrain_symmetry,
        )
        eff_params = {}
    else:
        eff_relaxer = relaxer
        eff_params = params_filter

    # Cache initial energy before relaxation (ASE reuses it at step 0,
    # so no extra forward pass is incurred).
    initial_atoms = atoms.copy()
    atoms.get_potential_energy()

    t0 = time.time()
    try:
        converged, relaxed = eff_relaxer.relax(
            atoms, steps=steps, fmax=fmax,
            params_filter=eff_params, verbose=False,
        )
        elapsed = time.time() - t0

        final_forces = relaxed.get_forces()
        final_energy = float(relaxed.get_potential_energy())

        task_doc = build_relax_task_doc(
            initial_atoms=initial_atoms,
            relaxed_atoms=relaxed,
            converged=converged,
            elapsed=elapsed,
            fmax=fmax,
            steps=steps,
            relax_cell=is_periodic and (eff_relaxer.filter is not None),
            constrain_symmetry=constrain_symmetry,
        )

        result = {
            "index": i,
            "formula": relaxed.get_chemical_formula(),
            "converged": converged,
            "energy_eV": final_energy,
            "energy_per_atom_eV": final_energy / len(relaxed),
            "max_force_eV_per_A": float(np.max(np.linalg.norm(final_forces, axis=1))),
            "rms_force_eV_per_A": float(np.sqrt(np.mean(final_forces**2))),
            "n_atoms": len(relaxed),
            "elapsed_seconds": round(elapsed, 2),
            "relaxed_atoms": relaxed,
            "task_doc": task_doc,
        }

    except RuntimeError as e:
        elapsed = time.time() - t0
        result = {
            "index": i,
            "formula": atoms.get_chemical_formula(),
            "converged": False,
            "error": str(e),
            "elapsed_seconds": round(elapsed, 2),
            "relaxed_atoms": atoms,
            "task_doc": None,
        }
        if "out of memory" in str(e).lower() and torch.cuda.is_available():
            torch.cuda.empty_cache()

    results.append(result)

# =============================================================================
# Step 4 — Print results to terminal
# =============================================================================

n_converged = sum(1 for r in results if r.get("converged"))

print("\n" + "=" * 80)
print("STRUCTURE RELAXATION RESULTS")
print("=" * 80)
print(f"Model: {model_name}  |  Device: {device}  |  Optimizer: {optimizer}")
print(f"fmax: {fmax} eV/Å  |  Max steps: {steps}")
if constrain_symmetry:
    print("Symmetry: constrained (FixSymmetry)")
if filter_name:
    pressure_str = f"  |  Pressure: {pressure} {pressure_unit}" if pressure else ""
    print(f"Cell filter: {filter_name}{pressure_str}")
print("-" * 80)
print(f"{'#':>3} {'Formula':<16} {'Conv':>5} {'Energy (eV)':>14} "
      f"{'E/atom (eV)':>13} {'F_max (eV/Å)':>13} {'Time (s)':>9}")
print("-" * 80)

for r in results:
    if "error" in r:
        print(f"{r['index']:>3} {r['formula']:<16} {'ERR':>5} {'—':>14} "
              f"{'—':>13} {'—':>13} {r['elapsed_seconds']:>9.1f}"
              f"  ⚠ {r['error'][:40]}")
    else:
        conv = "  ✓" if r["converged"] else "  ✗"
        print(f"{r['index']:>3} {r['formula']:<16} {conv:>5} "
              f"{r['energy_eV']:>14.6f} {r['energy_per_atom_eV']:>13.6f} "
              f"{r['max_force_eV_per_A']:>13.6f} {r['elapsed_seconds']:>9.1f}")

print("=" * 80)
print(f"Converged: {n_converged}/{len(results)}")

# =============================================================================
# Step 5 — Save results
# =============================================================================

timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
output_dir = os.path.join("mattersim_relax_results", timestamp)
os.makedirs(output_dir, exist_ok=True)

for r in results:
    stem = r["formula"].replace(" ", "")
    prefix = f"{r['index']:03d}_{stem}"

    # --- ForceFieldTaskDocument JSON (atomate2-compatible) ---
    if r["task_doc"] is not None:
        task_json_path = os.path.join(output_dir, f"task_{prefix}.json")
        with open(task_json_path, "w") as f:
            f.write(r["task_doc"].model_dump_json(indent=2))

    # --- Structure file (CIF for periodic, XYZ for non-periodic) ---
    if "error" not in r:
        relaxed = r["relaxed_atoms"]
        if all(relaxed.pbc):
            ase_write(os.path.join(output_dir, f"relax_{prefix}.cif"), relaxed)
        else:
            ase_write(os.path.join(output_dir, f"relax_{prefix}.xyz"), relaxed)

print(f"Results saved to: {output_dir}/")
for fn in sorted(os.listdir(output_dir)):
    print(f"  {fn}")
print("=" * 80)
