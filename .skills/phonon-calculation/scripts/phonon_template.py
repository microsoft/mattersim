"""
Reference implementation for the phonon-calculation skill.

AI agents should adapt this script to the user's specific request, modifying
the structure construction, model selection, relaxation settings, and phonon
parameters as needed.

This script is NOT meant to be run directly — see examples/phonon_bulk_si.py
for a runnable example. This file serves as the canonical template that agents
copy and adapt.
"""

import json
import os
import time
from datetime import datetime, timezone

import matplotlib
matplotlib.use("Agg")  # headless-safe backend — must be before any pyplot import

import numpy as np
import torch
from ase.io import read as ase_read
from ase.io import write as ase_write
from ase.units import GPa as GPa_unit

from mattersim.applications.phonon import PhononWorkflow
from mattersim.applications.relax import Relaxer
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
# Step 3 — (Optional) Relax structures
# =============================================================================
# Set relax_first=False to skip relaxation.

relax_first = True
relax_optimizer = "FIRE"
relax_fmax = 0.01               # eV/Å
relax_steps = 500
relax_filter = "FrechetCellFilter"  # cell relaxation on by default for phonon
relax_pressure = None               # external pressure value
relax_pressure_unit = "GPa"         # "GPa", "kbar", or "eV/A^3"
relax_constrain_symmetry = True     # preserve symmetry by default for phonon

# --- Convert pressure to eV/Å³ ---
relax_params_filter = {}
if relax_pressure is not None:
    from ase.units import GPa as _GPa
    if relax_pressure_unit == "GPa":
        scalar_pressure = relax_pressure * _GPa
    elif relax_pressure_unit == "kbar":
        scalar_pressure = relax_pressure * _GPa / 10.0
    elif relax_pressure_unit == "eV/A^3":
        scalar_pressure = relax_pressure
    else:
        raise ValueError(f"Unknown pressure_unit: {relax_pressure_unit!r}")
    relax_params_filter["scalar_pressure"] = scalar_pressure
    if relax_filter is None:
        relax_filter = "FrechetCellFilter"
elif relax_filter is not None:
    relax_params_filter["scalar_pressure"] = 0.0

# =============================================================================
# Step 4 — Phonon parameters
# =============================================================================
# Adapt to the user's request.

find_prim = False
amplitude = 0.01                # Å, finite-difference displacement
supercell_matrix = None         # [n1, n2, n3] or [[...],[...],[...]] or None (auto)
qpoints_mesh = None             # [n1, n2, n3] or None (auto)
max_atoms = None                # int or None (auto)

# --- Validate array inputs ---
if supercell_matrix is not None:
    supercell_matrix = np.array(supercell_matrix, dtype=int)
    if supercell_matrix.shape not in ((3,), (3, 3)):
        raise ValueError(
            f"supercell_matrix must have shape (3,) or (3,3), "
            f"got {supercell_matrix.shape}"
        )
if qpoints_mesh is not None:
    qpoints_mesh = np.array(qpoints_mesh, dtype=int)
    if qpoints_mesh.shape != (3,):
        raise ValueError(
            f"qpoints_mesh must have shape (3,), got {qpoints_mesh.shape}"
        )

# =============================================================================
# Step 5 — Run relaxation + phonon for each structure
# =============================================================================

timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
output_dir = os.path.abspath(os.path.join("mattersim_phonon_results", timestamp))
os.makedirs(output_dir, exist_ok=True)

results = []
for i, atoms in enumerate(atoms_list):
    formula = atoms.get_chemical_formula()
    struct_dir = os.path.join(output_dir, f"{i:03d}_{formula}")
    os.makedirs(struct_dir, exist_ok=True)

    # --- Validate periodicity ---
    if not all(atoms.pbc):
        results.append({
            "index": i,
            "formula": formula,
            "error": "Structure is not fully periodic (pbc must be True in all "
                     "directions). Phonon calculations require periodic structures.",
        })
        continue

    relax_info = None
    phonon_atoms = atoms  # may be replaced by relaxed structure

    # --- Pre-phonon relaxation ---
    if relax_first:
        relax_dir = os.path.join(struct_dir, "relax")
        os.makedirs(relax_dir, exist_ok=True)

        relaxer = Relaxer(
            optimizer=relax_optimizer,
            filter=relax_filter,
            constrain_symmetry=relax_constrain_symmetry,
        )

        t0 = time.time()
        try:
            converged, relaxed = relaxer.relax(
                atoms, steps=relax_steps, fmax=relax_fmax,
                params_filter=relax_params_filter, verbose=False,
            )
            relax_elapsed = time.time() - t0
            forces = relaxed.get_forces()
            energy = relaxed.get_potential_energy()
            max_force = float(np.max(np.linalg.norm(forces, axis=1)))

            relax_info = {
                "ran": True,
                "converged": converged,
                "energy_eV": float(energy),
                "energy_per_atom_eV": float(energy / len(relaxed)),
                "max_force_eV_per_A": max_force,
                "elapsed_seconds": round(relax_elapsed, 2),
                "relaxed_cell": relaxed.cell.tolist(),
                "relaxed_positions": relaxed.positions.tolist(),
                "relaxed_symbols": list(relaxed.symbols),
            }

            # Save relaxed structure
            ase_write(os.path.join(struct_dir, "relaxed.cif"), relaxed)

            # Save relax JSON
            relax_json = {
                "converged": converged,
                "energy_eV": float(energy),
                "max_force_eV_per_A": max_force,
                "optimizer": relax_optimizer,
                "fmax": relax_fmax,
                "steps": relax_steps,
                "filter": relax_filter,
            }
            with open(os.path.join(relax_dir, "relax_results.json"), "w") as f:
                json.dump(relax_json, f, indent=2)

            if not converged:
                print(f"  ⚠ Structure {i} ({formula}): relaxation did not "
                      f"converge (max_force={max_force:.4f} eV/Å)")

            # Use relaxed structure for phonon
            phonon_atoms = relaxed

        except RuntimeError as e:
            relax_elapsed = time.time() - t0
            relax_info = {
                "ran": True,
                "converged": False,
                "error": str(e),
                "elapsed_seconds": round(relax_elapsed, 2),
            }
            if "out of memory" in str(e).lower() and torch.cuda.is_available():
                torch.cuda.empty_cache()
            results.append({
                "index": i,
                "formula": formula,
                "relaxation": relax_info,
                "phonon": None,
                "error": f"Relaxation failed: {e}",
            })
            continue

    # --- Run phonon ---
    phonon_dir = os.path.join(struct_dir, "phonon")
    os.makedirs(phonon_dir, exist_ok=True)

    # Ensure calculator is attached
    phonon_atoms.calc = calc

    t0 = time.time()
    try:
        workflow = PhononWorkflow(
            atoms=phonon_atoms,
            find_prim=find_prim,
            work_dir=os.path.abspath(phonon_dir),
            amplitude=amplitude,
            supercell_matrix=supercell_matrix,
            qpoints_mesh=qpoints_mesh,
            max_atoms=max_atoms,
        )
        has_imaginary, phonon_obj = workflow.run()
        phonon_elapsed = time.time() - t0

        # Collect output files
        output_files = [
            f for f in os.listdir(phonon_dir)
            if os.path.isfile(os.path.join(phonon_dir, f))
        ]

        phonon_info = {
            "has_imaginary": bool(has_imaginary) if isinstance(has_imaginary, bool) else None,
            "supercell_matrix": workflow.supercell_matrix.tolist()
                if isinstance(workflow.supercell_matrix, np.ndarray)
                else workflow.supercell_matrix,
            "amplitude": amplitude,
            "find_prim": find_prim,
            "elapsed_seconds": round(phonon_elapsed, 2),
            "work_dir": phonon_dir,
            "output_files": sorted(output_files),
        }

    except Exception as e:
        phonon_elapsed = time.time() - t0
        phonon_info = {
            "has_imaginary": None,
            "error": str(e),
            "elapsed_seconds": round(phonon_elapsed, 2),
        }
        if "out of memory" in str(e).lower() and torch.cuda.is_available():
            torch.cuda.empty_cache()

    results.append({
        "index": i,
        "formula": formula,
        "relaxation": relax_info,
        "phonon": phonon_info,
    })

# =============================================================================
# Step 6 — Print results to terminal
# =============================================================================

print("\n" + "=" * 80)
print("PHONON CALCULATION RESULTS")
print("=" * 80)
print(f"Model: {model_name}  |  Device: {device}")
if relax_first:
    print(f"Relaxation: {relax_optimizer}, fmax={relax_fmax} eV/Å, "
          f"filter={relax_filter}, symmetry={'on' if relax_constrain_symmetry else 'off'}")
else:
    print("Relaxation: skipped")
print(f"Phonon: amplitude={amplitude} Å, find_prim={find_prim}, "
      f"supercell={'auto' if supercell_matrix is None else supercell_matrix.tolist()}")
print("-" * 80)
print(f"{'#':>3} {'Formula':<16} {'Relax':>8} {'Imaginary':>10} "
      f"{'Supercell':>16} {'Time (s)':>10}")
print("-" * 80)

for r in results:
    if "error" in r and r.get("phonon") is None:
        print(f"{r['index']:>3} {r['formula']:<16} "
              f"{'—':>8} {'—':>10} {'—':>16} {'—':>10}"
              f"  ⚠ {r['error'][:40]}")
        continue

    relax_str = "—"
    if r.get("relaxation"):
        relax_str = "✓" if r["relaxation"].get("converged") else "✗"

    ph = r.get("phonon", {})
    if ph.get("error"):
        imag_str = "ERR"
        sc_str = "—"
        time_str = f"{ph.get('elapsed_seconds', 0):>10.1f}"
    else:
        imag = ph.get("has_imaginary")
        imag_str = "Yes" if imag else ("No" if imag is False else "—")
        sc = ph.get("supercell_matrix")
        if sc is not None:
            if isinstance(sc, list) and len(sc) == 3 and not isinstance(sc[0], list):
                sc_str = f"{sc[0]}×{sc[1]}×{sc[2]}"
            else:
                sc_str = str(sc)
        else:
            sc_str = "auto"
        total_time = ph.get("elapsed_seconds", 0)
        if r.get("relaxation"):
            total_time += r["relaxation"].get("elapsed_seconds", 0)
        time_str = f"{total_time:>10.1f}"

    print(f"{r['index']:>3} {r['formula']:<16} {relax_str:>8} {imag_str:>10} "
          f"{sc_str:>16} {time_str}")

print("=" * 80)
print(f"Results saved to: {output_dir}/")
print("=" * 80)

# =============================================================================
# Step 7 — Save combined JSON
# =============================================================================

output_data = {
    "schema_version": "1.0",
    "task": "phonon_calculation",
    "metadata": {
        "model": model_name,
        "device": device,
        "relax_first": relax_first,
        "relaxation_params": {
            "optimizer": relax_optimizer,
            "fmax_eV_per_A": relax_fmax,
            "max_steps": relax_steps,
            "filter": relax_filter,
            "pressure": relax_pressure,
            "pressure_unit": relax_pressure_unit,
            "constrain_symmetry": relax_constrain_symmetry,
        } if relax_first else None,
        "phonon_params": {
            "find_prim": find_prim,
            "amplitude_A": amplitude,
            "supercell_matrix": supercell_matrix.tolist()
                if isinstance(supercell_matrix, np.ndarray) else supercell_matrix,
            "qpoints_mesh": qpoints_mesh.tolist()
                if isinstance(qpoints_mesh, np.ndarray) else qpoints_mesh,
            "max_atoms": max_atoms,
        },
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "n_structures": len(results),
    },
    "units": {
        "energy": "eV",
        "forces": "eV/Å",
        "stress": "GPa",
        "positions": "Å",
        "cell": "Å",
        "frequency": "THz",
    },
    "structures": [],
}

for r in results:
    idx = r["index"]
    entry = {
        "index": idx,
        "formula": r["formula"],
        "input": {
            "formula": atoms_list[idx].get_chemical_formula(),
            "n_atoms": len(atoms_list[idx]),
            "pbc": atoms_list[idx].pbc.tolist(),
            "cell": atoms_list[idx].cell.tolist(),
            "symbols": list(atoms_list[idx].symbols),
            "positions": atoms_list[idx].positions.tolist(),
        },
        "relaxation": r.get("relaxation"),
        "phonon": r.get("phonon"),
    }
    if r.get("error"):
        entry["error"] = r["error"]
    output_data["structures"].append(entry)

with open(os.path.join(output_dir, "phonon_results.json"), "w") as f:
    json.dump(output_data, f, indent=2)

print(f"\nJSON saved: {os.path.join(output_dir, 'phonon_results.json')}")
