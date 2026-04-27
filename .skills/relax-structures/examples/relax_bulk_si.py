"""
Example: Relax a bulk silicon cubic cell with MatterSim.

Usage:
    python .skills/relax-structures/examples/relax_bulk_si.py
    python .skills/relax-structures/examples/relax_bulk_si.py --model mattersim-v1.0.0-5m
    python .skills/relax-structures/examples/relax_bulk_si.py --fmax 0.001 --steps 1000
    python .skills/relax-structures/examples/relax_bulk_si.py --pressure 10 --pressure-unit GPa
    python .skills/relax-structures/examples/relax_bulk_si.py --constrain-symmetry
"""

import argparse
import json
import os
import time
from datetime import datetime, timezone

import numpy as np
import torch
from ase.build import bulk
from ase.io import write as ase_write
from ase.units import GPa as GPa_unit

from mattersim.applications.relax import Relaxer
from mattersim.forcefield import MatterSimCalculator


def main():
    parser = argparse.ArgumentParser(description="Relax bulk Si with MatterSim")
    parser.add_argument("--model", default="mattersim-v1.0.0-1m",
                        choices=["mattersim-v1.0.0-1m", "mattersim-v1.0.0-5m"])
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--optimizer", default="FIRE", choices=["FIRE", "BFGS"])
    parser.add_argument("--fmax", type=float, default=0.01)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--pressure", type=float, default=None,
                        help="External target pressure for cell relaxation.")
    parser.add_argument("--pressure-unit", default="GPa",
                        choices=["GPa", "kbar", "eV/A^3"],
                        help="Unit of --pressure value.")
    parser.add_argument("--constrain-symmetry", action="store_true",
                        help="Preserve space-group symmetry during relaxation.")
    parser.add_argument("--filter", default=None,
                        choices=["FrechetCellFilter", "ExpCellFilter"],
                        help="Cell filter for variable-cell relaxation.")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Build structure ---
    atoms = bulk("Si", "diamond", a=5.43, cubic=True)
    atoms_list = [atoms]

    # --- Load calculator ---
    calc = MatterSimCalculator(load_path=args.model, device=device)
    for a in atoms_list:
        a.calc = calc

    # --- Configure relaxation ---
    filter_name = args.filter
    params_filter = {}
    if args.pressure is not None:
        from ase.units import GPa as _GPa
        if args.pressure_unit == "GPa":
            scalar_pressure = args.pressure * _GPa
        elif args.pressure_unit == "kbar":
            scalar_pressure = args.pressure * _GPa / 10.0
        elif args.pressure_unit == "eV/A^3":
            scalar_pressure = args.pressure
        else:
            raise ValueError(f"Unknown pressure unit: {args.pressure_unit}")
        params_filter["scalar_pressure"] = scalar_pressure
        if filter_name is None:
            filter_name = "FrechetCellFilter"
    elif filter_name is not None:
        params_filter["scalar_pressure"] = 0.0

    relaxer = Relaxer(
        optimizer=args.optimizer,
        filter=filter_name,
        constrain_symmetry=args.constrain_symmetry,
    )
    results = []

    for i, a in enumerate(atoms_list):
        t0 = time.time()
        converged, relaxed = relaxer.relax(
            a, steps=args.steps, fmax=args.fmax,
            params_filter=params_filter, verbose=False,
        )
        elapsed = time.time() - t0
        forces = relaxed.get_forces()
        energy = relaxed.get_potential_energy()
        stress_voigt = relaxed.get_stress()

        results.append({
            "index": i,
            "formula": relaxed.get_chemical_formula(),
            "converged": converged,
            "energy_eV": float(energy),
            "energy_per_atom_eV": float(energy / len(relaxed)),
            "max_force_eV_per_A": float(np.max(np.linalg.norm(forces, axis=1))),
            "rms_force_eV_per_A": float(np.sqrt(np.mean(forces**2))),
            "stress_GPa": (stress_voigt / GPa_unit).tolist(),
            "n_atoms": len(relaxed),
            "elapsed_seconds": round(elapsed, 2),
            "relaxed_atoms": relaxed,
        })

    # --- Print results ---
    print("\n" + "=" * 80)
    print("STRUCTURE RELAXATION RESULTS")
    print("=" * 80)
    print(f"Model: {args.model}  |  Device: {device}  |  Optimizer: {args.optimizer}")
    print(f"fmax: {args.fmax} eV/Å  |  Max steps: {args.steps}")
    if args.constrain_symmetry:
        print("Symmetry: constrained (FixSymmetry)")
    if filter_name:
        pressure_str = (f"  |  Pressure: {args.pressure} {args.pressure_unit}"
                        if args.pressure else "")
        print(f"Cell filter: {filter_name}{pressure_str}")
    print("-" * 80)
    print(f"{'#':>3} {'Formula':<16} {'Conv':>5} {'Energy (eV)':>14} "
          f"{'E/atom (eV)':>13} {'F_max (eV/Å)':>13} {'Time (s)':>9}")
    print("-" * 80)

    for r in results:
        conv = "  ✓" if r["converged"] else "  ✗"
        print(f"{r['index']:>3} {r['formula']:<16} {conv:>5} "
              f"{r['energy_eV']:>14.6f} {r['energy_per_atom_eV']:>13.6f} "
              f"{r['max_force_eV_per_A']:>13.6f} {r['elapsed_seconds']:>9.1f}")

    n_converged = sum(1 for r in results if r.get("converged"))
    print("=" * 80)
    print(f"Converged: {n_converged}/{len(results)}")

    # --- Save results ---
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or os.path.join("mattersim_relax_results", timestamp)
    os.makedirs(output_dir, exist_ok=True)

    output_data = {
        "schema_version": "1.0",
        "task": "structure_relaxation",
        "metadata": {
            "model": args.model,
            "device": device,
            "optimizer": args.optimizer,
            "filter": filter_name,
            "fmax_eV_per_A": args.fmax,
            "max_steps": args.steps,
            "pressure": args.pressure,
            "pressure_unit": args.pressure_unit,
            "constrain_symmetry": args.constrain_symmetry,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "n_structures": len(results),
            "n_converged": n_converged,
        },
        "units": {
            "energy": "eV",
            "forces": "eV/Å",
            "stress": "GPa",
            "positions": "Å",
            "cell": "Å",
        },
        "structures": [],
    }

    for r in results:
        idx = r["index"]
        relaxed = r["relaxed_atoms"]
        entry = {
            "index": idx,
            "input": {
                "formula": atoms_list[idx].get_chemical_formula(),
                "n_atoms": len(atoms_list[idx]),
                "pbc": atoms_list[idx].pbc.tolist(),
                "cell": atoms_list[idx].cell.tolist(),
                "symbols": list(atoms_list[idx].symbols),
                "positions": atoms_list[idx].positions.tolist(),
            },
            "result": {
                "converged": r["converged"],
                "energy_eV": r["energy_eV"],
                "energy_per_atom_eV": r["energy_per_atom_eV"],
                "max_force_eV_per_A": r["max_force_eV_per_A"],
                "rms_force_eV_per_A": r["rms_force_eV_per_A"],
                "stress_GPa": r["stress_GPa"],
                "elapsed_seconds": r["elapsed_seconds"],
                "n_atoms": len(relaxed),
                "formula": relaxed.get_chemical_formula(),
                "pbc": relaxed.pbc.tolist(),
                "cell": relaxed.cell.tolist(),
                "symbols": list(relaxed.symbols),
                "positions": relaxed.positions.tolist(),
                "forces": relaxed.get_forces().tolist(),
            },
        }
        output_data["structures"].append(entry)

    with open(os.path.join(output_dir, "relax_results.json"), "w") as f:
        json.dump(output_data, f, indent=2)

    for r in results:
        relaxed = r["relaxed_atoms"]
        stem = r["formula"].replace(" ", "")
        ase_write(os.path.join(output_dir, f"relax_{r['index']:03d}_{stem}.cif"),
                  relaxed)

    print(f"Results saved to: {output_dir}/")
    for fn in sorted(os.listdir(output_dir)):
        print(f"  {fn}")
    print("=" * 80)


if __name__ == "__main__":
    main()
