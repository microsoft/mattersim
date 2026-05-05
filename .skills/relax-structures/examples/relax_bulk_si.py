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
import os
import time
from datetime import datetime, timezone

import numpy as np
import torch
from ase.build import bulk
from ase.io import write as ase_write

from mattersim.applications.relax import Relaxer
from mattersim.applications.schemas import build_relax_task_doc
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
        is_periodic = all(a.pbc)

        if not is_periodic:
            eff_relaxer = Relaxer(
                optimizer=args.optimizer, filter=None,
                constrain_symmetry=args.constrain_symmetry,
            )
            eff_params = {}
        else:
            eff_relaxer = relaxer
            eff_params = params_filter

        initial_atoms = a.copy()
        a.get_potential_energy()  # cache initial energy for build_relax_task_doc

        t0 = time.time()
        try:
            converged, relaxed = eff_relaxer.relax(
                a, steps=args.steps, fmax=args.fmax,
                params_filter=eff_params, verbose=False,
            )
            elapsed = time.time() - t0

            final_energy = float(relaxed.get_potential_energy())
            final_forces = relaxed.get_forces()

            task_doc = build_relax_task_doc(
                initial_atoms=initial_atoms,
                relaxed_atoms=relaxed,
                converged=converged,
                elapsed=elapsed,
                fmax=args.fmax,
                steps=args.steps,
                relax_cell=is_periodic and (eff_relaxer.filter is not None),
                constrain_symmetry=args.constrain_symmetry,
            )

            results.append({
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
            })

        except RuntimeError as e:
            elapsed = time.time() - t0
            results.append({
                "index": i,
                "formula": a.get_chemical_formula(),
                "converged": False,
                "error": str(e),
                "elapsed_seconds": round(elapsed, 2),
                "relaxed_atoms": a,
                "task_doc": None,
            })
            if "out of memory" in str(e).lower() and torch.cuda.is_available():
                torch.cuda.empty_cache()

    # --- Print results ---
    n_converged = sum(1 for r in results if r.get("converged"))

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

    # --- Save results ---
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or os.path.join("mattersim_relax_results", timestamp)
    os.makedirs(output_dir, exist_ok=True)

    for r in results:
        stem = r["formula"].replace(" ", "")
        prefix = f"{r['index']:03d}_{stem}"

        if r["task_doc"] is not None:
            task_json_path = os.path.join(output_dir, f"task_{prefix}.json")
            with open(task_json_path, "w") as f:
                f.write(r["task_doc"].model_dump_json(indent=2))

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


if __name__ == "__main__":
    main()
