"""
Example: Phonon calculation for bulk silicon with MatterSim.

This example relaxes a bulk Si primitive cell, then computes phonon properties
(band structure, DOS, force constants) using phonopy.

Usage:
    python .skills/phonon-calculation/examples/phonon_bulk_si.py
    python .skills/phonon-calculation/examples/phonon_bulk_si.py --model mattersim-v1.0.0-5m
    python .skills/phonon-calculation/examples/phonon_bulk_si.py --no-relax
    python .skills/phonon-calculation/examples/phonon_bulk_si.py --supercell 2 2 2
    python .skills/phonon-calculation/examples/phonon_bulk_si.py --find-prim
"""

import argparse
import json
import os
import time
from datetime import datetime, timezone

import matplotlib
matplotlib.use("Agg")

import numpy as np
import torch
from ase.build import bulk
from ase.io import write as ase_write
from ase.units import GPa as GPa_unit

from mattersim.applications.phonon import PhononWorkflow
from mattersim.applications.relax import Relaxer
from mattersim.forcefield import MatterSimCalculator


def main():
    parser = argparse.ArgumentParser(
        description="Phonon calculation for bulk Si with MatterSim"
    )
    # Model / device
    parser.add_argument("--model", default="mattersim-v1.0.0-1m",
                        choices=["mattersim-v1.0.0-1m", "mattersim-v1.0.0-5m"])
    parser.add_argument("--device", default="auto",
                        choices=["auto", "cpu", "cuda"])
    # Relaxation
    parser.add_argument("--no-relax", action="store_true",
                        help="Skip pre-phonon relaxation.")
    parser.add_argument("--relax-optimizer", default="FIRE",
                        choices=["FIRE", "BFGS"])
    parser.add_argument("--relax-fmax", type=float, default=0.01)
    parser.add_argument("--relax-steps", type=int, default=500)
    parser.add_argument("--relax-filter", default="FrechetCellFilter",
                        choices=["FrechetCellFilter", "ExpCellFilter", "none"])
    parser.add_argument("--relax-pressure", type=float, default=None)
    parser.add_argument("--relax-pressure-unit", default="GPa",
                        choices=["GPa", "kbar", "eV/A^3"])
    # Phonon
    parser.add_argument("--find-prim", action="store_true")
    parser.add_argument("--amplitude", type=float, default=0.01)
    parser.add_argument("--supercell", type=int, nargs=3, default=None,
                        metavar=("N1", "N2", "N3"),
                        help="Supercell matrix diagonal [n1 n2 n3].")
    parser.add_argument("--qmesh", type=int, nargs=3, default=None,
                        metavar=("Q1", "Q2", "Q3"),
                        help="Q-point mesh [q1 q2 q3].")
    parser.add_argument("--max-atoms", type=int, default=None)
    # Output
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    relax_first = not args.no_relax
    relax_filter = None if args.relax_filter == "none" else args.relax_filter

    supercell_matrix = np.array(args.supercell, dtype=int) if args.supercell else None
    qpoints_mesh = np.array(args.qmesh, dtype=int) if args.qmesh else None

    # --- Build structure ---
    atoms = bulk("Si", "diamond", a=5.43, cubic=True)
    atoms_list = [atoms]

    # --- Load calculator ---
    calc = MatterSimCalculator(load_path=args.model, device=device)
    for a in atoms_list:
        a.calc = calc

    # --- Output directory ---
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or os.path.join(
        "mattersim_phonon_results", timestamp
    )
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # --- Process each structure ---
    results = []
    for i, atoms in enumerate(atoms_list):
        formula = atoms.get_chemical_formula()
        struct_dir = os.path.join(output_dir, f"{i:03d}_{formula}")
        os.makedirs(struct_dir, exist_ok=True)

        relax_info = None
        phonon_atoms = atoms

        # --- Relaxation ---
        if relax_first:
            relax_dir = os.path.join(struct_dir, "relax")
            os.makedirs(relax_dir, exist_ok=True)

            # Pressure conversion
            relax_params_filter = {}
            if args.relax_pressure is not None:
                from ase.units import GPa as _GPa
                if args.relax_pressure_unit == "GPa":
                    sp = args.relax_pressure * _GPa
                elif args.relax_pressure_unit == "kbar":
                    sp = args.relax_pressure * _GPa / 10.0
                elif args.relax_pressure_unit == "eV/A^3":
                    sp = args.relax_pressure
                else:
                    raise ValueError(f"Unknown unit: {args.relax_pressure_unit}")
                relax_params_filter["scalar_pressure"] = sp
                if relax_filter is None:
                    relax_filter = "FrechetCellFilter"
            elif relax_filter is not None:
                relax_params_filter["scalar_pressure"] = 0.0

            relaxer = Relaxer(
                optimizer=args.relax_optimizer,
                filter=relax_filter,
                constrain_symmetry=True,
            )

            print(f"[{i}] Relaxing {formula}...")
            t0 = time.time()
            converged, relaxed = relaxer.relax(
                atoms, steps=args.relax_steps, fmax=args.relax_fmax,
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
            ase_write(os.path.join(struct_dir, "relaxed.cif"), relaxed)

            conv_str = "✓" if converged else "✗"
            print(f"    Relaxation: {conv_str}  E={energy:.6f} eV  "
                  f"F_max={max_force:.6f} eV/Å  ({relax_elapsed:.1f}s)")

            phonon_atoms = relaxed

        # --- Phonon ---
        phonon_dir = os.path.join(struct_dir, "phonon")
        os.makedirs(phonon_dir, exist_ok=True)
        phonon_atoms.calc = calc

        print(f"[{i}] Running phonon for {formula}...")
        t0 = time.time()
        try:
            workflow = PhononWorkflow(
                atoms=phonon_atoms,
                find_prim=args.find_prim,
                work_dir=os.path.abspath(phonon_dir),
                amplitude=args.amplitude,
                supercell_matrix=supercell_matrix,
                qpoints_mesh=qpoints_mesh,
                max_atoms=args.max_atoms,
            )
            has_imaginary, phonon_obj = workflow.run()
            phonon_elapsed = time.time() - t0

            output_files = sorted([
                f for f in os.listdir(phonon_dir)
                if os.path.isfile(os.path.join(phonon_dir, f))
            ])

            phonon_info = {
                "has_imaginary": bool(has_imaginary) if isinstance(has_imaginary, bool) else None,
                "supercell_matrix": workflow.supercell_matrix.tolist()
                    if isinstance(workflow.supercell_matrix, np.ndarray)
                    else workflow.supercell_matrix,
                "amplitude": args.amplitude,
                "find_prim": args.find_prim,
                "elapsed_seconds": round(phonon_elapsed, 2),
                "output_files": output_files,
            }

            imag_str = "Yes ⚠" if has_imaginary else "No ✓"
            print(f"    Phonon: imaginary={imag_str}  ({phonon_elapsed:.1f}s)")

        except Exception as e:
            phonon_elapsed = time.time() - t0
            phonon_info = {
                "has_imaginary": None,
                "error": str(e),
                "elapsed_seconds": round(phonon_elapsed, 2),
            }
            print(f"    Phonon ERROR: {e}")

        results.append({
            "index": i,
            "formula": formula,
            "relaxation": relax_info,
            "phonon": phonon_info,
        })

    # --- Print summary ---
    print("\n" + "=" * 80)
    print("PHONON CALCULATION RESULTS")
    print("=" * 80)
    print(f"Model: {args.model}  |  Device: {device}")
    if relax_first:
        print(f"Relaxation: {args.relax_optimizer}, fmax={args.relax_fmax} eV/Å, "
              f"filter={relax_filter}, symmetry=on")
    else:
        print("Relaxation: skipped")
    print(f"Phonon: amplitude={args.amplitude} Å, find_prim={args.find_prim}")
    print("-" * 80)
    print(f"{'#':>3} {'Formula':<16} {'Relax':>8} {'Imaginary':>10} "
          f"{'Supercell':>16} {'Time (s)':>10}")
    print("-" * 80)

    for r in results:
        rlx = r.get("relaxation")
        relax_str = "—"
        if rlx:
            relax_str = "✓" if rlx.get("converged") else "✗"

        ph = r.get("phonon", {})
        if ph.get("error"):
            print(f"{r['index']:>3} {r['formula']:<16} {relax_str:>8} "
                  f"{'ERR':>10} {'—':>16} {'—':>10}  ⚠ {ph['error'][:30]}")
        else:
            imag = ph.get("has_imaginary")
            imag_str = "Yes" if imag else ("No" if imag is False else "—")
            sc = ph.get("supercell_matrix")
            if isinstance(sc, list) and len(sc) == 3 and not isinstance(sc[0], list):
                sc_str = f"{sc[0]}×{sc[1]}×{sc[2]}"
            else:
                sc_str = str(sc) if sc else "auto"
            total_time = ph.get("elapsed_seconds", 0)
            if rlx:
                total_time += rlx.get("elapsed_seconds", 0)
            print(f"{r['index']:>3} {r['formula']:<16} {relax_str:>8} "
                  f"{imag_str:>10} {sc_str:>16} {total_time:>10.1f}")

    print("=" * 80)

    # --- Save JSON ---
    output_data = {
        "schema_version": "1.0",
        "task": "phonon_calculation",
        "metadata": {
            "model": args.model,
            "device": device,
            "relax_first": relax_first,
            "phonon_params": {
                "find_prim": args.find_prim,
                "amplitude_A": args.amplitude,
                "supercell_matrix": supercell_matrix.tolist() if supercell_matrix is not None else None,
                "qpoints_mesh": qpoints_mesh.tolist() if qpoints_mesh is not None else None,
                "max_atoms": args.max_atoms,
            },
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "n_structures": len(results),
        },
        "units": {
            "energy": "eV", "forces": "eV/Å", "stress": "GPa",
            "positions": "Å", "cell": "Å", "frequency": "THz",
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
        output_data["structures"].append(entry)

    with open(os.path.join(output_dir, "phonon_results.json"), "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"Results saved to: {output_dir}/")
    for fn in sorted(os.listdir(output_dir)):
        fpath = os.path.join(output_dir, fn)
        if os.path.isdir(fpath):
            subfiles = sorted(os.listdir(fpath))
            print(f"  {fn}/")
            for sf in subfiles:
                sfpath = os.path.join(fpath, sf)
                if os.path.isdir(sfpath):
                    print(f"    {sf}/")
                    for ssf in sorted(os.listdir(sfpath)):
                        print(f"      {ssf}")
                else:
                    print(f"    {sf}")
        else:
            print(f"  {fn}")
    print("=" * 80)


if __name__ == "__main__":
    main()
