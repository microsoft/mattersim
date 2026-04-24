#!/usr/bin/env python
"""Benchmark graph construction: CPU vs GPU on Materials Project structures.

Downloads ~1000 structures from the Materials Project with a distribution
over different atom counts, then benchmarks:
  1. CPU graph construction (old GraphConvertor)
  2. GPU graph construction (new BatchGraphConverter)
  3. Full inference pipeline (graph + model forward)

Produces publication-quality comparison plots.

Usage:
    python scripts/benchmark_mp_structures.py
    python scripts/benchmark_mp_structures.py --num-structures 500 --output-dir results
"""

import argparse
import gc
import json
import os
import pickle
import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from ase import Atoms

matplotlib.use("Agg")

# ── Plot style ──────────────────────────────────────────────────────────────


def setup_plot_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
    })


# ── Data acquisition ────────────────────────────────────────────────────────


def download_mp_structures(
    num_structures: int = 1000,
    cache_path: str = "benchmark_results/mp_structures.pkl",
    seed: int = 42,
) -> list[Atoms]:
    """Download structures from Materials Project with good size distribution.

    Samples structures across atom-count bins to ensure coverage from
    small (1-10 atoms) to large (200+ atoms) systems.
    """
    if os.path.exists(cache_path):
        print(f"Loading cached structures from {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    from mp_api.client import MPRester
    from pymatgen.io.ase import AseAtomsAdaptor

    print("Downloading structures from Materials Project...")

    # Define bins and how many structures to sample per bin
    bins = [
        (1, 10, 100),
        (11, 20, 100),
        (21, 30, 100),
        (31, 50, 150),
        (51, 80, 150),
        (81, 120, 100),
        (121, 180, 100),
        (181, 300, 100),
        (301, 500, 50),
        (501, 1000, 50),
    ]

    rng = np.random.default_rng(seed)
    all_atoms = []

    with MPRester() as mpr:
        for lo, hi, target_count in bins:
            print(f"  Fetching structures with {lo}-{hi} atoms "
                  f"(target: {target_count})...", end="", flush=True)
            try:
                docs = mpr.materials.summary.search(
                    nsites=(lo, hi),
                    fields=["material_id", "structure", "nsites"],
                    num_chunks=None,
                )
                if len(docs) > target_count:
                    indices = rng.choice(len(docs), target_count, replace=False)
                    docs = [docs[i] for i in indices]

                for doc in docs:
                    try:
                        atoms = AseAtomsAdaptor.get_atoms(doc.structure)
                        atoms.info["mp_id"] = str(doc.material_id)
                        all_atoms.append(atoms)
                    except Exception:
                        continue
                print(f" got {len(docs)}")
            except Exception as e:
                print(f" error: {e}")
                continue

    print(f"Total structures downloaded: {len(all_atoms)}")

    # Shuffle
    rng.shuffle(all_atoms)

    # Cache
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(all_atoms, f)
    print(f"Cached to {cache_path}")

    return all_atoms


# ── Benchmarking ────────────────────────────────────────────────────────────


def benchmark_graph_construction(
    atoms_list: list[Atoms],
    cutoff: float = 5.0,
    threebody_cutoff: float = 4.0,
    n_bins: int = 12,
) -> dict:
    """Benchmark graph construction: CPU 1-by-1, GPU 1-by-1, GPU batched per bin.

    For each atom-count bin, structures are replicated to ≥200 so that actual
    GPU compute dominates over kernel launch overhead. All 3 methods are timed
    on the same replicated set for fair comparison.

    Returns dict with per-bin summaries and totals.
    """
    from mattersim.datasets.utils.build import build_dataloader

    # ── Per atom-count bin benchmarks ──
    # Replicate each bin to a target count so GPU compute (not launch overhead)
    # dominates the measurement.
    min_structs_per_bin = 200
    print(f"  [3/3] GPU batched (per atom-count bin, "
          f"min {min_structs_per_bin} structs/bin)...")
    sizes = np.array([len(a) for a in atoms_list])
    lo, hi = sizes.min(), sizes.max()
    bin_edges = np.logspace(np.log10(max(lo, 1)), np.log10(hi), n_bins + 1)
    bin_indices = np.digitize(sizes, bin_edges)

    per_bin = []  # list of dicts with bin-level results
    gpu_batch_total_t0 = time.perf_counter()
    for b in range(1, n_bins + 1):
        mask = bin_indices == b
        if mask.sum() < 1:
            continue
        bin_atoms_orig = [atoms_list[j] for j in np.where(mask)[0]]
        bin_n_orig = len(bin_atoms_orig)
        bin_center = float(np.mean(sizes[mask]))

        # Replicate to reach min_structs_per_bin
        reps = max(1, (min_structs_per_bin + bin_n_orig - 1) // bin_n_orig)
        bin_atoms = (bin_atoms_orig * reps)[:max(min_structs_per_bin, bin_n_orig)]
        bin_n = len(bin_atoms)

        # GPU batched timing
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        try:
            dl = build_dataloader(
                bin_atoms, batch_size=64, model_type="m3gnet",
                only_inference=True, cutoff=cutoff,
                threebody_cutoff=threebody_cutoff, batch_converter=True,
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            bin_gpu_batch_ms = (time.perf_counter() - t0) * 1000
        except Exception as e:
            bin_gpu_batch_ms = None
            print(f"    Bin {b} error: {e}")

        # CPU 1-by-1 timing on the same replicated set
        t0 = time.perf_counter()
        for atoms in bin_atoms:
            build_dataloader(
                [atoms], batch_size=1, model_type="m3gnet",
                only_inference=True, cutoff=cutoff,
                threebody_cutoff=threebody_cutoff, batch_converter=False,
            )
        bin_cpu_ms = (time.perf_counter() - t0) * 1000

        # GPU 1-by-1 timing on the same replicated set
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for atoms in bin_atoms:
            build_dataloader(
                [atoms], batch_size=1, model_type="m3gnet",
                only_inference=True, cutoff=cutoff,
                threebody_cutoff=threebody_cutoff, batch_converter=True,
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        bin_gpu1by1_ms = (time.perf_counter() - t0) * 1000

        per_bin.append({
            "bin_center": bin_center,
            "count": bin_n,
            "count_orig": bin_n_orig,
            "cpu_total_ms": bin_cpu_ms,
            "gpu_1by1_total_ms": bin_gpu1by1_ms,
            "gpu_batch_total_ms": bin_gpu_batch_ms,
            "cpu_per_struct_ms": bin_cpu_ms / bin_n if bin_n > 0 else 0,
            "gpu_1by1_per_struct_ms": bin_gpu1by1_ms / bin_n if bin_n > 0 else 0,
            "gpu_batch_per_struct_ms": (
                bin_gpu_batch_ms / bin_n
                if bin_gpu_batch_ms is not None and bin_n > 0
                else None
            ),
        })
        status = (
            f"    Bin ~{int(bin_center)} atoms "
            f"({bin_n_orig} orig → {bin_n} structs):"
        )
        status += f" cpu={bin_cpu_ms / bin_n:.2f}"
        status += f"  gpu_1by1={bin_gpu1by1_ms / bin_n:.2f}"
        if bin_gpu_batch_ms is not None:
            status += f"  gpu_batch={bin_gpu_batch_ms / bin_n:.2f} ms/struct"
        print(status, flush=True)

    # Compute totals from per-bin sums
    cpu_total_ms = sum(b["cpu_total_ms"] for b in per_bin)
    gpu1by1_total_ms = sum(b["gpu_1by1_total_ms"] for b in per_bin)
    gpu_batch_total_ms = sum(
        b["gpu_batch_total_ms"] for b in per_bin
        if b.get("gpu_batch_total_ms") is not None
    )
    total_structs = sum(b["count"] for b in per_bin)

    summary = {
        "per_bin": per_bin,
        "cpu_total_ms": cpu_total_ms,
        "gpu_1by1_total_ms": gpu1by1_total_ms,
        "gpu_batch_total_ms": gpu_batch_total_ms,
        "n_structures": len(atoms_list),
        "n_structures_benchmarked": total_structs,
    }
    return summary


def benchmark_md_simulation(
    atoms_list: list[Atoms],
    per_bin_info: list[dict],
    device: str = "cuda",
    md_steps: int = 10,
) -> list[dict]:
    """Run short MD simulations comparing CPU vs GPU graph construction.

    Picks one representative structure per bin and runs NVE MD for ``md_steps``
    steps using ``MatterSimCalculator`` with both ``batch_converter=False``
    (CPU graph) and ``batch_converter=True`` (GPU graph).

    Returns per-bin MD timing results.
    """
    from ase.md.verlet import VelocityVerlet
    from ase import units as ase_units

    from mattersim.forcefield.potential import MatterSimCalculator, Potential

    # Load model once
    potential = Potential.from_checkpoint(device=device)
    potential.model.eval()

    results = []
    sizes = np.array([len(a) for a in atoms_list])

    # Use the same bins as the graph benchmark
    n_bins = len(per_bin_info)
    lo, hi = sizes.min(), sizes.max()
    bin_edges = np.logspace(np.log10(max(lo, 1)), np.log10(hi), n_bins + 1)
    bin_indices = np.digitize(sizes, bin_edges)

    for b_idx, binfo in enumerate(per_bin_info):
        b = b_idx + 1  # bin_indices are 1-based from digitize
        mask = bin_indices == b
        if mask.sum() < 1:
            continue

        # Pick the structure closest to bin center
        bin_idxs = np.where(mask)[0]
        center = binfo["bin_center"]
        closest = bin_idxs[np.argmin(np.abs(sizes[bin_idxs] - center))]
        atoms_orig = atoms_list[closest].copy()
        n_atoms = len(atoms_orig)

        record = {"n_atoms": n_atoms, "bin_center": center}

        for mode, use_gpu in [("cpu_graph", False), ("gpu_graph", True)]:
            atoms = atoms_orig.copy()
            calc = MatterSimCalculator(
                potential=potential,
                compute_stress=True,
                device=device,
                batch_converter=use_gpu,
            )
            atoms.calc = calc

            # Initialize velocities
            from ase.md.velocitydistribution import (
                MaxwellBoltzmannDistribution,
            )

            MaxwellBoltzmannDistribution(atoms, temperature_K=300)

            dyn = VelocityVerlet(atoms, timestep=1.0 * ase_units.fs)

            # Warmup: 1 step
            dyn.run(1)
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            # Timed MD
            t0 = time.perf_counter()
            dyn.run(md_steps)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - t0) * 1000
            per_step_ms = elapsed_ms / md_steps

            record[f"{mode}_total_ms"] = elapsed_ms
            record[f"{mode}_per_step_ms"] = per_step_ms

        speedup = (
            record["cpu_graph_per_step_ms"] / record["gpu_graph_per_step_ms"]
            if record["gpu_graph_per_step_ms"] > 0
            else 0
        )
        record["speedup"] = speedup

        print(
            f"    {n_atoms:>4} atoms: "
            f"cpu={record['cpu_graph_per_step_ms']:.1f} ms/step  "
            f"gpu={record['gpu_graph_per_step_ms']:.1f} ms/step  "
            f"speedup={speedup:.2f}×"
        )
        results.append(record)

    del potential
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
    gc.collect()
    return results


def plot_md_benchmark(md_results: list[dict], output_path: str, md_steps: int):
    """Bar + line plot for MD benchmark."""
    setup_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

    n_atoms = [r["n_atoms"] for r in md_results]
    cpu_ms = [r["cpu_graph_per_step_ms"] for r in md_results]
    gpu_ms = [r["gpu_graph_per_step_ms"] for r in md_results]
    speedups = [r["speedup"] for r in md_results]
    x = np.arange(len(n_atoms))
    w = 0.35

    # Left: per-step time
    ax = axes[0]
    ax.bar(x - w / 2, cpu_ms, w, label="CPU Graph",
           color=COLORS["CPU 1-by-1"], edgecolor="white", linewidth=0.5)
    ax.bar(x + w / 2, gpu_ms, w, label="GPU Graph",
           color=COLORS["GPU Batched"], edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Number of Atoms")
    ax.set_ylabel("Time per MD Step (ms)")
    ax.set_title(f"(a) MD Simulation — {md_steps} NVE Steps per Structure")
    ax.set_xticks(x)
    ax.set_xticklabels(n_atoms)
    ax.legend(frameon=True, fancybox=True, framealpha=0.9)
    ax.set_ylim(bottom=0)

    # Right: speedup
    ax = axes[1]
    bars = ax.bar(x, speedups, 0.6, color=COLORS["GPU Batched"],
                  edgecolor="white", linewidth=0.5)
    ax.axhline(y=1.0, color="gray", linestyle=":", linewidth=1.5)
    for bar, s in zip(bars, speedups):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{s:.2f}×", ha="center", va="bottom", fontsize=10,
                fontweight="bold", color=COLORS["GPU Batched"])
    ax.set_xlabel("Number of Atoms")
    ax.set_ylabel("Speedup (CPU Graph / GPU Graph)")
    ax.set_title("(b) GPU Graph Speedup in MD")
    ax.set_xticks(x)
    ax.set_xticklabels(n_atoms)
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved → {output_path}")


def benchmark_full_inference(
    atoms_list: list[Atoms],
    potential,
    device: str = "cuda",
    cutoff: float = 5.0,
    threebody_cutoff: float = 4.0,
) -> dict:
    """Benchmark full pipeline: CPU 1-by-1, GPU 1-by-1, GPU batched."""
    from mattersim.datasets.utils.build import build_dataloader
    from mattersim.forcefield.potential import batch_to_dict

    per_structure = []
    batch_size = 64

    # ── 1) CPU one-by-one (graph + inference) ──
    print("\n  [1/3] CPU graph → inference (one-by-one)...")
    cpu_total_t0 = time.perf_counter()
    for i, atoms in enumerate(atoms_list):
        record = {"n_atoms": len(atoms), "index": i}
        try:
            t0 = time.perf_counter()
            dl = build_dataloader(
                [atoms], batch_size=1, model_type="m3gnet",
                only_inference=True, cutoff=cutoff,
                threebody_cutoff=threebody_cutoff, batch_converter=False,
            )
            for batch in dl:
                inp = batch_to_dict(batch, device=device)
                res = potential.forward(
                    inp, include_forces=True, include_stresses=True,
                )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            record["cpu_total_ms"] = (t1 - t0) * 1000
            record["cpu_energy"] = res["total_energy"].detach().cpu().item()
        except Exception as e:
            record["cpu_total_ms"] = None
            record["cpu_error"] = str(e)
        per_structure.append(record)
        if (i + 1) % 200 == 0:
            print(f"    {i+1}/{len(atoms_list)}", flush=True)
    cpu_total_ms = (time.perf_counter() - cpu_total_t0) * 1000

    # ── 2) GPU one-by-one (graph + inference) ──
    print("  [2/3] GPU graph → inference (one-by-one)...")
    gpu1by1_total_t0 = time.perf_counter()
    for i, atoms in enumerate(atoms_list):
        try:
            t0 = time.perf_counter()
            dl = build_dataloader(
                [atoms], batch_size=1, model_type="m3gnet",
                only_inference=True, cutoff=cutoff,
                threebody_cutoff=threebody_cutoff, batch_converter=True,
            )
            for batch in dl:
                inp = batch_to_dict(batch, device=device)
                res = potential.forward(
                    inp, include_forces=True, include_stresses=True,
                )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            per_structure[i]["gpu_1by1_total_ms"] = (t1 - t0) * 1000
            per_structure[i]["gpu_energy"] = res["total_energy"].detach().cpu().item()
        except Exception as e:
            per_structure[i]["gpu_1by1_total_ms"] = None
            per_structure[i]["gpu_1by1_error"] = str(e)
        if (i + 1) % 200 == 0:
            print(f"    {i+1}/{len(atoms_list)}", flush=True)
    gpu1by1_total_ms = (time.perf_counter() - gpu1by1_total_t0) * 1000

    # ── 3) GPU batched (graph + inference, using dataloader batches) ──
    print("  [3/3] GPU batched graph → batched inference...")
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    gpu_batch_t0 = time.perf_counter()
    try:
        dl = build_dataloader(
            atoms_list, batch_size=batch_size, model_type="m3gnet",
            only_inference=True, cutoff=cutoff,
            threebody_cutoff=threebody_cutoff, batch_converter=True,
        )
        gpu_batch_energies = []
        n_failed = 0
        for batch_idx, batch in enumerate(dl):
            try:
                inp = batch_to_dict(batch, device=device)
                res = potential.forward(
                    inp, include_forces=True, include_stresses=True,
                )
                gpu_batch_energies.extend(
                    res["total_energy"].detach().cpu().tolist()
                )
            except RuntimeError:
                n_failed += 1
                # Fill with NaN for failed batches
                gpu_batch_energies.extend(
                    [float("nan")] * batch.num_graphs
                )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        gpu_batch_total_ms = (time.perf_counter() - gpu_batch_t0) * 1000
        if n_failed > 0:
            print(f"    Warning: {n_failed} batches failed during inference")
    except Exception as e:
        gpu_batch_total_ms = None
        gpu_batch_energies = []
        print(f"    GPU batch error: {e}")

    summary = {
        "per_structure": per_structure,
        "cpu_total_ms": cpu_total_ms,
        "gpu_1by1_total_ms": gpu1by1_total_ms,
        "gpu_batch_total_ms": gpu_batch_total_ms,
        "gpu_batch_energies": gpu_batch_energies,
        "n_structures": len(atoms_list),
    }
    return summary


# ── Plotting ────────────────────────────────────────────────────────────────

COLORS = {
    "CPU": "#4C72B0",
    "CPU 1-by-1": "#4C72B0",
    "GPU 1-by-1": "#DD8452",
    "GPU Batched": "#55A868",
}
MARKERS = {"CPU": "o", "GPU 1-by-1": "s", "GPU Batched": "^"}


def plot_atom_distribution(atoms_list: list[Atoms], output_path: str):
    """Histogram of atom counts in the benchmark dataset."""
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(8, 4.5))

    sizes = [len(a) for a in atoms_list]
    ax.hist(sizes, bins=50, color="#6A5ACD", edgecolor="white", linewidth=0.5,
            alpha=0.8)
    ax.set_xlabel("Number of Atoms")
    ax.set_ylabel("Count")
    ax.set_title(
        f"Materials Project Benchmark Dataset ({len(atoms_list)} structures)"
    )
    ax.axvline(np.median(sizes), color="red", linestyle="--", linewidth=1.5,
               label=f"Median = {int(np.median(sizes))}")
    ax.legend(frameon=True, fancybox=True, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved → {output_path}")


def bin_by_atoms(records, keys, n_bins=12):
    """Bin records by n_atoms and compute mean/std for each key."""
    valid = [r for r in records if all(r.get(k) is not None for k in keys)]
    if not valid:
        return None
    n_atoms = np.array([r["n_atoms"] for r in valid])
    lo, hi = n_atoms.min(), n_atoms.max()
    bin_edges = np.logspace(np.log10(max(lo, 1)), np.log10(hi), n_bins + 1)
    bin_idx = np.digitize(n_atoms, bin_edges)

    result = {"bin_centers": [], "counts": []}
    for k in keys:
        result[f"{k}_mean"] = []
        result[f"{k}_std"] = []

    vals = {k: np.array([r[k] for r in valid]) for k in keys}

    for b in range(1, n_bins + 1):
        mask = bin_idx == b
        if mask.sum() < 3:
            continue
        result["bin_centers"].append(float(np.mean(n_atoms[mask])))
        result["counts"].append(int(mask.sum()))
        for k in keys:
            result[f"{k}_mean"].append(float(np.mean(vals[k][mask])))
            result[f"{k}_std"].append(float(np.std(vals[k][mask])))

    for k in result:
        result[k] = np.array(result[k])
    return result


def plot_graph_scatter(graph_summary, output_path):
    """Line plot: per-structure amortized time for all 3 methods per bin."""
    setup_plot_style()
    per_bin = graph_summary["per_bin"]
    if not per_bin:
        return

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

    bc = np.array([b["bin_center"] for b in per_bin])
    cpu_t = np.array([b["cpu_per_struct_ms"] for b in per_bin])
    gpu1_t = np.array([b["gpu_1by1_per_struct_ms"] for b in per_bin])
    gpub_t = np.array([
        b["gpu_batch_per_struct_ms"] if b.get("gpu_batch_per_struct_ms") else 0
        for b in per_bin
    ])

    # Left: timing lines
    ax = axes[0]
    ax.plot(bc, cpu_t, "o-", color=COLORS["CPU 1-by-1"], linewidth=2,
            markersize=8, markeredgecolor="white", markeredgewidth=1,
            label="CPU 1-by-1")
    ax.plot(bc, gpu1_t, "s-", color=COLORS["GPU 1-by-1"], linewidth=2,
            markersize=8, markeredgecolor="white", markeredgewidth=1,
            label="GPU 1-by-1")
    ax.plot(bc, gpub_t, "^-", color=COLORS["GPU Batched"], linewidth=2.5,
            markersize=9, markeredgecolor="white", markeredgewidth=1.5,
            label="GPU Batched")
    ax.set_xlabel("Number of Atoms")
    ax.set_ylabel("Graph Construction Time (ms / structure)")
    ax.set_title("(a) Amortized Per-Structure Time")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(frameon=True, fancybox=True, framealpha=0.9)

    # Right: speedup vs CPU
    ax = axes[1]
    sp_1by1 = cpu_t / np.maximum(gpu1_t, 1e-6)
    sp_batch = np.where(gpub_t > 0, cpu_t / gpub_t, 0)
    ax.plot(bc, sp_1by1, "s-", color=COLORS["GPU 1-by-1"], linewidth=2,
            markersize=8, markeredgecolor="white", markeredgewidth=1,
            label="GPU 1-by-1")
    ax.plot(bc, sp_batch, "^-", color=COLORS["GPU Batched"], linewidth=2.5,
            markersize=9, markeredgecolor="white", markeredgewidth=1.5,
            label="GPU Batched")
    ax.axhline(y=1.0, color="gray", linestyle=":", linewidth=1.5,
               label="1× (no speedup)")
    ax.set_xlabel("Number of Atoms")
    ax.set_ylabel("Speedup vs CPU 1-by-1")
    ax.set_title("(b) GPU Speedup by System Size")
    ax.set_xscale("log")
    ax.legend(frameon=True, fancybox=True, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved → {output_path}")


def plot_graph_total_bar(graph_summary, output_path):
    """Bar chart: total wall time for all structures (3 methods)."""
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    methods = ["CPU 1-by-1", "GPU 1-by-1", "GPU Batched"]
    times_s = [
        graph_summary["cpu_total_ms"] / 1000,
        graph_summary["gpu_1by1_total_ms"] / 1000,
        (graph_summary["gpu_batch_total_ms"] or 0) / 1000,
    ]
    colors = [COLORS[m] for m in methods]

    bars = ax.bar(methods, times_s, color=colors, edgecolor="white",
                  linewidth=0.5, width=0.55)
    for bar, t in zip(bars, times_s):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{t:.2f}s", ha="center", va="bottom", fontsize=12,
                fontweight="bold")

    n = graph_summary["n_structures"]
    ax.set_ylabel("Total Wall Time (s)")
    ax.set_title(f"Graph Construction — {n} MP Structures Total Wall Time")
    ax.set_ylim(bottom=0, top=max(times_s) * 1.2)

    # Annotate speedups relative to CPU
    cpu_t = times_s[0]
    for i, (m, t) in enumerate(zip(methods[1:], times_s[1:]), 1):
        if t > 0:
            sp = cpu_t / t
            ax.annotate(f"{sp:.1f}× vs CPU",
                        xy=(bars[i].get_x() + bars[i].get_width() / 2,
                            bars[i].get_height() * 0.5),
                        ha="center", fontsize=10, color="white",
                        fontweight="bold")

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved → {output_path}")


def plot_graph_binned(graph_summary, output_path):
    """Binned bar chart: all 3 methods side by side per atom-count bin."""
    setup_plot_style()
    per_bin = graph_summary["per_bin"]
    if not per_bin:
        print(f"  Skipped {output_path} (no bin data)")
        return

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    bc = [b["bin_center"] for b in per_bin]
    x = np.arange(len(bc))
    w = 0.25

    cpu_vals = [b["cpu_per_struct_ms"] for b in per_bin]
    gpu1_vals = [b["gpu_1by1_per_struct_ms"] for b in per_bin]
    gpub_vals = [
        b["gpu_batch_per_struct_ms"] if b["gpu_batch_per_struct_ms"] else 0
        for b in per_bin
    ]
    counts = [b["count"] for b in per_bin]

    # Left: absolute times — 3 grouped bars
    ax = axes[0]
    ax.bar(x - w, cpu_vals, w, label="CPU 1-by-1",
           color=COLORS["CPU 1-by-1"], edgecolor="white", linewidth=0.5)
    ax.bar(x, gpu1_vals, w, label="GPU 1-by-1",
           color=COLORS["GPU 1-by-1"], edgecolor="white", linewidth=0.5)
    ax.bar(x + w, gpub_vals, w, label="GPU Batched",
           color=COLORS["GPU Batched"], edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Average Atoms per Bin")
    ax.set_ylabel("Time (ms / structure)")
    ax.set_title("(a) Graph Construction per Structure (binned)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(c)}" for c in bc], rotation=45, ha="right")
    ax.legend(frameon=True, fancybox=True, framealpha=0.9)
    ax.set_ylim(bottom=0)
    for xi, cnt in zip(x, counts):
        ax.annotate(f"n={cnt}", (xi, 0), textcoords="offset points",
                    xytext=(0, -18), ha="center", fontsize=8, color="gray")

    # Right: speedup — bars for 1by1, triangles for batched
    ax = axes[1]
    sp_1by1 = np.array(cpu_vals) / np.maximum(np.array(gpu1_vals), 1e-6)
    sp_batch = np.array(cpu_vals) / np.maximum(np.array(gpub_vals), 1e-6)
    # Cap infinite speedups where gpub_vals is 0
    sp_batch = np.where(np.array(gpub_vals) > 0, sp_batch, 0)

    bars = ax.bar(x - w / 2, sp_1by1, w * 1.5, color=COLORS["GPU 1-by-1"],
                  edgecolor="white", linewidth=0.5, label="GPU 1-by-1", alpha=0.7)
    ax.bar(x + w, sp_batch, w * 1.5, color=COLORS["GPU Batched"],
           edgecolor="white", linewidth=0.5, label="GPU Batched", alpha=0.7)
    ax.axhline(y=1.0, color="gray", linestyle=":", linewidth=1.5)

    for xi, s1, sb in zip(x, sp_1by1, sp_batch):
        ax.text(xi - w / 2, s1 + 0.15, f"{s1:.1f}×", ha="center",
                va="bottom", fontsize=7, fontweight="bold",
                color=COLORS["GPU 1-by-1"])
        if sb > 0:
            ax.text(xi + w, sb + 0.15, f"{sb:.1f}×", ha="center",
                    va="bottom", fontsize=7, fontweight="bold",
                    color=COLORS["GPU Batched"])

    ax.set_xlabel("Average Atoms per Bin")
    ax.set_ylabel("Speedup vs CPU 1-by-1")
    ax.set_title("(b) Speedup by System Size")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(c)}" for c in bc], rotation=45, ha="right")
    ax.set_ylim(bottom=0)
    ax.legend(frameon=True, fancybox=True, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved → {output_path}")


def plot_inference_total_bar(inf_summary, output_path):
    """Bar chart: total wall time for full inference (3 methods)."""
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    methods = ["CPU 1-by-1", "GPU 1-by-1", "GPU Batched"]
    times_s = [
        inf_summary["cpu_total_ms"] / 1000,
        inf_summary["gpu_1by1_total_ms"] / 1000,
        (inf_summary["gpu_batch_total_ms"] or 0) / 1000,
    ]
    colors = [COLORS[m] for m in methods]

    bars = ax.bar(methods, times_s, color=colors, edgecolor="white",
                  linewidth=0.5, width=0.55)
    for bar, t in zip(bars, times_s):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                f"{t:.1f}s", ha="center", va="bottom", fontsize=12,
                fontweight="bold")

    n = inf_summary["n_structures"]
    ax.set_ylabel("Total Wall Time (s)")
    ax.set_title(
        f"Full Pipeline (Graph + E/F/S) — {n} MP Structures Total Wall Time"
    )
    ax.set_ylim(bottom=0, top=max(times_s) * 1.2)

    cpu_t = times_s[0]
    for i, (m, t) in enumerate(zip(methods[1:], times_s[1:]), 1):
        if t > 0:
            sp = cpu_t / t
            ax.annotate(f"{sp:.1f}× vs CPU",
                        xy=(bars[i].get_x() + bars[i].get_width() / 2,
                            bars[i].get_height() * 0.5),
                        ha="center", fontsize=10, color="white",
                        fontweight="bold")

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved → {output_path}")


def plot_energy_parity(inf_summary, output_path):
    """Energy parity: CPU path vs GPU batched path."""
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(7, 7))

    records = inf_summary["per_structure"]
    gpu_batch_energies = inf_summary.get("gpu_batch_energies", [])

    valid = [
        r for r in records
        if r.get("cpu_energy") is not None and r.get("gpu_energy") is not None
    ]
    cpu_e = np.array([r["cpu_energy"] for r in valid])
    gpu_e = np.array([r["gpu_energy"] for r in valid])
    n_atoms = np.array([r["n_atoms"] for r in valid])
    diff = np.abs(cpu_e - gpu_e)

    sc = ax.scatter(cpu_e, gpu_e, c=n_atoms, cmap="viridis", s=12, alpha=0.6,
                    edgecolors="none")
    mn, mx = min(cpu_e.min(), gpu_e.min()), max(cpu_e.max(), gpu_e.max())
    margin = (mx - mn) * 0.03
    ax.plot([mn - margin, mx + margin], [mn - margin, mx + margin],
            "r--", linewidth=1.5, label="y = x")
    ax.set_xlabel("Energy — CPU graph (eV)")
    ax.set_ylabel("Energy — GPU graph (eV)")
    ax.set_title("Energy Parity: CPU vs GPU Graph Construction")
    plt.colorbar(sc, ax=ax, label="Number of Atoms", pad=0.02)

    textstr = (
        f"N = {len(valid)}\n"
        f"MAE = {np.mean(diff):.2e} eV\n"
        f"Max |ΔE| = {np.max(diff):.2e} eV\n"
        f"MAE/atom = {np.mean(diff / n_atoms)*1000:.2e} meV/atom"
    )
    props = dict(boxstyle="round", facecolor="wheat", alpha=0.7)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment="top", bbox=props)
    ax.legend(frameon=True, fancybox=True, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved → {output_path}")


def plot_summary_4panel(graph_summary, inf_summary, output_path, gpu_name):
    """Combined 2×2 summary panel."""
    setup_plot_style()
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    methods = ["CPU\n1-by-1", "GPU\n1-by-1", "GPU\nBatched"]
    colors = [COLORS["CPU"], COLORS["GPU 1-by-1"], COLORS["GPU Batched"]]

    # ── (a) Graph total bar ──
    ax = axes[0, 0]
    g_times = [
        graph_summary["cpu_total_ms"] / 1000,
        graph_summary["gpu_1by1_total_ms"] / 1000,
        (graph_summary["gpu_batch_total_ms"] or 0) / 1000,
    ]
    bars = ax.bar(methods, g_times, color=colors, edgecolor="white",
                  linewidth=0.5, width=0.55)
    for bar, t in zip(bars, g_times):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{t:.1f}s", ha="center", va="bottom", fontsize=10,
                fontweight="bold")
    ax.set_ylabel("Total Wall Time (s)")
    ax.set_title(f"(a) Graph Construction — Total ({graph_summary['n_structures_benchmarked']} structs)")
    ax.set_ylim(bottom=0, top=max(g_times) * 1.25)

    # ── (b) Binned bar chart ──
    ax = axes[0, 1]
    per_bin = graph_summary["per_bin"]
    if per_bin:
        bc = [b["bin_center"] for b in per_bin]
        x = np.arange(len(bc))
        w = 0.25
        cpu_v = [b["cpu_per_struct_ms"] for b in per_bin]
        gpu1_v = [b["gpu_1by1_per_struct_ms"] for b in per_bin]
        gpub_v = [b.get("gpu_batch_per_struct_ms") or 0 for b in per_bin]
        ax.bar(x - w, cpu_v, w, label="CPU 1-by-1", color=colors[0],
               edgecolor="white", linewidth=0.5)
        ax.bar(x, gpu1_v, w, label="GPU 1-by-1", color=colors[1],
               edgecolor="white", linewidth=0.5)
        ax.bar(x + w, gpub_v, w, label="GPU Batched", color=colors[2],
               edgecolor="white", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{int(c)}" for c in bc], rotation=45, ha="right")
        ax.legend(fontsize=9, frameon=True, fancybox=True, framealpha=0.9)
    ax.set_xlabel("Average Atoms per Bin")
    ax.set_ylabel("Time (ms / structure)")
    ax.set_title("(b) Per-Structure Time by Bin")
    ax.set_ylim(bottom=0)

    # ── (c) Per-bin line plot ──
    ax = axes[1, 0]
    per_bin = graph_summary["per_bin"]
    if per_bin:
        bc = np.array([b["bin_center"] for b in per_bin])
        cpu_t = np.array([b["cpu_per_struct_ms"] for b in per_bin])
        gpu1_t = np.array([b["gpu_1by1_per_struct_ms"] for b in per_bin])
        gpub_t = np.array([
            b["gpu_batch_per_struct_ms"]
            if b.get("gpu_batch_per_struct_ms") else 0
            for b in per_bin
        ])
        ax.plot(bc, cpu_t, "o-", color=colors[0], linewidth=2, markersize=6,
                markeredgecolor="white", markeredgewidth=1, label="CPU 1-by-1")
        ax.plot(bc, gpu1_t, "s-", color=colors[1], linewidth=2, markersize=6,
                markeredgecolor="white", markeredgewidth=1, label="GPU 1-by-1")
        ax.plot(bc, gpub_t, "^-", color=colors[2], linewidth=2, markersize=7,
                markeredgecolor="white", markeredgewidth=1, label="GPU Batched")
    ax.set_xlabel("Number of Atoms")
    ax.set_ylabel("Time (ms / structure)")
    ax.set_title("(c) Per-Structure Amortized Time")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(frameon=True, fancybox=True, framealpha=0.9)

    # ── (d) Speedup lines ──
    ax = axes[1, 1]
    if per_bin:
        sp_1by1 = cpu_t / np.maximum(gpu1_t, 1e-6)
        sp_batch = np.where(gpub_t > 0, cpu_t / gpub_t, 0)
        ax.plot(bc, sp_1by1, "s-", color=colors[1], linewidth=2,
                markersize=6, markeredgecolor="white", markeredgewidth=1,
                label="GPU 1-by-1")
        ax.plot(bc, sp_batch, "^-", color=colors[2], linewidth=2.5,
                markersize=7, markeredgecolor="white", markeredgewidth=1.5,
                label="GPU Batched")
        ax.axhline(y=1.0, color="gray", linestyle=":", linewidth=1.5)
    ax.set_xlabel("Number of Atoms")
    ax.set_ylabel("Speedup vs CPU 1-by-1")
    ax.set_title("(d) Speedup by System Size")
    ax.set_xscale("log")
    ax.legend(frameon=True, fancybox=True, framealpha=0.9)

    n = graph_summary["n_structures"]
    fig.suptitle(
        f"MatterSim Benchmark — {gpu_name}\n"
        f"{n} Materials Project structures",
        fontsize=15, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved → {output_path}")


# ── Main ────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark graph construction on MP structures",
    )
    parser.add_argument("--num-structures", type=int, default=1000)
    parser.add_argument("--output-dir", default="benchmark_results")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--skip-inference", action="store_true",
                        help="Skip full inference benchmark (faster)")
    parser.add_argument("--skip-md", action="store_true",
                        help="Skip MD simulation benchmark")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    gpu_name = "CPU"
    if args.device == "cuda" and torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name()

    print("=" * 70)
    print("MatterSim Graph Construction Benchmark")
    print("=" * 70)
    print(f"Device:     {args.device} ({gpu_name})")
    print(f"PyTorch:    {torch.__version__}")
    print(f"Structures: {args.num_structures}")
    print(f"Output:     {output_dir}")

    # 1. Download / load structures
    cache_path = os.path.join(output_dir, "mp_structures.pkl")
    atoms_list = download_mp_structures(
        num_structures=args.num_structures,
        cache_path=cache_path,
        seed=args.seed,
    )

    sizes = [len(a) for a in atoms_list]
    print(f"\nDataset statistics:")
    print(f"  Total structures: {len(atoms_list)}")
    print(f"  Atom counts: min={min(sizes)}, median={int(np.median(sizes))}, "
          f"max={max(sizes)}, mean={np.mean(sizes):.0f}")

    # Plot distribution
    print("\nPlotting atom distribution...")
    plot_atom_distribution(
        atoms_list, os.path.join(output_dir, "atom_distribution.png")
    )

    # 2. GPU warmup
    if torch.cuda.is_available():
        print("\nGPU warmup...")
        from mattersim.datasets.utils.build import build_dataloader
        _ = build_dataloader(
            [atoms_list[0]], batch_size=1, model_type="m3gnet",
            only_inference=True, batch_converter=True,
        )
        torch.cuda.synchronize()

    # 3. Graph construction benchmark (3-way)
    print("\n" + "=" * 70)
    print("Benchmarking graph construction: CPU 1-by-1, GPU 1-by-1, GPU batched")
    print("=" * 70)
    graph_summary = benchmark_graph_construction(atoms_list)

    with open(os.path.join(output_dir, "graph_results.json"), "w") as f:
        json.dump(graph_summary, f, indent=2, default=str)

    print(f"\n  Graph construction summary ({graph_summary['n_structures']} structures):")
    print(f"    CPU 1-by-1 total:  {graph_summary['cpu_total_ms']:.0f} ms")
    print(f"    GPU 1-by-1 total:  {graph_summary['gpu_1by1_total_ms']:.0f} ms")
    if graph_summary['gpu_batch_total_ms']:
        print(f"    GPU batched total: {graph_summary['gpu_batch_total_ms']:.0f} ms")
        print(f"    Speedup (GPU batched vs CPU): "
              f"{graph_summary['cpu_total_ms'] / graph_summary['gpu_batch_total_ms']:.1f}×")

    # 4. Full inference benchmark (optional)
    inf_summary = None
    if not args.skip_inference:
        print("\n" + "=" * 70)
        print("Benchmarking full pipeline: CPU 1-by-1, GPU 1-by-1, GPU batched")
        print("=" * 70)

        from mattersim.forcefield.potential import Potential
        potential = Potential.from_checkpoint(device=args.device)
        potential.model.eval()

        inf_summary = benchmark_full_inference(
            atoms_list, potential, device=args.device,
        )

        with open(os.path.join(output_dir, "inference_results.json"), "w") as f:
            json.dump(inf_summary, f, indent=2, default=str)

        print(f"\n  Full pipeline summary ({inf_summary['n_structures']} structures):")
        print(f"    CPU 1-by-1 total:  {inf_summary['cpu_total_ms'] / 1000:.1f} s")
        print(f"    GPU 1-by-1 total:  {inf_summary['gpu_1by1_total_ms'] / 1000:.1f} s")
        if inf_summary['gpu_batch_total_ms']:
            print(f"    GPU batched total: {inf_summary['gpu_batch_total_ms'] / 1000:.1f} s")
            print(f"    Speedup (GPU batched vs CPU): "
                  f"{inf_summary['cpu_total_ms'] / inf_summary['gpu_batch_total_ms']:.1f}×")

        del potential
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
        gc.collect()

    # 5. MD simulation benchmark
    md_steps = 10
    md_results = []
    if not args.skip_md:
        print("\n" + "=" * 70)
        print(f"Benchmarking MD simulation ({md_steps} NVE steps per structure)")
        print("  1 structure per atom-count bin, CPU vs GPU graph construction")
        print("=" * 70)

        md_results = benchmark_md_simulation(
            atoms_list,
            graph_summary["per_bin"],
            device=args.device,
            md_steps=md_steps,
        )

        with open(os.path.join(output_dir, "md_results.json"), "w") as f:
            json.dump(md_results, f, indent=2, default=str)

    # 6. Generate plots
    print("\n" + "=" * 70)
    print("Generating plots...")
    print("=" * 70)

    plot_graph_scatter(
        graph_summary, os.path.join(output_dir, "graph_scatter.png"),
    )
    plot_graph_total_bar(
        graph_summary, os.path.join(output_dir, "graph_total_bar.png"),
    )
    plot_graph_binned(
        graph_summary, os.path.join(output_dir, "graph_binned.png"),
    )

    if md_results:
        plot_md_benchmark(
            md_results,
            os.path.join(output_dir, "md_benchmark.png"),
            md_steps,
        )

    if inf_summary:
        plot_inference_total_bar(
            inf_summary, os.path.join(output_dir, "inference_total_bar.png"),
        )
        plot_energy_parity(
            inf_summary, os.path.join(output_dir, "energy_parity.png"),
        )

    # Summary panel (always generate with graph data)
    plot_summary_4panel(
        graph_summary, inf_summary,
        os.path.join(output_dir, "summary_mp.png"),
        gpu_name,
    )

    print(f"\n{'=' * 70}")
    print(f"All outputs saved to {output_dir}/")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
