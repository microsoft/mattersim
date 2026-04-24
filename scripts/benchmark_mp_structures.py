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
COLORS = {
    "CPU": "#4C72B0",
    "GPU": "#55A868",
}


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
) -> dict:
    """Benchmark CPU vs GPU graph construction on the given structures.

    Returns per-structure timings and metadata.
    """
    from mattersim.datasets.utils.build import build_dataloader

    results = []

    for i, atoms in enumerate(atoms_list):
        n = len(atoms)
        record = {"n_atoms": n, "index": i}

        # CPU path
        try:
            t0 = time.perf_counter()
            dl = build_dataloader(
                [atoms], batch_size=1, model_type="m3gnet",
                only_inference=True, cutoff=cutoff,
                threebody_cutoff=threebody_cutoff, batch_converter=False,
            )
            t1 = time.perf_counter()
            record["cpu_ms"] = (t1 - t0) * 1000
        except Exception as e:
            record["cpu_ms"] = None
            record["cpu_error"] = str(e)

        # GPU path
        try:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            dl = build_dataloader(
                [atoms], batch_size=1, model_type="m3gnet",
                only_inference=True, cutoff=cutoff,
                threebody_cutoff=threebody_cutoff, batch_converter=True,
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            record["gpu_ms"] = (t1 - t0) * 1000
        except Exception as e:
            record["gpu_ms"] = None
            record["gpu_error"] = str(e)

        results.append(record)

        if (i + 1) % 100 == 0 or i == len(atoms_list) - 1:
            print(f"  Processed {i+1}/{len(atoms_list)} structures", flush=True)

    return results


def benchmark_full_inference(
    atoms_list: list[Atoms],
    potential,
    device: str = "cuda",
    cutoff: float = 5.0,
    threebody_cutoff: float = 4.0,
) -> dict:
    """Benchmark full pipeline (graph + inference) CPU vs GPU."""
    from mattersim.datasets.utils.build import build_dataloader
    from mattersim.forcefield.potential import batch_to_dict

    results = []

    for i, atoms in enumerate(atoms_list):
        n = len(atoms)
        record = {"n_atoms": n, "index": i}

        # CPU graph → inference
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

        # GPU graph → inference
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
            record["gpu_total_ms"] = (t1 - t0) * 1000
            record["gpu_energy"] = res["total_energy"].detach().cpu().item()
        except Exception as e:
            record["gpu_total_ms"] = None
            record["gpu_error"] = str(e)

        results.append(record)

        if (i + 1) % 100 == 0 or i == len(atoms_list) - 1:
            print(f"  Processed {i+1}/{len(atoms_list)} structures", flush=True)

    return results


# ── Plotting ────────────────────────────────────────────────────────────────


def bin_results(results: list[dict], key_cpu: str, key_gpu: str, n_bins: int = 15):
    """Bin results by atom count and compute mean/std per bin."""
    valid = [
        r for r in results
        if r.get(key_cpu) is not None and r.get(key_gpu) is not None
    ]
    if not valid:
        return None

    n_atoms = np.array([r["n_atoms"] for r in valid])
    cpu_vals = np.array([r[key_cpu] for r in valid])
    gpu_vals = np.array([r[key_gpu] for r in valid])

    # Create log-spaced bins
    lo, hi = n_atoms.min(), n_atoms.max()
    bin_edges = np.logspace(np.log10(max(lo, 1)), np.log10(hi), n_bins + 1)
    bin_indices = np.digitize(n_atoms, bin_edges)

    bin_centers, cpu_means, cpu_stds, gpu_means, gpu_stds, counts = (
        [], [], [], [], [], [],
    )
    for b in range(1, n_bins + 1):
        mask = bin_indices == b
        if mask.sum() < 3:
            continue
        bin_centers.append(np.mean(n_atoms[mask]))
        cpu_means.append(np.mean(cpu_vals[mask]))
        cpu_stds.append(np.std(cpu_vals[mask]))
        gpu_means.append(np.mean(gpu_vals[mask]))
        gpu_stds.append(np.std(gpu_vals[mask]))
        counts.append(int(mask.sum()))

    return {
        "bin_centers": np.array(bin_centers),
        "cpu_means": np.array(cpu_means),
        "cpu_stds": np.array(cpu_stds),
        "gpu_means": np.array(gpu_means),
        "gpu_stds": np.array(gpu_stds),
        "counts": np.array(counts),
    }


def plot_scatter_timing(results: list[dict], key_cpu: str, key_gpu: str,
                        title: str, ylabel: str, output_path: str):
    """Scatter plot: per-structure CPU vs GPU timing."""
    setup_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

    valid = [
        r for r in results
        if r.get(key_cpu) is not None and r.get(key_gpu) is not None
    ]
    n_atoms = np.array([r["n_atoms"] for r in valid])
    cpu_vals = np.array([r[key_cpu] for r in valid])
    gpu_vals = np.array([r[key_gpu] for r in valid])

    # Left: scatter of individual timings
    ax = axes[0]
    ax.scatter(n_atoms, cpu_vals, alpha=0.35, s=12, color=COLORS["CPU"],
               label="CPU", edgecolors="none", zorder=2)
    ax.scatter(n_atoms, gpu_vals, alpha=0.35, s=12, color=COLORS["GPU"],
               label="GPU", edgecolors="none", zorder=3)
    ax.set_xlabel("Number of Atoms")
    ax.set_ylabel(ylabel)
    ax.set_title(f"(a) {title}")
    ax.legend(frameon=True, fancybox=True, framealpha=0.9, markerscale=2)
    ax.set_xscale("log")
    ax.set_yscale("log")

    # Right: speedup scatter
    ax = axes[1]
    speedup = cpu_vals / np.maximum(gpu_vals, 1e-6)
    sc = ax.scatter(n_atoms, speedup, c=n_atoms, cmap="viridis",
                    alpha=0.5, s=15, edgecolors="none")
    ax.axhline(y=1.0, color="gray", linestyle=":", linewidth=1.5, label="1× (no speedup)")
    ax.set_xlabel("Number of Atoms")
    ax.set_ylabel("Speedup (CPU / GPU)")
    ax.set_title("(b) GPU Speedup Factor")
    ax.set_xscale("log")
    ax.legend(frameon=True, fancybox=True, framealpha=0.9)
    plt.colorbar(sc, ax=ax, label="Number of Atoms", pad=0.02)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved → {output_path}")


def plot_binned_comparison(results: list[dict], key_cpu: str, key_gpu: str,
                           title: str, ylabel: str, output_path: str):
    """Binned bar chart with error bars: mean CPU vs GPU time per atom-count bin."""
    setup_plot_style()

    binned = bin_results(results, key_cpu, key_gpu)
    if binned is None:
        print(f"  Skipped {output_path} (no valid data)")
        return

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

    bc = binned["bin_centers"]
    x = np.arange(len(bc))
    w = 0.35

    # Left: absolute times
    ax = axes[0]
    ax.bar(x - w / 2, binned["cpu_means"], w, yerr=binned["cpu_stds"],
           label="CPU", color=COLORS["CPU"], edgecolor="white", linewidth=0.5,
           capsize=3, error_kw={"linewidth": 0.8})
    ax.bar(x + w / 2, binned["gpu_means"], w, yerr=binned["gpu_stds"],
           label="GPU", color=COLORS["GPU"], edgecolor="white", linewidth=0.5,
           capsize=3, error_kw={"linewidth": 0.8})
    ax.set_xlabel("Average Atoms per Bin")
    ax.set_ylabel(ylabel)
    ax.set_title(f"(a) {title}")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(c)}" for c in bc], rotation=45, ha="right")
    ax.legend(frameon=True, fancybox=True, framealpha=0.9)
    ax.set_ylim(bottom=0)

    # Annotate sample counts
    for xi, cnt in zip(x, binned["counts"]):
        ax.annotate(f"n={cnt}", (xi, 0), textcoords="offset points",
                    xytext=(0, -18), ha="center", fontsize=8, color="gray")

    # Right: speedup
    ax = axes[1]
    speedup = binned["cpu_means"] / np.maximum(binned["gpu_means"], 1e-6)
    bars = ax.bar(x, speedup, 0.6, color=COLORS["GPU"], edgecolor="white",
                  linewidth=0.5)
    ax.axhline(y=1.0, color="gray", linestyle=":", linewidth=1.5)
    ax.set_xlabel("Average Atoms per Bin")
    ax.set_ylabel("Speedup (CPU / GPU)")
    ax.set_title("(b) GPU Speedup by System Size")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(c)}" for c in bc], rotation=45, ha="right")
    ax.set_ylim(bottom=0)

    for bar, s in zip(bars, speedup):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                f"{s:.1f}×", ha="center", va="bottom", fontsize=9,
                fontweight="bold", color=COLORS["GPU"])

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved → {output_path}")


def plot_atom_distribution(atoms_list: list[Atoms], output_path: str):
    """Histogram of atom counts in the benchmark dataset."""
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(8, 4.5))

    sizes = [len(a) for a in atoms_list]
    ax.hist(sizes, bins=50, color="#6A5ACD", edgecolor="white", linewidth=0.5,
            alpha=0.8)
    ax.set_xlabel("Number of Atoms")
    ax.set_ylabel("Count")
    ax.set_title(f"Materials Project Benchmark Dataset ({len(atoms_list)} structures)")
    ax.axvline(np.median(sizes), color="red", linestyle="--", linewidth=1.5,
               label=f"Median = {int(np.median(sizes))}")
    ax.legend(frameon=True, fancybox=True, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved → {output_path}")


def plot_energy_parity(results: list[dict], output_path: str):
    """Energy parity: CPU path vs GPU path."""
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(7, 7))

    valid = [
        r for r in results
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

    # Error stats
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


def plot_summary_4panel(graph_results, inference_results, output_path, gpu_name):
    """Combined 2×2 summary panel."""
    setup_plot_style()

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # ── (a) Graph construction scatter ──
    ax = axes[0, 0]
    valid = [r for r in graph_results
             if r.get("cpu_ms") is not None and r.get("gpu_ms") is not None]
    n_a = np.array([r["n_atoms"] for r in valid])
    cpu_g = np.array([r["cpu_ms"] for r in valid])
    gpu_g = np.array([r["gpu_ms"] for r in valid])
    ax.scatter(n_a, cpu_g, alpha=0.3, s=10, color=COLORS["CPU"],
               label="CPU", edgecolors="none")
    ax.scatter(n_a, gpu_g, alpha=0.3, s=10, color=COLORS["GPU"],
               label="GPU", edgecolors="none")
    ax.set_xlabel("Number of Atoms")
    ax.set_ylabel("Graph Construction Time (ms)")
    ax.set_title("(a) Graph Construction Time")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(frameon=True, fancybox=True, framealpha=0.9, markerscale=2)

    # ── (b) Graph construction speedup ──
    ax = axes[0, 1]
    speedup_g = cpu_g / np.maximum(gpu_g, 1e-6)
    sc = ax.scatter(n_a, speedup_g, c=n_a, cmap="viridis", alpha=0.5, s=12,
                    edgecolors="none")
    ax.axhline(y=1.0, color="gray", linestyle=":", linewidth=1.5)
    ax.set_xlabel("Number of Atoms")
    ax.set_ylabel("Speedup (CPU / GPU)")
    ax.set_title("(b) Graph Construction Speedup")
    ax.set_xscale("log")
    plt.colorbar(sc, ax=ax, label="Atoms", pad=0.02)

    # ── (c) Full inference scatter ──
    ax = axes[1, 0]
    valid_inf = [r for r in inference_results
                 if r.get("cpu_total_ms") is not None
                 and r.get("gpu_total_ms") is not None]
    n_a_inf = np.array([r["n_atoms"] for r in valid_inf])
    cpu_inf = np.array([r["cpu_total_ms"] for r in valid_inf])
    gpu_inf = np.array([r["gpu_total_ms"] for r in valid_inf])
    ax.scatter(n_a_inf, cpu_inf, alpha=0.3, s=10, color=COLORS["CPU"],
               label="CPU graph", edgecolors="none")
    ax.scatter(n_a_inf, gpu_inf, alpha=0.3, s=10, color=COLORS["GPU"],
               label="GPU graph", edgecolors="none")
    ax.set_xlabel("Number of Atoms")
    ax.set_ylabel("Total Time: Graph + Inference (ms)")
    ax.set_title("(c) Full Pipeline Time (Graph + E/F/S Inference)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(frameon=True, fancybox=True, framealpha=0.9, markerscale=2)

    # ── (d) Energy parity ──
    ax = axes[1, 1]
    valid_e = [r for r in inference_results
               if r.get("cpu_energy") is not None and r.get("gpu_energy") is not None]
    cpu_e = np.array([r["cpu_energy"] for r in valid_e])
    gpu_e = np.array([r["gpu_energy"] for r in valid_e])
    n_e = np.array([r["n_atoms"] for r in valid_e])
    sc = ax.scatter(cpu_e, gpu_e, c=n_e, cmap="viridis", s=10, alpha=0.5,
                    edgecolors="none")
    mn, mx = min(cpu_e.min(), gpu_e.min()), max(cpu_e.max(), gpu_e.max())
    margin = (mx - mn) * 0.03
    ax.plot([mn - margin, mx + margin], [mn - margin, mx + margin],
            "r--", linewidth=1.5, label="y = x")
    ax.set_xlabel("Energy — CPU graph (eV)")
    ax.set_ylabel("Energy — GPU graph (eV)")
    ax.set_title("(d) Energy Parity")
    plt.colorbar(sc, ax=ax, label="Atoms", pad=0.02)
    diff = np.abs(cpu_e - gpu_e)
    textstr = f"MAE = {np.mean(diff):.2e} eV"
    ax.text(0.05, 0.92, textstr, transform=ax.transAxes, fontsize=11,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.7))
    ax.legend(frameon=True, fancybox=True, framealpha=0.9)

    fig.suptitle(
        f"MatterSim Graph Construction Benchmark — {gpu_name}\n"
        f"{len(graph_results)} Materials Project structures",
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

    # 3. Graph construction benchmark
    print("\n" + "=" * 70)
    print("Benchmarking graph construction (CPU vs GPU)...")
    print("=" * 70)
    graph_results = benchmark_graph_construction(atoms_list)

    # Save raw results
    with open(os.path.join(output_dir, "graph_results.json"), "w") as f:
        json.dump(graph_results, f, indent=2, default=str)

    # Stats
    valid_graph = [
        r for r in graph_results
        if r.get("cpu_ms") is not None and r.get("gpu_ms") is not None
    ]
    if valid_graph:
        cpu_times = np.array([r["cpu_ms"] for r in valid_graph])
        gpu_times = np.array([r["gpu_ms"] for r in valid_graph])
        speedups = cpu_times / np.maximum(gpu_times, 1e-6)
        print(f"\nGraph construction summary ({len(valid_graph)} structures):")
        print(f"  CPU total: {cpu_times.sum():.0f} ms")
        print(f"  GPU total: {gpu_times.sum():.0f} ms")
        print(f"  Overall speedup: {cpu_times.sum() / gpu_times.sum():.1f}×")
        print(f"  Median per-structure speedup: {np.median(speedups):.1f}×")

    # 4. Full inference benchmark (optional)
    inference_results = []
    if not args.skip_inference:
        print("\n" + "=" * 70)
        print("Benchmarking full inference pipeline (graph + E/F/S)...")
        print("=" * 70)

        from mattersim.forcefield.potential import Potential
        potential = Potential.from_checkpoint(device=args.device)
        potential.model.eval()

        inference_results = benchmark_full_inference(
            atoms_list, potential, device=args.device,
        )

        with open(os.path.join(output_dir, "inference_results.json"), "w") as f:
            json.dump(inference_results, f, indent=2, default=str)

        valid_inf = [
            r for r in inference_results
            if r.get("cpu_total_ms") is not None
            and r.get("gpu_total_ms") is not None
        ]
        if valid_inf:
            cpu_t = np.array([r["cpu_total_ms"] for r in valid_inf])
            gpu_t = np.array([r["gpu_total_ms"] for r in valid_inf])
            print(f"\nFull inference summary ({len(valid_inf)} structures):")
            print(f"  CPU total: {cpu_t.sum() / 1000:.1f} s")
            print(f"  GPU total: {gpu_t.sum() / 1000:.1f} s")
            print(f"  Overall speedup: {cpu_t.sum() / gpu_t.sum():.1f}×")

        del potential
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    # 5. Generate plots
    print("\n" + "=" * 70)
    print("Generating plots...")
    print("=" * 70)

    plot_scatter_timing(
        graph_results, "cpu_ms", "gpu_ms",
        "Graph Construction Time", "Time (ms)",
        os.path.join(output_dir, "graph_scatter.png"),
    )
    plot_binned_comparison(
        graph_results, "cpu_ms", "gpu_ms",
        "Graph Construction Time", "Time (ms)",
        os.path.join(output_dir, "graph_binned.png"),
    )

    if inference_results:
        plot_scatter_timing(
            inference_results, "cpu_total_ms", "gpu_total_ms",
            "Full Pipeline (Graph + Inference)", "Time (ms)",
            os.path.join(output_dir, "inference_scatter.png"),
        )
        plot_binned_comparison(
            inference_results, "cpu_total_ms", "gpu_total_ms",
            "Full Pipeline (Graph + Inference)", "Time (ms)",
            os.path.join(output_dir, "inference_binned.png"),
        )
        plot_energy_parity(
            inference_results,
            os.path.join(output_dir, "energy_parity.png"),
        )
        plot_summary_4panel(
            graph_results, inference_results,
            os.path.join(output_dir, "summary_mp.png"),
            gpu_name,
        )

    print(f"\n{'=' * 70}")
    print(f"All outputs saved to {output_dir}/")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
