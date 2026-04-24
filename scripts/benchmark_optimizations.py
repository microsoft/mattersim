#!/usr/bin/env python
"""Benchmark MatterSim optimizations: Original vs Optimized vs Checkpointed.

Compares three model configurations across varying system sizes:
  1. Original   — code from the main branch (torch_runstats scatter, no ckpt)
  2. Optimized  — native scatter_sum, same speed, checkpoint-ready
  3. Optimized + Gradient Checkpointing — trades ~20-30% speed for ~50% memory

The script runs the original model by temporarily switching to the main branch
in a subprocess (editable install), then runs optimized variants in-process.

Produces publication-quality plots saved to the output directory.

Usage:
    python scripts/benchmark_optimizations.py
    python scripts/benchmark_optimizations.py --sizes 8,64,216,512,1000,2744
    python scripts/benchmark_optimizations.py --output-dir results/benchmark
"""

import argparse
import gc
import json
import os
import subprocess
import sys
import tempfile
import textwrap
import time

import numpy as np
import torch
from ase.build import bulk, make_supercell

# ---------------------------------------------------------------------------
# Structure generation
# ---------------------------------------------------------------------------


def generate_structures(sizes: list[int]) -> dict[int, "ase.Atoms"]:
    """Generate perturbed bulk Si supercells targeting *sizes* atom counts."""
    base = bulk("Si", "diamond", a=5.43, cubic=True)  # 8 atoms
    structures = {}
    for target in sorted(sizes):
        n = max(1, round((target / len(base)) ** (1 / 3)))
        atoms = make_supercell(base, [[n, 0, 0], [0, n, 0], [0, 0, n]])
        rng = np.random.default_rng(42)
        atoms.positions += rng.normal(scale=0.01, size=atoms.positions.shape)
        structures[len(atoms)] = atoms
    return structures


# ---------------------------------------------------------------------------
# GPU memory helpers
# ---------------------------------------------------------------------------


def reset_gpu():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
    gc.collect()


def peak_memory_mb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024**2
    return 0.0


# ---------------------------------------------------------------------------
# In-process timing (optimized branch)
# ---------------------------------------------------------------------------


def time_inference_inprocess(
    potential,
    atoms,
    device: str,
    warmup: int = 2,
    repeats: int = 5,
) -> dict:
    """Time inference (E+F+S) and record peak GPU memory."""
    from mattersim.datasets.utils.build import build_dataloader
    from mattersim.forcefield.potential import batch_to_dict

    cutoff = potential.model.model_args.get("cutoff", 5.0)
    threebody_cutoff = potential.model.model_args.get("threebody_cutoff", 4.0)
    dl = build_dataloader(
        [atoms], batch_size=1, model_type="m3gnet",
        only_inference=True, cutoff=cutoff, threebody_cutoff=threebody_cutoff,
    )
    graph_batch = next(iter(dl))
    inp = batch_to_dict(graph_batch, device=device)

    potential.model.eval()

    # warmup
    for _ in range(warmup):
        potential.forward(inp, include_forces=True, include_stresses=True)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    reset_gpu()
    times = []
    for _ in range(repeats):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        res = potential.forward(inp, include_forces=True, include_stresses=True)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    return {
        "mean_ms": float(np.mean(times) * 1000),
        "std_ms": float(np.std(times) * 1000),
        "peak_mem_mb": float(peak_memory_mb()),
        "energy": float(res["total_energy"].detach().cpu().item()),
    }


# ---------------------------------------------------------------------------
# Subprocess timing helper (for the main-branch / original model)
# ---------------------------------------------------------------------------

_SUBPROCESS_SCRIPT = textwrap.dedent(r'''
"""Timing helper — runs inside a subprocess on the main branch."""
import gc, json, sys, time
import numpy as np, torch
from ase.build import bulk, make_supercell

def generate(sizes):
    base = bulk("Si", "diamond", a=5.43, cubic=True)
    out = {}
    for t in sorted(sizes):
        n = max(1, round((t / len(base)) ** (1.0 / 3.0)))
        atoms = make_supercell(base, [[n,0,0],[0,n,0],[0,0,n]])
        rng = np.random.default_rng(42)
        atoms.positions += rng.normal(scale=0.01, size=atoms.positions.shape)
        out[len(atoms)] = atoms
    return out

def reset():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
    gc.collect()

def peak_mb():
    return torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0.0

def bench(potential, atoms, device, warmup=2, repeats=5):
    from mattersim.datasets.utils.build import build_dataloader
    from mattersim.forcefield.potential import batch_to_dict
    ca = potential.model.model_args.get("cutoff", 5.0)
    cb = potential.model.model_args.get("threebody_cutoff", 4.0)
    dl = build_dataloader([atoms], batch_size=1, model_type="m3gnet",
                          only_inference=True, cutoff=ca, threebody_cutoff=cb)
    gb = next(iter(dl))
    inp = batch_to_dict(gb, device=device)
    potential.model.eval()
    for _ in range(warmup):
        potential.forward(inp, include_forces=True, include_stresses=True)
    if torch.cuda.is_available(): torch.cuda.synchronize()
    reset()
    times = []
    for _ in range(repeats):
        if torch.cuda.is_available(): torch.cuda.synchronize()
        t0 = time.perf_counter()
        res = potential.forward(inp, include_forces=True, include_stresses=True)
        if torch.cuda.is_available(): torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return {"mean_ms": float(np.mean(times)*1000),
            "std_ms": float(np.std(times)*1000),
            "peak_mem_mb": float(peak_mb()),
            "energy": float(res["total_energy"].detach().cpu().item())}

def main():
    cfg = json.loads(sys.argv[1])
    device = cfg["device"]
    sizes = cfg["sizes"]
    repeats = cfg["repeats"]
    from mattersim.forcefield.potential import Potential
    pot = Potential.from_checkpoint(device=device)
    pot.model.eval()
    structs = generate(sizes)
    results = {}
    for n, atoms in sorted(structs.items()):
        try:
            r = bench(pot, atoms, device, repeats=repeats)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                r = None
                if torch.cuda.is_available(): torch.cuda.empty_cache()
                gc.collect()
            else: raise
        results[n] = r
    print("__BENCH_RESULTS__")
    print(json.dumps(results))

if __name__ == "__main__":
    main()
''')


def run_original_benchmark(
    repo_root: str,
    device: str,
    sizes: list[int],
    repeats: int,
    branch: str,
) -> dict:
    """Run the original (main-branch) model via subprocess.

    Temporarily checks out *branch*, runs timing, then restores the
    current branch.  Safe because this process has already loaded the
    optimized modules into memory.
    """
    current_branch = (
        subprocess.check_output(
            ["git", "branch", "--show-current"], cwd=repo_root
        )
        .decode()
        .strip()
    )

    # Write helper script
    fd, script_path = tempfile.mkstemp(suffix=".py", prefix="bench_orig_")
    os.write(fd, _SUBPROCESS_SCRIPT.encode())
    os.close(fd)

    cfg = json.dumps({"device": device, "sizes": sizes, "repeats": repeats})

    try:
        # switch to main
        subprocess.check_call(
            ["git", "checkout", branch, "--quiet"], cwd=repo_root
        )
        # Clear __pycache__ to avoid stale bytecode
        subprocess.run(
            ["find", "src", "-name", "__pycache__", "-exec", "rm", "-rf",
             "{}", "+"],
            cwd=repo_root, capture_output=True,
        )

        proc = subprocess.run(
            [sys.executable, script_path, cfg],
            capture_output=True, text=True, cwd=repo_root,
        )
        if proc.returncode != 0:
            print(f"  [subprocess stderr]\n{proc.stderr[-2000:]}")
            raise RuntimeError(
                f"Original benchmark subprocess failed (rc={proc.returncode})"
            )

        # parse results
        for line in proc.stdout.splitlines():
            if line.startswith("__BENCH_RESULTS__"):
                raw = proc.stdout.split("__BENCH_RESULTS__")[1].strip()
                results_raw = json.loads(raw)
                # Keys are strings from JSON, convert to int
                return {int(k): v for k, v in results_raw.items()}

        raise RuntimeError(
            "Could not find __BENCH_RESULTS__ marker in subprocess output.\n"
            f"stdout (last 1000 chars): {proc.stdout[-1000:]}"
        )
    finally:
        subprocess.check_call(
            ["git", "checkout", current_branch, "--quiet"], cwd=repo_root
        )
        subprocess.run(
            ["find", "src", "-name", "__pycache__", "-exec", "rm", "-rf",
             "{}", "+"],
            cwd=repo_root, capture_output=True,
        )
        os.unlink(script_path)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def setup_plot_style():
    """Configure matplotlib for publication-quality plots."""
    import matplotlib as mpl
    import matplotlib.pyplot as plt

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
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
    })


COLORS = {
    "Original": "#4C72B0",
    "Optimized": "#55A868",
    "Optimized + Ckpt": "#C44E52",
}
MARKERS = {"Original": "o", "Optimized": "s", "Optimized + Ckpt": "^"}


def plot_inference_time(all_results: dict, output_dir: str):
    """Bar plot: inference time (ms) for each config × system size."""
    import matplotlib.pyplot as plt

    setup_plot_style()

    configs = list(all_results.keys())
    atom_counts = sorted(
        {n for cfg in all_results.values() for n in cfg if cfg[n] is not None}
    )

    fig, ax = plt.subplots(figsize=(10, 5.5))
    n_configs = len(configs)
    bar_width = 0.8 / n_configs
    x = np.arange(len(atom_counts))

    for i, cfg_name in enumerate(configs):
        cfg_data = all_results[cfg_name]
        means = []
        stds = []
        for n in atom_counts:
            d = cfg_data.get(n)
            if d is not None:
                means.append(d["mean_ms"])
                stds.append(d["std_ms"])
            else:
                means.append(0)
                stds.append(0)
        offset = (i - (n_configs - 1) / 2) * bar_width
        bars = ax.bar(
            x + offset, means, bar_width * 0.9, yerr=stds,
            label=cfg_name, color=COLORS[cfg_name],
            edgecolor="white", linewidth=0.5,
            capsize=3, error_kw={"linewidth": 1},
        )
        # Add value labels on bars
        for bar, m in zip(bars, means):
            if m > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f"{m:.0f}", ha="center", va="bottom", fontsize=8,
                )

    ax.set_xlabel("Number of Atoms")
    ax.set_ylabel("Inference Time (ms)")
    ax.set_title("Inference Time: Energy + Forces + Stresses")
    ax.set_xticks(x)
    ax.set_xticklabels(atom_counts)
    ax.legend(frameon=True, fancybox=True, shadow=False, framealpha=0.9)
    ax.set_ylim(bottom=0)

    path = os.path.join(output_dir, "inference_time.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved → {path}")


def plot_peak_memory(all_results: dict, output_dir: str):
    """Bar plot: peak GPU memory (MB) for each config × system size."""
    import matplotlib.pyplot as plt

    setup_plot_style()

    configs = list(all_results.keys())
    atom_counts = sorted(
        {n for cfg in all_results.values() for n in cfg if cfg[n] is not None}
    )

    fig, ax = plt.subplots(figsize=(10, 5.5))
    n_configs = len(configs)
    bar_width = 0.8 / n_configs
    x = np.arange(len(atom_counts))

    for i, cfg_name in enumerate(configs):
        cfg_data = all_results[cfg_name]
        mems = []
        for n in atom_counts:
            d = cfg_data.get(n)
            mems.append(d["peak_mem_mb"] if d else 0)
        offset = (i - (n_configs - 1) / 2) * bar_width
        bars = ax.bar(
            x + offset, mems, bar_width * 0.9,
            label=cfg_name, color=COLORS[cfg_name],
            edgecolor="white", linewidth=0.5,
        )
        for bar, m in zip(bars, mems):
            if m > 0:
                label = f"{m:.0f}" if m < 1000 else f"{m / 1024:.1f}G"
                ax.text(
                    bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                    label, ha="center", va="bottom", fontsize=8,
                )

    ax.set_xlabel("Number of Atoms")
    ax.set_ylabel("Peak GPU Memory (MB)")
    ax.set_title("Peak GPU Memory During Inference")
    ax.set_xticks(x)
    ax.set_xticklabels(atom_counts)
    ax.legend(frameon=True, fancybox=True, shadow=False, framealpha=0.9)
    ax.set_ylim(bottom=0)

    path = os.path.join(output_dir, "peak_memory.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved → {path}")


def plot_memory_savings(all_results: dict, output_dir: str):
    """Line plot: % memory saved by checkpointing relative to original."""
    import matplotlib.pyplot as plt

    setup_plot_style()

    atom_counts = sorted(
        {n for cfg in all_results.values() for n in cfg if cfg[n] is not None}
    )

    fig, ax = plt.subplots(figsize=(9, 5))

    # Memory savings: checkpointed vs original
    orig = all_results.get("Original", {})
    ckpt = all_results.get("Optimized + Ckpt", {})

    savings = []
    valid_counts = []
    for n in atom_counts:
        o = orig.get(n)
        c = ckpt.get(n)
        if o and c and o["peak_mem_mb"] > 0:
            pct = (1 - c["peak_mem_mb"] / o["peak_mem_mb"]) * 100
            savings.append(pct)
            valid_counts.append(n)

    if savings:
        ax.plot(
            valid_counts, savings, "o-",
            color=COLORS["Optimized + Ckpt"], linewidth=2.5,
            markersize=8, markeredgecolor="white", markeredgewidth=1.5,
            label="Memory Saved vs Original",
        )
        ax.fill_between(
            valid_counts, 0, savings,
            alpha=0.15, color=COLORS["Optimized + Ckpt"],
        )

        # Annotate values
        for xv, yv in zip(valid_counts, savings):
            ax.annotate(
                f"{yv:.0f}%", (xv, yv),
                textcoords="offset points", xytext=(0, 12),
                ha="center", fontsize=10, fontweight="bold",
                color=COLORS["Optimized + Ckpt"],
            )

    ax.set_xlabel("Number of Atoms")
    ax.set_ylabel("Memory Saved (%)")
    ax.set_title("GPU Memory Savings with Gradient Checkpointing")
    ax.set_ylim(bottom=0, top=max(savings + [60]) * 1.15)
    ax.legend(frameon=True, fancybox=True, framealpha=0.9)

    path = os.path.join(output_dir, "memory_savings.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved → {path}")


def plot_scaling(all_results: dict, output_dir: str):
    """Log-log scaling plot: time vs atoms for all configs."""
    import matplotlib.pyplot as plt

    setup_plot_style()

    fig, ax = plt.subplots(figsize=(9, 5.5))

    for cfg_name, cfg_data in all_results.items():
        ns = sorted(n for n in cfg_data if cfg_data[n] is not None)
        means = [cfg_data[n]["mean_ms"] for n in ns]
        stds = [cfg_data[n]["std_ms"] for n in ns]
        ax.errorbar(
            ns, means, yerr=stds,
            marker=MARKERS[cfg_name], linestyle="-", linewidth=2,
            markersize=8, markeredgecolor="white", markeredgewidth=1.5,
            color=COLORS[cfg_name], label=cfg_name,
            capsize=4, capthick=1.5,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of Atoms")
    ax.set_ylabel("Inference Time (ms)")
    ax.set_title("Inference Scaling: Time vs System Size")
    ax.legend(frameon=True, fancybox=True, framealpha=0.9)

    path = os.path.join(output_dir, "scaling.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved → {path}")


def plot_combined_summary(all_results: dict, output_dir: str, gpu_name: str):
    """2×2 panel summary figure."""
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    setup_plot_style()

    configs = list(all_results.keys())
    atom_counts = sorted(
        {n for cfg in all_results.values() for n in cfg if cfg[n] is not None}
    )
    n_configs = len(configs)
    bar_width = 0.8 / n_configs
    x = np.arange(len(atom_counts))

    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(2, 2, hspace=0.32, wspace=0.28)

    # ── Panel (a): Inference time bars ──
    ax1 = fig.add_subplot(gs[0, 0])
    for i, cfg_name in enumerate(configs):
        cd = all_results[cfg_name]
        means = [cd.get(n, {}).get("mean_ms", 0) if cd.get(n) else 0
                 for n in atom_counts]
        stds = [cd.get(n, {}).get("std_ms", 0) if cd.get(n) else 0
                for n in atom_counts]
        offset = (i - (n_configs - 1) / 2) * bar_width
        ax1.bar(
            x + offset, means, bar_width * 0.9, yerr=stds,
            label=cfg_name, color=COLORS[cfg_name],
            edgecolor="white", linewidth=0.5,
            capsize=2, error_kw={"linewidth": 0.8},
        )
    ax1.set_xlabel("Number of Atoms")
    ax1.set_ylabel("Inference Time (ms)")
    ax1.set_title("(a) Inference Time (E + F + S)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(atom_counts)
    ax1.legend(fontsize=9, frameon=True, fancybox=True, framealpha=0.9)
    ax1.set_ylim(bottom=0)

    # ── Panel (b): Peak memory bars ──
    ax2 = fig.add_subplot(gs[0, 1])
    for i, cfg_name in enumerate(configs):
        cd = all_results[cfg_name]
        mems = [cd.get(n, {}).get("peak_mem_mb", 0) if cd.get(n) else 0
                for n in atom_counts]
        offset = (i - (n_configs - 1) / 2) * bar_width
        ax2.bar(
            x + offset, mems, bar_width * 0.9,
            label=cfg_name, color=COLORS[cfg_name],
            edgecolor="white", linewidth=0.5,
        )
    ax2.set_xlabel("Number of Atoms")
    ax2.set_ylabel("Peak GPU Memory (MB)")
    ax2.set_title("(b) Peak GPU Memory")
    ax2.set_xticks(x)
    ax2.set_xticklabels(atom_counts)
    ax2.legend(fontsize=9, frameon=True, fancybox=True, framealpha=0.9)
    ax2.set_ylim(bottom=0)

    # ── Panel (c): Scaling (log-log) ──
    ax3 = fig.add_subplot(gs[1, 0])
    for cfg_name, cfg_data in all_results.items():
        ns = sorted(n for n in cfg_data if cfg_data[n] is not None)
        means = [cfg_data[n]["mean_ms"] for n in ns]
        ax3.plot(
            ns, means, marker=MARKERS[cfg_name], linestyle="-", linewidth=2,
            markersize=7, markeredgecolor="white", markeredgewidth=1,
            color=COLORS[cfg_name], label=cfg_name,
        )
    ax3.set_xscale("log")
    ax3.set_yscale("log")
    ax3.set_xlabel("Number of Atoms")
    ax3.set_ylabel("Inference Time (ms)")
    ax3.set_title("(c) Inference Scaling")
    ax3.legend(fontsize=9, frameon=True, fancybox=True, framealpha=0.9)

    # ── Panel (d): Memory savings line ──
    ax4 = fig.add_subplot(gs[1, 1])
    orig = all_results.get("Original", {})
    ckpt = all_results.get("Optimized + Ckpt", {})
    savings, valid_ns = [], []
    for n in atom_counts:
        o, c = orig.get(n), ckpt.get(n)
        if o and c and o["peak_mem_mb"] > 0:
            savings.append((1 - c["peak_mem_mb"] / o["peak_mem_mb"]) * 100)
            valid_ns.append(n)
    if savings:
        ax4.plot(
            valid_ns, savings, "o-", color=COLORS["Optimized + Ckpt"],
            linewidth=2.5, markersize=8,
            markeredgecolor="white", markeredgewidth=1.5,
        )
        ax4.fill_between(
            valid_ns, 0, savings, alpha=0.15,
            color=COLORS["Optimized + Ckpt"],
        )
        for xv, yv in zip(valid_ns, savings):
            ax4.annotate(
                f"{yv:.0f}%", (xv, yv),
                textcoords="offset points", xytext=(0, 10),
                ha="center", fontsize=10, fontweight="bold",
                color=COLORS["Optimized + Ckpt"],
            )
    ax4.set_xlabel("Number of Atoms")
    ax4.set_ylabel("Memory Saved (%)")
    ax4.set_title("(d) Memory Savings (Checkpointing vs Original)")
    ax4.set_ylim(bottom=0, top=max(savings + [60]) * 1.15)

    fig.suptitle(
        f"MatterSim Optimization Benchmark  ·  {gpu_name}",
        fontsize=15, fontweight="bold", y=0.98,
    )

    path = os.path.join(output_dir, "summary.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved → {path}")


# ---------------------------------------------------------------------------
# Console report
# ---------------------------------------------------------------------------

DIVIDER = "=" * 82


def print_report(all_results: dict):
    """Print a combined table to the console."""
    atom_counts = sorted(
        {n for cfg in all_results.values() for n in cfg if cfg[n] is not None}
    )
    configs = list(all_results.keys())

    print(f"\n{DIVIDER}")
    print("RESULTS")
    print(DIVIDER)

    # Time table
    header = f"{'Atoms':>8}"
    for c in configs:
        header += f" │ {c + ' (ms)':>22}"
    print(header)
    print("─" * len(header.encode("utf-8")))

    for n in atom_counts:
        row = f"{n:>8}"
        for c in configs:
            d = all_results[c].get(n)
            if d:
                row += f" │ {d['mean_ms']:>15.1f} ± {d['std_ms']:<4.1f}"
            else:
                row += f" │ {'OOM':>22}"
        print(row)

    # Memory table
    print()
    header = f"{'Atoms':>8}"
    for c in configs:
        header += f" │ {c + ' (MB)':>22}"
    print(header)
    print("─" * len(header.encode("utf-8")))

    for n in atom_counts:
        row = f"{n:>8}"
        for c in configs:
            d = all_results[c].get(n)
            if d:
                row += f" │ {d['peak_mem_mb']:>22.1f}"
            else:
                row += f" │ {'OOM':>22}"
        print(row)

    # Numerical consistency
    print(f"\nNumerical consistency (ΔE vs Original):")
    orig = all_results.get("Original", {})
    for c in configs:
        if c == "Original":
            continue
        for n in atom_counts:
            o = orig.get(n)
            d = all_results[c].get(n)
            if o and d:
                diff = abs(o["energy"] - d["energy"])
                print(f"  {c:>25}  {n:>6} atoms: ΔE = {diff:.2e} eV")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Original vs Optimized vs Checkpointed M3Gnet",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--sizes", default="8,64,216,512,1000,2744",
        help="Comma-separated target atom counts",
    )
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--output-dir", default="benchmark_results",
        help="Directory for output plots and data",
    )
    parser.add_argument(
        "--main-branch", default="main",
        help="Git branch name for the original (baseline) model",
    )
    args = parser.parse_args()

    device = args.device
    sizes = [int(s.strip()) for s in args.sizes.split(",")]
    repeats = args.repeats
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    gpu_name = "CPU"
    if device == "cuda" and torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name()

    print(DIVIDER)
    print("MatterSim Optimization Benchmark")
    print(DIVIDER)
    print(f"Device:       {device} ({gpu_name})")
    print(f"PyTorch:      {torch.__version__}")
    print(f"Target sizes: {sizes}")
    print(f"Repeats:      {repeats}")
    print(f"Output dir:   {output_dir}")

    # Generate structures for display
    structures = generate_structures(sizes)
    actual_sizes = sorted(structures.keys())
    print(f"Actual sizes: {actual_sizes}")

    # ==================================================================
    # 1. Original model (main branch, via subprocess)
    # ==================================================================
    print(f"\n{DIVIDER}")
    print("(1/3) Benchmarking ORIGINAL model (main branch)...")
    print(DIVIDER)
    try:
        original_results = run_original_benchmark(
            repo_root, device, sizes, repeats, branch=args.main_branch,
        )
        for n in sorted(original_results):
            r = original_results[n]
            if r:
                print(
                    f"  {n:>6} atoms: {r['mean_ms']:.1f} ms  "
                    f"mem={r['peak_mem_mb']:.0f} MB"
                )
            else:
                print(f"  {n:>6} atoms: OOM")
    except Exception as e:
        print(f"  ⚠ Original benchmark failed: {e}")
        print("  Falling back: using optimized (no ckpt) as 'Original'")
        original_results = None

    # ==================================================================
    # 2. Optimized model (current branch, no checkpointing)
    # ==================================================================
    print(f"\n{DIVIDER}")
    print("(2/3) Benchmarking OPTIMIZED model (no checkpointing)...")
    print(DIVIDER)

    from mattersim.forcefield.potential import Potential

    potential = Potential.from_checkpoint(device=device)
    potential.model.eval()
    potential.enable_gradient_checkpointing(False)

    optimized_results = {}
    for n, atoms in sorted(structures.items()):
        try:
            r = time_inference_inprocess(potential, atoms, device, repeats=repeats)
            print(
                f"  {n:>6} atoms: {r['mean_ms']:.1f} ms  "
                f"mem={r['peak_mem_mb']:.0f} MB"
            )
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                r = None
                print(f"  {n:>6} atoms: OOM")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
            else:
                raise
        optimized_results[n] = r

    # ==================================================================
    # 3. Optimized + Gradient Checkpointing
    # ==================================================================
    print(f"\n{DIVIDER}")
    print("(3/3) Benchmarking OPTIMIZED + GRADIENT CHECKPOINTING...")
    print(DIVIDER)

    potential.enable_gradient_checkpointing(True)

    ckpt_results = {}
    for n, atoms in sorted(structures.items()):
        try:
            r = time_inference_inprocess(potential, atoms, device, repeats=repeats)
            print(
                f"  {n:>6} atoms: {r['mean_ms']:.1f} ms  "
                f"mem={r['peak_mem_mb']:.0f} MB"
            )
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                r = None
                print(f"  {n:>6} atoms: OOM")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
            else:
                raise
        ckpt_results[n] = r

    potential.enable_gradient_checkpointing(False)

    # ==================================================================
    # Combine & Report
    # ==================================================================
    if original_results is None:
        original_results = optimized_results  # fallback

    all_results = {
        "Original": original_results,
        "Optimized": optimized_results,
        "Optimized + Ckpt": ckpt_results,
    }

    print_report(all_results)

    # Save raw data
    data_path = os.path.join(output_dir, "benchmark_data.json")
    # Convert keys to strings for JSON
    serializable = {
        cfg: {str(k): v for k, v in data.items()}
        for cfg, data in all_results.items()
    }
    with open(data_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nRaw data saved → {data_path}")

    # Generate plots
    print("\nGenerating plots...")
    plot_inference_time(all_results, output_dir)
    plot_peak_memory(all_results, output_dir)
    plot_memory_savings(all_results, output_dir)
    plot_scaling(all_results, output_dir)
    plot_combined_summary(all_results, output_dir, gpu_name)

    print(f"\n{DIVIDER}")
    print(f"All outputs saved to {output_dir}/")
    print(DIVIDER)


if __name__ == "__main__":
    main()
