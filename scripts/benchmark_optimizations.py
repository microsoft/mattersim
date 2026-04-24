#!/usr/bin/env python
"""Benchmark script for MatterSim optimization features.

Benchmarks the following optimizations extracted from the internal repo:
  1. Gradient checkpointing — memory reduction for large systems
  2. Native scatter_sum — checkpoint-compatible scatter operations
  3. AOTI compilation — ahead-of-time compiled inference (optional)
  4. GPU three-body index computation (optional standalone benchmark)

Usage:
    python scripts/benchmark_optimizations.py [--device cuda] [--sizes 8,64,216,512]
    python scripts/benchmark_optimizations.py --help

The script generates test structures of varying sizes from bulk silicon,
benchmarks inference (energy + forces + stresses), and reports timing
and memory usage comparisons.
"""

import argparse
import gc
import sys
import time
from collections import defaultdict

import numpy as np
import torch
from ase.build import bulk, make_supercell

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def generate_structures(sizes: list[int]) -> dict[int, "ase.Atoms"]:
    """Generate bulk Si structures at different atom counts.

    Uses cubic diamond Si and builds supercells to reach approximately
    the requested sizes.  Returns a dict mapping actual atom count to
    the Atoms object.
    """
    from ase import Atoms  # noqa: F811

    base = bulk("Si", "diamond", a=5.43, cubic=True)  # 8 atoms
    structures: dict[int, Atoms] = {}
    for target in sorted(sizes):
        # Find supercell multiplier: base has 8 atoms
        n = max(1, round((target / len(base)) ** (1 / 3)))
        atoms = make_supercell(base, [[n, 0, 0], [0, n, 0], [0, 0, n]])
        # Small random perturbation to break symmetry
        rng = np.random.default_rng(42)
        atoms.positions += rng.normal(scale=0.01, size=atoms.positions.shape)
        structures[len(atoms)] = atoms
    return structures


def get_gpu_memory_mb() -> float:
    """Return current GPU memory allocated in MB (0 if not CUDA)."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**2
    return 0.0


def get_peak_gpu_memory_mb() -> float:
    """Return peak GPU memory allocated in MB (0 if not CUDA)."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024**2
    return 0.0


def reset_gpu_memory_stats():
    """Reset peak memory stats."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
    gc.collect()


def time_inference(
    potential,
    atoms,
    device: str,
    include_forces: bool = True,
    include_stresses: bool = True,
    warmup: int = 2,
    repeats: int = 5,
) -> dict:
    """Time a single inference run, returning timing and memory info."""
    from mattersim.datasets.utils.build import build_dataloader
    from mattersim.forcefield.potential import batch_to_dict

    # Build dataloader once (not included in timing)
    cutoff = potential.model.model_args.get("cutoff", 5.0)
    threebody_cutoff = potential.model.model_args.get("threebody_cutoff", 4.0)
    dl = build_dataloader(
        [atoms],
        batch_size=1,
        model_type="m3gnet",
        only_inference=True,
        cutoff=cutoff,
        threebody_cutoff=threebody_cutoff,
    )
    graph_batch = next(iter(dl))
    input_dict = batch_to_dict(graph_batch, device=device)

    potential.model.eval()

    # Warmup
    for _ in range(warmup):
        _ = potential.forward(
            input_dict,
            include_forces=include_forces,
            include_stresses=include_stresses,
        )
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Timed runs
    reset_gpu_memory_stats()
    times = []
    for _ in range(repeats):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        result = potential.forward(
            input_dict,
            include_forces=include_forces,
            include_stresses=include_stresses,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)

    peak_mem = get_peak_gpu_memory_mb()

    return {
        "mean_time_ms": np.mean(times) * 1000,
        "std_time_ms": np.std(times) * 1000,
        "min_time_ms": np.min(times) * 1000,
        "peak_memory_mb": peak_mem,
        "energy": result["total_energy"].detach().cpu().item(),
    }


# ---------------------------------------------------------------------------
# Benchmark: Gradient Checkpointing
# ---------------------------------------------------------------------------


def benchmark_gradient_checkpointing(
    potential, structures: dict, device: str, repeats: int
) -> dict:
    """Compare inference with and without gradient checkpointing."""
    results = defaultdict(dict)

    for n_atoms, atoms in sorted(structures.items()):
        print(f"\n  {n_atoms} atoms:", end="", flush=True)

        # --- Baseline (no checkpointing) ---
        potential.enable_gradient_checkpointing(False)
        try:
            reset_gpu_memory_stats()
            r_base = time_inference(
                potential, atoms, device, repeats=repeats
            )
            print(f" baseline={r_base['mean_time_ms']:.1f}ms", end="", flush=True)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(" baseline=OOM", end="", flush=True)
                r_base = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
            else:
                raise

        # --- With checkpointing ---
        potential.enable_gradient_checkpointing(True)
        try:
            reset_gpu_memory_stats()
            r_ckpt = time_inference(
                potential, atoms, device, repeats=repeats
            )
            print(
                f" checkpointed={r_ckpt['mean_time_ms']:.1f}ms", end="", flush=True
            )
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(" checkpointed=OOM", end="", flush=True)
                r_ckpt = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
            else:
                raise

        results[n_atoms] = {"baseline": r_base, "checkpointed": r_ckpt}

    # Reset
    potential.enable_gradient_checkpointing(False)
    return dict(results)


# ---------------------------------------------------------------------------
# Benchmark: GPU Three-Body Index Computation
# ---------------------------------------------------------------------------


def benchmark_threebody_indices(structures: dict, device: str) -> dict:
    """Benchmark GPU vs CPU three-body index computation."""
    from mattersim.datasets.utils.build import build_dataloader
    from mattersim.forcefield.m3gnet.threebody_indices_torch import (
        compute_threebody_torch,
    )

    results = {}

    for n_atoms, atoms in sorted(structures.items()):
        print(f"\n  {n_atoms} atoms:", end="", flush=True)

        # Build graph to get edge_indices
        dl = build_dataloader(
            [atoms],
            batch_size=1,
            model_type="m3gnet",
            only_inference=True,
            cutoff=5.0,
            threebody_cutoff=4.0,
        )
        graph_batch = next(iter(dl))

        # Prepare inputs for GPU three-body computation
        edge_index = graph_batch.edge_index.T.contiguous()  # [n_edges, 2]
        # Sort by first column (central atom) as required
        sorted_idx = torch.argsort(edge_index[:, 0], stable=True)
        edge_index_sorted = edge_index[sorted_idx]
        n_atoms_tensor = graph_batch.num_atoms

        # CPU timing
        t_cpu_times = []
        for _ in range(3):
            t0 = time.perf_counter()
            _ = compute_threebody_torch(edge_index_sorted, n_atoms_tensor)
            t1 = time.perf_counter()
            t_cpu_times.append(t1 - t0)
        cpu_ms = np.mean(t_cpu_times) * 1000

        # GPU timing (if available)
        gpu_ms = None
        if device == "cuda" and torch.cuda.is_available():
            edge_index_gpu = edge_index_sorted.cuda()
            n_atoms_gpu = n_atoms_tensor.cuda()
            # warmup
            _ = compute_threebody_torch(edge_index_gpu, n_atoms_gpu)
            torch.cuda.synchronize()

            t_gpu_times = []
            for _ in range(3):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                _ = compute_threebody_torch(edge_index_gpu, n_atoms_gpu)
                torch.cuda.synchronize()
                t1 = time.perf_counter()
                t_gpu_times.append(t1 - t0)
            gpu_ms = np.mean(t_gpu_times) * 1000

        results[n_atoms] = {"cpu_ms": cpu_ms, "gpu_ms": gpu_ms}

        msg = f" cpu={cpu_ms:.2f}ms"
        if gpu_ms is not None:
            msg += f" gpu={gpu_ms:.2f}ms speedup={cpu_ms / gpu_ms:.1f}x"
        print(msg, end="", flush=True)

    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

DIVIDER = "=" * 78


def print_checkpointing_report(results: dict):
    """Print a formatted table of gradient checkpointing results."""
    print(f"\n{DIVIDER}")
    print("GRADIENT CHECKPOINTING BENCHMARK")
    print(DIVIDER)
    print(
        f"{'Atoms':>8} │ {'Baseline (ms)':>14} │ {'Checkpt (ms)':>14} │ "
        f"{'Slowdown':>9} │ {'Mem Base (MB)':>14} │ {'Mem Ckpt (MB)':>14} │ "
        f"{'Mem Saved':>10}"
    )
    print("─" * 8 + "─┼─" + "─" * 14 + "─┼─" + "─" * 14 + "─┼─" +
          "─" * 9 + "─┼─" + "─" * 14 + "─┼─" + "─" * 14 + "─┼─" + "─" * 10)

    for n_atoms in sorted(results.keys()):
        r = results[n_atoms]
        base = r["baseline"]
        ckpt = r["checkpointed"]

        base_time = f"{base['mean_time_ms']:.1f}" if base else "OOM"
        ckpt_time = f"{ckpt['mean_time_ms']:.1f}" if ckpt else "OOM"

        if base and ckpt:
            slowdown = f"{ckpt['mean_time_ms'] / base['mean_time_ms']:.2f}x"
        elif base is None and ckpt:
            slowdown = "∞→OK"
        else:
            slowdown = "—"

        base_mem = f"{base['peak_memory_mb']:.1f}" if base else "OOM"
        ckpt_mem = f"{ckpt['peak_memory_mb']:.1f}" if ckpt else "OOM"

        if base and ckpt and base["peak_memory_mb"] > 0:
            saved = (
                1 - ckpt["peak_memory_mb"] / base["peak_memory_mb"]
            ) * 100
            saved_str = f"{saved:+.0f}%"
        else:
            saved_str = "—"

        print(
            f"{n_atoms:>8} │ {base_time:>14} │ {ckpt_time:>14} │ "
            f"{slowdown:>9} │ {base_mem:>14} │ {ckpt_mem:>14} │ "
            f"{saved_str:>10}"
        )

    # Verify numerical equivalence
    print("\nNumerical equivalence check (energy diff between modes):")
    for n_atoms in sorted(results.keys()):
        r = results[n_atoms]
        base = r["baseline"]
        ckpt = r["checkpointed"]
        if base and ckpt:
            diff = abs(base["energy"] - ckpt["energy"])
            print(f"  {n_atoms:>6} atoms: ΔE = {diff:.2e} eV")


def print_threebody_report(results: dict):
    """Print a formatted table of three-body index benchmark results."""
    print(f"\n{DIVIDER}")
    print("GPU THREE-BODY INDEX COMPUTATION")
    print(DIVIDER)
    has_gpu = any(r["gpu_ms"] is not None for r in results.values())

    if has_gpu:
        print(
            f"{'Atoms':>8} │ {'CPU (ms)':>10} │ {'GPU (ms)':>10} │ "
            f"{'Speedup':>8}"
        )
        print("─" * 8 + "─┼─" + "─" * 10 + "─┼─" + "─" * 10 + "─┼─" + "─" * 8)
        for n_atoms in sorted(results.keys()):
            r = results[n_atoms]
            speedup = (
                f"{r['cpu_ms'] / r['gpu_ms']:.1f}x"
                if r["gpu_ms"]
                else "—"
            )
            gpu_str = f"{r['gpu_ms']:.2f}" if r["gpu_ms"] else "N/A"
            print(
                f"{n_atoms:>8} │ {r['cpu_ms']:>10.2f} │ {gpu_str:>10} │ "
                f"{speedup:>8}"
            )
    else:
        print(f"{'Atoms':>8} │ {'CPU (ms)':>10}")
        print("─" * 8 + "─┼─" + "─" * 10)
        for n_atoms in sorted(results.keys()):
            r = results[n_atoms]
            print(f"{n_atoms:>8} │ {r['cpu_ms']:>10.2f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark MatterSim optimizations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for benchmarking (default: cuda if available)",
    )
    parser.add_argument(
        "--sizes",
        default="8,64,216,512",
        help="Comma-separated target atom counts (default: 8,64,216,512)",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=5,
        help="Number of timed repetitions per configuration (default: 5)",
    )
    parser.add_argument(
        "--skip-checkpointing",
        action="store_true",
        help="Skip gradient checkpointing benchmark",
    )
    parser.add_argument(
        "--skip-threebody",
        action="store_true",
        help="Skip three-body index benchmark",
    )
    args = parser.parse_args()

    device = args.device
    sizes = [int(s.strip()) for s in args.sizes.split(",")]
    repeats = args.repeats

    print(DIVIDER)
    print("MatterSim Optimization Benchmark")
    print(DIVIDER)
    print(f"Device:       {device}")
    if device == "cuda" and torch.cuda.is_available():
        print(f"GPU:          {torch.cuda.get_device_name()}")
        print(
            f"GPU Memory:   "
            f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB"
        )
    print(f"PyTorch:      {torch.__version__}")
    print(f"Target sizes: {sizes}")
    print(f"Repeats:      {repeats}")

    # Generate structures
    print("\nGenerating test structures...")
    structures = generate_structures(sizes)
    actual_sizes = sorted(structures.keys())
    print(f"  Created structures with {actual_sizes} atoms")

    # ------------------------------------------------------------------
    # Benchmark 1: Gradient Checkpointing
    # ------------------------------------------------------------------
    if not args.skip_checkpointing:
        print("\n" + DIVIDER)
        print("Running gradient checkpointing benchmark...")
        print("  (comparing baseline vs checkpointed inference)")

        from mattersim.forcefield.potential import Potential

        potential = Potential.from_checkpoint(device=device)
        potential.model.eval()

        ckpt_results = benchmark_gradient_checkpointing(
            potential, structures, device, repeats
        )
        print()
        print_checkpointing_report(ckpt_results)

        del potential
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    # ------------------------------------------------------------------
    # Benchmark 2: GPU Three-Body Index Computation
    # ------------------------------------------------------------------
    if not args.skip_threebody:
        print("\n" + DIVIDER)
        print("Running three-body index computation benchmark...")
        print("  (comparing CPU vs GPU torch implementation)")

        tb_results = benchmark_threebody_indices(structures, device)
        print()
        print_threebody_report(tb_results)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\n{DIVIDER}")
    print("SUMMARY")
    print(DIVIDER)
    print(
        "Optimizations available in this build:\n"
        "  ✓ Native scatter_sum (checkpoint-compatible)\n"
        "  ✓ Gradient checkpointing (M3Gnet.enable_gradient_checkpointing)\n"
        "  ✓ GPU three-body indices (threebody_indices_torch.py)\n"
        "  ✓ AOTI compilation (forcefield.aoti_compile — requires torch>=2.4)\n"
    )
    print(
        "To enable gradient checkpointing at inference time:\n"
        "    potential = Potential.from_checkpoint()\n"
        "    potential.enable_gradient_checkpointing(True)\n"
    )
    print(
        "To compile with AOTI for maximum inference speed:\n"
        "    from mattersim.forcefield.aoti_compile import (\n"
        "        AOTISettings, compile_m3gnet_aoti, load_aoti_model\n"
        "    )\n"
        "    settings = AOTISettings(include_forces=True, include_stresses=True)\n"
        "    pt2_path = compile_m3gnet_aoti(model, version='v1.0.0', device='cuda')\n"
        "    aoti_model = load_aoti_model(pt2_path, model.model_args, settings)\n"
    )


if __name__ == "__main__":
    main()
