import os
import time
import random
from loguru import logger
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm
from mattersim.datasets.utils.build import build_dataloader
from mattersim.forcefield import Potential
from ase.io import read as ase_read
from ase.io import write as ase_write


def run_benchmark(atoms_list, batch_converter, potential=None):
    """Run graph conversion benchmark and optionally inference, return elapsed times."""
    if not atoms_list:
        return 0.0, 0.0
    
    # Graph conversion time
    t0 = time.time()
    dataloader = build_dataloader(atoms_list, only_inference=True, batch_converter=batch_converter)
    t1 = time.time()
    time_graph = t1 - t0
    
    # Inference time (if potential provided)
    time_inference = 0.0
    if potential is not None:
        t2 = time.time()
        _ = potential.predict_properties(
            dataloader, 
            include_forces=False, 
            include_stresses=False
        )
        t3 = time.time()
        time_inference = t3 - t2
    
    return time_graph, time_inference


def compute_energy_errors_and_speedup(potential, atoms_list):
    """
    Compute energy prediction errors between batch_converter=True and batch_converter=False,
    and measure the speedup.
    
    Uses batch_converter=False as reference (original implementation).
    
    Args:
        potential: MatterSim Potential model
        atoms_list: List of ASE Atoms objects
    
    Returns:
        dict with error metrics (MAE, RMSE, Max Error), per-atom versions, and timing info
    """
    logger.info(f"Processing {len(atoms_list)} structures...")
    
    # -------------------------------------------------------------------------
    # Reference: batch_converter=False (original/slow)
    # -------------------------------------------------------------------------
    logger.info("Running prediction with batch_converter=False (reference)...")
    t0_ref = time.time()
    dataloader_ref = build_dataloader(
        atoms_list, 
        only_inference=True, 
        batch_converter=False
    )
    t1_ref = time.time()
    time_graph_ref = t1_ref - t0_ref
    
    t0_pred_ref = time.time()
    ref_energies, _, _ = potential.predict_properties(
        dataloader_ref, 
        include_forces=False, 
        include_stresses=False
    )
    t1_pred_ref = time.time()
    time_pred_ref = t1_pred_ref - t0_pred_ref
    time_total_ref = time_graph_ref + time_pred_ref
    
    logger.info(f"  Graph conversion time: {time_graph_ref:.4f}s")
    logger.info(f"  Prediction time: {time_pred_ref:.4f}s")
    logger.info(f"  Total time: {time_total_ref:.4f}s")
    
    # -------------------------------------------------------------------------
    # Test: batch_converter=True (new/fast)
    # -------------------------------------------------------------------------
    logger.info("Running prediction with batch_converter=True (test)...")
    t0_test = time.time()
    dataloader_test = build_dataloader(
        atoms_list, 
        only_inference=True, 
        batch_converter=True
    )
    t1_test = time.time()
    time_graph_test = t1_test - t0_test
    
    t0_pred_test = time.time()
    pred_energies, _, _ = potential.predict_properties(
        dataloader_test, 
        include_forces=False, 
        include_stresses=False
    )
    t1_pred_test = time.time()
    time_pred_test = t1_pred_test - t0_pred_test
    time_total_test = time_graph_test + time_pred_test
    
    logger.info(f"  Graph conversion time: {time_graph_test:.4f}s")
    logger.info(f"  Prediction time: {time_pred_test:.4f}s")
    logger.info(f"  Total time: {time_total_test:.4f}s")
    
    # -------------------------------------------------------------------------
    # Compute errors and speedup
    # -------------------------------------------------------------------------
    ref_energies = np.array(ref_energies)
    pred_energies = np.array(pred_energies)
    
    # Compute errors
    errors = pred_energies - ref_energies
    abs_errors = np.abs(errors)
    
    # Per-structure metrics
    mae = np.mean(abs_errors)
    rmse = np.sqrt(np.mean(errors**2))
    max_error = np.max(abs_errors)
    
    # Per-atom metrics
    num_atoms = np.array([len(a) for a in atoms_list])
    per_atom_errors = abs_errors / num_atoms
    mae_per_atom = np.mean(per_atom_errors)
    rmse_per_atom = np.sqrt(np.mean((errors / num_atoms)**2))
    max_error_per_atom = np.max(per_atom_errors)
    
    # Speedup calculations
    speedup_graph = time_graph_ref / time_graph_test if time_graph_test > 0 else float('inf')
    speedup_total = time_total_ref / time_total_test if time_total_test > 0 else float('inf')
    
    return {
        "mae": mae,
        "rmse": rmse,
        "max_error": max_error,
        "mae_per_atom": mae_per_atom,
        "rmse_per_atom": rmse_per_atom,
        "max_error_per_atom": max_error_per_atom,
        "ref_energies": ref_energies,
        "pred_energies": pred_energies,
        "num_structures": len(atoms_list),
        "num_atoms": num_atoms,
        # Timing info
        "time_graph_ref": time_graph_ref,
        "time_pred_ref": time_pred_ref,
        "time_total_ref": time_total_ref,
        "time_graph_test": time_graph_test,
        "time_pred_test": time_pred_test,
        "time_total_test": time_total_test,
        "speedup_graph": speedup_graph,
        "speedup_total": speedup_total,
    }


def run_speedup_benchmark(atoms_list, potential=None):
    """Run speedup benchmark across different atom count categories.
    
    Args:
        atoms_list: List of ASE Atoms objects
        potential: MatterSim Potential model (if provided, also benchmarks inference)
    
    Returns:
        dict with benchmark results including graph and total speedups
    """
    # Categorize by number of atoms (using bins)
    bins = defaultdict(list)
    bin_size = 10
    
    for atoms in atoms_list:
        n_atoms = len(atoms)
        bin_idx = (n_atoms // bin_size) * bin_size
        bins[bin_idx].append(atoms)
        
    sorted_bin_keys = sorted(bins.keys())
    
    x_axis = []              # Avg atoms in bin
    speedups_graph = []      # Speedup factor for graph conversion
    speedups_total = []      # Speedup factor for total (graph + inference)
    
    include_inference = potential is not None
    if include_inference:
        logger.info("Starting speedup benchmark (graph conversion + inference)...")
    else:
        logger.info("Starting speedup benchmark (graph conversion only)...")
    
    for bin_start in tqdm(sorted_bin_keys):
        current_atoms = bins[bin_start]
        if not current_atoms:
            continue
            
        avg_atoms = np.mean([len(a) for a in current_atoms])
        
        # Expand list to 100 structures for robust timing
        target_count = 100
        multiplier = (target_count // len(current_atoms)) + 1
        bench_atoms = (current_atoms * multiplier)[:target_count]

        # Benchmark batch_converter=True (New/Fast)
        time_graph_true, time_inf_true = run_benchmark(bench_atoms, batch_converter=True, potential=potential)
        time_total_true = time_graph_true + time_inf_true
        
        # Benchmark batch_converter=False (Old/Slow)
        time_graph_false, time_inf_false = run_benchmark(bench_atoms, batch_converter=False, potential=potential)
        time_total_false = time_graph_false + time_inf_false
        
        # Calculate speedups
        speedup_graph = time_graph_false / time_graph_true if time_graph_true > 0 else 0.0
        speedup_total = time_total_false / time_total_true if time_total_true > 0 else 0.0
            
        x_axis.append(avg_atoms)
        speedups_graph.append(speedup_graph)
        speedups_total.append(speedup_total)
        
        if include_inference:
            logger.info(f"Bin {bin_start}-{bin_start+bin_size} (Avg {avg_atoms:.1f} atoms): "
                        f"Count={len(bench_atoms)}, "
                        f"Graph: {time_graph_false:.4f}s vs {time_graph_true:.4f}s ({speedup_graph:.2f}x), "
                        f"Total: {time_total_false:.4f}s vs {time_total_true:.4f}s ({speedup_total:.2f}x)")
        else:
            logger.info(f"Bin {bin_start}-{bin_start+bin_size} (Avg {avg_atoms:.1f} atoms): "
                        f"Count={len(bench_atoms)}, Time(True)={time_graph_true:.4f}s, "
                        f"Time(False)={time_graph_false:.4f}s, Speedup={speedup_graph:.2f}x")

    return {
        "x_axis": x_axis,
        "speedups_graph": speedups_graph,
        "speedups_total": speedups_total,
        "include_inference": include_inference,
    }


def plot_speedup(benchmark_results, output_path='speedup_plot.png'):
    """Plot speedup results for both graph conversion and total inference."""
    x_axis = benchmark_results["x_axis"]
    speedups_graph = benchmark_results["speedups_graph"]
    speedups_total = benchmark_results["speedups_total"]
    include_inference = benchmark_results["include_inference"]
    
    plt.figure(figsize=(10, 6))
    plt.plot(x_axis, speedups_graph, marker='o', linestyle='-', color='b', label='Graph Conversion Speedup')
    
    if include_inference:
        plt.plot(x_axis, speedups_total, marker='s', linestyle='--', color='r', label='Total (Graph + Inference) Speedup')
    
    plt.title('Speedup of BatchConverter vs Original Implementation', fontsize=16)
    plt.xlabel('Average Number of Atoms', fontsize=14)
    plt.ylabel('Speedup (Time(False) / Time(True))', fontsize=14)
    plt.axhline(y=1.0, color='gray', linestyle=':', alpha=0.7, label='No speedup (1x)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(output_path, dpi=300)
    logger.info(f"Speedup plot saved to {output_path}")
    plt.close()


def plot_energy_parity(error_results, output_path='energy_parity_plot.png'):
    """Plot energy parity (predicted vs reference)."""
    ref = error_results["ref_energies"]
    pred = error_results["pred_energies"]
    
    plt.figure(figsize=(8, 8))
    plt.scatter(ref, pred, alpha=0.6, edgecolors='none', s=20)
    
    # Plot y=x line
    min_val = min(ref.min(), pred.min())
    max_val = max(ref.max(), pred.max())
    margin = (max_val - min_val) * 0.05
    plt.plot([min_val - margin, max_val + margin], 
             [min_val - margin, max_val + margin], 
             'r--', linewidth=2, label='y = x')
    
    plt.xlabel('Reference Energy (batch_converter=False) [eV]', fontsize=14)
    plt.ylabel('Predicted Energy (batch_converter=True) [eV]', fontsize=14)
    plt.title('Energy Parity: BatchConverter vs Original', fontsize=16)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add error metrics as text
    textstr = (f"MAE: {error_results['mae']:.4e} eV\n"
               f"RMSE: {error_results['rmse']:.4e} eV\n"
               f"MAE/atom: {error_results['mae_per_atom']*1000:.4e} meV/atom\n"
               f"Speedup (graph): {error_results['speedup_graph']:.2f}x\n"
               f"Speedup (total): {error_results['speedup_total']:.2f}x")
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=11,
             verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    logger.info(f"Energy parity plot saved to {output_path}")
    plt.close()


def main():
    if not os.path.exists("test_structures.xyz"):
        logger.error("test_structures.xyz not found! Please generate it first.")
        return

    atoms_list = ase_read("test_structures.xyz", ":")
    logger.info(f"Loaded {len(atoms_list)} structures.")

    # =========================================================================
    # Part 1: Energy Error & Speedup Analysis
    # Compare batch_converter=True vs batch_converter=False (reference)
    # =========================================================================
    logger.info("=" * 60)
    logger.info("Part 1: Energy Error & Speedup Analysis")
    logger.info("Reference: batch_converter=False (original implementation)")
    logger.info("Test: batch_converter=True (batch implementation)")
    logger.info("=" * 60)
    
    # Load the potential model
    logger.info("Loading MatterSim potential...")
    potential = Potential.load()
    
    # Compute energy errors and speedup
    error_results = compute_energy_errors_and_speedup(potential, atoms_list)
    
    logger.info("-" * 40)
    logger.info("Energy Error Summary (batch_converter=True vs False):")
    logger.info(f"  Number of structures: {error_results['num_structures']}")
    logger.info(f"  MAE:           {error_results['mae']:.6e} eV")
    logger.info(f"  RMSE:          {error_results['rmse']:.6e} eV")
    logger.info(f"  Max Error:     {error_results['max_error']:.6e} eV")
    logger.info(f"  MAE/atom:      {error_results['mae_per_atom']*1000:.6e} meV/atom")
    logger.info(f"  RMSE/atom:     {error_results['rmse_per_atom']*1000:.6e} meV/atom")
    logger.info(f"  Max Error/atom: {error_results['max_error_per_atom']*1000:.6e} meV/atom")
    logger.info("-" * 40)
    logger.info("Timing Summary:")
    logger.info(f"  Graph conversion (ref):  {error_results['time_graph_ref']:.4f}s")
    logger.info(f"  Graph conversion (test): {error_results['time_graph_test']:.4f}s")
    logger.info(f"  Speedup (graph only):    {error_results['speedup_graph']:.2f}x")
    logger.info(f"  Total time (ref):        {error_results['time_total_ref']:.4f}s")
    logger.info(f"  Total time (test):       {error_results['time_total_test']:.4f}s")
    logger.info(f"  Speedup (total):         {error_results['speedup_total']:.2f}x")
    logger.info("-" * 40)
    
    # Plot energy parity
    plot_energy_parity(error_results)
    
    # =========================================================================
    # Part 2: Speedup Benchmark by Atom Count (Graph + Inference)
    # =========================================================================
    logger.info("=" * 60)
    logger.info("Part 2: Speedup Benchmark by Atom Count (Graph + Inference)")
    logger.info("=" * 60)
    
    benchmark_results = run_speedup_benchmark(atoms_list, potential=potential)
    
    # Plot speedup results
    plot_speedup(benchmark_results)
    
    # Summary statistics
    avg_speedup_graph = np.mean(benchmark_results["speedups_graph"])
    avg_speedup_total = np.mean(benchmark_results["speedups_total"])
    logger.info("-" * 40)
    logger.info("Speedup Summary Across All Bins:")
    logger.info(f"  Average Graph Conversion Speedup: {avg_speedup_graph:.2f}x")
    logger.info(f"  Average Total (Graph + Inference) Speedup: {avg_speedup_total:.2f}x")
    logger.info("-" * 40)
    
    logger.info("Benchmark complete!")


if __name__ == "__main__":
    main()