import argparse
import torch
import numpy as np
from ase.io import read, write
from ase.calculators.singlepoint import SinglePointCalculator
from mattersim.forcefield.potential import Potential
from mattersim.datasets.utils.build import build_dataloader
import os

def get_properties_batch(atoms_list, model_path, batch_size):
    """
    Predict properties (energies, forces, stresses) for a batch of atoms.
    
    Parameters:
        atoms_list (list): List of ASE Atoms objects.
        model_path (str): Path to the MatterSim model checkpoint.
        batch_size (int): Batch size for inference.

    Returns:
        tuple: energies, forces, stresses
    """
    # Load the MatterSim model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running MatterSim on {device}")
    potential = Potential.from_checkpoint(model_path, device=device)

    # Build the dataloader compatible with MatterSim
    dataloader = build_dataloader(atoms_list, batch_size=batch_size, only_inference=True)

    # Make predictions
    predictions = potential.predict_properties(dataloader, include_forces=True, include_stresses=True)

    return predictions  # Returns tuple (energies, forces, stresses)

def assign_properties_to_atoms(atoms_list, energies, forces, stresses):
    """
    Assign energies, forces, and stresses to ASE Atoms objects.

    Parameters:
        atoms_list (list): List of ASE Atoms objects.
        energies (list): Predicted energies.
        forces (list): Predicted forces.
        stresses (list): Predicted stresses.

    Returns:
        list: Updated ASE Atoms objects with assigned properties.
    """
    for atoms, energy, force, stress in zip(atoms_list, energies, forces, stresses):
        atoms.calc = SinglePointCalculator(atoms, energy=energy, forces=force, stress=stress)
    return atoms_list

def save_atoms_to_xyz(atoms_list, filename):
    """
    Save ASE Atoms objects to an .xyz file.

    Parameters:
        atoms_list (list): List of ASE Atoms objects.
        filename (str): Output file name.
    """
    write(filename, atoms_list)

def main():
    """
    Main function to perform batch inference using MatterSim.
    """
    parser = argparse.ArgumentParser(description="Batch inference using MatterSim.")
    parser.add_argument(
        "--load_path", 
        type=str, 
        default="MatterSim-v1.0.0-1M.pth", 
        help="Path to the MatterSim model checkpoint."
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=64, 
        help="Batch size for inference."
    )
    parser.add_argument(
        "--input_file", 
        type=str, 
        required=True, 
        help="Path to the input .xyz file containing structures."
    )
    parser.add_argument(
        "--output_file", 
        type=str, 
        default=None, 
        help="Path to save the output .xyz file with predicted properties. Defaults to '<input_file_name>_MatterSim_inference.xyz'."
    )
    args = parser.parse_args()

    # Set default output file name if not provided
    if args.output_file is None:
        base_name = os.path.splitext(os.path.basename(args.input_file))[0]
        args.output_file = f"{base_name}_MatterSim_inference.xyz"

    # Read input structures from the .xyz file
    atoms_list = read(args.input_file, index=":")

    # Perform batch inference
    energies, forces, stresses = get_properties_batch(
        atoms_list, model_path=args.load_path, batch_size=args.batch_size
    )

    # Assign predicted properties to atoms
    atoms_with_properties = assign_properties_to_atoms(atoms_list, energies, forces, stresses)

    # Save updated structures to the output .xyz file
    save_atoms_to_xyz(atoms_with_properties, filename=args.output_file)

    print(f"Inference completed. Results saved to {args.output_file}")

if __name__ == "__main__":
    main()
