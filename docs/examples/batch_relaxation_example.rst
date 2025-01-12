Batch Structure Optimization
================

This is a simple example of how to use MatterSim to efficiently relax a list of structures.


Import the necessary modules
----------------------------

First we import the necessary modules.

.. code-block:: python
    :linenos:

    from ase.build import bulk
    from mattersim.applications.batch_relax import BatchRelaxer
    from mattersim.forcefield.potential import Potential

Set up the MatterSim batch relaxer
----------------------------------

.. code-block:: python
    :linenos:

    # initialize the default MatterSim Potential
    potential = Potential.from_checkpoint()

    # initialize the batch relaxer with a EXPCELLFILTER for cell relaxation and a FIRE optimizer
    relaxer = BatchRelaxer(potential, fmax=0.01, filter="EXPCELLFILTER", optimizer="FIRE")


Relax the structures
--------------------

.. code-block:: python
    :linenos:

    # Here, we generate a list of ASE Atoms objects we want to relax
    atoms = [bulk("C"), bulk("Mg"), bulk("Si"), bulk("Ni")]

    # And then perturb them a bit so that relaxation is not trivial
    for atom in atoms:
        atom.rattle(stdev=0.1)

    # Run the relaxation
    relaxation_trajectories = relaxer.relax(atoms)


Inspect the relaxed structures
------------------------------

.. code-block:: python
    :linenos:
    
    # Extract the relaxed structures and corresponding energies
    relaxed_structures = [traj[-1] for traj in relaxation_trajectories.values()]
    relaxed_energies = [structure.info['total_energy'] for structure in relaxed_structures]

    # Do the same with the initial structures and energies
    initial_structures = [traj[0] for traj in relaxation_trajectories.values()]
    initial_energies = [structure.info['total_energy'] for structure in initial_structures]

    # verify by inspection that total energy has decreased in all instances
    for initial_energy, relaxed_energy in zip(initial_energies, relaxed_energies):
        print(f"Initial energy: {initial_energy} eV, relaxed energy: {relaxed_energy} eV")
