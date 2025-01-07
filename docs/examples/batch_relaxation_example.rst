Batch relaxation
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

    # Run the relaxation
    relaxation_trajectories = relaxer.relax(atoms)


Inspect the relaxed structures
------------------------------

.. code-block:: python
    :linenos:
    
    # Extract the relaxed relaxed_structures
    relaxed_structures = [traj[-1] for traj in relaxation_trajectories]

    # And the corresponding total energies
    relaxed_energies = [structure.info['total_energy'] for structure in relaxed_structures]