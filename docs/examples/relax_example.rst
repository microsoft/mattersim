Structure Optimization
======================

This is a simple example of how to perform a structure optimization using the MatterSim.

Import the necessary modules
----------------------------

.. code-block:: python

    import numpy as np
    from ase.build import bulk
    from mattersim.forcefield.potential import Potential
    from mattersim.forcefield.potential import DeepCalculator

Set up the structure to relax
-----------------------------

.. code-block:: python

    # initialize the structure of silicon
    si = bulk("Si", "diamond", a=5.43)

    # perturb the structure
    si.positions += 0.1 * np.random.randn(len(si), 3)

    # load the model
    potential = Potential.load(load_path="/path/to/checkpoint", device="cuda:0")

    # create a calculator from the model
    calculator = DeepCalculator(potential=potential)

    # attach the calculator to the atoms object
    si.calc = calculator

Create the optimizer
--------------------

MatterSim implements a built-in relaxation class to support the relaxation of ase atoms.
