Structure Optimization
======================

This is a simple example of how to perform a structure optimization using the MatterSim.

Import the necessary modules
----------------------------

.. code-block:: python

    import numpy as np
    from ase.build import bulk
    from ase.units import GPa
    from mattersim.forcefield.potential import MatterSimCalculator
    from mattersim.applications.relax import Relaxer

Set up the structure to relax
-----------------------------

.. code-block:: python

    # initialize the structure of silicon
    si = bulk("Si", "diamond", a=5.43)

    # perturb the structure
    si.positions += 0.1 * np.random.randn(len(si), 3)

    # attach the calculator to the atoms object
    si.calc = MatterSimCalculator()

Run the relaxation
--------------------

MatterSim implements a built-in relaxation class to support the relaxation of ase atoms.

.. code-block:: python

    # initialize the relaxation object
    relaxer = Relaxer(
        optimizer="BFGS", # the optimization method
        filter="ExpCellFilter", # filter to apply to the cell
        constrain_symmetry=True, # whether to constrain the symmetry
    )

    relaxed_structure = relaxer.relax(si, steps=500)
