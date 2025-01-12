Phonon Dispersion
=================

This is a simple example of how to compute the phonon dispersion using the MatterSim.

Import the necessary modules
----------------------------

First we import the necessary modules. It is worth noting
that we have a built-in workflow for phonon calculations using ``phonopy`` in MatterSim.

.. code-block:: python
    :linenos:

    import numpy as np
    from ase.build import bulk
    from ase.units import GPa
    from ase.visualize import view
    from mattersim.forcefield.potential import MatterSimCalculator
    from mattersim.applications.phonon import PhononWorkflow

Set up the MatterSim calculator
-------------------------------

.. code-block:: python
    :linenos:

    # initialize the structure of silicon
    si = bulk("Si")

    # attach the calculator to the atoms object
    si.calc = MatterSimCalculator()

Set up the phonon workflow
--------------------------

.. code-block:: python
    :linenos:

    ph = PhononWorkflow(
        atoms=si,
        find_prim = False,
        work_dir = "/tmp/phonon_si_example",
        amplitude = 0.01,
        supercell_matrix = np.diag([4,4,4]),
    )

Compute the phonon dispersion
-----------------------------

.. code-block:: python
    :linenos:

    has_imag, phonons = ph.run()
    print(f"Has imaginary phonon: {has_imag}")
    print(f"Phonon frequencies: {phonons}")

Inspect the phonon dispersion
-----------------------------

Once the calculation is done, you can find the phonon plot in the work directory.
In this case, you find the plot in the directory ``/tmp/phonon_si_example``,
and here is the phonon dispersion plot for the example above.

.. image:: /_static/phonon_dispersion.png
    :width: 400
    :align: center
    :alt: Phonon dispersion of silicon
