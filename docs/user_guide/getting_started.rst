Getting Started
===============

A minimal example using ASE calculator
--------------------------------------

MatterSim provides an interface to the Atomic Simulation Environment (ASE) to
facilitate the use of MatterSim potentials in the popular ASE package.

.. code-block:: python
    :linenos:

    import torch
    from ase.build import bulk
    from ase.units import GPa
    from mattersim.forcefield import MatterSimCalculator

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running MatterSim on {device}")

    si = bulk("Si", "diamond", a=5.43)
    si.calc = MatterSimCalculator(device=device)
    print(f"Energy (eV)                 = {si.get_potential_energy()}")
    print(f"Energy per atom (eV/atom)   = {si.get_potential_energy()/len(si)}")
    print(f"Forces of first atom (eV/A) = {si.get_forces()[0]}")
    print(f"Stress[0][0] (eV/A^3)       = {si.get_stress(voigt=False)[0][0]}")
    print(f"Stress[0][0] (GPa)          = {si.get_stress(voigt=False)[0][0] / GPa}")


In the example above, the ``MatterSimCalculator`` class implements the ASE calculator interface.
However, with ``MatterSimCalculator``, one can only predict the properties of a single structure at a time,
which is not efficient for large-scale calculations to effectively utilize the GPU.
Thus, we also provide a more efficient way to predict the properties of multiple structures using the ``Potential`` class.

Batch prediction using the ``Potential`` class
----------------------------------------------

The ``Potential`` class provides a more efficient way to predict the properties of
multiple structures using the ``predict_properties`` method.
In the following example, we demonstrate how to predict the properties of
a list of structures using the ``Potential`` class.

.. code-block:: python
    :linenos:

    import numpy as np
    from ase.build import bulk
    from ase.units import GPa
    from mattersim.forcefield.potential import Potential
    from mattersim.datasets.utils.build import build_dataloader

    # set up the structure
    si = bulk("Si", "diamond", a=5.43)

    # replicate the structures to form a list
    structures = [si] * 10

    # load the model
    potential = Potential.load(load_path="/path/to/checkpoint", device="cuda:0")

    # build the dataloader that is compatible with MatterSim
    dataloader = build_dataloader(structures, only_inference=True)

    # make predictions
    predictions = potential.predict_properties(dataloader, include_forces=True, include_stresses=True)

    # print the predictions
    print(f"Total energy in eV: {predictions[0]}")
    print(f"Forces in eV/Angstrom: {predictions[1]}")
    print(f"Stresses in GPa: {predictions[2]}")
    print(f"Stresses in eV/A^3: {np.array(predictions[2])*GPa}")

.. warning ::
    By default, MatterSim ``potential.predict_properties`` predicts stress tensors in GPa.
    To convert the stress tensor to :math:`\mathrm{eV}\cdot\mathrm{\mathring{A}}^{-3}`,
    multiply the stress tensor by the conversion factor ``GPa``.
