Getting Started
===============

A minimal example
-----------------

The following example demonstrates how to load a pre-trained potential and make predictions for a single structure.

.. code-block:: python
    :linenos:

    from ase.build import bulk
    from mattersim.forcefield.potential import Potential
    from mattersim.datasets.utils.build import build_dataloader

    # set up the structure
    si = bulk("Si", "diamond", a=5.43)

    # load the model
    potential = Potential.load(load_path="/path/to/checkpoint", device="cuda:0")

    # build the dataloader that is compatible with MatterSim
    dataloader = build_dataloader([si], only_inference=True, model_type=model_name)

    # make predictions
    predictions = potential.predict_properties(dataloader, include_forces=True, include_stresses=True)

    # print the predictions
    print(f"Total energy in eV: {predictions[0]}")
    print(f"Forces in eV/Angstrom: {predictions[1]}")
    print(f"Stresses in GPa: {predictions[2]}")


Interface to ASE
----------------

MatterSim provides an interface to the Atomic Simulation Environment (ASE) to facilitate the use of MatterSim potentials in the popular ASE package.

.. code-block:: python
    :linenos:

    from ase.build import bulk
    from ase.units import GPa
    from mattersim.forcefield.potential import DeepCalculator

    # same as before
    si = bulk("Si", "diamond", a=5.43)
    potential = Potential.load(load_path="/path/to/checkpoint", device="cuda:0")

    # set up the calculator
    calculator = DeepCalculator(
        potential=potential,
        # important! convert GPa to eV/Angstrom^3
        stress_weight=GPa,
    )

    si.calc = calculator
    # or
    si.set_calculator(calculator)

    print(si.get_potential_energy())
    print(si.get_forces())
    print(si.get_stress(voigt=False))


In the example above, the `DeepCalculator` class implements the ASE calculator interface. The **stress_weight** parameter is used to convert the stress tensor from GPa to :math:`\mathrm{eV}\cdot\mathrm{\mathring{A}}^{-3}`.

.. warning ::
    By default, the ASE package assumes :math:`\mathrm{eV}\cdot\mathrm{\mathring{A}}^{-3}` for the stress tensor. However, MatterSim uses GPa for the stress tensor. Therefore, the **stress_weight** parameter is necessary to convert the stress tensor from GPa to :math:`\mathrm{eV}\cdot\mathrm{\mathring{A}}^{-3}`.
