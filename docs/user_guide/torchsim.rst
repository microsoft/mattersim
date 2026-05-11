Simulations with TorchSim
=========================

`TorchSim <https://github.com/TorchSim/torch-sim>`_ is a GPU-accelerated
molecular simulation engine built on PyTorch. MatterSim provides a
``TorchSimWrapper`` that adapts MatterSim potentials as a TorchSim
``ModelInterface``, so you can use TorchSim's optimizers and integrators
directly with MatterSim models.


Creating a TorchSim wrapper
---------------------------

The easiest way to create a wrapper is via the ``get_torchsim_wrapper``
helper, which accepts a model identifier, a checkpoint path, or an
already-loaded ``Potential`` object.

.. code-block:: python
    :linenos:

    import torch
    from mattersim.torchsim import get_torchsim_wrapper

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # From a model identifier (downloads the checkpoint automatically)
    wrapper = get_torchsim_wrapper(potential="mattersim-v1.0.0-1M", device=device)

    # Or from an already-loaded Potential
    from mattersim.forcefield import Potential

    potential = Potential.from_checkpoint(device=device)
    wrapper = get_torchsim_wrapper(potential=potential, device=device)

You can also construct the wrapper directly if you need full control:

.. code-block:: python
    :linenos:

    from mattersim.torchsim import TorchSimWrapper
    from mattersim.forcefield import Potential

    potential = Potential.from_checkpoint(device=device)
    wrapper = TorchSimWrapper(model=potential, device=device)


Structure relaxation
--------------------

TorchSim provides several optimizers for structure relaxation, including
``fire``, ``lbfgs``, ``bfgs``, and ``gradient_descent``.

.. code-block:: python
    :linenos:

    import numpy as np
    import torch
    import torch_sim as ts
    from ase.build import bulk
    from mattersim.torchsim import get_torchsim_wrapper

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create the wrapper
    wrapper = get_torchsim_wrapper(potential="mattersim-v1.0.0-1M", device=device)

    # Set up the structure
    si = bulk("Si", "diamond", a=5.43)
    si.positions += 0.1 * np.random.randn(len(si), 3)

    # Initialize a TorchSim state from the ASE Atoms object
    state = ts.initialize_state([si], device=device)

    # Run relaxation with the FIRE optimizer
    relaxed_state = ts.optimize(
        system=state,
        model=wrapper,
        optimizer=ts.Optimizer.fire,
        max_steps=500,
        pbar=True,
    )

    # Convert back to ASE Atoms
    relaxed_atoms = relaxed_state.to_atoms()[0]
    print(f"Relaxed energy: {relaxed_state.energy[0].item():.4f} eV")

You can also customize convergence criteria using
``ts.generate_force_convergence_fn``:

.. code-block:: python
    :linenos:

    convergence_fn = ts.generate_force_convergence_fn(force_tol=0.01)

    relaxed_state = ts.optimize(
        system=state,
        model=wrapper,
        optimizer=ts.Optimizer.fire,
        max_steps=500,
        convergence_fn=convergence_fn,
        pbar=True,
    )


Molecular dynamics
------------------

TorchSim supports various integrators for molecular dynamics simulations.
Below we demonstrate NVT dynamics using a Langevin thermostat.

.. code-block:: python
    :linenos:

    import torch
    import torch_sim as ts
    from ase.build import bulk
    from mattersim.torchsim import get_torchsim_wrapper

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create the wrapper
    wrapper = get_torchsim_wrapper(potential="mattersim-v1.0.0-1M", device=device)

    # Set up the structure
    si = bulk("Si", "diamond", a=5.43)
    state = ts.initialize_state([si], device=device)

    # Run NVT MD at 300 K for 1000 steps
    final_state = ts.integrate(
        system=state,
        model=wrapper,
        integrator=ts.Integrator.nvt_langevin,
        n_steps=1000,
        temperature=300.0,
        timestep=1e-3,
        pbar=True,
    )

    # Convert back to ASE Atoms
    final_atoms = final_state.to_atoms()[0]

Available integrators include:

- **NVE**: ``ts.Integrator.nve`` — microcanonical ensemble
- **NVT Langevin**: ``ts.Integrator.nvt_langevin`` — Langevin thermostat
- **NVT Nosé–Hoover**: ``ts.Integrator.nvt_nose_hoover`` — Nosé–Hoover thermostat
- **NPT Langevin**: ``ts.Integrator.npt_langevin_isotropic`` — Langevin barostat (isotropic)
- **NPT Nosé–Hoover**: ``ts.Integrator.npt_nose_hoover_isotropic`` — Nosé–Hoover barostat (isotropic)


Saving trajectories
-------------------

TorchSim can save trajectory frames to HDF5 files during the simulation
using a ``TrajectoryReporter``.

.. code-block:: python
    :linenos:

    import torch
    import torch_sim as ts
    from ase.build import bulk
    from mattersim.torchsim import get_torchsim_wrapper

    device = "cuda" if torch.cuda.is_available() else "cpu"
    wrapper = get_torchsim_wrapper(potential="mattersim-v1.0.0-1M", device=device)

    si = bulk("Si", "diamond", a=5.43)
    state = ts.initialize_state([si], device=device)

    # Configure a trajectory reporter to save every 100 steps
    reporter = ts.TrajectoryReporter(
        filenames=["md_trajectory.h5md"],
        state_frequency=100,
        state_kwargs=dict(save_velocities=True, save_forces=True),
    )

    final_state = ts.integrate(
        system=state,
        model=wrapper,
        integrator=ts.Integrator.nvt_langevin,
        n_steps=1000,
        temperature=300.0,
        timestep=1e-3,
        trajectory_reporter=reporter,
        pbar=True,
    )


.. note::

    For more details on TorchSim's API, including autobatching, advanced
    trajectory handling, and custom integrators, please refer to the
    `TorchSim documentation <https://torch-sim.readthedocs.io/>`_.
