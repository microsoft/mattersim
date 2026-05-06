LAMMPS Integration
==================

MatterSim can be used as an interatomic potential inside
`LAMMPS <https://www.lammps.org/>`_ via the ``ML-IAP`` interface.
This enables large-scale molecular dynamics with MPI domain decomposition
and multi-GPU support.

Prerequisites
-------------

- LAMMPS built with ``ML-IAP`` (``MLIAP_ENABLE_PYTHON``), ``PKG_PYTHON``,
  and Kokkos CUDA support
- Python packages: ``mattersim``, ``lammps``, ``torch``

Quick Start
-----------

**1. Export the model**

.. code-block:: python

    from mattersim.lammps.mliap_wrapper import MatterSimMLIAP

    # Choose your model: "mattersim-v1.0.0-1M" or "mattersim-v1.0.0-5M"
    mliap = MatterSimMLIAP.from_checkpoint("mattersim-v1.0.0-1M", device="cpu")
    mliap.save("mattersim-v1.0.0-1M-mliap.pt")

The checkpoint is downloaded automatically on first use.

**2. Run LAMMPS**

Use ``pair_style mliap unified`` with the exported model in your input file:

.. code-block:: lammps

    pair_style      mliap unified mattersim-v1.0.0-1M-mliap.pt
    pair_coeff      * * Cu

.. code-block:: bash

    lmp -in input.in -k on g 1 -sf kk -pk kokkos newton on neigh half

**3. Validate (optional)**

Run the included example and verify LAMMPS output matches Python recomputation:

.. code-block:: bash

    cd src/mattersim/lammps/examples
    lmp -in cu_nve.in -var MODEL_PATH /path/to/mattersim_mliap.pt \
        -k on g 1 -sf kk -pk kokkos newton on neigh half
    python validate.py --version mattersim-v1.0.0-1M --plot

Expected errors: ~1e-6 eV/atom, ~1e-6 eV/Å, ~1e-6 GPa.

Multi-GPU
---------

Multi-GPU uses MPI domain decomposition. Each M3GNet layer syncs ghost atom
embeddings across ranks via LAMMPS exchange. Requires a CUDA-aware MPI stack.

.. code-block:: bash

    mpirun -np 4 lmp -in input.in \
        -k on g 4 -sf kk \
        -pk kokkos newton on neigh half gpu/aware on \
        comm/pair/forward device comm/pair/reverse device

.. note::

    The exact MPI flags may vary depending on your MPI implementation and
    GPU-aware transport layer (e.g. UCX, libfabric). The above works with
    OpenMPI + Kokkos.

How It Works
------------

1. **Graph over all atoms:** The graph is built over ``ntotal`` atoms
   (local + ghost) using the LAMMPS neighbor list directly.

2. **The pbc_offsets trick:** With ``pos=0``, ``cell=I``,
   ``pbc_offsets=rij``, M3GNet computes ``edge_vector = -rij``.
   Differentiating energy w.r.t. ``pbc_offsets`` yields per-edge pair
   forces for LAMMPS.

3. **Inter-layer exchange:** In multi-GPU mode, ``LammpsExchange`` syncs
   ghost atom embeddings via ``forward_exchange`` in the forward pass and
   accumulates gradients via ``reverse_exchange`` in the backward pass.

Limitations
-----------

- **CUDA-aware MPI required for multi-GPU.**
- **mattersim must be installed.** The ``.pt`` file contains pickled Python
  objects that require the ``mattersim`` package.

Docker Image
------------

A Dockerfile is provided in ``dockerfiles/lammps.Dockerfile`` that builds
LAMMPS with Kokkos GPU support, CUDA-aware MPI (UCX + OpenMPI), and
MatterSim pre-installed. This is the easiest way to get started.

**Build the image**

.. code-block:: bash

    docker build -t mattersim-lammps \
        --build-arg KOKKOS_ARCH=AMPERE80 \
        -f dockerfiles/lammps.Dockerfile .

Set ``KOKKOS_ARCH`` to match your GPU:

===========  ============================
Arch         GPUs
===========  ============================
VOLTA70      V100
AMPERE80     A100
AMPERE86     RTX A6000, A5000, A4000
HOPPER90     H100, H200
===========  ============================

**Try the example**

The image is self-contained — no mounting required. To persist output files
or use your own inputs, add ``-v $(pwd):/work -w /work`` to the
``docker run`` command.

.. code-block:: bash

    docker run --gpus all -it mattersim-lammps

    # Inside the container:
    mkdir /tmp/example && cd /tmp/example

    # Export the model and copy the example input (one-time):
    python -c "
    from mattersim.lammps.mliap_wrapper import MatterSimMLIAP
    mliap = MatterSimMLIAP.from_checkpoint('mattersim-v1.0.0-5M')
    mliap.save('model.pt')
    import shutil, mattersim.lammps.examples as e, os
    shutil.copy(os.path.join(os.path.dirname(e.__file__), 'cu_nve.in'), '.')
    "

    # Run (single GPU):
    lmp -in cu_nve.in -k on g 1 -sf kk \
        -pk kokkos newton on neigh half \
        -var MODEL_PATH model.pt

**Run your own simulations**

Mount your input files and run:

.. code-block:: bash

    docker run --gpus all -it -v $(pwd):/work -w /work mattersim-lammps

    # Single GPU:
    lmp -in input.in -k on g 1 -sf kk \
        -pk kokkos newton on neigh half \
        -var MODEL_PATH /path/to/mattersim-v1.0.0-5M-mliap.pt

    # Multi-GPU (4 GPUs):
    mpirun -np 4 --mca pml ucx --mca osc ucx \
        lmp -in input.in \
        -k on g 4 -sf kk \
        -pk kokkos newton on neigh half gpu/aware on \
        comm/pair/forward device comm/pair/reverse device \
        -var MODEL_PATH /path/to/mattersim-v1.0.0-5M-mliap.pt

.. note::

    The ``--mca pml ucx --mca osc ucx`` flags select UCX as the MPI
    transport for GPU-aware communication. These are included in the
    Docker image by default but may not be needed on all systems — omit
    them if your MPI stack already uses UCX or another CUDA-aware transport.
