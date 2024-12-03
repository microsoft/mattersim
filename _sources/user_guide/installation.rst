
Installation
============

Install from PyPI
-----------------

To install the package, run the following command:

.. code-block:: console

    pip install mattersim

In case you want to install the package with the latest version, you can run the following command:

.. code-block:: console

    pip install git+https://github.com/microsoft/mattersim.git

Install from source code
------------------------



To install the package, run the following command under the root of the folder:

.. code-block:: console

    conda env create -f environment.yaml
    conda activate mattersim
    pip install -e .
    python setup.py build_ext --inplace


Please note that the installation process may take a while due to the installation of the dependencies.
For faster installation, we recommend the users to install with
`mamba or micromamba <https://mamba.readthedocs.io/en/latest/index.html>`_,
and the `uv <https://docs.astral.sh/uv/>`_ package manager.

.. code-block:: console

    mamba env create -f environment.yaml
    mamba activate mattersim
    uv pip install -e .
    python setup.py build_ext --inplace

Model checkpoints
----------------------------

The currently available model checkpoints can be found in the `MatterSim GitHub repository <https://github.com/microsoft/mattersim/tree/main/src/mattersim/pretrained_models>`_.
