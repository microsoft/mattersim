.. mattersim documentation master file, created by
   sphinx-quickstart on Thu Aug 22 14:01:40 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to the MatterSim Documentation!
=======================================

Overview
--------

`MatterSim <https://arxiv.org/abs/2405.04967>`_ is an advanced deep learning model designed to simulate 
the properties of materials across a wide range of elements, temperatures, and pressures. 
The model leverages state-of-the-art deep learning techniques to deliver high accuracy and 
efficiency in atomistic simulations, making it a valuable tool for researchers 
in the field of materials science.

MatterSim is still in active development, and more checkpoints may be
released in appropriate time, so please stay tuned for updates.

Pre-trained Models
------------------

We currently offer two pre-trained versions of MatterSim with **M3GNet** architecture:

1. **mattersim-mini-v1.0.0**: A mini version of the model that is faster to run. 
2. **mattersim-medium-v1.0.0**: A medium version of the model that is more accurate.

These models have been trained using the data generated through the workflows 
introduced in the `MatterSim <https://arxiv.org/abs/2405.04967>`_ manuscript, which provides an in-depth 
explanation of the methodologies underlying the MatterSim model.

FAQ
---

**Q1**: What is the difference between the mini and medium versions of MatterSim?

   **A**: The mini version is a smaller model that is faster to run, while the medium version is more accurate.

**Q2**: Are you going to release the pre-trained models of MatterSim with transformer-based architectures?

   **A**: The transformer-based MatterSim is still under development. Please contact us for more information.

Bibliography
------------

.. note::

   If you use MatterSim in your research, please cite the following paper:

   .. code-block:: bibtex

      @article{yang2024mattersim,
         title={
            Mattersim: A deep learning atomistic model across elements,
            temperatures and pressures
         },
         author={
            Yang, Han and Hu, Chenxi and Zhou, Yichi and Liu, Xixian
            and Shi, Yu and Li, Jielan and Li, Guanzhi and Chen, Zekun
            and Chen, Shuizhou and Zeni, Claudio and others
         },
         journal={arXiv preprint arXiv:2405.04967},
         year={2024}
      }



.. toctree::
   :maxdepth: 2
   :caption: User Guide:
   :hidden:

   user_guide/installation
   user_guide/getting_started
   examples/examples
