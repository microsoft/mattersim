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

MatterSim is still in active development, and newer models may be released
in appropriate time. Please stay tuned for updates.

Pre-trained Models
------------------

We currently offer two pre-trained **MatterSim-v1** models based on the **M3GNet** architecture

1. **MatterSim-v1.0.0-1M**: A mini version of the model that is faster to run.
2. **MatterSim-v1.0.0-5M**: A larger version of the model that is more accurate.

These models have been trained using the data generated through the workflows
introduced in the `MatterSim <https://arxiv.org/abs/2405.04967>`_ manuscript, which provides an in-depth
explanation of the methodologies underlying the MatterSim model.

More advanced and fully-supported pretrained versions of MatterSim,
and additional materials capabilities are available in
`Azure Quantum Elements <https://quantum.microsoft.com/en-us/solutions/azure-quantum-elements>`_.

Bibliography
------------

.. note::

   If you use MatterSim in your research, please cite the following paper:

   .. code-block:: bibtex

      @article{yang2024mattersim,
         title={MatterSim: A Deep Learning Atomistic Model Across Elements, Temperatures and Pressures},
         author={Han Yang and Chenxi Hu and Yichi Zhou and Xixian Liu and Yu Shi and Jielan Li and Guanzhi Li and Zekun Chen and Shuizhou Chen and Claudio Zeni and Matthew Horton and Robert Pinsler and Andrew Fowler and Daniel Zügner and Tian Xie and Jake Smith and Lixin Sun and Qian Wang and Lingyu Kong and Chang Liu and Hongxia Hao and Ziheng Lu},
         year={2024},
         eprint={2405.04967},
         archivePrefix={arXiv},
         primaryClass={cond-mat.mtrl-sci},
         url={https://arxiv.org/abs/2405.04967},
         journal={arXiv preprint arXiv:2405.04967
      }


Frequently Asked Questions
--------------------------

**Q1**: I have identified a system where MatterSim produces inaccurate or unexpected results. How can I report this?

   **A**: We welcome and appreciate detailed bug reports to help improve MatterSim. Please raise an issue on our GitHub repository: https://github.com/microsoft/mattersim/issues.

**Q2**: I have specific need for some features that are not currently supported by MatterSim. How can I request these features?

   **A**: We are always looking to improve MatterSim and welcome feature requests. Please suggest a feature on our GitHub repository: https://github.com/microsoft/mattersim/issues.

**Q3**: Do you have any plans to release more pre-trained models in the future?

   **A**: Yes, we are actively working on developing more pre-trained models for MatterSim. Please stay tuned for updates.

**Q4**: How can I contribute to the development of MatterSim?

   **A**: We warmly welcome contributions! Please help improve MatterSim by reporting bugs, suggesting features, or submitting pull requests to our GitHub repository: https://github.com/microsoft/mattersim/pulls.


.. toctree::
   :maxdepth: 2
   :caption: User Guide:
   :hidden:

   user_guide/installation
   user_guide/getting_started
   examples/examples
