<h1>
<p align="center">
    <img src="https://github.com/microsoft/mattersim/blob/main/docs/_static/mattersim-banner.png?raw=true" alt="MatterSim logo" width="600"/>
</p>
</h1>

<!-- <h1 align="center">MatterSim</h1> -->

<h4 align="center">

[![arXiv](https://img.shields.io/badge/arXiv-2405.04967-blue?logo=arxiv&logoColor=white.svg)](https://arxiv.org/abs/2405.04967)
[![Requires Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg?logo=python&logoColor=white)](https://python.org/downloads)
[![PyPI Downloads](https://static.pepy.tech/badge/mattersim)](https://pepy.tech/projects/mattersim)
</h4>


MatterSim is a deep learning atomistic model across elements, temperatures and pressures.

## Documentation

This README provides a quick start guide. For more comprehensive information, please refer to the [MatterSim documentation](https://microsoft.github.io/mattersim/).

## Installation

### Prerequisite
* `Python >= 3.10`


### Install from PyPI
> [!TIP]
> While not mandatory, we recommend creating a clean conda environment before installing MatterSim to avoid potential package conflicts. You can create and activate a conda environment with the following commands:
>
> ```bash
> # create the environment
> conda create -n mattersim python=3.10
>
> # activate the environment
> conda activate mattersim
> ```
>

To install MatterSim, use the following command. Please note that downloading the dependencies may take some time:
```bash
pip install mattersim
```

In case you want to install the package with the latest version, you can run the following command:

```bash
pip install git+https://github.com/microsoft/mattersim.git
```

### Install from source code
1. Download the source code of MatterSim and change to the directory

```bash
git clone git@github.com:microsoft/mattersim.git
cd mattersim
```

2. Install MatterSim

> [!WARNING]
> We strongly recommend that users install MatterSim using [mamba or micromamba](https://mamba.readthedocs.io/en/latest/index.html), because *conda* can be significantly slower when resolving the dependencies in environment.yaml.

To install the package, run the following command under the root of the folder:

```bash
mamba env create -f environment.yaml
mamba activate mattersim
uv pip install -e .
```

## Pre-trained Models

We currently offer two pre-trained **MatterSim-v1** models based on the **M3GNet** architecture in the [pretrained_models](./pretrained_models/) folder:

1. **MatterSim-v1.0.0-1M**: A mini version of the model that is faster to run.
2. **MatterSim-v1.0.0-5M**: A larger version of the model that is more accurate.

These models have been trained using the data generated through the workflows
introduced in the [MatterSim manuscript](https://arxiv.org/abs/2405.04967), which provides an in-depth
explanation of the methodologies underlying the MatterSim model.

More advanced and fully-supported pretrained versions of MatterSim,
and additional materials capabilities are available in
**[Azure Quantum Elements](https://quantum.microsoft.com/en-us/solutions/azure-quantum-elements)**.

## Usage

> [!TIP]
> **Note for macOS Users:** If you are using macOS with Apple Silicon, please be aware of potential numerical instability with the MPS backend. We recommend using the CPU device for MatterSim on Mac to avoid these issues.

### A minimal test
```python
import torch
from loguru import logger
from ase.build import bulk
from ase.units import GPa
from mattersim.forcefield import MatterSimCalculator

device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Running MatterSim on {device}")

si = bulk("Si", "diamond", a=5.43)
si.calc = MatterSimCalculator(device=device)
logger.info(f"Energy (eV)                 = {si.get_potential_energy()}")
logger.info(f"Energy per atom (eV/atom)   = {si.get_potential_energy()/len(si)}")
logger.info(f"Forces of first atom (eV/A) = {si.get_forces()[0]}")
logger.info(f"Stress[0][0] (eV/A^3)       = {si.get_stress(voigt=False)[0][0]}")
logger.info(f"Stress[0][0] (GPa)          = {si.get_stress(voigt=False)[0][0] / GPa}")
```

In this release, we provide two checkpoints: `MatterSim-v1.0.0-1M.pth` and `MatterSim-v1.0.0-5M.pth`. By default, the `1M` version is loaded.
To switch to the `5M` version, manually set the `load_path` of `MatterSimCalculator` as shown below:

```python
MatterSimCalculator(load_path="MatterSim-v1.0.0-5M.pth", device=device)
```

## Finetune
> [!TIP]
> MatterSim provides a finetune script to finetune the pre-trained MatterSim model on a custom dataset.
> Please refer to the [MatterSim documentation](https://microsoft.github.io/mattersim/) for more details.

### A minimal finetune example

```bash
torchrun --nproc_per_node=1 src/mattersim/training/finetune_mattersim.py --load_model_path mattersim-v1.0.0-1m --train_data_path tests/data/high_level_water.xyz
```




## Reference

We kindly request that users of MatterSim version 1.0.0 cite our preprint available on arXiv:
```
@article{yang2024mattersim,
      title={MatterSim: A Deep Learning Atomistic Model Across Elements, Temperatures and Pressures},
      author={Han Yang and Chenxi Hu and Yichi Zhou and Xixian Liu and Yu Shi and Jielan Li and Guanzhi Li and Zekun Chen and Shuizhou Chen and Claudio Zeni and Matthew Horton and Robert Pinsler and Andrew Fowler and Daniel Zügner and Tian Xie and Jake Smith and Lixin Sun and Qian Wang and Lingyu Kong and Chang Liu and Hongxia Hao and Ziheng Lu},
      year={2024},
      eprint={2405.04967},
      archivePrefix={arXiv},
      primaryClass={cond-mat.mtrl-sci},
      url={https://arxiv.org/abs/2405.04967},
      journal={arXiv preprint arXiv:2405.04967}
}
```

> [!IMPORTANT]
> We kindly ask users to **explicitly** specify the exact model version and checkpoint (e.g., **MatterSim-v1.0.0-1M**) when reporting results in academic papers or technical reports, rather than referring to the model generically as **MatterSim**. Precise versioning is crucial for ensuring reproducibility. For instance, the statement "_This study was conducted using MatterSim-v1.0.0-1M_" serves as a good example.

## Limitations

**MatterSim-v1** is designed specifically for atomistic simulations of bulk materials. Applications or interpretations beyond this scope should be approached with caution. For instance, when using the model for simulations involving surfaces, interfaces, or properties influenced by long-range interactions, the results may be qualitatively accurate but are not suitable for quantitative analysis. In such cases, we recommend fine-tuning the model to better align with the specific application.

## Trademarks

This project may contain trademarks or logos for projects, products, or services.
Authorized use of Microsoft trademarks or logos is subject to and must follow [Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.

## Responsible AI Transparency Documentation

The responsible AI transparency documentation can be found [here](MODEL_CARD.md).


## Researcher and Developers
MatterSim is actively under development, and we welcome community engagement. If you have research interests related to this model, ideas you’d like to contribute, or issues to report, we encourage you to reach out to us at [ai4s-materials@microsoft.com](mailto:ai4s-materials@microsoft.com).
