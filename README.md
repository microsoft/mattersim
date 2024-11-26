<h1>
<p align="center">
    <img src="docs/_static/mattersim-banner.jpg" alt="MatterSim logo" width="600"/>
</p>
</h1>

<!-- <h1 align="center">MatterSim</h1> -->

<h4 align="center">

[![arXiv](https://img.shields.io/badge/arXiv-2405.04967-blue?logo=arxiv&logoColor=white.svg)](https://arxiv.org/abs/2405.04967)
[![Requires Python 3.9+](https://img.shields.io/badge/Python-3.9+-blue.svg?logo=python&logoColor=white)](https://python.org/downloads)

</h4>


MatterSim is a deep learning atomistic model across elements, temperatures and pressures.

## Installation
### Install from source code
Requirements:
- Python == 3.9

To install the package, run the following command under the root of the folder:
```bash
conda env create -f environment.yaml
conda activate mattersim
pip install -e .
python setup.py build_ext --inplace
```

## Usage
### A minimal test
```python
from mattersim.forcefield.potential import Potential
from mattersim.datasets.utils.build import build_dataloader

potential = Potential.load(load_path="/path/to/checkpoint", device="cuda:0")
from ase.build import bulk
si = bulk("Si", "diamond", a=5.43)
dataloader = build_dataloader([si], only_inference=True, model_type=model_name)
predictions = potential.predict_properties(dataloader, include_forces=True, include_stresses=True)
print(predictions)
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
> We kindly ask users to **explicitly** specify the exact model version and checkpoint (e.g., **MatterSim-v1.0.0-1M**) when reporting results in academic papers or technical reports, rather than referring to the model generically as **MatterSim**. Precise versioning is crucial for ensuring reproducibility. For instance, the statement "_This benchmark was conducted using MatterSim-v1.0.0-1M_" serves as a good example.

## Limitations
MatterSim is designed specifically for atomistic simulations of bulk materials. Applications or interpretations beyond this scope should be approached with caution. For instance, when using the model for simulations involving surfaces, interfaces, or properties influenced by long-range interactions, the results may be qualitatively accurate but are not suitable for quantitative analysis. In such cases, we recommend fine-tuning the model to better align with the specific application.

## Trademarks

This project may contain trademarks or logos for projects, products, or services.
Authorized use of Microsoft trademarks or logos is subject to and must follow [Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.

## Responsible AI Transparency Documentation
The responsible AI transparency documentation can be found [here](MODEL_CARD.md).


## Researcher and Developers
MatterSim is actively under development, and we welcome community engagement. If you have research interests related to this model, ideas you’d like to contribute, or issues to report, we encourage you to reach out to us.
