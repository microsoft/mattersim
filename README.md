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
If you use MatterSim, please cite our preprint on [arXiv](https://arxiv.org/abs/2405.04967):
```
@article{yang2024mattersim,
  title={Mattersim: A deep learning atomistic model across elements, temperatures and pressures},
  author={Yang, Han and Hu, Chenxi and Zhou, Yichi and Liu, Xixian and Shi, Yu and Li, Jielan and Li, Guanzhi and Chen, Zekun and Chen, Shuizhou and Zeni, Claudio and others},
  journal={arXiv preprint arXiv:2405.04967},
  year={2024}
}
```

## Trademarks

This project may contain trademarks or logos for projects, products, or services.
Authorized use of Microsoft trademarks or logos is subject to and must follow [Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.

## Responsible AI Transparency Documentation
The responsible AI transparency documentation can be found [here](MODEL_CARD.md).


## Researcher and Developers
MatterSim is currently in active development. If you have any specific research interests related to this model or encounter any issues, please don't hesitate to reach out to us.
