---
license: mit
license_link: https://opensource.org/license/mit

arxiv: 2405.04967
language:
- en
tags:
- materials-science
- force-field
- molecular-dynamics
---

# MatterSim

MatterSim is a large-scale pretrained deep learning model for efficient materials emulations and property predictions.

## Model Details

### Model Description

MatterSim is a deep learning model for general materials design tasks. It supports efficient atomistic simulations at first-principles level and accurate prediction of broad material properties across the periodic table, spanning temperatures from 0 to 5000 K and pressures up to 1000 GPa. Out-of-the-box, the model serves as a machine learning force field, and shows remarkable capabilities not only in predicting ground-state material structures and energetics, but also in simulating their behavior under realistic temperatures and pressures. MatterSim also serves as a platform for continuous learning and customization by integrating domain-specific data. The model can be fine-tuned for atomistic simulations at a desired level of theory or for direct structure-to-property predictions with high data efficiency.

Please refer to the [MatterSim](https://arxiv.org/abs/2405.04967) manuscript for more details on the model.

- **Developed by:** Han Yang, Chenxi Hu, Yichi Zhou, Xixian Liu, Yu Shi, Jielan Li, Guanzhi Li, Zekun Chen, Shuizhou Chen, Claudio Zeni, Matthew Horton, Robert Pinsler, Andrew Fowler, Daniel Zügner, Tian Xie, Jake Smith, Lixin Sun, Qian Wang, Lingyu Kong, Chang Liu, Hongxia Hao, Ziheng Lu
- **Funded by:** Microsoft Research AI for Science
- **Model type:** Currently, we only release the models trained with **M3GNet** architecture.
- **License:** MIT License

### Model Sources

- **Repository:** https://github.com/microsoft/mattersim
- **Paper:** https://arxiv.org/abs/2405.04967

### Available Models

|                    | mattersim-v1.0.0-1M   | mattersim-v1.0.0-5M     |
| ------------------ | --------------------- | ----------------------- |
| Training Data Size | 3M                    | 6M                      |
| Model Parameters   | 880K                  | 4.5M                    |


## Uses

The MatterSim model is intended for property predictions of materials.

### Direct Use

The model is used for materials simulation and property prediction tasks. An interface to atomic simulation environment is provided. Examples of direct usages include but not limited to

- Direct prediction of energy, forces and stress of a given materials
- Phonon prediction using finite difference
- Molecular dynamics

## Evaluation

### Testing Data, Factors & Metrics

#### Testing Data

To evaluate the model performance, we created the following test sets

- **MPtrj-random-1k:** 1k structures randomly sampled from MPtrj dataset
- **MPtrj-highest-stress-1k:** 1k structures with highest stress magnitude sampled from MPtrj dataset
- **Alexandria-1k:** 1k structures randomly sampled from Alexandria
- **MPF-Alkali-TP:** For detailed description of the generation of the dataset, please refer to the SI of the [MatterSim manuscript](https://arxiv.org/abs/2405.04967)
- **MPF-TP:** For detailed description of the generation of the dataset, please refer to the SI of the [MatterSim manuscript](https://arxiv.org/abs/2405.04967)
- **Random-TP:** For detailed description of the generation of the dataset, please refer to the SI of the [MatterSim manuscript](https://arxiv.org/abs/2405.04967)

We released the test datasets in pickle files and each of them contains the `ase.Atoms` objects. To access the structures and corresponding labels in the datasets, you do use the following snippet to get started,

```python
import pickle
from ase.units import GPa

atoms_list = pickle.load(open("/path/to/datasets.pkl", "rb"))
atoms = atoms_list[0]

print(f"Energy: {atoms.get_potential_energy()} eV")
print(f"Forces: {atoms.get_forces()} eV/A")
print(f"Stress: {atoms.get_stress(voigt=False)} eV/A^3, or {atoms.get_stress(voigt=False)/GPa}")
```

#### Metrics

We evaluate the performance by computing the mean absolute errors (MAEs) of energy (E), forces (F) and stress (S) of each structures within the same dataset. The MAEs are defined as follows,
<p align="center">
      <img src="https://latex.codecogs.com/svg.latex?\mathrm{MAE}_E=\frac{1}{N}\sum_{i}^N\frac{1}{N_{at}^{(i)}}|E_i-\tilde{E}_i|" alt="MAE_E equation">
</p>
<p align="center">
      <img src="https://latex.codecogs.com/svg.latex?\mathrm{MAE}_F=\frac{1}{N}\sum_i^N\frac{1}{N_{at}^{(i)}}\sum_{j}^{N^{(i)}_{at}}||F_{ij}-\tilde{F}_{ij}||_2," alt="MAE_F equation">
</p>
<p align="center">
      <img src="https://latex.codecogs.com/svg.latex?\mathrm{MAE}_S=\frac{1}{N}\sum_i^{N}||S_{i}-\tilde{S}_{i}||_2," alt="MAE_S equation">
</p>
where N is the number of structures in the same dataset, <img src="https://latex.codecogs.com/svg.image?\inline&space;&space;N_{at}^{(i)}"> is the number of atoms in the i-th structure and E, F and S represent ground-truth energy, forces and stress, respectively.


### Results

| Dataset              | Dataset Size | MAE               | mattersim-v1.0.0-1M | mattersim-v1.0.0-5M |
| -------------------- | ------------ | ----------------- | ------------ | ------------ |
| MPtrj-random-1k      | 1000         | Energy [eV/atom]  | 0.030        | 0.024        |
|                      |              | Forces [eV/<img src="https://latex.codecogs.com/svg.latex?\AA" alt="\AA">] | 0.149        | 0.109        |
|                      |              | Stress [GPa]      | 0.241        | 0.186        |
| MPtrj-high-stress-1k | 1000         | Energy [eV/atom]  | 0.110        | 0.108        |
|                      |              | Forces [eV/<img src="https://latex.codecogs.com/svg.latex?\AA" alt="\AA">] | 0.417        | 0.361        |
|                      |              | Stress [GPa]      | 6.230        | 6.003        |
| Alexandria-1k        | 1000         | Energy [eV/atom]  | 0.058        | 0.016        |
|                      |              | Forces [eV/<img src="https://latex.codecogs.com/svg.latex?\AA" alt="\AA">] | 0.086        | 0.042        |
|                      |              | Stress [GPa]      | 0.761        | 0.205        |
| MPF-Alkali-TP        | 460          | Energy [eV/atom]  | 0.024        | 0.021        |
|                      |              | Forces [eV/<img src="https://latex.codecogs.com/svg.latex?\AA" alt="\AA">] | 0.331        | 0.293        |
|                      |              | Stress [GPa]      | 0.845        | 0.714        |
| MPF-TP               | 1069         | Energy [eV/atom]  | 0.029        | 0.026        |
|                      |              | Forces [eV/<img src="https://latex.codecogs.com/svg.latex?\AA" alt="\AA">] | 0.418        | 0.364        |
|                      |              | Stress [GPa]      | 1.159        | 1.144        |
| Random-TP            | 693          | Energy [eV/atom]  | 0.208        | 0.199        |
|                      |              | Forces [eV/<img src="https://latex.codecogs.com/svg.latex?\AA" alt="\AA">] | 0.933        | 0.824        |
|                      |              | Stress [GPa]      | 2.065        | 1.999        |

## Technical Specifications [optional]

### Model Architecture and Objective

The checkpoints released in this repository are those trained on an internal implementation of the **M3GNet** architecture.

#### Software

- Python == 3.9

## Citation

**BibTeX:**
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

## Model Card Contact

- Han Yang (hanyang@microsoft.com)
- Ziheng Lu (zihenglu@microsoft.com)




### Out-of-Scope Use

The model only supports atomistic simulations of materials and molecules. Any attempt and interpretation beyond that should be avoided.
The model does not support generation of new materials as it is designed for materials simulation and property prediction only.
The model is intended for research and experimental purposes. Further testing/development are needed before considering its application in real-world scenarios.

## Bias, Risks, and Limitations

The current model has relatively low accuracy for organic polymeric systems.
Accuracy is inferior to the best (more computationally expensive) methods available.
The model is trained on a specific variant of Density Functional Theory (PBE) that has known limitations across chemical space which will affect accuracy of prediction, such as the ability to simulate highly-correlated systems. (The model can be fine-tuned with higher accuracy data.)
The model does not support all capabilities of some of the latest models such as predicting Born effective charges or simulating a material in an applied electric field.
We have evaluated the model on many examples, but there are many examples that are beyond our available resources to test.

### Recommendations

For any applications related simulations of surfaces, interfaces, and systems with long-range interactions, the results are often qualitatively correct. For quantitative results, the model needs to be fine-tuned.
