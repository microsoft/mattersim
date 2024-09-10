# -*- coding: utf-8 -*-
import numpy as np
from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext as _build_ext


class build_ext(_build_ext):
    def finalize_options(self):
        _build_ext.finalize_options(self)
        if self.distribution.ext_modules:
            import pyximport

            pyximport.install()


setup(
    name="mattersim",
    version="1.0",
    description=(
        "MatterSim: MatterSim: A Deep Learning Atomistic Model Across "
        "Elements, Temperatures and Pressures."
    ),
    author=(
        "Han Yang, Chenxi Hu, Yichi Zhou, Xixian Liu, Yu Shi, Jielan Li, "
        "Guanzhi Li, Zekun Chen, Shuizhou Chen, Claudio Zeni, Matthew Horton, "
        "Robert Pinsler, Andrew Fowler, Daniel Zügner, Tian Xie, Jake Smith, "
        "Lixin Sun, Qian Wang, Lingyu Kong, Chang Liu, Hongxia Hao, Ziheng Lu"
    ),
    author_email=(
        "hanyang@microsoft.com; jielanli@microsoft.com; "
        "hongxiahao@microsoft.com; zihenglu@microsoft.com;"
    ),
    python_requires=">=3.9",
    packages=find_packages(where="src", include=["mattersim", "mattersim.*"]),
    package_dir={"": "src"},
    url="https://github.com/msr-ai4science/mattersim",
    requires=[
        "torch",
    ],
    install_requires=[
        "ase>=3.23.0",
        "Cython>=0.29.32",
        "e3nn==0.5.0",
        "numpy<2",
        "pymatgen",
        "torch_geometric==2.0.4",
        "torch_runstats==0.2.0",
        "torchmetrics>=0.10.0",
        "torch-ema==0.3",
        "opt_einsum_fx",
        "pre-commit",
        "Pathlib",
        "pytest",
        "pytest-testmon",
        "azure-storage-blob",
        "azure-identity",
    ],
    setup_requires=[
        "Cython>=0.29.32",
    ],
    cmdclass={"build_ext": build_ext},
    ext_modules=[
        Extension(
            "mattersim.datasets.utils.threebody_indices",
            ["src/mattersim/datasets/utils/threebody_indices.pyx"],
            include_dirs=[np.get_include()],
        )
    ],
)
