from setuptools import setup, Extension
import numpy as np
from Cython.Build import cythonize

extensions = [
    Extension(
        "mattersim.datasets.utils.threebody_indices",
        ["src/mattersim/datasets/utils/threebody_indices.pyx"],
        include_dirs=[np.get_include()]
    )
]

setup(
    ext_modules=cythonize(extensions)
)