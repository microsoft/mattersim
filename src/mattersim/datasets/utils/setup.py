# -*- coding: utf-8 -*-
from distutils.core import Extension, setup

import numpy
from Cython.Build import cythonize

package = Extension(
    "threebody_indices",
    ["threebody_indices.pyx"],
    include_dirs=[numpy.get_include()],  # noqa: E501
)
setup(ext_modules=cythonize([package]))

# usage:
# python setup.py build_ext --inplace
