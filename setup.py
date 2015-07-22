'''from setuptools import setup, find_packages
setup (
    name = "ir-reduce",
    packages = find_packages(),

    install_requires = ["astropy", "scipy", "numpy", "pillow", "kivy"], 
)'''

from distutils.core import setup
from Cython.Build import cythonize
import numpy as np

setup(
    name = "irreduc",
    ext_modules = cythonize(["fits_class.pyx","calib.pyx", "datatypes.pyx"]), include_dirs=[np.get_include()]
)
