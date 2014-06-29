from setuptools import setup, find_packages
setup (
    name = "ir-reduce",
    packages = find_packages(),

    install_requires = ["astropy", "scipy", "numpy", "pillow", "kivy"], 
)
