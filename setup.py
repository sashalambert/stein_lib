import os
import sys
from setuptools import setup, find_packages


if sys.version_info.major != 3:
    print("This Python is only compatible with Python 3, but you are running "
          "Python {}. The installation will likely fail.".format(sys.version_info.major))

setup(
    name='steinlib',
    version='1.0.0',
    packages=find_packages(),
    description='Stein Variational Inference Library',
    url='https://github.com/sashalambert/stein_lib.git',
    author='Sasha Alexander Lambert',
    install_requires=[
        'numpy',
        'torch',
        'pyro-ppl',
        'matplotlib',
    ],
)
