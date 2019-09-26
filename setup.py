# TODO Requirements are not minimal
from setuptools import setup, find_packages

setup(name='py-tmps',
      version='1.0.1',
      description='Implementation of the tmps algorithm for real and imaginary time evolution of quantum states '
                  'represented by mps, mpo or pmps for chain and star geometries with a focus on impurity models. '
                  'For the chain geometry, the algorithm described by '
                  'Schollwoeck in Annals of Physics 326 (2011), 96-192; doi: 10.1016/j.aop.2010.09.012 is used. '
                  'For the star geometry, the algorithm is described by Hans Gerd Evertz in '
                  'DMRG for Multiband Impurity Solvers, Institute for Theoretical and Computational Physics. '
                  'Graz University of Technology, Austria, is used.',
      author='Fabian Hoeb, Ish Dhand, Alexander Nuesseler',
      install_requires=['numpy>=1.12', 'scipy>=0.19', 'mpnum>=1.0.3', 'pytest>=3.7.1'],
      author_email='fabian.hoeb@uni-ulm.de, ish.dhand@uni-ulm.de, alexander.nuesseler@uni-ulm.de',
      packages=find_packages(where='.'))
