## A matrix product state time evolution library for Python based on mpnum

* support for well-known matrix product representations (see mpnum documentation for details), such as:
  * matrix product states (MPS)
  * matrix product operators (MPO)
  * local purification matrix product states (PMPS)
* support for chain (next nearest neighbor coupling) and star (long range coupling) geometries
* real time evolution, using either second or fourth order trotter decomposition, based on algorithms described in: Annals of Physics 326 (2011), 96-192; doi: 10.1016/j.aop.2010.09.012 by Schollwoeck and DMRG for Multiband Impurity Solvers, Institute for Theoretical and Computational Physics, Graz University of Technology by Hans Gerd Evertz
* thermal and ground state generation via imaginary time evolution
* support for generating chain geometry Hamiltonians, reduced states, mixed state purification and local svd compression

To install the latest stable version run

    pip install tmps


Required packages:

* numpy, scipy, mpnum

Supported Python versions:

* 3.5, 3.6, 3.7


## Contributors

* Fabian Hoeb, <fabian.hoeb@uni-ulm.de>, [University of Ulm]


## License

Distributed under the terms of the BSD 3-Clause License (see [LICENSE](LICENSE)).
