# Brief documentation of the project structure and naming conventions

## Project structure:
The project can be divided into three sub-modules:
* The chain module, which contains the tmps code for different kinds of mpa types (mpnum array types)
* The star module, which contains the swap based tmps code for different kinds of mpa types (mpnum array types)
* The utils module, which contains several useful utility functions

#### The chain and star modules
Both modules follow the same basic structure, which is can be summarized as follows:
There is a propagtor container object, which contains the mpos that are required for the time evolution.
In the case of the chain, that is handled by a single class called MPPropagator, in the case of the star, there is
a base-class/inheritace structure due to the added complexity of building the propagator mpos.

There is a tmpbase class, which is inherited by TMPS/TMPO/TPMPS classes, that then implement the actual time evolution
based on the propagator container object, which is passed to the class as handle during construction.
The tmpbase class implements constructors, which allow building the objects directly from arrays of components of the
Hamiltonian and provides an interface to interact with.

There are imaginary time evolution subclasses of the respective TMPS/TMPO/TPMPS classes, which disable some tracking
mechanisms, that would overflow during imaginary time evolution and also track the trace of the state for the
computation of the partition function of calculated thermal states.

All of the above mentioned parts have their own respective factory functions, which generate Propagator/TMP/ITMP objects
(Note, that in the example scripts TMP objects are usually referred to as propagator).

There also exist interfaces for the the generation of ground and thermal states of the full star/chain via imaginary 
time evolution.

Sidenote:
While both star and chain tmp objects contain the evolve and fast_evolve functions, the star fast_evolve is an alias for evolve.
The chain fast_evolve skips the final dot product of the propagation step whenever the end-Parameter is not set True.

#### Utility functions:
* Embedding chain Hamiltonians as mpos, calculating the norm of chain Hamiltonians
* Generating number ground states as mps
* Generating product states as numpy arrays 
* Interfaces for easier canonicalization and compression of mps/mpo/pmps represented states
* Compression of the site local tensors for pmps
* Pauli matrices and fock number/ladder operators
* State purification (of numpy-arrays) into numpy arrays or mpnum arrays
* Swap gate generation (as numpy array or mpnum array)
* Random state generation interface for easier generation of mps/mpo/pmps represented states
* Interfaces, which can be used to generate reduced states from mps, pmps and mpo and can also calculate expectation values of reduced states
* Expectation value calculation for full mps/mpo/pmps
* Generation of mpnum shape tuples from individual mpa axes
* Thermal state generation for pmps/mpo, where the Hamiltonian contains only site local terms
* Generation of maximally mixed pmps and mpos
* Interface for the single bond svd compresssion of mps/pmps/mpo represented states 

## Notation and conventions throughout the project
For both star and chain type time evolutions:
* h_site refers to an iterable (of numpy arrays) of the site local parts of the Hamiltonian 
* h_bond refers to an iterable (of numpy arrays) of the couplings between the sites in the Hamiltonian
* hi_list refers to a list/tuple (of numpy arrays), which contains both site local parts and couplings of a Hamiltonian that can be written as sum of the terms in this list
* tau refers to the time step (which for all externally accessible interfaces is always passed as real number float, for imaginary time evolution)
* mpa_type is always a type identifier string for the mpnum array, that is to be evolved in time.
* psi_0 refers to the initial state of the propagation
* state_compression_kwargs, psi_0_compression_kwargs and op_compression_kwargs refer to dictionaries, which contain information about various types of compressions to be performed by the algorithm

For the star type time evolution only:
* system_index refers to the index of the system site in the mpa-chain to which the star geometry is mapped. Must be smaller than the length of the chain minus 1.

