"""
    Thermal state generation for the 1D Ising model H = - \\sum_{i=1}^{N-1} J_i S^Z_i x S^Z_i+1
"""

import tmps
from tmps.utils import pauli
import numpy as np


# Number of spins:
N_spin = 60

# Isotropic coupling J_i = J:
J = 0.2

# Target inverse temperature:
beta = 1

# Number of Trotter timesteps to reach beta:
nof_steps = 50

# Using second order Trotter decomposition:
second_order_trotter = False

# MPArray type (mpo or pmps here)
mpa_type = 'pmps'

# Compression for the initial state, the state during the propagation and the operators
psi_0_compression = None
state_compression = {'method': 'svd', 'relerr': 1e-7}
op_compression = {'method': 'svd', 'relerr': 1e-10}

# Coupling operators in the Hamiltonian H = sum_i h_i
coupling_operator = np.kron(pauli.Z, pauli.Z)
hi_list = [J*coupling_operator] * (N_spin-1)

# Generate the thermal state and partition function:
psi_beta, info = tmps.chain.thermal.from_hi(beta, mpa_type, [2]*N_spin, hi_list, nof_steps=nof_steps,
                                            state_compression_kwargs=state_compression,
                                            op_compression_kwargs=op_compression,
                                            second_order_trotter=second_order_trotter,
                                            psi_0_compression_kwargs=psi_0_compression, verbose=True,
                                            get_partition_function=True)

# Exact solution of the partition function:
Z_exact = 2**N_spin * np.cosh(beta*J)**(N_spin-1)
print('Relative error of the partition function: ')
print(np.abs(info['Z'] - Z_exact)/np.abs(Z_exact))