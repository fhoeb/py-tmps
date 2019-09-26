"""
    Thermal state generation for the 1D Ising model H = - \\sum_{i=1}^{N-1} J_i S^Z_i x S^Z_i+1
"""

import tmps
from tmps.utils import pauli
import numpy as np


# Number of spins:
N_spin = 25

# Isotropic coupling J_i = J:
J = 0.2

# Random initial state properties
rank = 1
seed = 102

# Imaginary time step:
tau = 0.05

# Parameters for the convergence:
# Number of steps to do before any convergence check is performed
start = 100
# Number of steps between convergence checks
step = 1
# Maximum number of steps before propagation is aborted
stop = 500
# Desired accuracy of the state
eps = 1e-4

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

psi_beta, info = tmps.chain.ground.from_hi_convergence(tau, mpa_type, [2] * N_spin, hi_list, rank=rank, seed=seed,
                                                       start=start, step=step, stop=stop, eps=eps,
                                                       state_compression_kwargs=state_compression,
                                                       op_compression_kwargs=op_compression,
                                                       second_order_trotter=second_order_trotter,
                                                       psi_0_compression_kwargs=psi_0_compression, verbose=True,
                                                       get_energy=True)
E_0_exact = - J*(N_spin-1)
print('Relative error of the ground state: ')
print(np.abs(info['energy'] - E_0_exact)/np.abs(E_0_exact))
print(info)

# The above can also be accomplished b direct propagation without testing for convergence via:
# psi_beta, info = tmps.chain.ground.from_hi(tau, nof_steps, mpa_type, [2] * N_spin, hi_list, rank=rank,
#                                            state_compression_kwargs=state_compression,
#                                            op_compression_kwargs=op_compression,
#                                            second_order_trotter=second_order_trotter,
#                                            psi_0_compression_kwargs=psi_0_compression, verbose=True,
#                                            get_energy=True)
