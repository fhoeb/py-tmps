"""
    Time evolution for the spin-boson quantum impurity model with an ohmic spectral density.
    Using a chain mapping
"""

import tmps
from tmps.utils import fock, pauli, convert, state_reduction_as_array
import numpy as np
import mpnum as mp
from scipy.special import gamma
import matplotlib.pyplot as plt

# Parameters:
# Number of bath sites:
N_bath = 30

# Spin energy gap
omega_0 = 1

# Parameters for the ohmic spectral density:
alpha = 1
s = 1
omega_c = 1

# Parameter for the initial state Psi_0 = cos(theta) |1>  + sin(theta) |0>
theta = np.pi/4

# Local dimension of the bosonic bath
loc_dim = 10

# Trotterization timestep:
tau = 0.01

# Number of steps for the time evolution:
nof_steps = 100

# Using second order Trotter decomposition:
second_order_trotter = False

# Compression for the initial state, the state during the propagation and the operators
psi_0_compression = None
state_compression = {'method': 'svd', 'relerr': 1e-7}
op_compression = {'method': 'svd', 'relerr': 1e-10}

# Initial state of the System:
ground = np.array([0.0, np.sin(theta)])
excited = np.array([np.cos(theta), 0.0])
sys_psi_0 = (ground + excited)

# Initial state of the bath:
bath_psi_0 = tmps.utils.broadcast_number_ground_state(loc_dim, N_bath)

# Initial state of the full chain:
psi_0 = mp.chain([convert.to_mparray(sys_psi_0, 'mps'), bath_psi_0])

# Local Hamiltonian of the System:
spin_loc = omega_0/2 * pauli.Z

# Coupling between System and bath:
spin_coupl = pauli.Z

# Energies and couplings for the ohmic bath (see Chin et al., J. Math. Phys. 51, 092109 (2010)):
n = np.arange(N_bath)
omega = omega_c * (2*n + 1 + s)
t = omega_c * np.sqrt((n[:-1] + 1)*(n[:-1] + s + 1))

# Spin-Bath coupling
c0 = np.sqrt(alpha*omega_c**2 * gamma(s+1))

print('Building the Hamiltonian')

# Local Hamiltonian of the bath
fock_n = fock.n(loc_dim)
bath_loc = [energy * fock_n for energy in omega]

# Bath coupling
bath_coupling_op = np.kron(fock.a(loc_dim), fock.a_dag(loc_dim)) + \
                   np.kron(fock.a_dag(loc_dim), fock.a(loc_dim))
bath_bath_coupl = [coupling * bath_coupling_op for coupling in t]

# Spin-Bath coupling
spin_bath_coupl = c0 * (np.kron(spin_coupl, fock.a_dag(loc_dim)) + np.kron(spin_coupl.conj().T, fock.a(loc_dim)))

# Create propagator object:
print('Creating the propagator')
propagator = tmps.chain.from_hamiltonian(psi_0, 'mps', [spin_loc] + bath_loc, [spin_bath_coupl] + bath_bath_coupl,
                                         tau=tau, state_compression_kwargs=state_compression,
                                         op_compression_kwargs=op_compression,
                                         second_order_trotter=second_order_trotter,
                                         psi_0_compression_kwargs=psi_0_compression)

# Generate arrays for coherences and population in the excited state and fill those with the respective values
# of the initial state
coherence = np.empty(nof_steps + 1)
population = np.empty(nof_steps + 1)
reduced_state = state_reduction_as_array(propagator.psi_t, 'mps', startsite=0, nof_sites=1)
coherence[0] = np.abs(reduced_state[0, 1])
population[0] = np.abs(reduced_state[0, 0])

# Main propagation loop
print('Propagation starts')
for psi_t in propagator(nof_steps=nof_steps):
    print('Current ranks of the state: ', psi_t.ranks)
    reduced_state = state_reduction_as_array(psi_t, 'mps', startsite=0, nof_sites=1)
    coherence[propagator.step] = np.abs(reduced_state[0, 1])
    population[propagator.step] = np.abs(reduced_state[0, 0])
print('Propagation finished')


plt.figure(1)
plt.plot(np.abs(coherence), '.-')
plt.title('Absolute value of the coherences for the spin')
plt.xlabel('t')
plt.ylabel('$\\left| \\rho_{0,1} \\right|$')
plt.figure(2)
plt.plot(np.abs(population), '.-')
plt.title('Absolute value of the excited state populations for the spin')
plt.xlabel('t')
plt.ylabel('$\\left| \\rho_{0,0} \\right|$')
plt.ylim((0, 1))
plt.show()

# The above is shorthand for the to the following construction
# for step in range(nof_steps):
#     propagator.evolve()
#     reduced_state = state_reduction_as_array(propagator.psi_t, 'mps', startsite=0, nof_sites=1)
#     coherence[step] = np.abs(reduced_state[0, 1])
#     population[step] = np.abs(reduced_state[0, 0])

# Can also do:
# for psi_t, t in propagator(nof_steps=nof_steps, get_time=True):
#     ...
# which is equivalent to:
# for step in range(nof_steps):
#     propagator.evolve()
#     psi_t = propagator.psi_t
#     t = propagator.t
# Which also provides the current time t
