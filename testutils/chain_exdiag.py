"""
exdiag.py - Auxiliary class for reference exact diagonalization of chain Hamiltonians of the form:
            sum_i^L H_{i} + sum_i^(L-1) H_{i, i+1}
"""
import numpy as np
from scipy.linalg import eigh


class ExDiagPropagator:

    def __init__(self, psi_0, site_dims, site_ops, bond_ops, tau, state_type='vec'):
        """
            Constructor for ExactDiag class, does necessary setup for propagation
        :param psi_0: state vector or density matrix as numpy array
        :param site_dims: List of the physical dimensions of each site on the chain (e.g. [2,3,2,4]
        :param site_ops: List or Generator of single site operators (H_(i)) as numpy arrays
        :param bond_ops: List or Generator of nearest neighbor/bond operators (H_{i, i+1}) as numpy arrays
        :param tau: timestep for the propagation
        :param state_type: switch to propagate psi_0 either as a state vector: 'vec' or density matrix: 'op'
        """
        self.psi_t = psi_0
        self.site_dims = site_dims
        self.L = len(site_dims)
        self.site_ops = site_ops
        self.bond_ops = bond_ops
        self.tau = tau
        self.state_type = state_type
        self.H = self._embed_hamiltonian()
        self.propagator = self._generate_propagator()

    def _embed_hamiltonian(self):
        """
            Generates hamiltonian H from site local and bond local terms: sum_i^L H_{i} + sum_i^(L-1) H_{i, i+1}
        :return:
        """
        H = 0
        site = 0
        for site_op in self.site_ops:
            site_op = site_op.astype(complex)
            d = np.prod(self.site_dims[:site], dtype=int)
            d_ = np.prod(self.site_dims[site+1:], dtype=int)
            H += np.kron(np.kron(np.eye(int(d)), site_op), np.eye(int(d_)))
            site += 1
        site = 0
        for bond_op in self.bond_ops:
            bond_op = bond_op.astype(complex)
            d = np.prod(self.site_dims[:site], dtype=int)
            d_ = np.prod(self.site_dims[site+2:], dtype=int)
            H += np.kron(np.kron(np.eye(int(d)), bond_op), np.eye(int(d_)))
            site += 1
        return H

    def _generate_propagator(self):
        """
            Performs exact diagonalization on the Hamiltonian and builds propagator U(tau)
        :return: Propagator U(tau)
        """
        D, V = eigh(self.H)
        return V @ np.diag(np.exp(-1j * self.tau * D)) @ V.T.conj()

    def reset(self, psi_0):
        """
            Reets back to some new initial state
        :param psi_0: New state of same shape (Matrix for 'op', vector for 'vec') as old one
        :return:
        """
        self.psi_t = psi_0

    def evolve(self):
        """
            Evolves the quantum state in time by tau
        :return:
        """
        if self.state_type == 'vec':
            self.psi_t = np.dot(self.propagator, self.psi_t)
        elif self.state_type == 'op':
            self.psi_t = self.propagator @ self.psi_t @ self.propagator.T.conj()
        else:
            assert False


def generate_thermal_state(beta, site_dims, site_ops, bond_ops):
    """
        Generates a normalized thermal state rho = 1/Z exp(- beta * H) via exact diagonalization.
        Uses a dummy ExDiagPropagator for the diagonalization.
    :param beta: Inverse temperature
    :param site_dims: Site physical dimensions
    :param site_ops: Iterator over site local operators
    :param bond_ops: Iterator over bond operators
    :return:
    """
    tau = -1j*beta
    dummy_propagator = ExDiagPropagator(None, site_dims, site_ops, bond_ops, tau)
    return dummy_propagator.propagator/np.trace(dummy_propagator.propagator)

