"""
    Test for the calculation of the norm of a chain Hamiltonian
"""
from tmps.utils.norm import get_norm_of_hamiltonian
from testutils.chain_exdiag import ExDiagPropagator
import tmps.utils.pauli as pauli
import numpy as np
import pytest


@pytest.mark.fast
@pytest.mark.parametrize("L", [2, 3, 4, 5, 6, 7])
def test_equal_site_hamiltonian(L):
    """
        Pytest test for calculating the norm  of nearest neighbor Hamiltonians
        with equal local dimension (d=2) on all sites
    :param L: Size of the chain
    :return:
    """
    site_dims = [2] * L
    site_ops = [pauli.X] * L
    bond_ops = [np.kron(pauli.Z, pauli.Z)] * (L - 1)
    full_hamiltonian = ExDiagPropagator(None, site_dims, site_ops, bond_ops, 0.01).H
    mp_norm = get_norm_of_hamiltonian(site_ops, bond_ops)
    assert abs(np.linalg.norm(full_hamiltonian) - mp_norm) == pytest.approx(0.0)


@pytest.mark.fast
def test_unequal_site_hamiltonian():
    """
        Pytest test for calculating the norm of nearest neighbor Hamiltonians
        with unequal local dimension ([2, 4, 4])
    :return:
    """
    site_dims = [2, 4, 4]
    site_ops = [pauli.X, np.kron(pauli.X, pauli.X), np.kron(pauli.X, pauli.X)]
    bond_ops = [np.kron(pauli.Z, np.kron(pauli.X, pauli.X)),
                np.kron(np.kron(np.kron(pauli.X, pauli.X), pauli.Z), pauli.X)]
    full_hamiltonian = ExDiagPropagator(None, site_dims, site_ops, bond_ops, 0.01).H
    mp_norm = get_norm_of_hamiltonian(site_ops, bond_ops)
    assert abs(np.linalg.norm(full_hamiltonian) - mp_norm) == pytest.approx(0.0)