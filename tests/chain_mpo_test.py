"""
    Test for chain propagation of a mixed quantum state represented by a mpo
"""
import numpy as np
import mpnum as mp
from tmps.chain.factory import from_hamiltonian
import pytest
import testutils.chain_exdiag as exdiag
import tmps.utils.kron as kron
import tmps.utils.pauli as pauli
from tmps.utils.random import get_random_mpa
from itertools import repeat


@pytest.mark.fast
@pytest.mark.parametrize("L", [2, 3, 4, 5])
@pytest.mark.parametrize("sot, nof_steps, tau", [(True, 250, 0.001), (False, 50, 0.001)])
def test_regular_chain(L, sot, nof_steps, tau):
    """
        Pytest test for time evolution of a mixed quantum state in mpo form under a Hamiltonian of the form:
        sum_i^L H_{i} + sum_i^(L-1) H_{i, i+1}
        where the H_(i) are called site ops and the H_{i, i+1} are called bond ops. Those are constructed from
        pauli spin operators.
        The physical dimension of each site (site_dim) is constant d!
        Tests are performed for different chain lengths and different time discretizations
        The initial state is a superposition product state
    :param L: Chain L
    :param nof_steps: Number of time evolution steps
    :param tau: Timestep in each step of the time evolution
    :return:
    """
    # Parameters:
    psi_i = np.array([1.0, 1.0])
    psi_i /= np.linalg.norm(psi_i)
    site_dim = 2
    site_dims = [site_dim] * L
    site_op = pauli.Z
    bond_op = np.kron(pauli.Z, pauli.X)

    # Setup for exact diagonalization
    psi_0 = kron.generate_product_state(psi_i, L)
    rho_0 = np.outer(psi_0, psi_0.conj())
    exdiag_mixed_propagator = exdiag.ExDiagPropagator(rho_0, site_dims, repeat(site_op, L),
                                                      repeat(bond_op, L - 1), tau, state_type='op')

    state_compress_kwargs = {'method': 'svd', 'relerr': 1e-10}
    op_compression_kwargs = {'method': 'svd', 'relerr': 1e-13}
    mpo_rho_0 = np.copy(rho_0).reshape(*(*site_dims, *site_dims))
    mpo = mp.MPArray.from_array_global(mpo_rho_0, ndims=2)
    tmps_propagator = from_hamiltonian(mpo, 'mpo', site_op, bond_op, tau=tau,
                                       state_compression_kwargs=state_compress_kwargs,
                                       op_compression_kwargs=op_compression_kwargs,
                                       second_order_trotter=sot)

    # Propagataion loop
    normdist = []
    for step in range(nof_steps):
        exdiag_mixed_propagator.evolve()
        tmps_propagator.evolve()

        rho_t = exdiag_mixed_propagator.psi_t
        mpo_t_array = mp.MPArray.to_array_global(tmps_propagator.psi_t).reshape(site_dim ** L, site_dim ** L)
        normdist.append(np.linalg.norm(mpo_t_array - rho_t))
    max_normdist = np.max(np.array(normdist))
    if sot:
        assert np.max(max_normdist) < 1e-6
    else:
        assert np.max(max_normdist) < 1e-8


@pytest.mark.slow
@pytest.mark.parametrize("sot, nof_steps, tau", [(True, 250, 0.001), (False, 50, 0.001)])
def test_irregular_chain(sot, nof_steps, tau):
    """
        Pytest test for time evolution of a mixed quantum state in mpo form under a Hamiltonian of the form:
        sum_i^L H_{i} + sum_i^(L-1) H_{i, i+1}
        where the H_(i) are called site ops and the H_{i, i+1} are called bond ops. Those are constructed from
        random hermitian matrices
        The physical dimension of each site (site_dim) is allowed to vary from site to site!
        The test is perfomed for one random Hamiltonian for a chain of the form [2, 4, 3, 5]
        for different time discretizations.
    :param nof_steps: Number of time evolution steps
    :param tau: Timestep in each step of the time evolution
    :return:
    """
    # Parameters for the chain:
    # Chain shape:
    site_dims = [2, 4, 3, 2]
    # Site local initial states
    psi_1, psi_2, psi_3, psi_4 = np.array([0, 1], dtype=float), np.array([0, 1, 0, 0], dtype=float), \
                                 np.array([0, 0, 1], dtype=float), np.array([0, 1], dtype=float)
    # Generate site and bond operators of random hamiltonian
    random_sites = [np.random.rand(2, 2), np.random.rand(4, 4), np.random.rand(3, 3), np.random.rand(2, 2)]
    random_bonds = [np.random.rand(8, 8), np.random.rand(12, 12), np.random.rand(6, 6)]
    # Make them hermitian
    site_ops = [(site + site.T.conj()) / 2 for site in random_sites]
    bond_ops = [(bond + bond.T.conj()) / 2 for bond in random_bonds]
    # Put initial states together
    psi_0 = np.kron(np.kron(np.kron(psi_1, psi_2), psi_3), psi_4)
    rho_0 = np.outer(psi_0, psi_0.conj())
    # Setup for exact diagonalization
    exdiag_mixed_propagator = exdiag.ExDiagPropagator(rho_0, site_dims, site_ops, bond_ops, tau, state_type='op')

    # Setup for tMPS evolution
    state_compress_kwargs = {'method': 'svd', 'relerr': 1e-10}
    op_compression_kwargs = {'method': 'svd', 'relerr': 1e-13}
    mpo_rho_0 = np.copy(rho_0).reshape(*(*site_dims, *site_dims))
    mpo = mp.MPArray.from_array_global(mpo_rho_0, ndims=2)
    tmps_propagator = from_hamiltonian(mpo, 'mpo', site_ops, bond_ops, tau=tau,
                                       state_compression_kwargs=state_compress_kwargs,
                                       op_compression_kwargs=op_compression_kwargs,
                                       second_order_trotter=sot)

    # Propagataion loop
    normdist = []
    for step in range(nof_steps):
        exdiag_mixed_propagator.evolve()
        tmps_propagator.evolve()

        rho_t = exdiag_mixed_propagator.psi_t
        mpo_t_array = mp.MPArray.to_array_global(tmps_propagator.psi_t).reshape(int(np.prod(site_dims)),
                                                                                int(np.prod(site_dims)))
        normdist.append(np.linalg.norm(mpo_t_array - rho_t))

    max_normdist = np.max(np.array(normdist))
    if sot:
        assert np.max(max_normdist) < 1e-6
    else:
        assert np.max(max_normdist) < 1e-8


@pytest.mark.fast
@pytest.mark.parametrize("L", [2, 3, 4, 5])
@pytest.mark.parametrize("sot, nof_steps, tau", [(True, 250, 0.001), (False, 50, 0.001)])
def test_regular_chain_random(L, sot, nof_steps, tau):
    """
        Pytest test for time evolution of a mixed quantum state in mpo form under a Hamiltonian of the form:
        sum_i^L H_{i} + sum_i^(L-1) H_{i, i+1}
        where the H_(i) are called site ops and the H_{i, i+1} are called bond ops, they are generated randomly.
        The physical dimension of each site (site_dim) is constant
        Tests are performed for different chain lengths and different time discretizations.
        The initial state is randomized.
    :param L: Chain L
    :param nof_steps: Number of time evolution steps
    :param tau: Timestep in each step of the time evolution
    :return:
    """
    # Parameters:
    site_dim = 2
    site_dims = [site_dim] * L

    mpo_rho_0 = get_random_mpa('mpo', site_dims, rank=2, seed=103)
    rho_0 = mp.MPArray.to_array_global(mpo_rho_0).reshape(site_dim ** L, site_dim ** L)

    site_op = pauli.Z
    bond_op = np.kron(pauli.Z, pauli.X)

    # Setup for exact diagonalization
    exdiag_mixed_propagator = exdiag.ExDiagPropagator(rho_0, site_dims, repeat(site_op, L),
                                                      repeat(bond_op, L - 1),
                                                      tau, state_type='op')

    # Setup for tMPS evolution
    mps_compress_kwargs = {'method': 'svd', 'relerr': 1e-10}
    op_compression_kwargs = {'method': 'svd', 'relerr': 1e-13}

    tmps_propagator = from_hamiltonian(mpo_rho_0, 'mpo', site_op, bond_op, tau=tau,
                                       state_compression_kwargs=mps_compress_kwargs,
                                       op_compression_kwargs=op_compression_kwargs,
                                       second_order_trotter=sot)

    # Propagataion loop
    normdist = []
    for step in range(nof_steps):
        exdiag_mixed_propagator.evolve()
        tmps_propagator.evolve()

        rho_t = exdiag_mixed_propagator.psi_t
        mpo_t_array = mp.MPArray.to_array_global(tmps_propagator.psi_t).reshape(site_dim ** L, site_dim ** L)
        normdist.append(np.linalg.norm(mpo_t_array - rho_t))
    max_normdist = np.max(np.array(normdist))
    if sot:
        assert np.max(max_normdist) < 1e-6
    else:
        assert np.max(max_normdist) < 1e-8
