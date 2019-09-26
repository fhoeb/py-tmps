"""
    Test for star propagation of a mixed quantum state represented by a mpo
"""
import pytest
from testutils import star_test_generator
from testutils.star_exdiag import StarExDiagPropagator
from tmps.star.factory import from_hamiltonian
import tmps.utils.pauli as pauli
import tmps.utils.fock as fock
import numpy as np
import mpnum as mp


@pytest.mark.fast
@pytest.mark.parametrize("L, system_index", [(3, 1), (4, 0), (5, 2), (6, 1), (7, 0)])
@pytest.mark.parametrize("sot, nof_steps, tau", [(False, 50, 0.001), (True, 250, 0.001)])
def test_regular_chain(L, system_index, sot, nof_steps, tau):
    """
        Pytest test for time evolution of a mixed quantum state in mpo form under a Hamiltonian of the form:
        sum_i^L H_{i} + sum_j^(L-1) H_{system_index, j}
        where the H_(i) are called site ops and the H_{i, i+1} are called bond ops. Those are constructed from
        pauli spin operators.
        The physical dimension of each site (site_dim) is constant d!
        Tests are performed for different chain lengths and different time discretizations
        The initial state is a random state
    :param L: Chain L
    :param system_index: Index of the system site in the chain
    :param sot: second order trotter evolution switch
    :param nof_steps: Number of time evolution steps
    :param tau: Timestep in each step of the time evolution
    :return:
    """
    site_dim = 2
    sys_site_op = pauli.Z
    bath_site_op = pauli.Z
    sys_couple_op = pauli.X
    bath_couple_op = pauli.X
    seed = 42
    rank = 1

    exdiag_state, dims, ed_site_ops, ed_bond_ops, tm_state, tm_site_ops, tm_bond_ops = \
        star_test_generator.generate_regular_mpo_test(L, system_index, sys_site_op, bath_site_op,
                                                      sys_couple_op, bath_couple_op, seed=seed, rank=rank)
    op_compression = {'method': 'svd', 'relerr': 1e-12}
    state_compression = {'method': 'svd', 'relerr': 1e-10}

    tm_prop = from_hamiltonian(tm_state, 'mpo', system_index, tm_site_ops, tm_bond_ops, tau=tau,
                               op_compression_kwargs=op_compression,
                               state_compression_kwargs=state_compression,
                               second_order_trotter=sot)

    ed_prop = StarExDiagPropagator(exdiag_state, dims, system_index, ed_site_ops, ed_bond_ops, tau, state_type='op')
    normdist = []
    for i in range(nof_steps):
        tm_prop.evolve()
        ed_prop.evolve()
        tmps_psi_t_array = mp.MPArray.to_array_global(tm_prop.psi_t).reshape(site_dim ** L,
                                                                             site_dim ** L)
        diff = tmps_psi_t_array - ed_prop.psi_t
        normdist.append(np.linalg.norm(diff))
    max_normdist = np.max(np.array(normdist))
    if sot:
        assert np.max(max_normdist) < 1e-6
    else:
        assert np.max(max_normdist) < 1e-8


@pytest.mark.fast
@pytest.mark.parametrize("L, system_index", [(3, 1), (4, 0), (5, 2), (6, 1)])
@pytest.mark.parametrize("sot, nof_steps, tau", [(True, 250, 0.001), (False, 50, 0.001)])
def test_irregular_chain(L, system_index, sot, nof_steps, tau):
    """
        Pytest test for time evolution of a mixed quantum state in mpo form under a Hamiltonian of the form:
        sum_i^L H_{i} + sum_j^(L-1) H_{system_index, j}
        where the H_(i) are called site ops and the H_{i, i+1} are called bond ops, constructed from ladder operators
        and pauli operators.
        The physical dimension of the system site is different from the bath sites
        Tests are performed for different chain lengths and different time discretizations
        The initial state is a random state
    :param L: Chain L
    :param system_index: Index of the system site in the chain
    :param sot: second order trotter evolution switch
    :param nof_steps: Number of time evolution steps
    :param tau: Timestep in each step of the time evolution
    :return:
    """
    sys_site_op = fock.n(3)
    bath_site_op = pauli.Z
    sys_couple_op = fock.a(3) + fock.a_dag(3)
    bath_couple_op = pauli.X
    seed = 42
    rank = 1

    exdiag_state, dims, ed_site_ops, ed_bond_ops, tm_state, tm_site_ops, tm_bond_ops = \
        star_test_generator.generate_irregular_mpo_test(L, system_index, sys_site_op, bath_site_op,
                                                        sys_couple_op, bath_couple_op, seed=seed, rank=rank)
    op_compression = {'method': 'svd', 'relerr': 1e-12}
    state_compression = {'method': 'svd', 'relerr': 1e-10}

    tm_prop = from_hamiltonian(tm_state, 'mpo', system_index, tm_site_ops, tm_bond_ops, tau=tau,
                               op_compression_kwargs=op_compression,
                               state_compression_kwargs=state_compression,
                               second_order_trotter=sot)

    ed_prop = StarExDiagPropagator(exdiag_state, dims, system_index, ed_site_ops, ed_bond_ops, tau, state_type='op')
    normdist = []
    for i in range(nof_steps):
        tm_prop.evolve()
        ed_prop.evolve()
        tmps_psi_t_array = mp.MPArray.to_array_global(tm_prop.psi_t).reshape(np.prod(dims, dtype=int),
                                                                             np.prod(dims, dtype=int))
        diff = tmps_psi_t_array - ed_prop.psi_t
        normdist.append(np.linalg.norm(diff))
    max_normdist = np.max(np.array(normdist))
    if sot:
        assert np.max(max_normdist) < 1e-6
    else:
        assert np.max(max_normdist) < 1e-8
