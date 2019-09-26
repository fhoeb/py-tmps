"""
    This module exists, because the conventions by which star_exdiag and the StarTMP classes take the site and
    coupling operators differ. For the TMP classes, they may be passed AS IF system site and the coupled-to bath
    site are right next to each other. For the star_exdiag they must be passed as if they were embedded in a chain
    with the necessary identities between them.
"""
import numpy as np
import mpnum as mp
import tmps.utils.kron as kron


def get_operators(L, system_index, sys_site_op, bath_site_op, sys_couple_op, bath_couple_op):
    """
        Constructs operators for the star exact diagonalization and the StarTMP
    :param L: Length of the chain for which to generate operators
    :param system_index: INdex of the system site
    :param sys_site_op: Site operator for the system
    :param bath_site_op: Site operator for the bath (broadcast on all bath sites)
    :param sys_couple_op: Coupling operator from sthe system site
    :param bath_couple_op: Coupling operator from the bath site
    :return: Site operators for the exact diagonalization, bond operators for the exact diagonalization,
             Site operators for tmps evolution, bond operators for the tmps evolution
    """
    # Exdiag operators:
    ed_site_ops = list()
    ed_bond_ops = list()
    bath_site_dim = bath_site_op.shape[0]
    for site in range(system_index):
        ed_site_ops.append(bath_site_op)
    ed_site_ops.append(sys_site_op)
    for site in range(L-system_index-1):
        ed_site_ops.append(bath_site_op)
    for site in range(system_index):
        id_between = np.identity(bath_site_dim**(system_index-site-1)) if system_index-site-1 > 0 else np.identity(1)
        ed_bond_ops.append(np.kron(np.kron(bath_couple_op, id_between), sys_couple_op))
    for site in range(system_index+1, L):
        id_between = np.identity(bath_site_dim**(site-system_index-1)) if site-system_index-1 > 0 else np.identity(1)
        ed_bond_ops.append(np.kron(sys_couple_op, np.kron(id_between, bath_couple_op)))
    # tmps operators:
    tm_site_ops = []
    tm_bond_ops = []
    for site in range(system_index):
        tm_site_ops.append(bath_site_op)
    tm_site_ops.append(sys_site_op)
    for site in range(L-system_index-1):
        tm_site_ops.append(bath_site_op)
    for site in range(system_index):
        tm_bond_ops.append(np.kron(bath_couple_op, sys_couple_op))
    for site in range(system_index+1, L):
        tm_bond_ops.append(np.kron(sys_couple_op, bath_couple_op))
    return ed_site_ops, ed_bond_ops, tm_site_ops, tm_bond_ops


def generate_regular_mps_test(L, system_index, sys_site_op, bath_site_op, sys_couple_op, bath_couple_op,
                              single_state=None, seed=42, rank=2):
    """
            See get_operators for an explanation for most of the parameters. Additionally generates a random MPS.
            Operators must be 2x2 Matrices
        :returns: Initial state for exact diagonalization, dimensions of the chain, exact diagonalization site operators,
                  exact diagonalization bond operators, TMP initial state, TMP site operators, TMP bond operators
    """
    assert L > 2
    assert sys_site_op.shape == bath_site_op.shape == sys_couple_op.shape == bath_couple_op.shape == (2, 2)
    dims = [2]*L
    if single_state is not None:
        exdiag_state = kron.generate_product_state(single_state, L)
        tm_state = mp.MPArray.from_array(exdiag_state.reshape(tuple([2]*L)), ndims=1)
    else:
        rng = np.random.RandomState(seed=seed)
        tm_state = mp.random_mpa(sites=L, ldim=2, rank=rank, randstate=rng, normalized=True)
        exdiag_state = tm_state.to_array().reshape(2**L)
    ed_site_ops, ed_bond_ops, tm_site_ops, tm_bond_ops = \
        get_operators(L, system_index, sys_site_op, bath_site_op, sys_couple_op, bath_couple_op)
    return exdiag_state, dims, ed_site_ops, ed_bond_ops, tm_state, tm_site_ops, tm_bond_ops


def generate_irregular_mps_test(L, system_index, sys_site_op, bath_site_op, sys_couple_op, bath_couple_op,
                                seed=42, rank=2):
    """
            See get_operators for an explanation for most of the parameters. Additionally generates a random MPS.
            System and Bath operators may be of any dimension
        :returns: Initial state for exact diagonalization, dimensions of the chain, exact diagonalization site operators,
                  exact diagonalization bond operators, TMP initial state, TMP site operators, TMP bond operators
    """
    assert L > 2
    assert sys_site_op.shape == sys_couple_op.shape  and bath_couple_op.shape == bath_site_op.shape
    bath_dim = bath_site_op.shape[0]
    sys_dim = sys_site_op.shape[0]
    dims = [bath_dim] * system_index + [sys_dim] + [bath_dim]*(L-system_index-1)
    rng = np.random.RandomState(seed=seed)
    ldim = [(bath_dim, )]*system_index + [(sys_dim, )] + [(bath_dim, )] * (L-system_index-1)
    tm_state = mp.random_mpa(sites=L, ldim=tuple(ldim), rank=rank, randstate=rng, normalized=True)
    exdiag_state = tm_state.to_array().reshape(bath_dim**(L-1) * sys_dim)
    ed_site_ops, ed_bond_ops, tm_site_ops, tm_bond_ops = \
        get_operators(L, system_index, sys_site_op, bath_site_op, sys_couple_op, bath_couple_op)
    return exdiag_state, dims, ed_site_ops, ed_bond_ops, tm_state, tm_site_ops, tm_bond_ops


def generate_regular_mpo_test(L, system_index, sys_site_op, bath_site_op, sys_couple_op, bath_couple_op,
                              single_state=None, seed=42, rank=2):
    """
            See get_operators for an explanation for most of the parameters. Additionally generates a random MPO.
            Operators must be 2x2 Matrices
        :returns: Initial state for exact diagonalization, dimensions of the chain, exact diagonalization site operators,
                  exact diagonalization bond operators, TMP initial state, TMP site operators, TMP bond operators
    """
    assert L > 2
    assert sys_site_op.shape == bath_site_op.shape == sys_couple_op.shape == bath_couple_op.shape == (2, 2)
    dims = [2]*L
    if single_state is not None:
        psi_0 = kron.generate_product_state(single_state, L)
        exdiag_state = np.outer(psi_0, psi_0.conj())
        psi_0_mps = mp.MPArray.from_array(exdiag_state.reshape(tuple([2]*L)), ndims=1)
        tm_state = mp.mps_to_mpo(psi_0_mps)
    else:
        rng = np.random.RandomState(seed=seed)
        psi_0_pmps = mp.random_mpa(sites=L, ldim=(2, 2), rank=rank, randstate=rng, normalized=True)
        tm_state = mp.pmps_to_mpo(psi_0_pmps)
        exdiag_state = mp.MPArray.to_array_global(tm_state).reshape(2 ** L, 2 ** L)
    ed_site_ops, ed_bond_ops, tm_site_ops, tm_bond_ops = \
        get_operators(L, system_index, sys_site_op, bath_site_op, sys_couple_op, bath_couple_op)
    return exdiag_state, dims, ed_site_ops, ed_bond_ops, tm_state, tm_site_ops, tm_bond_ops


def generate_irregular_mpo_test(L, system_index, sys_site_op, bath_site_op, sys_couple_op, bath_couple_op,
                                seed=42, rank=2):
    """
            See get_operators for an explanation for most of the parameters. Additionally generates a random MPO.
            System and Bath operators may be of any dimension
        :returns: Initial state for exact diagonalization, dimensions of the chain, exact diagonalization site operators,
                  exact diagonalization bond operators, TMP initial state, TMP site operators, TMP bond operators
    """
    assert L > 2
    assert sys_site_op.shape == sys_couple_op.shape and bath_couple_op.shape == bath_site_op.shape
    bath_dim = bath_site_op.shape[0]
    sys_dim = sys_site_op.shape[0]
    dims = [bath_dim] * system_index + [sys_dim] + [bath_dim]*(L-system_index-1)
    rng = np.random.RandomState(seed=seed)
    ldim = [(bath_dim, bath_dim)]*system_index + [(sys_dim, sys_dim)] + [(bath_dim, bath_dim)] * (L-system_index-1)
    pmps_rho_0 = mp.random_mpa(sites=L, ldim=tuple(ldim), rank=rank, randstate=rng, normalized=True)
    tm_state = mp.pmps_to_mpo(pmps_rho_0)
    exdiag_state = mp.MPArray.to_array_global(tm_state).reshape(bath_dim**(L-1) * sys_dim, bath_dim**(L-1) * sys_dim)
    ed_site_ops, ed_bond_ops, tm_site_ops, tm_bond_ops = \
        get_operators(L, system_index, sys_site_op, bath_site_op, sys_couple_op, bath_couple_op)
    return exdiag_state, dims, ed_site_ops, ed_bond_ops, tm_state, tm_site_ops, tm_bond_ops


def generate_regular_pmps_test(L, system_index, sys_site_op, bath_site_op, sys_couple_op, bath_couple_op,
                              single_state=None, seed=42, rank=2):
    """
            See get_operators for an explanation for most of the parameters. Additionally generates a random PMPS.
            Operators must be 2x2 Matrices
        :returns: Initial state for exact diagonalization, dimensions of the chain, exact diagonalization site operators,
                  exact diagonalization bond operators, TMP initial state, TMP site operators, TMP bond operators
    """
    assert L > 2
    assert sys_site_op.shape == bath_site_op.shape == sys_couple_op.shape == bath_couple_op.shape == (2, 2)
    dims = [2]*L
    if single_state is not None:
        psi_0 = kron.generate_product_state(single_state, L)
        exdiag_state = np.outer(psi_0, psi_0.conj())
        psi_0_mps = mp.MPArray.from_array(exdiag_state.reshape(tuple([2]*L)), ndims=1)
        tm_state = mp.mps_to_pmps(psi_0_mps)
    else:
        rng = np.random.RandomState(seed=seed)
        tm_state = mp.random_mpa(sites=L, ldim=(2, 2), rank=rank, randstate=rng, normalized=True)
        mpo_rho_0 = mp.pmps_to_mpo(tm_state)
        exdiag_state = mp.MPArray.to_array_global(mpo_rho_0).reshape(2 ** L, 2 ** L)
    ed_site_ops, ed_bond_ops, tm_site_ops, tm_bond_ops = \
        get_operators(L, system_index, sys_site_op, bath_site_op, sys_couple_op, bath_couple_op)
    return exdiag_state, dims,  ed_site_ops, ed_bond_ops, tm_state, tm_site_ops, tm_bond_ops


def generate_irregular_pmps_test(L, system_index, sys_site_op, bath_site_op, sys_couple_op, bath_couple_op,
                                 seed=42, rank=2):
    """
            See get_operators for an explanation for most of the parameters. Additionally generates a random PMPS.
            System and Bath operators may be of any dimension
        :returns: Initial state for exact diagonalization, dimensions of the chain, exact diagonalization site operators,
                  exact diagonalization bond operators, TMP initial state, TMP site operators, TMP bond operators
    """
    assert L > 2
    assert sys_site_op.shape == sys_couple_op.shape  and bath_couple_op.shape == bath_site_op.shape
    bath_dim = bath_site_op.shape[0]
    sys_dim = sys_site_op.shape[0]
    dims = [bath_dim] * system_index + [sys_dim] + [bath_dim]*(L-system_index-1)
    rng = np.random.RandomState(seed=seed)
    ldim = [(bath_dim, bath_dim)]*system_index + [(sys_dim, sys_dim)] + [(bath_dim, bath_dim)] * (L-system_index-1)
    tm_state = mp.random_mpa(sites=L, ldim=tuple(ldim), rank=rank, randstate=rng, normalized=True)
    mpo_rho_0 = mp.pmps_to_mpo(tm_state)
    exdiag_state = mp.MPArray.to_array_global(mpo_rho_0).reshape(bath_dim**(L-1) * sys_dim, bath_dim**(L-1) * sys_dim)
    ed_site_ops, ed_bond_ops, tm_site_ops, tm_bond_ops = \
        get_operators(L, system_index, sys_site_op, bath_site_op, sys_couple_op, bath_couple_op)
    return exdiag_state, dims, ed_site_ops, ed_bond_ops, tm_state, tm_site_ops, tm_bond_ops




