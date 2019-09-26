"""
    Tests for star thermal state generation via imaginary time evolution
"""
import numpy as np
import mpnum as mp
import testutils.star_exdiag as exdiag
from testutils import star_test_generator
import tmps.utils.pauli as pauli
import tmps.utils.fock as fock
import tmps.star.thermal as thermal
import pytest as pytest


@pytest.mark.slow
@pytest.mark.parametrize("L, system_index", [(3, 0), (4, 1), (5, 2), (6, 0), (7, 5)])
@pytest.mark.parametrize("beta, nof_steps", [(0.1, 10), (1, 100)])
@pytest.mark.parametrize("mpa_type", ['pmps', 'mpo'])
def test_thermal_state_generation(L, system_index, beta, nof_steps, mpa_type):
    """
        Pytest test for generating a thermal state via imaginary time evolution
    :param L: Size of the chain
    :param beta: Inverse temperature of the final state
    :param nof_steps: number of steps in the propagation grid
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
        star_test_generator.generate_regular_mps_test(L, system_index, sys_site_op, bath_site_op,
                                                      sys_couple_op, bath_couple_op, seed=seed, rank=rank)
    op_compression = {'method': 'svd', 'relerr': 1e-12}
    state_compression = {'method': 'svd', 'relerr': 1e-10}
    exdiag_thermal = exdiag.generate_thermal_state(beta, dims, system_index, ed_site_ops, ed_bond_ops)
    # Setup for tMPS evolution
    tmps_thermal, info = thermal.from_hamiltonian(beta, mpa_type, system_index, tm_site_ops, tm_bond_ops,
                                                  nof_steps=nof_steps,
                                                  state_compression_kwargs=state_compression,
                                                  op_compression_kwargs=op_compression)
    if mpa_type == 'pmps':
        tmps_thermal = mp.pmps_to_mpo(tmps_thermal)
    normdiff = np.linalg.norm(exdiag_thermal - tmps_thermal.to_array_global().reshape(site_dim ** L, site_dim ** L))
    assert normdiff < 1e-8


@pytest.mark.slow
@pytest.mark.parametrize("L, system_index", [(3, 0), (4, 1), (5, 2), (6, 0), (7, 5)])
@pytest.mark.parametrize("beta, nof_steps", [(0.1, 10), (1, 100)])
@pytest.mark.parametrize("mpa_type", ['pmps', 'mpo'])
def test_unequal_site_dim_thermal_state_generation(L, system_index, beta, nof_steps, mpa_type):
    """
        Pytest test for generating a thermal state via imaginary time evolution for a chain with unequal site dimensions
    :param beta: Inverse temperature of the final state
    :param nof_steps: number of steps in the propagation grid
    :return:
    """
    sys_site_op = fock.n(3)
    bath_site_op = pauli.Z
    sys_couple_op = fock.a(3) + fock.a_dag(3)
    bath_couple_op = pauli.X
    seed = 42
    rank = 1

    exdiag_state, dims, ed_site_ops, ed_bond_ops, tm_state, tm_site_ops, tm_bond_ops = \
        star_test_generator.generate_irregular_mps_test(L, system_index, sys_site_op, bath_site_op,
                                                        sys_couple_op, bath_couple_op, seed=seed, rank=rank)
    op_compression = {'method': 'svd', 'relerr': 1e-12}
    state_compression = {'method': 'svd', 'relerr': 1e-10}
    exdiag_thermal = exdiag.generate_thermal_state(beta, dims, system_index, ed_site_ops, ed_bond_ops)
    # Setup for tMPS evolution
    tmps_thermal, info = thermal.from_hamiltonian(beta, mpa_type, system_index, tm_site_ops, tm_bond_ops,
                                                  nof_steps=nof_steps,
                                                  state_compression_kwargs=state_compression,
                                                  op_compression_kwargs=op_compression)
    if mpa_type == 'pmps':
        tmps_thermal = mp.pmps_to_mpo(tmps_thermal)
    normdiff = np.linalg.norm(exdiag_thermal - tmps_thermal.to_array_global().reshape(np.prod(dims, dtype=int),
                                                                                      np.prod(dims, dtype=int)))
    assert normdiff < 1e-8
