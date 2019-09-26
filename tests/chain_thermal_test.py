"""
    Tests for chain thermal state generation via imaginary time evolution
"""
import numpy as np
import mpnum as mp
import testutils.chain_exdiag as exdiag
import tmps.utils.pauli as pauli
from itertools import repeat
import tmps.chain.thermal as thermal
import pytest as pytest


@pytest.mark.slow
@pytest.mark.parametrize("L", [2, 3, 4, 5, 6])
@pytest.mark.parametrize("beta, nof_steps", [(0.1, 10), (1, 100)])
@pytest.mark.parametrize("mpa_type", ['pmps', 'mpo'])
def test_thermal_state_generation(L, beta, nof_steps, mpa_type):
    """
        Pytest test for generating a thermal state via imaginary time evolution for a spin chain
    :param L: Size of the chain
    :param beta: Inverse temperature of the final state
    :param nof_steps: number of steps in the propagation grid
    :return:
    """
    site_dim = 2
    site_dims = [site_dim] * L

    site_op = pauli.X
    bond_op = np.kron(pauli.X, pauli.Z)
    exdiag_thermal = exdiag.generate_thermal_state(beta, site_dims, repeat(site_op, L), repeat(bond_op, L - 1))
    # Setup for tMPS evolution
    state_compress_kwargs = {'method': 'svd', 'relerr': 1e-10, 'sites_step': 3}
    tmps_thermal, info = thermal.from_hamiltonian(beta, mpa_type, [site_op]*L, [bond_op]*(L-1),
                                                  nof_steps=nof_steps,
                                                  state_compression_kwargs=state_compress_kwargs,
                                                  op_compression_kwargs=None)
    if mpa_type == 'pmps':
        tmps_thermal = mp.pmps_to_mpo(tmps_thermal)
    normdiff = np.linalg.norm(exdiag_thermal - tmps_thermal.to_array_global().reshape(site_dim ** L, site_dim ** L))
    assert normdiff < 1e-8


@pytest.mark.slow
@pytest.mark.parametrize("beta, nof_steps", [(0.1, 10), (1, 100)])
@pytest.mark.parametrize("mpa_type", ['mpo', 'pmps'])
def test_unequal_site_dim_thermal_state_generation(beta, nof_steps, mpa_type):
    """
        Pytest test for generating a thermal state via imaginary time evolution for a chain with unequal site dimensions
        for spin operators with local dimensions [2, 4, 4] in the chain
    :param beta: Inverse temperature of the final state
    :param nof_steps: number of steps in the propagation grid
    :return:
    """
    site_dims = [2, 4, 4]

    site_ops = [pauli.X, np.kron(pauli.X, pauli.X), np.kron(pauli.X, pauli.X)]
    dim4_op = np.kron(pauli.X, pauli.Z)
    bond_ops = [np.kron(pauli.X, dim4_op), np.kron(dim4_op, dim4_op)]
    exdiag_thermal = exdiag.generate_thermal_state(beta, site_dims, site_ops, bond_ops)
    # Setup for tMPS evolution
    state_compress_kwargs = {'method': 'svd', 'relerr': 1e-10}
    tmps_thermal, info = thermal.from_hamiltonian(beta, mpa_type, site_ops, bond_ops,
                                                  nof_steps=nof_steps,
                                                  state_compression_kwargs=state_compress_kwargs,
                                                  op_compression_kwargs=None)
    global_dim = np.prod(site_dims, dtype=int)
    if mpa_type == 'pmps':
        tmps_thermal = mp.pmps_to_mpo(tmps_thermal)
    normdiff = np.linalg.norm(np.abs(exdiag_thermal - tmps_thermal.to_array_global().reshape(global_dim, global_dim)))
    assert normdiff < 1e-8
