"""
    Test for the methods to probe a quantum system represented by an mparray
"""
import numpy as np
import mpnum as mp
import pytest
from tmps.utils import state_reduction_as_array, sandwich_state
import testutils.random as random
import tmps.utils.kron as kron
from itertools import repeat


@pytest.mark.fast
@pytest.mark.parametrize("trace_out", [1, 2])
@pytest.mark.parametrize("startsite", [-2, 0, 2])
@pytest.mark.parametrize("site_dim, L", [(2, 5), (2, 8)])
def test_oqs_reductions_mpo(trace_out, startsite, site_dim, L):
    """
        Pytest test for performing partial traces over a quantum system represented by an mpo
    :param trace_out: How many consecutive sites should be in the reduced system
    :param startsite: Where in the chain should the reduced system start
    :param site_dim: Local site dimension of the chain
    :param L: Length of the chain
    :return:
    """
    np.random.seed(seed=103)
    test_operator = np.random.rand(site_dim ** trace_out, site_dim ** trace_out)
    test_operator_mpo = mp.MPArray.from_array_global(test_operator.reshape(tuple([site_dim] * (2 * trace_out))),
                                                     ndims=2)
    local_states = random.generate_random_density_matrices(repeat(site_dim, L))
    # Generate reduced reference state, watch out for negative startsite here (convert it to a positive one)
    ref_startsite = startsite if startsite >= 0 else L+startsite
    reduced_density = kron.chain_product_state(local_states[ref_startsite:ref_startsite + trace_out])
    # generate reduced state from utils.probe
    mpo = mp.MPArray.from_array_global(kron.chain_product_ndarray(local_states), ndims=2)
    reduced = state_reduction_as_array(mpo, 'mpo', startsite=startsite, nof_sites=trace_out)
    assert np.linalg.norm(reduced_density - reduced) == pytest.approx(0)
    # Calculate expectation value for reference state
    sandwich_density = np.trace(reduced_density @ test_operator)
    # Calculate expectation value using utils.probe
    sandwich_reduced = sandwich_state(test_operator_mpo, mpo, 'mpo', startsite=startsite)
    assert sandwich_density - sandwich_reduced == pytest.approx(0)


@pytest.mark.fast
@pytest.mark.parametrize("trace_out", [1, 2])
@pytest.mark.parametrize("startsite", [-2, 0, 2])
@pytest.mark.parametrize("site_dim, L", [(2, 5), (2, 8)])
def test_oqs_reductions_mps(trace_out, startsite, site_dim, L):
    """
        Pytest test for performing partial traces over a quantum system represented by an mps
    :param trace_out: How many consecutive sites should be in the reduced system
    :param startsite: Where in the chain should the reduced system start
    :param site_dim: Local site dimension of the chain
    :param L: Length of the chain
    :return:
    """
    np.random.seed(seed=103)
    test_operator = np.random.rand(site_dim ** trace_out, site_dim ** trace_out)
    test_operator_mpo = mp.MPArray.from_array_global(test_operator.reshape(tuple([site_dim] * (2 * trace_out))),
                                                     ndims=2)
    local_states = random.generate_random_state_vectors(repeat(site_dim, L))
    # Generate reduced reference state, watch out for negative startsite here (convert it to a positive one)
    ref_startsite = startsite if startsite >= 0 else L+startsite
    reduced_nparray = kron.chain_product_state(local_states[ref_startsite:ref_startsite + trace_out])
    reduced_density = np.outer(reduced_nparray, reduced_nparray.conj())
    # generate reduced state from utils.probe
    mps = mp.MPArray.from_array_global(kron.chain_product_ndarray(local_states), ndims=1)
    reduced = state_reduction_as_array(mps, 'mps', startsite=startsite, nof_sites=trace_out)
    assert np.linalg.norm(reduced_density - reduced) == pytest.approx(0)
    # Calculate expectation value for reference state
    sandwich_density = np.trace(reduced_density @ test_operator)
    # Calculate expectation value using utils.probe
    sandwich_reduced = sandwich_state(test_operator_mpo, mps, 'mps', startsite=startsite)
    assert sandwich_density - sandwich_reduced == pytest.approx(0)


@pytest.mark.fast
@pytest.mark.parametrize("trace_out", [1, 2])
@pytest.mark.parametrize("startsite", [-2, 0, 2])
@pytest.mark.parametrize("site_dim, L", [(2, 5), (2, 8)])
def test_oqs_reductions_pmps(trace_out, startsite, site_dim, L):
    """
        Pytest test for performing partial traces over a quantum system represented by a pmps
    :param trace_out: How many consecutive sites should be in the reduced system
    :param startsite: Where in the chain should the reduced system start
    :param site_dim: Local site dimension of the chain
    :param L: Length of the chain
    :return:
    """
    np.random.seed(seed=103)
    test_operator = np.random.rand(site_dim ** trace_out, site_dim ** trace_out)
    test_operator_mpo = mp.MPArray.from_array_global(test_operator.reshape(tuple([site_dim] * (2 * trace_out))),
                                                     ndims=2)
    local_states = random.generate_random_state_vectors(repeat(site_dim, L))
    # Generate reduced reference state, watch out for negative startsite here (convert it to a positive one)
    ref_startsite = startsite if startsite >= 0 else L+startsite
    reduced_nparray = kron.chain_product_state(local_states[ref_startsite:ref_startsite + trace_out])
    reduced_density = np.outer(reduced_nparray, reduced_nparray.conj())
    # generate reduced state from utils.probe
    mps = mp.MPArray.from_array_global(kron.chain_product_ndarray(local_states), ndims=1)
    pmps = mp.mps_to_pmps(mps)
    reduced = state_reduction_as_array(pmps, 'pmps', startsite=startsite, nof_sites=trace_out)
    assert np.linalg.norm(reduced_density - reduced) == pytest.approx(0)
    # Calculate expectation value for reference state
    sandwich_density = np.trace(reduced_density @ test_operator)
    # Calculate expectation value using utils.probe
    sandwich_reduced = sandwich_state(test_operator_mpo, pmps, 'pmps', startsite=startsite)
    assert sandwich_density - sandwich_reduced == pytest.approx(0)