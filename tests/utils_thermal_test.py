import tmps.utils.fock as fock
from tmps.utils.thermal import  get_thermal_state
import numpy as np
import mpnum as mp
import pytest


@pytest.mark.fast
@pytest.mark.parametrize("L, dim", [(2, 7), (3, 5), (4, 5), (5, 5), (7, 6)])
@pytest.mark.parametrize("beta", [1, 10])
def test_pmps_thermal_state(L, dim, beta):
    """
        Pytest test for calculating the norm  of nearest neighbor Hamiltonians
        with equal site dimension
    :param L: Size of the chain
    :return:
    """
    n = fock.n(dim)
    thermal_exact = [np.diag(np.exp(-beta*np.diag(n)))/ np.sum(np.exp(-beta*np.diag(n)))]*L
    thermal_test = get_thermal_state(beta, [n]*L, 'pmps' , as_type='mparray_list', to_cform=None)
    normdist = []
    for exact, test in zip(thermal_exact, thermal_test):
        normdist.append(np.linalg.norm(exact - mp.MPArray.to_array_global(mp.pmps_to_mpo(test))))
    assert np.max(normdist) < 1e-8