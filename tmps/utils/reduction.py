"""
    Methods to probe parts of a quantum state described by a mparray
"""
import mpnum as mp
import numpy as np


def state_reduction(psi, mpa_type, startsite=0, nof_sites=1):
    """
        Calculates reduced state as mpo for psi as mps, mpo or pmps
        for site 1 to site nof_sites from the start of the chain:
        So we get a reduced state for the indices s_i,..,s_{nof_sites}
    :param psi: State to generate reduced state from as mps, pmps or mpo
    :param mpa_type: mpa type of psi (mps, pmo or pmps)
    :param startsite: first site of the reduced state (may be negative, indexing works like for python lists)
    :param nof_sites: number of sites up to which (starting from startsite) the reduced state should be generated
    :return: reduced state as mpo
    """
    assert len(psi) > 0
    if startsite < 0:
        # Out of bounds index check
        assert startsite >= -len(psi)
        # Convert negative startsite to a positive one
        startsite = len(psi)+startsite
    # Out of bounds index check
    assert startsite + nof_sites <= len(psi)
    if mpa_type == 'mps':
        return next(mp.reductions_mps_as_mpo(psi, width=nof_sites, startsites=range(startsite, startsite+1)))
    elif mpa_type == 'mpo':
        return next(mp.reductions_mpo(psi, width=nof_sites, startsites=range(startsite, startsite+1)))
    elif mpa_type == 'pmps':
        return mp.pmps_to_mpo(next(mp.reductions_pmps(psi, width=nof_sites, startsites=range(startsite, startsite+1))))
    else:
        raise AssertionError('Invalid propagator')


def state_reduction_as_ndarray(psi, mpa_type, startsite=0, nof_sites=1):
    """
        Calculates reduced state as numpy ndarray with indices in global order for psi as mps, mpo or pmps
        for site 1 to site nof_sites from the start of the chain:
        So we get a reduced state for the indices s_i,..,s_{nof_sites}
    :param psi: State to generate reduced state from as mps, pmps or mpo
    :param mpa_type: mpa type of psi (mps, pmo or pmps)
    :param startsite: first site of the reduced state
    :param nof_sites: number of sites up to which (starting from startsite) the reduced state should be generated
    :return: reduced state as numpy ndarray for indices in global order:
             psi_red^{s_i, .., s_{nof_sites}, s_i, .., s_{nof_sites}}
    """
    reduced_mpa = state_reduction(psi, mpa_type, startsite=startsite, nof_sites=nof_sites)
    return reduced_mpa.to_array_global()


def state_reduction_as_array(psi, mpa_type, startsite=0, nof_sites=1):
    """
        Calculates reduced state as numpy array with indices in matrix form for psi as mps, mpo or pmps
        for site 1 to site nof_sites from the start of the chain:
        So weg get a reduced state for the indices s_i,..,s_{nof_sites}
    :param psi: State to generate reduced state from as mps, pmps or mpo
    :param mpa_type: mpa type of psi (mps, pmo or pmps)
    :param startsite: first site of the reduced state (may be negative, indexing works like for python lists)
    :param nof_sites: number of sites up to which (starting from startsite) the reduced state should be generated
    :return: reduced state as numpy array for indices in matrix form:
             psi_red^{(s_i, .., s_{nof_sites}), (s_i, .., s_{nof_sites})}
    """
    reduced_mpa = state_reduction(psi, mpa_type, startsite=startsite, nof_sites=nof_sites)
    # Since reduced states are quadratic matrices, both axes have the same size
    axis_size = np.prod([site_shape[0] for site_shape in reduced_mpa.shape], dtype=int)
    return reduced_mpa.to_array_global().reshape(axis_size, axis_size)


def reduction(propagator, startsite=0, nof_sites=1):
    """
        Calculates reduced state as mpo for the psi_t of a TMPSPropagator object
        for site 1 to site nof_sites from the start of the chain:
        So weg get a reduced state for the indices s_i,..,s_{nof_sites}
    :param propagator: TMPSPropagator object from whcih we take the state psi_t for generation of the reduced state
    :param startsite: first site of the reduced state (may be negative, indexing works like for python lists)
    :param nof_sites: number of sites up to which (starting from startsite) the reduced state should be generated
    :return: reduced state as mpo
    """
    assert len(propagator.psi_t) > 0
    if startsite < 0:
        # Out of bounds index check
        assert startsite >= -len(propagator.psi_t)
        # Convert negative startsite to a positive one
        startsite = len(propagator.psi_t)+startsite
    # Out of bounds index check
    assert startsite + nof_sites <= len(propagator.psi_t)
    if propagator.mpa_type == 'mps':
        return next(mp.reductions_mps_as_mpo(propagator.psi_t, width=nof_sites,
                                             startsites=range(startsite, startsite+1)))
    elif propagator.mpa_type == 'mpo':
        return next(mp.reductions_mpo(propagator.psi_t, width=nof_sites,
                                      startsites=range(startsite, startsite+1)))
    elif propagator.mpa_type == 'pmps':
        return mp.pmps_to_mpo(next(mp.reductions_pmps(propagator.psi_t, width=nof_sites,
                                                      startsites=range(startsite, startsite+1))))
    else:
        raise AssertionError('Invalid propagator')


def reduction_as_ndarray(propagator, startsite=0, nof_sites=1):
    """
        Calculates reduced state as numpy ndarray with indices in global order for psi as mps, mpo or pmps
        for site 1 to site nof_sites from the start of the chain:
        So weg get a reduced state for the indices s_i,..,s_{nof_sites}
    :param propagator: TMPSPropagator object from which we take the state psi_t for generation of the reduced state
    :param startsite: first site of the reduced state (may be negative, indexing works like for python lists)
    :param nof_sites: number of sites up to which (starting from startsite) the reduced state should be generated
    :return: reduced state as numpy ndarray for indices in global order:
             psi_red^{s_i, .., s_{nof_sites}, s_i, .., s_{nof_sites}}
    """
    reduced_mpa = reduction(propagator, startsite=startsite, nof_sites=nof_sites)
    return reduced_mpa.to_array_global()


def reduction_as_array(propagator, startsite=0, nof_sites=1):
    """
        Calculates reduced state as numpy array with indices in matrix form for psi as mps, mpo or pmps
        for site 1 to site nof_sites from the start of the chain:
        So weg get a reduced state for the indices s_i,..,s_{nof_sites}
    :param propagator: TMPSPropagator object from which we take the state psi_t for generation of the reduced state
    :param startsite: first site of the reduced state (may be negative, indexing works like for python lists)
    :param nof_sites: number of sites up to which (starting from startsite) the reduced state should be generated
    :return: reduced state as numpy array for indices in matrix form:
             psi_red^{(s_i, .., s_{nof_sites}), (s_i, .., s_{nof_sites})}
    """
    reduced_mpa = reduction(propagator, startsite=startsite, nof_sites=nof_sites)
    # Since reduced states have quadratic matrices, both axes have the same size
    axis_size = np.prod([site_shape[0] for site_shape in reduced_mpa.shape])
    return reduced_mpa.to_array_global().reshape(axis_size, axis_size)


def sandwich_state(op, psi, mpa_type, startsite=0):
    """
        Calculates the expectation value of an operator in mpo form with a reduced state: <psi_red| mpo |psi_red>.
        nof_sites can be inferred from len(mpo) here
    :param op: Operator for which to take expectation value as mpo
    :param psi: State from which to generate reduced state |psi_red>
    :param startsite: first site of the reduced state
    :param mpa_type: mpa type of psi (mps, pmo or pmps)
    :return: expectation value <psi_red| mpo |psi_red>
    """
    reduced = state_reduction(psi, mpa_type, startsite=startsite, nof_sites=len(op))
    # TODO: maybe compress first after dot?
    return mp.trace(mp.dot(op, reduced))


def sandwich_state_as_array(op, psi, mpa_type, startsite=0, nof_sites=1):
    """
        Calculates the expectation value of an operator as numpy array with a reduced state: <psi_red| op |psi_red>
    :param op: Operator for which to take expectation value as numpy array. Specifically it must be passed
                in matrix form O^((s_i, .., s_{nof_sites}), (s_i, .., s_{nof_sites}))
    :param psi: State from which to generate reduced state |psi_red>
    :param mpa_type: mpa type of psi (mps, pmo or pmps)
    :param startsite: first site of the reduced state (may be negative, indexing works like for python lists)
    :param nof_sites: number of sites up to which (starting from startsite) the reduced state should be generated
    :return: expectation value <psi_red| op |psi_red>
    """
    reduced = state_reduction_as_array(psi, mpa_type, startsite=startsite, nof_sites=nof_sites).reshape(op.shape[0],
                                                                                                        op.shape[1])
    return np.trace(op @ reduced)


def sandwich(op, propagator, startsite=0):
    """
        Calculates the expectation value of an operator in mpo form with a reduced state: <psi_t_red| mpo |psi_t_red>.
        nof_sites can be inferred from len(mpo) here
    :param op: Operator for which to take expectation value as mpo
    :param propagator: TMPSPropagator object. The reduced state |psi_red> is generated from propagator.psi_t
    :param startsite: first site of the reduced state (may be negative, indexing works like for python lists)
    :return: expectation value <psi_t_red| mpo |psi_t_red>
    """
    reduced = reduction(propagator, startsite=startsite, nof_sites=len(op))
    # TODO: maybe compress first after dot?
    return mp.trace(mp.dot(op, reduced))


def sandwich_as_array(op, propagator, startsite=0, nof_sites=1):
    """
        Calculates the expectation value of an operator as numpy array with a reduced state: <psi_t_red| op |psi_t_red>
    :param op: Operator for which to take expectation value as numpy array. Specifically it must be passed
                in matrix form O^((s_i, .., s_{nof_sites}), (s_i, .., s_{nof_sites}))
    :param propagator: TMPSPropagator object. The reduced state |psi_red> is generated from propagator.psi_t
    :param startsite: first site of the reduced state (may be negative, indexing works like for python lists)
    :param nof_sites: number of sites up to which (starting from startsite) the reduced state should be generated
    :return: expectation value <psi_t_red| op |psi_t_red>
    """
    reduced = reduction_as_array(propagator, startsite=startsite, nof_sites=nof_sites).reshape(op.shape[0],
                                                                                               op.shape[1])
    return np.trace(op @ reduced)
