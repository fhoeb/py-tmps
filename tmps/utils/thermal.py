import numpy as np
import mpnum as mp
from scipy.linalg import expm
from tmps.utils.purify import purify_states, isdiag
from tmps.utils.cform import canonicalize_to


def get_thermal_state(beta, h_site, mpa_type='mpo', as_type='mparray', to_cform=None):
    """
        Generates a thermal state e^(-beta*H_i)/Z in pmps form or mpo form site local Hamiltonians (H_i) without
        couplings.
    :param beta: Inverse temperature
    :param h_site: local operator(s) of Hamiltonian as single numpy array or iterable of numpy arrays for each site.
                   If passed as 2d arrays, they are treated as full matrices. If they are passed as
                   vectors, they are treated as diagonal elements of a matrix
    :param mpa_type: Type of mpa to evolve (allowed are 'pmps' and 'mpo')
    :param as_type: 'mparray' means as one combined mparray.
                    'mparray_list' as list of individual mparrays in the same order as the h_site were passed
                    'ndarray' returns the states as a list of ndarrays in the same order as the h_site were passed
    :param to_cform: Desired canonical form of the mparray (if as_type was selected to be 'mparray')
    :return: thermal state in pmps or mpo form (Any ancilla sites have the same dimension as the physical ones),
             info object from the propagation (if beta was set 0 it returns an empty dict)
    """
    assert mpa_type == 'mpo' or mpa_type == 'pmps'
    thermal_states = []
    if isinstance(h_site, np.ndarray):
        if len(h_site.shape) == 2:
            if not isdiag(h_site):
                state = expm(-beta*h_site)
            else:
                state = np.diag(np.exp(-beta * np.diag(h_site)))
        elif len(h_site.shape) == 1:
            state = np.diag(np.exp(-beta * h_site))
        else:
            raise AssertionError('Passed numpy array must either be of matrix or vector shape')
        thermal_states.append(state)
    else:
        try:
            for site in h_site:
                if len(site.shape) == 2:
                    if not isdiag(site):
                        state = expm(-beta*site)
                        thermal_states.append(state/np.trace(state))
                    else:
                        state = np.diag(np.exp(-beta*np.diag(site)))
                        thermal_states.append(state/np.trace(state))
                elif len(site.shape) == 1:
                    state = np.diag(np.exp(-beta * site))
                    thermal_states.append(state / np.trace(state))
                else:
                    raise AssertionError('Passed numpy array(s) must either be of matrix or vector shape')
        except TypeError:
            raise AssertionError('h_site must be single numpy array or iterable of numpy arrays')
    if as_type == 'ndarray':
        return thermal_states
    elif as_type == 'mparray':
        if mpa_type == 'pmps':
            state = mp.chain(purify_states(thermal_states, to='mparray'))
        elif mpa_type == 'mpo':
            state = mp.chain([mp.MPArray.from_array_global(state, ndims=2) for state in thermal_states])
        else:
            raise AssertionError('Invalid mpa_type')
        canonicalize_to(state, to_cform=to_cform)
        return state
    elif as_type == 'mparray_list':
        if mpa_type == 'pmps':
            return purify_states(thermal_states, to='mparray')
        elif mpa_type == 'mpo':
            return [mp.MPArray.from_array_global(state, ndims=2) for state in thermal_states]
        else:
            raise AssertionError('Invalid mpa_type')
    else:
        raise AssertionError('Unrecognized return type')

