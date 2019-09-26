import numpy as np
import mpnum as mp
from scipy.linalg import eigh


def isdiag(M):
    """
        Returns True, if the matrix M is population. Else False
    """
    return np.all(M == np.diag(np.diagonal(M)))


def purify(rho):
    """
        Purifies a quantum state represented by the density matrix rho of dimensions NxN.
        Returns an equivalent one dimensional pure state ndarray of dimension N^2.
    """
    assert rho.shape[0] == rho.shape[1]
    if not isdiag(rho):
        w, v = eigh(rho)
    else:
        w = np.diagonal(rho)
        v = np.identity(rho.shape[0])
    purified = np.zeros(rho.shape[0]**2)
    ancilla = np.identity(rho.shape[0])
    for index, eval in enumerate(w):
        purified += np.sqrt(eval) * np.kron(v[:, index], ancilla[:, index])
    return purified


def purify_to_ndarray(rho):
    """
        Purifies a quantum state represented by the density matrix rho of dimensions NxN.
        Returns an equivalent one dimensional pure state ndarray as (N,N)-Tensor
    """
    purified_state = purify(rho)
    return purified_state.reshape(rho.shape)


def purify_to_mparray(rho):
    """
        Purifies a quantum state represented by the density matrix rho of dimensions NxN.
        Returns an equivalent one dimensional pure state as pmps
    """
    purified_state = purify_to_ndarray(rho)
    return mp.MPArray.from_array_global(purified_state, ndims=2)


def purify_states(rho_list, to='mparray'):
    """
        Purifies a list of quantum states represented by the density matrices rho of dimensions NxN.
        Returns an equivalent one list dimensional pure state of specified kind ('mparray', 'ndarray', 'state')
    """
    purified_list = []
    for rho in rho_list:
        if to == 'mparray':
            purified_list.append(purify_to_mparray(rho))
        elif to == 'ndarray':
            purified_list.append(purify_to_ndarray(rho))
        elif to == 'state':
            purified_list.append(purify(rho))
        else:
            print('unrecognized return type')
            raise AssertionError
    return purified_list
