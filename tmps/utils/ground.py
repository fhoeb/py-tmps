import numpy as np
import mpnum as mp
from itertools import repeat


def get_number_ground_state(dims):
    """
        Embeds ground states: (1 0 ... 0).T (of possibly different dimensions) onto a chain as mparrays
    :param dims: List of Hilbert space dimensions of all ground states
    :return: MPArray which contains psi_0[0], psi_0[1], ... psi_0[L] embedded in a chain of length L
             where psi_0[i] is a ground state of Hilbert space dimension dims[i]. Returns None if dims is empty
    """
    psi = []
    if len(dims) == 0:
        return None
    for dim in dims:
        next_psi = np.zeros(dim)
        next_psi[0] = 1
        psi.append(next_psi)
    return mp.MPArray.from_kron(psi)


def broadcast_number_ground_state(dim, L):
    """
        Broadcasts a ground state: (1 0 ... 0).T (dimension dim) onto a chain of length L as mparrays
    :param dim: Hilbert space dimension of all ground states
    :param L: Size of the chain to broadcast state onto
    :return: MPArray with L copies of the ground state specified above, ranks 1 between them. Returns None if L is <= 0
    """
    if L <= 0:
        return None
    psi = np.zeros(dim)
    psi[0] = 1
    return mp.MPArray.from_kron(repeat(psi, L))
