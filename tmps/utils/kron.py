"""
    Convenience functions to generate generate product vectors/matrices/ndarrays using kronecker products
"""

import numpy as np


def generate_product_state(psi_i, L):
    """
        Generates product state chain |psi_i> x |psi_i> x ... x |psi_i> for L sites
    :param psi_i: single site state (numpy vector or matrix)
    :param L: number of sites
    :return: Generated product state as matrix/vector (same ndims as psi_i). Returns None if L is 0
    """
    if L == 0:
        return None
    psi = np.array([1.0])
    for i in range(L):
        psi = np.kron(psi, psi_i)
    return psi


def generate_product_ndarray(psi_i, L):
    """
        Generates product state chain |psi_i> x |psi_i> x ... x |psi_i> for L sites
    :param psi_i: single site state (numpy vector or matrix)
    :param L: number of sites
    :return: Generated product state as (global shape) ndarray. Returns None if L is 0
    """
    psi = generate_product_state(psi_i, L)
    if psi is None:
        return psi
    if len(psi_i.shape) == 1:
        new_shape = tuple(([psi_i.shape[0]] * L))
    elif len(psi_i.shape) == 2:
        new_shape = tuple(([psi_i.shape[0]] * L) + ([psi_i.shape[1]] * L))
    else:
        assert False
    return psi.reshape(new_shape)


def chain_product_state(psi_it):
    """
        Chains product states |psi[0]> x |psi[1]> x ... x |psi[L]> from all elements in psi-iterable
    :param psi_it: Iterable of single site states (vector or matrix)
    :return: Generated product state as matrix/vector (same ndims as elements in psi_it).
             If psi_it is empty it returns None
    """
    psi = None
    for index, state in enumerate(psi_it):
        if index == 0:
            psi = state
        else:
            psi = np.kron(psi, state)
    return psi


def chain_product_ndarray(psi_it):
    """
        Chains product states |psi[0]> x |psi[1]> x ... x |psi[L]> from all elements in psi-iterable
    :param psi_it: Iterable of single site states (vector or matrix)
    :return: Generated product state as (global shape) ndarray. If psi_it is empty it returns None
    """
    psi = None
    shapes = []
    for index, state in enumerate(psi_it):
        if index == 0:
            psi = state
        else:
            psi = np.kron(psi, state)
        shapes.append(state.shape)
    # TODO: Assert all have equal ndims
    if len(shapes[0]) == 1:
        new_shape = tuple([shape[0] for shape in shapes])
    elif len(shapes[0]) == 2:
        rows = []
        cols = []
        for shape in shapes:
            rows.append(shape[0])
            cols.append(shape[1])
        new_shape = tuple(rows + cols)
    else:
        assert False
    return psi.reshape(new_shape) if psi is not None else psi
