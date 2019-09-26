import mpnum as mp
import numpy as np
from tmps.utils.purify import purify_to_mparray


def to_mparray(psi, mpa_type):
    """
        Convert numpy ndarray to mpnum MPArray. Allowed mpa_types are 'mps', 'mpo', and 'pmps'
    """
    if isinstance(psi, np.ndarray):
        if mpa_type == 'mps':
            if len(psi.shape) != 1:
                raise AssertionError('Can only convert vectors to mps')
            return mp.MPArray.from_array(psi)
        elif mpa_type == 'mpo':
            if len(psi.shape) == 1:
                return mp.MPArray.from_array_global(np.outer(psi, psi.conj()), ndims=2)
            elif len(psi.shape) == 2:
                return mp.MPArray.from_array_global(psi, ndims=2)
            else:
                raise AssertionError('Can only convert vectors or matrices to mpo')
        elif mpa_type == 'pmps':
            if len(psi.shape) == 1:
                return purify_to_mparray(np.outer(psi, psi.conj()))
            elif len(psi.shape) == 2:
                return purify_to_mparray(psi)
            else:
                raise AssertionError('Can only convert vectors or matrices to pmps')
        else:
            raise AssertionError('Unrecognized mpa_type')
    elif isinstance(psi, mp.MPArray):
        pass
    else:
        raise AssertionError('Unrecognized data type for psi')


def to_ndarray(psi, mpa_type):
    """
        Convert mpnum MPArray to numpy ndarray. Allowed mpa_types are 'mps', 'mpo', and 'pmps'
    """
    if isinstance(psi, mp.MPArray):
        if mpa_type == 'mps':
            return psi.to_array()
        elif mpa_type == 'mpo':
            return psi.to_array_global()
        elif mpa_type == 'pmps':
            return mp.pmps_to_mpo(psi).to_array_global()
        else:
            raise AssertionError('Unrecognized mpa_type')
    elif isinstance(psi, np.ndarray):
        pass
    else:
        raise AssertionError('Unrecognized data type for psi')


