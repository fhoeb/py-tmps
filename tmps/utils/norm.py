"""
    Tools for calculating and using the norm of a Chain type Hamiltonian
"""
from tmps.utils.hamiltonian import embed_hamiltonian_as_mpo, embed_hi_as_mpo
import mpnum as mp


def get_norm_of_hamiltonian(h_site, h_bond, compression_relerr=None):
    """
        Calculates norm of a Hamiltonian H = sum_i h_site_i + h_bond_i,i+1
    :param h_site: List of site local operators
    :param h_bond: List of bond operators
    :param compression_relerr: If not None, the result of the sum is compressed using svd compression for the
                               specified relative error
    :return: Norm of the Hamiltonian
    """
    return mp.norm(embed_hamiltonian_as_mpo(h_site, h_bond, compression_relerr=compression_relerr))


def get_norm_of_hi(dims, hi_list, compression_relerr=None):
    """
        Calculates norm of a Hamiltonian H = sum_i hi,i+1
    :param dims: Dimensions of the sites of the chain (axis 0 legs of mpnum shape) as tuple or list
    :param hi_list: List/Tuple for all coupling terms in the Hamiltonian
    :param compression_relerr: If not None, the result of the sum is compressed using svd compression for the
                               specified relative error
    :return: Norm of the Hamiltonian
    """
    return mp.norm(embed_hi_as_mpo(dims, hi_list, compression_relerr=compression_relerr))
