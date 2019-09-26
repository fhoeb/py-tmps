"""
    Tools for embedding states and Chain Hamiltonians on 1D-chains as MPArrays
"""
from tmps.utils.shape import get_shape_from_dims
import mpnum as mp
import numpy as np


# TODO: Improve _local_sum implementation to do fewer unneccessary add operations


def construct_hi(shape, h_site, h_bond):
    """
        Splits site local contributions of hamiltonian up in tow parts and constructs:
        hi_site = h_loc_i/2 + h_bond_i,i+1 + h_loc_i+1/2
        except at start and end of the chain, where the local operators contribute fully.
    :param shape: Shape of the state (or chain on which) to propagate (in mparray shape form).
                  Only the only axis 0 legs on each site are relevant.
                  This means, that shapes from mps, pmps and mpos are treated equally.
    :param h_site: Iterator over local site Hamiltonians.
    :param h_bond: Iterator over bond Hamiltonians.
    :return: List of all hi
    """
    L = len(shape)
    # contains all split up site contributions to hi
    hi_site = []
    # split up site local contributions and divide them among the bonds
    for site, h in enumerate(h_site):
        if site == 0:
            hi_site.append(np.kron(h, np.eye(shape[site + 1][0], dtype=complex)))
        elif site == L - 1:
            hi_site[site - 1] += np.kron(np.eye(shape[site - 1][0], dtype=complex), h)
        else:
            hi_site.append(np.kron(h / 2, np.eye(shape[site + 1][0], dtype=complex)))
            hi_site[site - 1] += np.kron(np.eye(shape[site - 1][0], dtype=complex), h / 2)
    # add the bond operators to the site local ones
    return [site_op + bond_op for site_op, bond_op in zip(hi_site, h_bond)]


def _local_sum(dims, hi_mpos, compression_relerr=1e-15):
    """
        Embeds the sum of bond local terms (nearest neighbor coupling operators) on a chain as MPArrays
    :param dims: Dimensions of the sites of the chain (axis 0 legs of mpnum shape) as tuple or list
    :param hi_mpos: Two-site operators represented by MPArrays, which are to be embedded on a chain
    :param compression_relerr: If not None, the result of the sum is compressed using svd compression for the
                               specified relative error
    :return: MPArray of hi_mpos embedded on a chain of shape *shape*
    """
    nof_sites = len(dims)
    # Nearest neighbor operators occupy 2 sites each
    op_width = 2
    local_terms = []
    for startpos, hi_mpo in enumerate(hi_mpos):
        left = [mp.factory.eye(startpos, dims[:startpos])] if startpos > 0 else []
        right = [mp.factory.eye(nof_sites - startpos - op_width, dims[startpos+op_width:])] \
            if nof_sites - startpos - op_width > 0 else []
        h_at_startpos = mp.chain(left + [hi_mpo] + right)
        local_terms.append(h_at_startpos)
    H = local_terms[0]
    for local_term in local_terms[1:]:
        H += local_term
    if compression_relerr is not None:
        H.compress(method='svd', relerr=1e-15)
    return H


def embed_hamiltonian_as_mpo(h_site, h_bond, compression_relerr=1e-15):
    """
        Constructs mpo representation of a Hamiltonian H given by site local and bond local operators:
        H = sum_i h_site_i + h_bond_i,i+1
    :param h_site: List of site local operators
    :param h_bond: List of bond operators
    :param compression_relerr: If not None, the result of the sum is compressed using svd compression for the
                               specified relative error
    :return: Hamiltonian H as MPArray
    """
    hi_mpos = []
    dims = []
    for site in h_site:
        assert site.shape[0] == site.shape[1]
        dims.append(site.shape[0])
    shape = get_shape_from_dims(dims)
    for i, hi_op in enumerate(construct_hi(shape, h_site, h_bond)):
        hi_mpo = mp.MPArray.from_array_global(hi_op.reshape(dims[i], dims[i + 1],
                                                            dims[i], dims[i + 1]), ndims=2)
        hi_mpos.append(hi_mpo)
    return _local_sum(dims, hi_mpos, compression_relerr=compression_relerr)


def embed_hi_as_mpo(dims, hi_list, compression_relerr=1e-15):
    """
        Constructs mpo representation of a Hamiltonian H given by bond local operators:
        H = \sum_i hi
    :param dims: Dimensions of the sites of the chain (axis 0 legs of mpnum shape) as tuple or list
    :param hi_list: Iterable for all (bond local) terms in the Hamiltonian
    :param compression_relerr: If not None, the result of the sum is compressed using svd compression for the
                               specified relative error
    :return: Hamiltonian H as MPArray
    """
    hi_mpos = []
    for i, hi_op in enumerate(hi_list):
        hi_mpo = mp.MPArray.from_array_global(hi_op.reshape(dims[i], dims[i + 1],
                                                            dims[i], dims[i + 1]), ndims=2)
        hi_mpos.append(hi_mpo)
    return _local_sum(dims, hi_mpos, compression_relerr=compression_relerr)
