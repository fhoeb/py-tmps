from tmps.utils.cform import canonicalize_to
import numpy as np
from scipy.linalg import svd


def compress(mpa, mpa_type, relerr=1e-10, rank=None, stable=False, to_cform=None, sites_relerr=1e-10, sites_rank=None,
             sites_stable=False, **kwargs):
    """
        Compresses mpa of specified type
    :param mpa: MPArray
    :param mpa_type: either 'mps', 'mpo', 'pmps'
    :param relerr: relative error for the compression
    :param rank: maximum rank of the compressed state
    :param stable: If the stable but slower variant of the svd compression should be used directly instead of trying the
                   possibly faster one first
    :param to_cform: target canonical form ('left', 'right' or None)
    :param sites_relerr: For pmps: Relative error for the compression of the site local tensors
    :param sites_rank: For pmps: Maximum rank for the compressione of the site local tensors
    :param sites_stable: For pmps: Same as stable bot for the local tensor compression
    """
    if mpa_type == 'pmps':
        compress_pmps(mpa, relerr=relerr, rank=rank, stable=stable, sites_relerr=sites_relerr, sites_rank=sites_rank,
                      sites_stable=sites_stable, to_cform=to_cform)
    elif mpa_type == 'mps' or mpa_type == 'pmps':
        compress_mpa(mpa, relerr=relerr, rank=rank, to_cform=to_cform)
    else:
        raise AssertionError('Unknown mpa_type')


def compress_pmps(pmps, relerr=1e-10, rank=None, stable=False, to_cform=None, sites_relerr=1e-10, sites_rank=None,
                  sites_stable=False, **kwargs):
    """
        Compresses pmps
    :param pmps: MPArray (of pmps form)
    :param relerr: relative error for the compression
    :param rank: maximum rank of the compressed state
    :param stable: If the stable but slower variant of the svd compression should be used directly instead of trying the
                   possibly faster one first
    :param to_cform: target canonical form ('left', 'right' or None)
    :param sites_relerr: For pmps: Relative error for the compression of the site local tensors
    :param sites_rank: For pmps: Maximum rank for the compressione of the site local tensors
    :param sites_stable: For pmps: Same as stable bot for the local tensor compression
    """
    compress_mpa(pmps, relerr=relerr, rank=rank, stable=stable, to_cform=to_cform)
    compress_pmps_sites(pmps, relerr=sites_relerr, rank=sites_rank, stable=sites_stable)


def compress_mpa(mpa, relerr=1e-10, rank=None, stable=False, to_cform=None, **kwargs):
    """
        Compresses mpa (either mps or mpo)
    :param mpa: MPArray
    :param relerr: relative error for the compression
    :param rank: maximum rank of the compressed state
    :param stable: If the stable but slower variant of the svd compression should be used directly instead of trying the
                   possibly faster one first
    :param to_cform: target canonical form ('left', 'right' or None)
    """
    mpa.compress(relerr=relerr, rank=rank, stable=stable, direction=to_cform)


def compress_pmps_sites(pmps, relerr=1e-10, rank=None, stable=False, to_cform=None):
    """
        Compresses pmps local sites
    :param pmps: MPArray (of pmps form)
    :param relerr: relative error for the compression
    :param rank: maximum rank of the compressed state
    :param to_cform: target canonical form ('left', 'right' or None)
    :param stable: If the stable but slower variant of the svd compression should be used directly instead of trying the
                   possibly faster one first
    """
    for site, lt in enumerate(pmps._lt):
        new_lt = _compressed_lt(lt, relerr, rank, stable)
        pmps._lt.update(site, new_lt, canonicalization=None)
    canonicalize_to(pmps, to_cform=to_cform)


def _compressed_lt(lt, relerr, rank, stable):
    """
        Compresses the local tensor lt in the case of a pmps
    """
    # Get sizes of virtual (i, j) and physical (m, n) legs
    i, m, n, j = lt.shape
    if not stable:
        try:
            u, sv, v = svd(lt.reshape(i*m, n*j), lapack_driver='gesdd', full_matrices=False)
        except np.linalg.LinAlgError:
            u, sv, v = svd(lt.reshape(i*m, n*j), lapack_driver='gesvd', overwrite_a=True, full_matrices=False)
    else:
        u, sv, v = svd(lt.reshape(i*m, n*j), lapack_driver='gesvd', overwrite_a=True, full_matrices=False)
    if relerr is None:
        k_prime = min(rank, len(sv))
        u = u[:, :k_prime]
        sv = sv[:k_prime]
        v = v[:k_prime, :]
        rank_t = len(sv)
    else:
        svsum = np.cumsum(sv) / np.sum(sv)
        rank_relerr = np.searchsorted(svsum, 1 - relerr) + 1
        rank_t = min(len(sv), rank, rank_relerr) if rank is not None else min(len(sv), rank_relerr)
    return np.dot(u[:, :rank_t], sv[:rank_t, None] * v[:rank_t, :]).reshape((i, m, n, j))
