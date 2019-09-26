from mpnum.utils.extmath import matdot
import numpy as np
from scipy.linalg import svd


class LocalCompression:
    """Container for local compression parameters (so they don't have to be passed every time)"""
    def __init__(self, relerr=1e-10, rank=None, stable=False, direction=None, canonicalize=True,
                 reduced_canonicalization=True, **kwargs):
        """
            See the StarTMP real time propagator factory docstring for an explanation of the parameters
        """
        self.relerr = relerr
        self.rank = rank
        self.stable = stable
        self.direction = direction
        self.canonicalize = canonicalize
        self.reduced_canonicalization = reduced_canonicalization

    def compress(self, mpa, bond):
        """
            Compresses the mpa bond with index 'site'
        """
        ln, rn = mpa.canonical_form
        default_direction = 'left' if len(mpa) - rn > ln else 'right'
        direction = default_direction if self.direction is None else self.direction
        if direction == 'right':
            if self.canonicalize:
                if self.reduced_canonicalization:
                    mpa.canonicalize(right=bond + 1)
                else:
                    mpa.canonicalize(right=1)
            self._local_svd_r_compression(mpa, bond)
        elif direction == 'left':
            if self.canonicalize:
                if self.reduced_canonicalization:
                    mpa.canonicalize(left=bond + 1)
                else:
                    mpa.canonicalize(left=len(mpa) - 1)
            self._local_svd_l_compression(mpa, bond + 1)
        else:
            raise ValueError('{} is not a valid direction'.format(direction))

    def _local_svd_r_compression(self, mpa, site):
        """
            SVD for a single site in the mpa chain for a right canonicalized chain
        :return:
        """
        rank = max(mpa.ranks) if self.rank is None else self.rank
        ltens = mpa._lt[site]
        matshape = (-1, ltens.shape[-1])
        if not self.stable:
            try:
                u, sv, v = svd(ltens.reshape(matshape), lapack_driver='gesdd', full_matrices=False)
            except np.linalg.LinAlgError:
                u, sv, v = svd(ltens.reshape(matshape), lapack_driver='gesvd', overwrite_a=True,
                               full_matrices=False)
        else:
            u, sv, v = svd(ltens.reshape(matshape), lapack_driver='gesvd', overwrite_a=True,
                           full_matrices=False)
        if self.relerr is None:
            k_prime = min(rank, len(sv))

            u = u[:, :k_prime]
            sv = sv[:k_prime]
            v = v[:k_prime, :]

            rank_t = len(sv)
        else:
            svsum = np.cumsum(sv) / np.sum(sv)
            rank_relerr = np.searchsorted(svsum, 1 - self.relerr) + 1
            rank_t = min(ltens.shape[-1], u.shape[1], rank, rank_relerr)
        # If the compressed u is small enough, copy it and let the gc clean up the original u
        if u[:rank_t, :].size / u.size < 0.7:
            newtens = (u[:, :rank_t].reshape(ltens.shape[:-1] + (rank_t,)).copy(),
                       matdot(sv[:rank_t, None] * v[:rank_t, :], mpa._lt[site + 1]))
        else:
            newtens = (u[:, :rank_t].reshape(ltens.shape[:-1] + (rank_t,)),
                       matdot(sv[:rank_t, None] * v[:rank_t, :], mpa._lt[site + 1]))
        mpa._lt.update(slice(site, site + 2), newtens, canonicalization=('left', None))

    def _local_svd_l_compression(self, mpa, site):
        """
            SVD for a single site in the mpa chain for a left canonicalized chain
        :return:
        """
        rank = max(mpa.ranks) if self.rank is None else self.rank
        ltens = mpa._lt[site]
        matshape = (ltens.shape[0], -1)
        if not self.stable:
            try:
                u, sv, v = svd(ltens.reshape(matshape), lapack_driver='gesdd', full_matrices=False)
            except np.linalg.LinAlgError:
                u, sv, v = svd(ltens.reshape(matshape), lapack_driver='gesvd', overwrite_a=True,
                               full_matrices=False)
        else:
            u, sv, v = svd(ltens.reshape(matshape), lapack_driver='gesvd', overwrite_a=True,
                           full_matrices=False)
        if self.relerr is None:
            k_prime = min(rank, len(sv))

            u = u[:, :k_prime]
            sv = sv[:k_prime]
            v = v[:k_prime, :]

            rank_t = len(sv)
        else:
            svsum = np.cumsum(sv) / np.sum(sv)
            rank_relerr = np.searchsorted(svsum, 1 - self.relerr) + 1
            rank_t = min(ltens.shape[0], v.shape[0], rank, rank_relerr)
        # If the compressed v is small enough, copy it and let the gc clean up the original v
        if v[:rank_t, :].size / v.size < 0.7:
            newtens = (matdot(mpa._lt[site - 1], u[:, :rank_t] * sv[None, :rank_t]),
                       v[:rank_t, :].reshape((rank_t,) + ltens.shape[1:]).copy())
        else:
            newtens = (matdot(mpa._lt[site - 1], u[:, :rank_t] * sv[None, :rank_t]),
                       v[:rank_t, :].reshape((rank_t,) + ltens.shape[1:]))
        mpa._lt.update(slice(site - 1, site + 1), newtens, canonicalization=(None, 'right'))

    def update(self, relerr=1e-10, rank=None, direction=None, stable=False, canonicalize_every_step=True,
               reduced_canonicalization=True, **kwargs):
        self.relerr = relerr
        self.rank = rank
        self.stable = stable
        self.direction = direction
        self.canonicalize = canonicalize_every_step
        self.reduced_canonicalization = reduced_canonicalization


def local_svd_compression(mpa, bond, relerr=1e-10, rank=None, stable=False, direction=None, canonicalize=True,
                          reduced_canonicalization=True, **kwargs):
    """
        SVD for a single bond in the mpa chain
    """
    lc = LocalCompression(relerr=relerr, rank=rank, stable=stable, direction=direction, canonicalize=canonicalize,
                          reduced_canonicalization=reduced_canonicalization)
    lc.compress(mpa, bond)
