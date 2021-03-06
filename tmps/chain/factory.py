"""
    Contains factory functions for generation of chain tmp objects (real time evolution objects, TMPS, TMPO, TPMPS)
"""

from tmps.chain.tmps import TMPS
from tmps.chain.tpmps import TPMPS
from tmps.chain.tmpo import TMPO
from tmps.utils.shape import check_shape


def from_hamiltonian(psi_0, mpa_type, h_site, h_bond, tau=0.01, state_compression_kwargs=None,
                     op_compression_kwargs=None, second_order_trotter=False, t0=0, psi_0_compression_kwargs=None,
                     force_op_cform=False):
    """
        Factory function for TMP-objects
        :param psi_0: Initial state as MPArray. Need not be normalized, as it is normalized before propagation
        :param mpa_type: Type of mpa to propagate, supported are 'mps', 'mpo', and 'pmps'
        :param h_site: Iterator over local site Hamiltonians. If a single numpy ndarray is passed
                       this element is broadcast over all sites
        :param h_bond: Iterator over bond Hamiltonians. If a single numpy ndarray is passed
                       this element is broadcast over all bonds
        :param tau: Timestep for each invocation of evolve
        :param state_compression_kwargs: Arguments for mpa compression after each dot product
                                         Allowed arguments are:
                                         'method': Compression method (default is 'svd', see mpnum docs)
                                         'rank': Maximum allowed rank (default is None, see mpnum docs)
                                         'relerr': Target relative error for svd compression (default is 1e-10)
                                         'stable': If the slower, but more stable variant of the svd should be chosen
                                                   directly instead of trying the faster one first (default is False)
                                         'canonicalize_every_step': If every compression should be accompanied by a
                                                                    canonicalization (default is True)
                                         'canonicalize_last_step': Force canonicalize the last compression step
                                                                   in one timestep (default is True)
                                         For pmps evomlutions, the following arguments are also recognized:
                                         'sites_step': After how many compressions of the chain bonds
                                                       should the local tensors of a pmps be compressed
                                                       (default is 3)
                                         'sites_relerr': To which target relative error should the site tensors be
                                                         compressed (default is 1e-12)
                                         'sites_rank': Maximum allowed rank between the pmps site tensors
                                         'sites_stable': If the slower, but more stable variant of the svd should be
                                                         chosen directly for the pmps site compression instead of
                                                         trying the faster one first (default is False)
        :param op_compression_kwargs: Arguments for trotter step operator pre-compression
                                      Allowed Arguments are:
                                      'method': Compression method (default is 'svd', see mpnum docs)
                                      'rank': Maximum allowed rank (default is None, see mpnum docs)
                                      'relerr': Target relative error for svd compression (default is 1e-12)
                                      'stable': If the slower, but more stable variant of the svd should be chosen
                                                   directly instead of trying the faster one first (default is False)
        :param second_order_trotter: Switch to use second order instead of fourth order trotter if desired
                                     By default fourth order Trotter is used
        :param force_op_cform: Force canonical form of time evolution operators to match state canonical form.
                               If not set True, a default is used, depending on the state canonical form.
                               Canonical form is also forced if state is not canonicalized before every compression.
        :param psi_0_compression_kwargs: Arguments for mpa compression before the time evolution proper
                                         Allowed arguments are:
                                         'method': Compression method (default is 'svd', see mpnum docs)
                                         'rank': Maximum allowed rank (default is None, see mpnum docs)
                                         'relerr': Target relative error for svd compression (default is 1e-10)
                                         'stable': If the slower, but more stable variant of the svd should be chosen
                                                   directly instead of trying the faster one first (default is False)
                                         For pmps evomlutions, the following arguments are also recognized:
                                         'sites_step': After how many compressions of the chain bonds
                                                       should the local tensors of a pmps be compressed
                                                       (default is 3)
                                         'sites_relerr': To which target relative error should the site tensors be
                                                         compressed (default is 1e-12)
                                         'sites_rank': Maximum allowed rank between the pmps site tensors
                                         'sites_stable': If the slower, but more stable variant of the svd should be
                                                         chosen directly for the pmps site compression instead of
                                                         trying the faster one first (default is False)
        :param t0: Initial time of the propagation
    :return: TMP object. If mpa_type is mps: TMPS obj., if mpa_type is mpo: TMPO obj.,
                         if mpa_type is 'pmps': TPMPS obj.
    """
    if not check_shape(psi_0, mpa_type):
        raise AssertionError('MPA shape of the initial state is not compatible with the chosen mpa_type')
    if mpa_type == 'mps':
        return TMPS.from_hamiltonian(psi_0, False, False, h_site, h_bond, tau=tau,
                                     state_compression_kwargs=state_compression_kwargs,
                                     op_compression_kwargs=op_compression_kwargs,
                                     second_order_trotter=second_order_trotter, t0=t0,
                                     psi_0_compression_kwargs=psi_0_compression_kwargs,
                                     force_op_cform=force_op_cform)
    elif mpa_type == 'pmps':
        return TPMPS.from_hamiltonian(psi_0, True, False, h_site, h_bond, tau=tau,
                                      state_compression_kwargs=state_compression_kwargs,
                                      op_compression_kwargs=op_compression_kwargs,
                                      second_order_trotter=second_order_trotter, t0=t0,
                                      psi_0_compression_kwargs=psi_0_compression_kwargs,
                                      force_op_cform=force_op_cform)
    elif mpa_type == 'mpo':
        return TMPO.from_hamiltonian(psi_0, False, True, h_site, h_bond, tau=tau,
                                     state_compression_kwargs=state_compression_kwargs,
                                     op_compression_kwargs=op_compression_kwargs,
                                     second_order_trotter=second_order_trotter, t0=t0,
                                     psi_0_compression_kwargs=psi_0_compression_kwargs,
                                     force_op_cform=force_op_cform)
    else:
        raise AssertionError('Unrecognized mpa_type')


def from_hi(psi_0, mpa_type, hi_list, tau=0.01, state_compression_kwargs=None, op_compression_kwargs=None,
            second_order_trotter=False, t0=0, psi_0_compression_kwargs=None, force_op_cform=False):
    """
        Factory function for TMP-objects
        :param psi_0: Initial state as MPArray. Need not be normalized, as it is normalized before propagation
        :param mpa_type: Type of mpa to propagate, supported are 'mps', 'mpo', and 'pmps'
        :param hi_list: List/Tuple of all terms in the Hamiltonian H = sum_i hi, where hi is local to one bond
        :param tau: Timestep for each invocation of evolve
        :param state_compression_kwargs: see from_hamiltonian
        :param op_compression_kwargs: see from_hamiltonian
        :param second_order_trotter: Switch to use second order instead of fourth order trotter if desired
                                     By default fourth order Trotter is used
        :param force_op_cform: Force canonical form of time evolution operators to match state canonical form.
                               If not set True, a default is used, depending on the state canonical form.
                               Canonical form is also forced if state is not canonicalized before every compression.
        :param psi_0_compression_kwargs: see from_hamiltonian
        :param t0: Initial time of the propagation
    :return: TMP object. If mpa_type is mps: TMPS obj., if mpa_type is mpo: TMPO obj.,
                         if mpa_type is 'pmps': TPMPS obj.
    """
    if not check_shape(psi_0, mpa_type):
        raise AssertionError('MPA shape of the initial state is not compatible with the chosen mpa_type')
    if mpa_type == 'mps':
        return TMPS.from_hi(psi_0, False, False, hi_list, tau=tau, state_compression_kwargs=state_compression_kwargs,
                            op_compression_kwargs=op_compression_kwargs,
                            second_order_trotter=second_order_trotter, t0=t0,
                            psi_0_compression_kwargs=psi_0_compression_kwargs,
                            force_op_cform=force_op_cform)
    elif mpa_type == 'pmps':
        return TPMPS.from_hi(psi_0, True, False, hi_list, tau=tau, state_compression_kwargs=state_compression_kwargs,
                             op_compression_kwargs=op_compression_kwargs,
                             second_order_trotter=second_order_trotter, t0=t0,
                             psi_0_compression_kwargs=psi_0_compression_kwargs,
                             force_op_cform=force_op_cform)
    elif mpa_type == 'mpo':
        return TMPO.from_hi(psi_0, False, True, hi_list, tau=tau, state_compression_kwargs=state_compression_kwargs,
                            op_compression_kwargs=op_compression_kwargs,
                            second_order_trotter=second_order_trotter, t0=t0,
                            psi_0_compression_kwargs=psi_0_compression_kwargs,
                            force_op_cform=force_op_cform)
    else:
        raise AssertionError('Unrecognized mpa_type')
