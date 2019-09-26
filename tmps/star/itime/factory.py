"""
    Main interface factory functions for imaginary time evolution propagators (ITMPS, ITMPO, ITPMPS)
"""

from tmps.star.itime.itmps import StarITMPS
from tmps.star.itime.itpmps import StarITPMPS
from tmps.star.itime.itmpo import StarITMPO
import numpy as np
from tmps.utils.shape import check_shape


def from_hamiltonian(psi_0, mpa_type, system_index, h_site, h_bond, tau=0.01, state_compression_kwargs=None,
                     op_compression_kwargs=None, second_order_trotter=False, t0=0, psi_0_compression_kwargs=None,
                     track_trace=False):
    """
        Factory function for imaginary time TMP-objects (ITMPS, ITMPO, ITPMPS)
    :param psi_0: Initial state as MPArray. Need not be normalized, as it is normalized before propagation
    :param mpa_type: Type of MPArray to propagate, supported are mps, mpo, and pmps
    :param system_index: Index of the system site in the chain (place of the system site operator in the h_site list)
    :param h_site: Iterator over local site Hamiltonians. If a single numpy ndarray is passed
                   this element is broadcast over all sites
    :param h_bond: Iterator over coupling Hamiltonians.
                   Ordered like this:
                       - Sites left of the system site (denoted by system index) couple (from left to right)
                         the current site to the system site AS IF they were directly adjacent
                       - Sites right of the system site (denoted by system index) couple (from left to right)
                         the system site to the current site AS IF they were directly adjacent
                   At system_index, the list/iterator may contain either None or the first coupling for the
                   site immediately to the right of the system. If a a single numpy ndarray is passed
                   this element is broadcast over all sites
    :param tau: Timestep for each invocation of evolve. Real timestep should be passed here. Default is .01
    :param state_compression_kwargs: Arguments for mpa compression after each dot product (see real time
                                     evolution factory function for details)
    :param op_compression_kwargs: Arguments for trotter step operator pre-compression (see real time evolution
                                  factory function for details)
    :param second_order_trotter: Switch to use second order instead of fourth order trotter if desired
                                 By default fourth order Trotter is used
    :param t0: Initial time of the propagation
    :param psi_0_compression_kwargs: Optional compresion kwargs for the initial state (see real time evolution
                                     factory function for details)
    :param track_trace: If the trace of the (effective) density matrix should be tracked during the
                        imaginary time evolution
    :return: TMP object. If mpa_type is mps: ITMPS obj., if mpa_type is mpo: ITMPO obj.,
             if mpa_type is pmps: ITPMPS obj.
    """
    if not check_shape(psi_0, mpa_type):
        raise AssertionError('MPA shape of the initial state is not compatible with the chosen mpa_type')
    assert np.imag(tau) == 0 and np.real(tau) != 0
    tau = 1j * tau
    if mpa_type == 'mps':
        return StarITMPS.from_hamiltonian(psi_0, False, False, system_index, h_site, h_bond, tau=tau,
                                          state_compression_kwargs=state_compression_kwargs,
                                          op_compression_kwargs=op_compression_kwargs,
                                          second_order_trotter=second_order_trotter, t0=t0,
                                          psi_0_compression_kwargs=psi_0_compression_kwargs,
                                          track_trace=track_trace)
    elif mpa_type == 'pmps':
        return StarITPMPS.from_hamiltonian(psi_0, True, False, system_index, h_site, h_bond, tau=tau,
                                           state_compression_kwargs=state_compression_kwargs,
                                           op_compression_kwargs=op_compression_kwargs,
                                           second_order_trotter=second_order_trotter, t0=t0,
                                           psi_0_compression_kwargs=psi_0_compression_kwargs,
                                           track_trace=track_trace)
    elif mpa_type == 'mpo':
        return StarITMPO.from_hamiltonian(psi_0, False, True, system_index, h_site, h_bond, tau=tau,
                                          state_compression_kwargs=state_compression_kwargs,
                                          op_compression_kwargs=op_compression_kwargs,
                                          second_order_trotter=second_order_trotter, t0=t0,
                                          psi_0_compression_kwargs=psi_0_compression_kwargs,
                                          track_trace=track_trace)
    else:
        raise AssertionError('Unsupported mpa_type')


def from_hi(psi_0, mpa_type, system_index, hi, tau=0.01, state_compression_kwargs=None,
            op_compression_kwargs=None, second_order_trotter=False, t0=0, psi_0_compression_kwargs=None,
            track_trace=False):
    """
        Factory function for imaginary time TMP-objects (ITMPS, ITMPO, ITPMPS)
    :param psi_0: Initial state as MPArray. Need not be normalized, as it is normalized before propagation
    :param mpa_type: Type of MPArray to propagate, supported are mps, mpo, and pmps
    :param system_index: Index of the system site in the chain (place of the system site operator in the hi_list)
    :param hi: List/tuple for all terms in the Hamiltonian H = sum_i hi
               Ordered like this:
               - Sites left of the system site (denoted by system index) couple (from left to right)
                 the current site to the system site (and contain the site local operators)
               - The term for the system site must be present and contains the local Hamiltonian only!
                 May be None, in which case the local Hamiltonian for the site is assumed to be 0
               - Sites right of the system site (denoted by system index) couple (from left to right)
                 the system site to the current site (and contain the site local operators)
    :param tau: Timestep for each invocation of evolve. Real timestep should be passed here. Default is .01
    :param state_compression_kwargs: Arguments for mpa compression after each dot product (see real time
                                     evolution factory function for details)
    :param op_compression_kwargs: Arguments for trotter step operator pre-compression (see real time evolution
                                  factory function for details)
    :param second_order_trotter: Switch to use second order instead of fourth order trotter if desired
                                 By default fourth order Trotter is used
    :param t0: Initial time of the propagation
    :param psi_0_compression_kwargs: Optional compresion kwargs for the initial state (see real time evolution
                                     factory function for details)
    :param track_trace: If the trace of the (effective) density matrix should be tracked during the
                        imaginary time evolution
    :return: TMP object. If mpa_type is mps: ITMPS obj., if mpa_type is mpo: ITMPO obj., if mpa_type is pmps: ITPMPS obj.
    """
    if not check_shape(psi_0, mpa_type):
        raise AssertionError('MPA shape of the initial state is not compatible with the chosen mpa_type')
    assert np.imag(tau) == 0 and np.real(tau) != 0
    tau = 1j * tau
    if mpa_type == 'mps':
        return StarITMPS.from_hi(psi_0, False, False, system_index, hi, tau=tau,
                                 state_compression_kwargs=state_compression_kwargs,
                                 op_compression_kwargs=op_compression_kwargs,
                                 second_order_trotter=second_order_trotter, t0=t0,
                                 psi_0_compression_kwargs=psi_0_compression_kwargs,
                                 track_trace=track_trace)
    elif mpa_type == 'pmps':
        return StarITPMPS.from_hi(psi_0, True, False, system_index, hi, tau=tau,
                                  state_compression_kwargs=state_compression_kwargs,
                                  op_compression_kwargs=op_compression_kwargs,
                                  second_order_trotter=second_order_trotter, t0=t0,
                                  psi_0_compression_kwargs=psi_0_compression_kwargs,
                                  track_trace=track_trace)
    elif mpa_type == 'mpo':
        return StarITMPO.from_hi(psi_0, False, True, system_index, hi, tau=tau,
                                 state_compression_kwargs=state_compression_kwargs,
                                 op_compression_kwargs=op_compression_kwargs,
                                 second_order_trotter=second_order_trotter, t0=t0,
                                 psi_0_compression_kwargs=psi_0_compression_kwargs,
                                 track_trace=track_trace)
    else:
        raise AssertionError('Unsupported mpa_type')
