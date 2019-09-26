"""
    Functions for thermal state generation
"""
import mpnum as mp
from tmps.utils.mixed import get_maximally_mixed_state
from tmps.star.itime.factory import from_hamiltonian as propagator_from_hamiltonian, from_hi as propagator_from_hi
from operator import mul
from functools import reduce


def _propagation(propagator, nof_steps, to_mpo, verbose):
    """
        Performs imaginary time evolution of the passed propagator. If to_mpo is set True, the state
        will be converted to an mpo before returning
    """
    if verbose:
        print('Starting propagtion')
    for step in range(nof_steps):
        propagator.evolve()
        if verbose:
            print('Step {:d}:'.format(step+1))
            print(propagator.psi_t.ranks)
    if to_mpo:
        psi_t = mp.pmps_to_mpo(propagator.psi_t)
    else:
        psi_t = propagator.psi_t
    if verbose:
        print('Propagation finished')
    return psi_t, propagator.info()


def get_propagator(beta, mpa_type, system_index, h_site, h_bond, nof_steps=10, state_compression_kwargs=None,
                   op_compression_kwargs=None, second_order_trotter=False,
                   psi_0_compression_kwargs=None):
    """
        Returns the propagator object for imaginary time evolution ground state generation
    :param beta: Inverse temperature
    :param mpa_type: Type of mpa to evolve (allowed are 'pmps' and 'mpo')
    :param system_index: Index of the system site in the chain (place of the system site operator in h_site)
    :param h_site: local operators of hamiltonian as list or tuple
    :param h_bond: iterator over bond (coupling) operators of the Hamiltonian
    :param nof_steps: number of steps for the imaginary time evolution
    :param state_compression_kwargs: Arguments for mps compression after each second order trotter step U(tau_i).
                                     (see real time evolution factory function for details)
    :param op_compression_kwargs: Arguments for trotter step operator precompression (see real time evolution
                                  factory function for details)
    :param second_order_trotter: Uses second order trotter steps for the propagation
    :param psi_0_compression_kwargs: Optional compression kwargs for the initial state (see real time evolution
                                     factory function for details)
    """
    assert mpa_type == 'mpo' or mpa_type == 'pmps'
    dims = []
    for index, site in enumerate(h_site):
        assert site.shape[0] == site.shape[1]
        dims.append(site.shape[0])
    tau = -(beta / 2) / nof_steps
    psi_0 = get_maximally_mixed_state(mpa_type, dims, normalized=False)
    return propagator_from_hamiltonian(psi_0, mpa_type, system_index, h_site, h_bond, tau=tau,
                                       state_compression_kwargs=state_compression_kwargs,
                                       op_compression_kwargs=op_compression_kwargs,
                                       second_order_trotter=second_order_trotter,
                                       psi_0_compression_kwargs=psi_0_compression_kwargs)


def get_propagator_from_hi(beta, mpa_type, system_index, dims, hi_list, nof_steps=10, state_compression_kwargs=None,
                           op_compression_kwargs=None, second_order_trotter=False, psi_0_compression_kwargs=None):
    """
        Returns the propagator object for imaginary time evolution ground state generation
    :param beta: Inverse temperature
    :param mpa_type: Type of mpa to evolve (allowed are 'pmps' and 'mpo')
    :param system_index: Index of the system site in the chain (place of the system site operator in the hi_list)
    :param hi_list: List/Tuple of all terms in the Hamiltonian H = sum_i hi, where hi is local to one bond
    :param nof_steps: number of steps for the imaginary time evolution
    :param state_compression_kwargs: Arguments for mps compression after each second order trotter step U(tau_i).
                                     (see real time evolution factory function for details)
    :param op_compression_kwargs: Arguments for trotter step operator precompression (see real time evolution
                                  factory function for details)
    :param second_order_trotter: Uses second order trotter steps for the propagation
    :param psi_0_compression_kwargs: Optional compresion kwargs for the initial state (see real time evolution
                                     factory function for details)
    """
    assert mpa_type == 'mpo' or mpa_type == 'pmps'
    tau = -(beta / 2) / nof_steps
    psi_0 = get_maximally_mixed_state(mpa_type, dims, normalized=False)
    return propagator_from_hi(psi_0, mpa_type, system_index, hi_list, tau=tau,
                              state_compression_kwargs=state_compression_kwargs,
                              op_compression_kwargs=op_compression_kwargs,
                              second_order_trotter=second_order_trotter,
                              psi_0_compression_kwargs=psi_0_compression_kwargs)


def from_hamiltonian(beta, mpa_type, system_index, h_site, h_bond, nof_steps=10, state_compression_kwargs=None,
                     op_compression_kwargs=None, second_order_trotter=False, psi_0_compression_kwargs=None,
                     force_pmps_evolution=False, verbose=False, get_partition_function=False):
    """
        Generates a thermal state e^(-beta*H)/Z in pmps form or mpo form by using imaginary time evolution.
    :param beta: Inverse temperature
    :param mpa_type: Type of mpa to evolve (allowed are 'pmps' and 'mpo')
    :param system_index: Index of the system site in the chain (place of the system site operator in h_site)
    :param h_site: local operators of hamiltonian as list or tuple
    :param h_bond: iterator over bond (coupling) operators of the Hamiltonian
    :param nof_steps: number of steps for the imaginary time evolution
    :param state_compression_kwargs: Arguments for mps compression after each second order trotter step U(tau_i).
                                     (see real time evolution factory function for details)
    :param op_compression_kwargs: Arguments for trotter step operator precompression (see real time evolution
                                  factory function for details)
    :param second_order_trotter: Uses second order trotter steps for the propagation
    :param psi_0_compression_kwargs: Optional compresion kwargs for the initial state (see real time evolution
                                     factory function for details)
    :param force_pmps_evolution: Forces pmps evolution for the thermal state instead of mpo evolution if
                                 the map_type is 'mpo'
    :param verbose: If updates on the time evolution should be given via stdout
    :param get_partition_function: If set True, the trace of the state is tracked during the imaginary time evolution.
                                   The infinite temperature partition function (Z0) and the partition function for the
                                   returned state (Z) are then included within the info dict.
    :return: thermal state in pmps or mpo form (Any ancilla sites have the same dimension as the physical ones),
             info object from the propagation (if beta was set 0 it returns an empty dict)
    """
    assert mpa_type == 'mpo' or mpa_type == 'pmps'
    to_mpo = True if (mpa_type == 'mpo') and force_pmps_evolution else False
    dims = []
    for index, site in enumerate(h_site):
        assert site.shape[0] == site.shape[1]
        dims.append(site.shape[0])
    if beta == 0:
        info = dict()
        if get_partition_function:
            Z0 = reduce(mul, dims, 1)
            info['Z'] = Z0
            info['Z0'] = Z0
        return get_maximally_mixed_state(mpa_type, dims, normalized=True), dict()
    evo_mpa_type = 'pmps' if force_pmps_evolution else mpa_type
    thermal = get_propagator(beta, evo_mpa_type, system_index, h_site, h_bond, nof_steps=nof_steps,
                             state_compression_kwargs=state_compression_kwargs,
                             op_compression_kwargs=op_compression_kwargs,
                             second_order_trotter=second_order_trotter,
                             psi_0_compression_kwargs=psi_0_compression_kwargs)
    psi_beta, info = _propagation(thermal, nof_steps, to_mpo, verbose)
    if get_partition_function:
        Z0 = reduce(mul, dims, 1)
        info['Z0'] = Z0
        try:
            Z = Z0 * info['state_trace']
        except OverflowError:
            print('Overflow error during calculation of the partition function.')
            Z = None
        except TypeError:
            print('Partition function could not be calculated, because state trace was too large')
            Z = None
        info['Z'] = Z
    return psi_beta, info


def from_hi(beta, mpa_type, system_index, dims, hi_list, nof_steps=10, state_compression_kwargs=None,
            op_compression_kwargs=None, second_order_trotter=False, psi_0_compression_kwargs=None,
            force_pmps_evolution=True, verbose=False, get_partition_function=False):
    """
        Generates a thermal state e^(-beta*H)/Z in pmps form or mpo form by using imaginary time evolution.
    :param beta: Inverse temperature
    :param mpa_type: Type of mpa to evolve (allowed are 'pmps' and 'mpo')
    :param system_index: Index of the system site in the chain (place of the system site operator in the hi_list)
    :param dims: Physical dimensions in the chain
    :param hi_list: List/Tuple of all terms in the Hamiltonian H = sum_i hi, where hi is local to one bond
    :param nof_steps: number of steps for the imaginary time evolution
    :param state_compression_kwargs: Arguments for mps compression after each second order trotter step U(tau_i).
                                     (see real time evolution factory function for details)
    :param op_compression_kwargs: Arguments for trotter step operator precompression (see real time evolution
                                  factory function for details)
    :param second_order_trotter: Uses second order trotter steps for the propagation
    :param psi_0_compression_kwargs: Optional compresion kwargs for the initial state (see real time evolution
                                     factory function for details)
    :param force_pmps_evolution: Forces pmps evolution for the thermal state instead of mpo evolution if
                                 the map_type is 'mpo'
    :param get_partition_function: If set True, the trace of the state is tracked during the imaginary time evolution.
                                   The infinite temperature partition function (Z0) and the partition function for the
                                   returned state (Z) are then included within the info dict.
    :param verbose: If updates on the time evolution should be given via stdout
    """
    assert mpa_type == 'mpo' or mpa_type == 'pmps'
    to_mpo = True if (mpa_type == 'mpo') and force_pmps_evolution else False
    if beta == 0:
        info = dict()
        if get_partition_function:
            Z0 = reduce(mul, dims, 1)
            info['Z'] = Z0
            info['Z0'] = Z0
        return get_maximally_mixed_state(mpa_type, dims, normalized=True), dict()
    evo_mpa_type = 'pmps' if force_pmps_evolution else mpa_type
    thermal = get_propagator_from_hi(beta, evo_mpa_type, system_index, dims, hi_list, nof_steps=nof_steps,
                                     state_compression_kwargs=state_compression_kwargs,
                                     op_compression_kwargs=op_compression_kwargs,
                                     second_order_trotter=second_order_trotter,
                                     psi_0_compression_kwargs=psi_0_compression_kwargs)
    psi_beta, info = _propagation(thermal, nof_steps, to_mpo, verbose)
    if get_partition_function:
        Z0 = reduce(mul, dims, 1)
        info['Z0'] = Z0
        try:
            Z = Z0 * info['state_trace']
        except OverflowError:
            print('Overflow error during calculation of the partition function.')
            Z = None
        except TypeError:
            print('Partition function could not be calculated, because state trace was too large')
            Z = None
        info['Z'] = Z
    return psi_beta, info
