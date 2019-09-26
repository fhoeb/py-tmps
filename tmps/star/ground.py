"""
    Functions for ground state generation for star geometries via imaginary time evolution
"""

import mpnum as mp
from tmps.star.itime.factory import from_hi as propagator_from_hi, from_hamiltonian as propagator_from_hamiltonian
from tmps.utils.random import get_random_mpa


def _propagation(propagator, nof_steps, verbose):
    """
        Performs imaginary time evolution of the passed propagator.
    """
    if verbose:
        print('Starting propagtion')
    for step in range(nof_steps):
        propagator.evolve()
        if verbose:
            print('Step {:d}:'.format(step+1))
            print(propagator.psi_t.ranks)
    if verbose:
        print('Propagation finished')
    return propagator.psi_t, propagator.info()


def _convergence(propagator, start, step, stop, err, verbose):
    """
        Performs imaginary time evolution of the passed propagator until a convergence condition is met
        (absolute l2-norm difference between the states from successive timesteps is small enough)
    """
    nof_steps = stop
    if verbose:
        print('Starting propagtion')
    for i in range(start):
        propagator.evolve()
        if verbose:
            print('Step {:d}:'.format(i+1))
            print(propagator.psi_t.ranks)
    last_psi_t = propagator.psi_t.copy()
    check = None
    for i in range(1, (stop - start)+1):
        propagator.evolve()
        if i % step == 0:
            check = mp.normdist(last_psi_t, propagator.psi_t)
            if check < err:
                if verbose:
                    print('Step {:d}:'.format(i+start))
                    print(propagator.psi_t.ranks)
                nof_steps = i+start
                break
            else:
                last_psi_t = propagator.psi_t.copy()
        if verbose:
            print('Step {:d}:'.format(i+start))
            print(propagator.psi_t.ranks)
    if nof_steps == stop:
        print('Did not reach convergence in ' + str(nof_steps) + ' steps!')
    info = propagator.info()
    info['nof_steps'] = nof_steps
    info['error'] = check
    if verbose:
        print('Propagation finished')
    return propagator.psi_t, info


def get_propagator(beta, mpa_type, system_index, h_site, h_bond, rank=1, state_compression_kwargs=None,
                   op_compression_kwargs=None, psi_0_compression_kwargs=None, second_order_trotter=False,
                   seed=102):
    """
        Returns the propagator object for imaginary time evolution ground state generation
    :param beta: Decay coefficient
    :param mpa_type: Type of mpa to evolve (allowed are 'mps', 'pmps' and 'mpo')
    :param system_index: Index of the system site in the chain (place of the system site operator in h_site)
    :param h_site: local operators of hamiltonian as list or tuple
    :param h_bond: iterator over bond (coupling) operators of the Hamiltonian
    :param rank: Rank of the random initial state (default is 1)
    :param state_compression_kwargs: Arguments for mps compression after each second order trotter step U(tau_i).
                                     (see real time evolution factory function for details)
    :param op_compression_kwargs: Arguments for trotter step operator precompression (see real time evolution
                                  factory function for details)
    :param second_order_trotter: Uses second order trotter steps for the propagation
    :param seed: Seed for the initial random state
    :param psi_0_compression_kwargs: Optional compression kwargs for the initial state (see real time evolution
                                     factory function for details)
    """
    tau = -beta
    axis_0 = []
    for index, site in enumerate(h_site):
        assert site.shape[0] == site.shape[1]
        axis_0.append(site.shape[0])
    psi_0 = get_random_mpa(mpa_type, axis_0, seed=seed, rank=rank)
    return propagator_from_hamiltonian(psi_0, mpa_type, system_index, h_site, h_bond, tau=tau,
                                       state_compression_kwargs=state_compression_kwargs,
                                       op_compression_kwargs=op_compression_kwargs,
                                       psi_0_compression_kwargs=psi_0_compression_kwargs,
                                       second_order_trotter=second_order_trotter, t0=0,
                                       track_trace=False)


def get_propagator_from_hi(beta, mpa_type, system_index, dims, hi_list, rank=1, state_compression_kwargs=None,
                           op_compression_kwargs=None, second_order_trotter=False, seed=102,
                           psi_0_compression_kwargs=None):
    """
        Returns the propagator object for imaginary time evolution ground state generation
    :param beta: Decay coefficient
    :param mpa_type: Type of mpa to evolve (allowed are 'mps', 'pmps' and 'mpo')
    :param system_index: Index of the system site in the chain (place of the system site operator in the hi_list)
    :param dims: Physical dimensions in the chain
    :param hi_list: List/Tuple of all terms in the Hamiltonian H = sum_i hi, where hi is local to one bond
    :param rank: Rank of the random initial state (default is 1)
    :param state_compression_kwargs: Arguments for mps compression after each second order trotter step U(tau_i).
                                     (see real time evolution factory function for details)
    :param op_compression_kwargs: Arguments for trotter step operator precompression (see real time evolution
                                  factory function for details)
    :param second_order_trotter: Uses second order trotter steps for the propagation
    :param seed: Seed for the initial random state
    :param psi_0_compression_kwargs: Optional compression kwargs for the initial state (see real time evolution
                                     factory function for details)
    """
    tau = -beta
    psi_0 = get_random_mpa(mpa_type, dims, seed=seed, rank=rank)
    return propagator_from_hi(psi_0, mpa_type, system_index, hi_list, tau=tau,
                              state_compression_kwargs=state_compression_kwargs,
                              op_compression_kwargs=op_compression_kwargs,
                              psi_0_compression_kwargs=psi_0_compression_kwargs,
                              second_order_trotter=second_order_trotter, t0=0,
                              track_trace=False)


def from_hamiltonian(beta, nof_steps, mpa_type, system_index, h_site, h_bond, rank=1, state_compression_kwargs=None,
                     op_compression_kwargs=None, psi_0_compression_kwargs=None, second_order_trotter=False, seed=102,
                     verbose=False):
    """
        Generates a ground state of a given Hamiltonian.
    :param beta: Decay coefficient
    :param nof_steps: number of steps for the imaginary time evolution
    :param mpa_type: Type of mpa to evolve (allowed are 'mps', 'pmps' and 'mpo')
    :param system_index: Index of the system site in the chain (place of the system site operator in h_site)
    :param h_site: local operators of hamiltonian as list or tuple
    :param h_bond: iterator over bond (coupling) operators of the Hamiltonian
    :param nof_steps: number of steps for the imaginary time evolution
    :param rank: Rank of the random initial state (default is 1)
    :param state_compression_kwargs: Arguments for mps compression after each second order trotter step U(tau_i).
                                     (see real time evolution factory function for details)
    :param op_compression_kwargs: Arguments for trotter step operator precompression (see real time evolution
                                  factory function for details)
    :param second_order_trotter: Uses second order trotter steps for the propagation
    :param seed: Seed for the initial random state
    :param psi_0_compression_kwargs: Optional compression kwargs for the initial state (see real time evolution
                                     factory function for details)
    :param verbose: If updates on the time evolution should be given via stdout
    :return: ground state from the imaginary time evolution,
             info dict from the propagation
    """
    ground = get_propagator(beta, mpa_type, system_index, h_site, h_bond, rank=rank,
                            state_compression_kwargs=state_compression_kwargs,
                            op_compression_kwargs=op_compression_kwargs,
                            psi_0_compression_kwargs=psi_0_compression_kwargs,
                            second_order_trotter=second_order_trotter, seed=seed)
    return _propagation(ground, nof_steps, verbose)


def from_convergence(beta, mpa_type, system_index, h_site, h_bond, rank=1, state_compression_kwargs=None,
                     op_compression_kwargs=None, psi_0_compression_kwargs=None, second_order_trotter=False, seed=102,
                     start=10, step=1, eps=1e-7, stop=1000, verbose=False):
    """
        Generates a ground state for a given Hamiltonian using |psi_t - psi_(t+dt)|_2 as convergence measure.
    :param beta: Decay coefficient
    :param mpa_type: Type of mpa to evolve (allowed are 'mps', 'pmps' and 'mpo')
    :param system_index: Index of the system site in the chain (place of the system site operator in h_site)
    :param h_site: local operators of hamiltonian as list or tuple
    :param h_bond: iterator over bond (coupling) operators of the Hamiltonian
    :param rank: Rank of the random initial state (default is 1)
    :param state_compression_kwargs: Arguments for mps compression after each second order trotter step U(tau_i).
                                     (see real time evolution factory function for details)
    :param op_compression_kwargs: Arguments for trotter step operator precompression (see real time evolution
                                  factory function for details)
    :param second_order_trotter: Uses second order trotter steps for the propagation
    :param seed: Seed for the initial random state
    :param psi_0_compression_kwargs: Optional compression kwargs for the initial state (see real time evolution
                                     factory function for details)
    :param verbose: If updates on the time evolution should be given via stdout
    :param start: Number of steps before any convergence measure is checked
    :param step: Number of steps between convergence measure is checked
    :param stop: Maximum number of steps before convergence is aborted
    :param eps: Maximum absolute (and relative) allowed value for the convergence measure
    :return: ground state from the imaginary time evolution as mps,
             info dict from the propagation
    """
    assert stop > start
    ground = get_propagator(beta, mpa_type, system_index, h_site, h_bond, rank=rank,
                            state_compression_kwargs=state_compression_kwargs,
                            op_compression_kwargs=op_compression_kwargs,
                            psi_0_compression_kwargs=psi_0_compression_kwargs,
                            second_order_trotter=second_order_trotter, seed=seed)
    return _convergence(ground, start, step, stop, eps, verbose)


def from_hi(beta, nof_steps, mpa_type, system_index, dims, hi_list, rank=1, state_compression_kwargs=None,
            op_compression_kwargs=None, psi_0_compression_kwargs=None, second_order_trotter=False, seed=102,
            verbose=False):
    """
        Generates a ground state of a given Hamiltonian.
    :param beta: Decay coefficient
    :param nof_steps: number of steps for the imaginary time evolution
    :param mpa_type: Type of mpa to evolve (allowed are 'mps', 'pmps' and 'mpo')
    :param system_index: Index of the system site in the chain (place of the system site operator in the hi_list)
    :param dims: Physical dimensions in the chain
    :param hi_list: List/Tuple of all terms in the Hamiltonian H = sum_i hi, where hi is local to one bond
    :param rank: Rank of the random initial state (default is 1)
    :param nof_steps: number of steps for the imaginary time evolution
    :param rank: Rank of the random initial state (default is 1)
    :param state_compression_kwargs: Arguments for mps compression after each second order trotter step U(tau_i).
                                     (see real time evolution factory function for details)
    :param op_compression_kwargs: Arguments for trotter step operator precompression (see real time evolution
                                  factory function for details)
    :param second_order_trotter: Uses second order trotter steps for the propagation
    :param seed: Seed for the initial random state
    :param psi_0_compression_kwargs: Optional compression kwargs for the initial state (see real time evolution
                                     factory function for details)
    :param verbose: If updates on the time evolution should be given via stdout
    :return: ground state from the imaginary time evolution,
             info dict from the propagation
    """
    ground = get_propagator_from_hi(beta, mpa_type, system_index, dims, hi_list, rank=rank,
                                    state_compression_kwargs=state_compression_kwargs,
                                    op_compression_kwargs=op_compression_kwargs,
                                    psi_0_compression_kwargs=psi_0_compression_kwargs,
                                    second_order_trotter=second_order_trotter, seed=seed)
    return _propagation(ground, nof_steps, verbose)


def from_hi_convergence(beta, mpa_type, system_index, dims, hi_list, rank=1, state_compression_kwargs=None,
                        op_compression_kwargs=None, psi_0_compression_kwargs=None, second_order_trotter=False, seed=102,
                        start=10, step=1, eps=1e-7, stop=1000, verbose=False):
    """
        Generates a ground state for a given Hamiltonian using |psi_t - psi_(t+dt)|_2 as convergence measure.
    :param beta: Decay coefficient
    :param mpa_type: Type of mpa to evolve (allowed are 'mps', 'pmps' and 'mpo')
    :param system_index: Index of the system site in the chain (place of the system site operator in the hi_list)
    :param dims: Physical dimensions in the chain
    :param hi_list: List/Tuple of all terms in the Hamiltonian H = sum_i hi, where hi is local to one bond
    :param rank: Rank of the random initial state (default is 1)
    :param state_compression_kwargs: Arguments for mps compression after each second order trotter step U(tau_i).
                                     (see real time evolution factory function for details)
    :param op_compression_kwargs: Arguments for trotter step operator precompression (see real time evolution
                                  factory function for details)
    :param second_order_trotter: Uses second order trotter steps for the propagation
    :param seed: Seed for the initial random state
    :param psi_0_compression_kwargs: Optional compression kwargs for the initial state (see real time evolution
                                     factory function for details)
    :param verbose: If updates on the time evolution should be given via stdout
    :param start: Number of steps before any convergence measure is checked
    :param step: Number of steps between convergence measure is checked
    :param stop: Maximum number of steps before convergence is aborted
    :param eps: Maximum absolute (and relative) allowed value for the convergence measure
    :return: ground state from the imaginary time evolution as mps,
             info dict from the propagation
    """
    assert stop > start
    ground = get_propagator_from_hi(beta, mpa_type, system_index, dims, hi_list, rank=rank,
                                    state_compression_kwargs=state_compression_kwargs,
                                    op_compression_kwargs=op_compression_kwargs,
                                    psi_0_compression_kwargs=psi_0_compression_kwargs,
                                    second_order_trotter=second_order_trotter, seed=seed)
    return _convergence(ground, start, step, stop, eps, verbose)
