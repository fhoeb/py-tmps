from tmps.star.propagator.mps_2o_propagator import StarMPS2OPropagator
from tmps.star.propagator.mps_4o_propagator import StarMPS4OPropagator
from tmps.star.propagator.mpo_2o_propagator import StarMPO2OPropagator
from tmps.star.propagator.mpo_4o_propagator import StarMPO4OPropagator
from tmps.star.propagator.pmps_2o_propagator import StarPMPS2OPropagator
from tmps.star.propagator.pmps_4o_propagator import StarPMPS4OPropagator


def get(shape, system_index, hi_list, tau=0.01, ancilla_sites=False, build_adj=False, op_compression_kwargs=None,
        second_order_trotter=False, to_cform=None):
    """
        Returns a chain propagator object. Can be used to instantiate a suitable tmp object.
    :param shape: Shape of the state (or chain on which) to propagate (in mparray shape form).
    :param system_index: Index of the system site in the chain (place of the system site operator in the hi_list)
    :param hi_list: List/tuple for all terms in the Hamiltonian H = sum_i hi
                    Ordered like this:
                    - Sites left of the system site (before system index) couple (from left to right)
                      the current site to the system site (and contain the site local operators for the
                      current sites only!)
                    - The term for the system site must be present and denotes the local Hamiltonian only!
                      May be None, in which case the local Hamiltonian for the site is assumed to be 0
                    - Sites right of the system site (after system index) couple (from left to right)
                      the system site to the current site (and contain the site local operators for the
                      current sites only!)
    :param tau: Timestep for each invocation of evolve
    :param ancilla_sites: If axis 1 legs on the chain are to be interpreted as pyhsical legs (for mpos) or
                          as ancilla legs untouched by time evolution as in the case of pmps.
    :param build_adj: If the adjoint trotter-operators should be pre-built as well (for mpo evolution)
    :param op_compression_kwargs: Arguments for operator precompression
    :param second_order_trotter: Switch to use second order instead of fourth order trotter if desired
                                 By default fourth order Trotter is used
    :param to_cform: Force canonical form of the trotter operators (None forces no canonical form, 'left' means
                     left canonical, 'right' means right canonical)
    """
    if ancilla_sites:
        if build_adj:
            raise AssertionError('No propagator with ancilla sites AND pre-built adjoint operators implemented')
        else:
            if second_order_trotter:
                return StarPMPS2OPropagator(shape, system_index, hi_list, tau=tau,
                                            op_compression_kwargs=op_compression_kwargs, to_cform=to_cform)
            else:
                return StarPMPS4OPropagator(shape, system_index, hi_list, tau=tau,
                                            op_compression_kwargs=op_compression_kwargs, to_cform=to_cform)
    else:
        if build_adj:
            if second_order_trotter:
                return StarMPO2OPropagator(shape, system_index, hi_list, tau=tau,
                                           op_compression_kwargs=op_compression_kwargs, to_cform=to_cform)
            else:
                return StarMPO4OPropagator(shape, system_index, hi_list, tau=tau,
                                           op_compression_kwargs=op_compression_kwargs, to_cform=to_cform)
        else:
            if second_order_trotter:
                return StarMPS2OPropagator(shape, system_index, hi_list, tau=tau,
                                           op_compression_kwargs=op_compression_kwargs, to_cform=to_cform)
            else:
                return StarMPS4OPropagator(shape, system_index, hi_list, tau=tau,
                                           op_compression_kwargs=op_compression_kwargs, to_cform=to_cform)


def get_from_hamiltonian(shape, system_index, h_site, h_bond, tau=0.01, ancilla_sites=False, build_adj=False,
                         op_compression_kwargs=None, second_order_trotter=False, to_cform=None):
    """
        Returns a chain propagator object. Can be used to instantiate a suitable tmp object.
    :param shape: Shape of the state (or chain on which) to propagate (in mparray shape form).
    :param system_index: Index of the system site in the chain (place of the system site operator in the h_site list)
    :param h_site: Iterator for all local terms in the Hamiltonian for all sites in the chain
    :param h_bond: Iterator for all System-Bath coupling terms in the Hamiltonian for all sites in the chain,
                    Ordered like this:
                    - Sites left of the system site (before system_index) couple (from left to right)
                      the current site to the system site
                    - Sites right of the system site (after system_index) couple (from left to right)
                      the system site to the current site
                    (list is shorter by one element compared to the h_site list)
    :param tau: Timestep for each invocation of evolve
    :param ancilla_sites: If axis 1 legs on the chain are to be interpreted as pyhsical legs (for mpos) or
                          as ancilla legs untouched by time evolution as in the case of pmps.
    :param build_adj: If the adjoint trotter-operators should be pre-built as well (for mpo evolution)
    :param op_compression_kwargs: Arguments for operator precompression
    :param second_order_trotter: Switch to use second order instead of fourth order trotter if desired
                                 By default fourth order Trotter is used
    :param to_cform: Force canonical form of the trotter operators (None forces no canonical form, 'left' means
                     left canonical, 'right' means right canonical)
    """
    if ancilla_sites:
        if build_adj:
            raise AssertionError('No propagator with ancilla sites AND pre-built adjoint operators implemented')
        else:
            if second_order_trotter:
                return StarPMPS2OPropagator.from_hamiltonian(shape, system_index, h_site, h_bond, tau=tau,
                                                             op_compression_kwargs=op_compression_kwargs,
                                                             to_cform=to_cform)
            else:
                return StarPMPS4OPropagator.from_hamiltonian(shape, system_index, h_site, h_bond, tau=tau,
                                                             op_compression_kwargs=op_compression_kwargs,
                                                             to_cform=to_cform)
    else:
        if build_adj:
            if second_order_trotter:
                return StarMPO2OPropagator.from_hamiltonian(shape, system_index, h_site, h_bond, tau=tau,
                                                            op_compression_kwargs=op_compression_kwargs,
                                                            to_cform=to_cform)
            else:
                return StarMPO4OPropagator.from_hamiltonian(shape, system_index, h_site, h_bond, tau=tau,
                                                            op_compression_kwargs=op_compression_kwargs,
                                                            to_cform=to_cform)
        else:
            if second_order_trotter:
                return StarMPS2OPropagator.from_hamiltonian(shape, system_index, h_site, h_bond, tau=tau,
                                                            op_compression_kwargs=op_compression_kwargs,
                                                            to_cform=to_cform)
            else:
                return StarMPS4OPropagator.from_hamiltonian(shape, system_index, h_site, h_bond, tau=tau,
                                                            op_compression_kwargs=op_compression_kwargs,
                                                            to_cform=to_cform)
