from tmps.chain.propagator.mppropagator import MPPropagator


def get(shape, hi_list, tau=0.01, ancilla_sites=False, op_compression_kwargs=None,
        second_order_trotter=False, to_cform=None, build_adj=False):
    """
        Returns a chain propagator object. Can be used to instantiate a suitable tmp object.
    :param shape: Shape (physical dimensions, as returned by mpnum arrays using mps.shape) of the chain to propagate
                  If ancilla_sites is not set True only axis 0 legs are taken into account for the construction of
                  the propagator, which suffices for mps and mpos
    :param hi_list: List/Tuple of all terms in the Hamiltonian H = sum_i hi, where hi is local to one bond
    :param tau: Timestep
    :param ancilla_sites: If the chain has ancilla sites (for pmps evolution)
    :param op_compression_kwargs: Arguments for trotter step operator pre-compression (see real time evolution
                                  factory function for details)
    :param second_order_trotter: Switch to use second order instead of fourth order trotter if desired
                                 By default fourth order Trotter is used
    :param to_cform: Force canonical form of the trotter operators (None forces no canonical form, 'left' means
                     left canonical, 'right' means right canonical)
    :param build_adj: If the adjoint trotter-operators should be pre-built as well (for mpo evolution)
    :return: MPPropagator object
    """
    return MPPropagator(shape, hi_list, tau=tau, ancilla_sites=ancilla_sites,
                        op_compression_kwargs=op_compression_kwargs,
                        second_order_trotter=second_order_trotter, to_cform=to_cform, build_adj=build_adj)


def get_from_hamiltonian(shape, h_site, h_bond, tau=0.01, ancilla_sites=False, op_compression_kwargs=None,
                         second_order_trotter=False, to_cform=None, build_adj=False):
    """
        Returns a chain propagator object. Can be used to instantiate a suitable tmp object.
    :param shape: Shape (physical dimensions, as returned by mpnum arrays using mps.shape) of the chain to propagate
                  If ancilla_sites is not set True only axis 0 legs are taken into account for the construction of
                  the propagator, which suffices for mps and mpos
    :param h_site: Iterator over local site Hamiltonians. If a list with only 1 element is passed
                   this element is broadcast over all sites
    :param h_bond: Iterator over bond Hamiltonians. If a list with only 1 element is passed
                   this element is broadcast over all bonds
    :param tau: Timestep
    :param ancilla_sites: If the chain has ancilla sites (for pmps evolution)
    :param op_compression_kwargs: Arguments for trotter step operator pre-compression (see real time evolution
                                  factory function for details)
    :param second_order_trotter: Switch to use second order instead of fourth order trotter if desired
                                 By default fourth order Trotter is used
    :param to_cform: Force canonical form of the trotter operators (None forces no canonical form, 'left' means
                     left canonical, 'right' means right canonical)
    :param build_adj: If the adjoint trotter-operators should be pre-built as well (for mpo evolution)
    :return: MPPropagator object
    """
    return MPPropagator.from_hamiltonian(shape, h_site, h_bond, tau=tau, ancilla_sites=ancilla_sites,
                                         op_compression_kwargs=op_compression_kwargs,
                                         second_order_trotter=second_order_trotter, to_cform=to_cform,
                                         build_adj=build_adj)
