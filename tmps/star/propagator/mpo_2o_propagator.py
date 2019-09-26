"""
    MPO Propagator related container class for the trotterized time evolution steps for a second order Trotter
    decomposition
    Essentially identical to the similar class for MPS Propagators, but also computes the adjoint operators
"""
from tmps.star.propagator.mps_2o_propagator import StarMPS2OPropagator


class StarMPO2OPropagator(StarMPS2OPropagator):
    def __init__(self, shape, system_index, hi_list, tau=0.01, op_compression_kwargs=None, to_cform=None):
        super().__init__(shape, system_index, hi_list, tau=tau, op_compression_kwargs=op_compression_kwargs,
                         to_cform=to_cform)
        self.build_adj = True

    def _combine_swap_and_timeevo(self, trotter_exponentials, post_swap, pre_swap):
        """
            Builds a list of tuples, which contain all the operators, that are necessary for the complete sweep of
            the system site through the chain and back to its initial position (with local time evolutions
            of the system as their own distinct operators)
            Each entry in the list contains a tuple, the first element of the tuple is the index of the left one
            of the two sites in the chain, for which the above mentioned operator applies. The second element is
            the operator itself, the third element of the tuple is the adjoint of the operator.
            Also pre-compresses the operators if op_compression_kwargs is not None
        :param trotter_exponentials: List of trotterized unitary operators
        :param post_swap: List of swap operators to be applied after the time-evo operators
        :param pre_swap: List of swap operators to be applied before the time-evo operators
        :return: List of tuples with entries as described above
        """
        timeevo_ops = super()._combine_swap_and_timeevo(trotter_exponentials, post_swap, pre_swap)
        return [(start_at, trotter_mpo, trotter_mpo.adj()) for (start_at, trotter_mpo) in timeevo_ops]