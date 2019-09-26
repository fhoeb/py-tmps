"""
    MPO Propagator related container class for the trotterized time evolution steps for a fourth order Trotter
    decomposition
    Essentially identical to the similar class for MPS Propagators, but also computes the adjoint operators
"""
from tmps.star.propagator.mps_4o_propagator import StarMPS4OPropagator


class StarMPO4OPropagator(StarMPS4OPropagator):
    def __init__(self, shape, system_index, hi_list, tau=0.01, op_compression_kwargs=None, to_cform=None):
        super().__init__(shape, system_index, hi_list, tau=tau, op_compression_kwargs=op_compression_kwargs,
                         to_cform=to_cform)
        self.build_adj = True

    def _combine_swap_and_timeevo(self, trotter_exponentials, post_swap, pre_swap):
        """
            Builds a list of tuples, which contain all the operators, that are necessary for the complete sweep of
            the system site through the chain and back to its initial position (with local time evolutions
            of the system as their own distinct operators) in the sense of a fourth order trotter evolution.
            Each entry in the list contains a tuple, the first element of the tuple is the index of the left one
            of the two sites in the chain, for which the above mentioned operator applies. The second element is
            the operator itself, the third element of the tuple is the adjoint of the operator.
            The entire evolution progresses through five full sweeps through the chain, which are built using
            _get_full_sweep() with the respective operators from U_2(tau_1)/U_2(tau_2).
            Also pre-compresses the operators if op_compression_kwargs is not None (the op_ranks variable then
            contains the ranks from the tau_1 operators first. Then from the tau_2 operators second)
        :param trotter_exponentials: [U_2(\tau_1)-exponentials (ordered by system site), U_2(\tau_2)-exponentials
                                      (ordered by system site]
        :param post_swap: List of swap operators to be applied after the time-evo operators
        :param pre_swap: List of swap operators to be applied before the time-evo operators
        :return: List of tuples with entries as described above
        """
        trotter_tau_1 = trotter_exponentials[0]
        trotter_tau_2 = trotter_exponentials[1]
        # First second order trotter step, contains start and end
        first_sweep = self._get_full_sweep(trotter_tau_1, post_swap, pre_swap)
        # Third sweep, contains the operators for the tau_2 timestep, has start and end
        third_sweep = self._get_full_sweep(trotter_tau_2, post_swap, pre_swap)
        # These are all the individual operators that are needed
        self._save_op_ranks(first_sweep+third_sweep)
        # And now reorder them to form a fourth order trotter sequence
        # Second sweep, does not contain start and end
        second_sweep = first_sweep[1:-1]
        # Fourth sweep, same as second sweep
        fourth_sweep = second_sweep
        # Last sweep, almost the same as the first but position of start and end operators swapped
        fifth_sweep = [first_sweep[-1]] + second_sweep + [first_sweep[0]]
        timeevo_ops = first_sweep + second_sweep + third_sweep + fourth_sweep + fifth_sweep
        first_sweep_adj = [trotter_mpo.adj() for (start_at, trotter_mpo) in first_sweep]
        third_sweep_adj = [trotter_mpo.adj() for (start_at, trotter_mpo) in third_sweep]
        second_sweep_adj = first_sweep_adj[1:-1]
        fourth_sweep_adj = second_sweep_adj
        fifth_sweep_adj = [first_sweep_adj[-1]] + second_sweep_adj + [first_sweep_adj[0]]
        timeevo_ops_adj = first_sweep_adj + second_sweep_adj + third_sweep_adj + fourth_sweep_adj + fifth_sweep_adj
        return [(start_at, trotter_mpo, trotter_mpo_adj)
                for ((start_at, trotter_mpo), trotter_mpo_adj) in zip(timeevo_ops, timeevo_ops_adj)]