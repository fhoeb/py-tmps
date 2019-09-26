"""
    Container object for PMPS fourth order trotter time evolution operators
"""
import mpnum as mp
import numpy as np
from scipy.linalg import expm
from tmps.utils.swap import get_pmps_swap_mpo
from tmps.star.propagator.propagator_base import StarMPPropagatorBase
from tmps.utils.cform import canonicalize_to

# TODO: Test if _compress_mpo is smarter before or after group_sites in the sweep functions


class StarPMPS4OPropagator(StarMPPropagatorBase):
    def __init__(self, shape, system_index, hi_list, tau=0.01, op_compression_kwargs=None, to_cform=None):
        """
            Constructor for the MPPropagator class. Constructs propagation operators which correspond to a particular
            shape of the chain for which we wish to propagate a state (mpo or mps shape).
            Uses fourth order trotter decomposition for the Hamiltonian:
                U(\tau) = U_2(\tau_1) U_2(\tau_1) U_2(\tau_2) U_2(\tau_1) U_2(\tau_1)
                where the U_2 are second order trotter decompositions of the form:
                U(\tau) = two sweeps of e^(-j*H_i*\tau/2),  e^(-j*H_i*\tau/2)
            Method of propagation is explained in detail in: DMRG for Multiband Impurity Solvers by Hans Gerd Evertz
        :param shape: Shape of the state (or chain on which) to propagate (in mparray shape form).
                      Only axis 0 legs are taken into account
        :param system_index: Index of the system site in the chain (place of the system site operator in the hi_list)
        :param hi_list: List/tuple for all terms in the Hamiltonian H = \sum_i h_i
                        Ordered like this:
                        - Sites left of the system site (denoted by system index) couple (from left to right)
                          the current site to the system site (and contain the site local operators for the
                          current sites only!)
                        - The term for the system site must be present and denotes the local Hamiltonian only!
                          May be None, in which case the local Hamiltonian for the site is assumed to be 0
                        - Sites right of the system site (denoted by system index) couple (from left to right)
                          the system site to the current site (and contain the site local operators for the
                          current sites only!)
        :param tau: Timestep for each invocation of evolve
        :param op_compression_kwargs: Arguments for second order trotter step U(\tau_i) operator precompression
        """
        self.ancilla_sites = True
        self.build_adj = False
        self._assert_ndims(shape)
        self.tau_1 = 1/(4 - 4**(1/3)) * tau
        self.tau_2 = tau - 4*self.tau_1
        self.step_trotter_error = tau ** 5
        super().__init__(shape, system_index, hi_list, tau=tau, op_compression_kwargs=op_compression_kwargs,
                         to_cform=to_cform)

    def _assert_ndims(self, shape):
        """
            Checks if ndims per site are all the same, and if they are smaller or equal to 2.
            For physical legs only with two legs per site we also check if the leg dimensions agree (quadratic
            operator)
        :param shape: state/chain shape to test for ndims
        :return:
        """
        init_site_legs = len(shape[0])
        assert init_site_legs <= 2
        for site_shape in shape:
            assert init_site_legs == len(site_shape)

    def _get_swaps(self):
        """
        :return: post_swap (list of swaps, which are to be applied after a trotterized unitary time step; List
                            is in the same order as the sites on the chain, Nones at both ends and at system site.
                            Index in the list indicates the bath site with which to swap the system site. Same for
                            post_swap),
                 pre_swap (list of swaps, which are to be applied before a trotterized unitary time step)
        """
        post_swap, pre_swap = [], []
        system_shape = self.shape[self.system_index]
        for site, site_shape in enumerate(self.shape):
            # If ancilla sites are present, one must group two sites of the four available ones together
            if 0 < site < self.system_index:
                post_swap.append(get_pmps_swap_mpo(site_shape, system_shape))
                pre_swap.append(get_pmps_swap_mpo(system_shape, site_shape))
            elif self.system_index < site < self.L - 1:
                post_swap.append(get_pmps_swap_mpo(system_shape, site_shape))
                pre_swap.append(get_pmps_swap_mpo(site_shape, system_shape))
            else:
                # Extend arrays to match length of propagator (exp^(iHt)-mpo) array
                post_swap.append(None)
                pre_swap.append(None)
        return post_swap, pre_swap

    def _build_2o_trotter_exponentials(self, hi_list, tau, start_tau, end_tau):
        """
            Builds list, which contains all trotterized bond-local exponentials (exp(-ij*tau*hi) in mpo form.
            At the system index, the list contains only a site local operator. Everywhere else we have operators, which
            act on one bond in the chain (assuming the system is right next to them)
            For the system site local time evolution we construct two operators, one with start_tau, the other with
            end_tau as timesteps (thus, the entry of the system_index contains a tuple of operators).
            This is done to be able to skip unnecessary double applications of the system operator between two second
            order trotter sweeps.
        :return: List of all trotterized bond-local exponentials (exp(-1j*\tau*hi) in mpo form.
        """
        propagator_mpos = []
        if self.system_index == 0:
            for site, hi in enumerate(hi_list):
                if site == 0:
                    # system site
                    mpo = (self._system_site_mpo(hi, start_tau), self._system_site_mpo(hi, end_tau))
                elif 0 < site < self.L-1:
                    # Couple from system to site
                    mpo = self._mpo_from_hi(hi, tau/2, lbond=self.system_index, rbond=site)
                else:
                    # final site
                    mpo = self._mpo_from_hi(hi, tau, lbond=self.system_index, rbond=self.L-1)
                propagator_mpos.append(mpo)
        else:
            for site, hi in enumerate(hi_list):
                if site == 0:
                    mpo = self._mpo_from_hi(hi, tau, lbond=0, rbond=self.system_index)
                elif 0 < site < self.system_index:
                    # Couple from site to system
                    mpo = self._mpo_from_hi(hi, tau/2, lbond=site, rbond=self.system_index)
                elif site == self.system_index:
                    # system site mpo
                    mpo = (self._system_site_mpo(hi, start_tau), self._system_site_mpo(hi, end_tau))
                elif self.system_index < site < self.L-1:
                    # Couple from system to site
                    mpo = self._mpo_from_hi(hi, tau/2, lbond=self.system_index, rbond=site)
                else:
                    # final site
                    mpo = self._mpo_from_hi(hi, tau, lbond=self.system_index, rbond=self.L-1)
                propagator_mpos.append(mpo)
        return propagator_mpos

    def _build_trotter_exponentials(self, hi_list):
        """
            Builds all required second order trotter exponentials for timesteps \tau_1 and \tau_2 using
            _build_2o_trotter_exponentials.
        :return: List of Lists: [U_2(\tau_1)-exponentials (ordered by system site), U_2(\tau_2)-exponentials (ordered by
                                 system site]
        """
        second_order_trotter_tau_1 = self._build_2o_trotter_exponentials(hi_list, self.tau_1, self.tau_1/2, self.tau_1)
        second_order_trotter_tau_2 = self._build_2o_trotter_exponentials(hi_list, self.tau_2, (self.tau_1 + self.tau_2)/2,
                                                                         (self.tau_1 + self.tau_2)/2)
        propagator_mpos = [second_order_trotter_tau_1, second_order_trotter_tau_2]
        return propagator_mpos

    def _system_site_mpo(self, h, tau):
        """
        :param h: System site local operator
        :param tau: timestep
        :return: trotterized exponential in mpo form for the system site and the ancilla
        """
        if h is None:
            return mp.chain([mp.eye(1, self.shape[self.system_index][0]),
                             mp.eye(1, self.shape[self.system_index][1])])
        propagator = expm(-1j * tau * h)
        propagator = propagator.reshape(self.shape[self.system_index][0], self.shape[self.system_index][0])
        # Add identity to ancilla bond
        mpo = mp.chain([mp.MPArray.from_array_global(propagator, ndims=2), mp.eye(1, self.shape[self.system_index][1])])
        return self._compress_mpo(mpo)

    def _mpo_from_hi(self, hi, tau, lbond, rbond):
        """
            Generates
            U^{(s1, s2, s3), (s1', s2', s3')} = U^{(s1, s3), (s1', s3')} * delta^{s2, s2'}
            for each hi with U^{(s1, s3), (s1', s3')} = e^(-1j*\tau*hi)
            with s2/s2' ancilla site. And then appends a delta^{s4, s4'} at the end for the second ancilla.
        :param hi: Bond operator (tuple of (Eigvals, Eigvecs))
        :param tau: timestep for propagator
        :param lbond: Bond index (i) for the left site in hi
        :param rbond: Bond index for the right site in hi
        :return: e^(-1j*\tau*hi) in mpo form. A two site four legs (two physical, two ancilla on each site)
                 ready for application to a state
        """

        # Generate e^(-j*tau*hi)
        physical_legs_exp = expm(-1j * tau * hi)
        # Tensorial shape of hi for the two physical sites i and i+1 in global form
        physical_legs_tensor_shape = (self.shape[lbond][0], self.shape[rbond][0],
                                      self.shape[lbond][0], self.shape[rbond][0])
        physical_legs_exp = physical_legs_exp.reshape(physical_legs_tensor_shape)
        # Here we need to consider that there is an ancilla between the physical sites for which
        # physical_legs_exp was constructed.
        ldim_first_ancilla = self.shape[lbond][1]
        ldim_second_ancilla = self.shape[rbond][1]
        # U^((s1, s3), (s1', s3')) * delta^(s2, s2')
        physical_and_first_ancilla = np.tensordot(physical_legs_exp, np.eye(ldim_first_ancilla), axes=0)
        # Slide indices s2 and s2' between s1 and s3/1' and s3' respectively
        physical_and_first_ancilla = np.moveaxis(physical_and_first_ancilla, [-2, -1], [1, 4])
        # Add identity for second ancilla bond
        mpo = mp.chain([mp.MPArray.from_array_global(physical_and_first_ancilla, ndims=2),
                        mp.eye(1, ldim_second_ancilla)])
        return self._compress_mpo(mpo)

    def _get_right_sweep(self, trotter_exponentials, post_swap, pre_swap):
        """
            Builds a list of tuples, which contain all the operators, that are necessary for the complete sweep
            from the system site, to the right edge of the chain and back.
            Sweeping to the right, we have: an evolution operator, followed by a post_swap
            Sweeping back to the left we have: a pre-swap, followed by an evolution operator
            Both combined to a single mpo.
            Each entry in the list contains a tuple, the first element of the tuple is the index of the left one
            of the two sites in the chain, for which the above mentioned operator applies. The second element is
            the operator itself
        :param trotter_exponentials: List of trotterized unitary operators
        :param post_swap: List of swap operators to be applied after the time-evo operators
        :param pre_swap: List of swap operators to be applied before the time-evo operators
        :return: List of tuples with entries as described above
        """
        right_sweep = list()
        # sweep right
        for site in range(self.system_index+1, self.L-1):
            right_sweep.append((site-1,
                                self._compress_mpo(mp.dot(post_swap[site], trotter_exponentials[site]).group_sites(2))))
        # right edge propagation
        right_sweep.append((self.L-2, trotter_exponentials[self.L-1].group_sites(2)))
        # sweep back to the start
        for site in range(self.L-2, self.system_index, -1):
            right_sweep.append((site-1,
                                self._compress_mpo(mp.dot(trotter_exponentials[site], pre_swap[site]).group_sites(2))))
        return right_sweep

    def _get_left_sweep(self, trotter_exponentials, post_swap, pre_swap):
        """
            Builds a list of tuples, which contain all the operators, that are necessary for the complete sweep
            from the system site, to the left edge of the chain and back.
            Sweeping to the left, we have: an evolution operator, followed by a post_swap
            Sweeping back to the right we have: a pre-swap, followed by an evolution operator
            Both combined to a single mpo.
            Each entry in the list contains a tuple, the first element of the tuple is the index of the left one
            of the two sites in the chain, for which the above mentioned operator applies. The second element is
            the operator itself
        :param trotter_exponentials: List of trotterized unitary operators
        :param post_swap: List of swap operators to be applied after the time-evo operators
        :param pre_swap: List of swap operators to be applied before the time-evo operators
        :return: List of tuples with entries as described above
        """
        # System site is back at the start
        left_sweep = []
        # sweep left
        for site in range(self.system_index-1, 0, -1):
            left_sweep.append((site,
                               self._compress_mpo(mp.dot(post_swap[site], trotter_exponentials[site]).group_sites(2))))
        # left edge propagataion
        left_sweep.append((0, trotter_exponentials[0].group_sites(2)))
        # sweep back to the start
        for site in range(1, self.system_index):
            left_sweep.append((site,
                               self._compress_mpo(mp.dot(trotter_exponentials[site], pre_swap[site]).group_sites(2))))
        return left_sweep

    def _get_full_sweep(self, trotter_exponentials, post_swap, pre_swap):
        """
            Builds a list of tuples, which contain all the operators, that are necessary for the complete sweep of
            the system site through the chain and back to its initial position (with local time evolutions
            of the system as their own distinct operators) in the sense of a second order trotter evolution.
            Each entry in the list contains a tuple, the first element of the tuple is the index of the left one
            of the two sites in the chain, for which the above mentioned operator applies. The second element is
            the operator itself.
        :param trotter_exponentials: List of trotterized unitary operators
        :param post_swap: List of swap operators to be applied after the time-evo operators
        :param pre_swap: List of swap operators to be applied before the time-evo operators
        :return: List of tuples with entries as described above
        """
        full_sweep = list()
        # first local system operator (start)
        full_sweep.append((self.system_index, trotter_exponentials[self.system_index][0].group_sites(2)))
        right_sweep = self._get_right_sweep(trotter_exponentials, post_swap, pre_swap)
        full_sweep += right_sweep
        if self.system_index != 0:
            # If system is not leftmost site we are not done yet, need a left sweep
            left_sweep = self._get_left_sweep(trotter_exponentials, post_swap, pre_swap)
            full_sweep += left_sweep
        # second local system operator (end)
        full_sweep.append((self.system_index, trotter_exponentials[self.system_index][1].group_sites(2)))
        return full_sweep

    def _combine_swap_and_timeevo(self, trotter_exponentials, post_swap, pre_swap):
        """
            Builds a list of tuples, which contain all the operators, that are necessary for the complete sweep of
            the system site through the chain and back to its initial position (with local time evolutions
            of the system as their own distinct operators) in the sense of a fourth order trotter evolution.
            Each entry in the list contains a tuple, the first element of the tuple is the index of the left one
            of the two sites in the chain, for which the above mentioned operator applies. The second element is
            the operator itself.
            The entire evolution progresses through five full sweeps through the chain, which are built using
            _get_full_sweep() with the respective operators from U_2(\tau_1)/U_2(\tau_2).
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
        return timeevo_ops
