"""
    Container object for MPS second order trotter time evolution operators
"""
import mpnum as mp
from scipy.linalg import expm
from tmps.utils.swap import get_swap_mpo
from tmps.star.propagator.propagator_base import StarMPPropagatorBase


class StarMPS2OPropagator(StarMPPropagatorBase):
    def __init__(self, shape, system_index, hi_list, tau=0.01, op_compression_kwargs=None, to_cform=None):
        """
            Constructor for the MPPropagator class. Constructs propagation operators which correspond to a particular
            shape of the chain for which we wish to propagate a state (mps shape).
            Uses second order trotter decomposition for the Hamiltonian:
              U(\tau) = two sweeps of e^(-j*H_i*\tau/2),  e^(-j*H_i*\tau/2)
            Method of propagation is explained in detail in: DMRG for Multiband Impurity Solvers by Hans Gerd Evertz
        :param shape: Shape of the state (or chain on which) to propagate (in mparray shape form).
                      Only axis 0 legs are taken into account for the construction of
                      the propagator, which suffices for mps
        :param system_index: Index of the system site in the chain (place of the system site operator in the hi_list)
        :param hi_list: List/tuple for all terms in the Hamiltonian H = \sum_i hi
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
        self.ancilla_sites = False
        self.build_adj = False
        self._assert_ndims(shape)
        self.step_trotter_error = tau ** 3
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
        # For mpos we have 2 physical legs per site, check that their dimensions agree
        if init_site_legs == 2:
            # For mpos we have 2 physical legs per site, check that their dimensions agree
            for site_shape in shape:
                assert init_site_legs == len(site_shape)
                assert site_shape[0] == site_shape[1]
        else:
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
                post_swap.append(get_swap_mpo(site_shape[0], system_shape[0]))
                pre_swap.append(get_swap_mpo(system_shape[0], site_shape[0]))
            elif self.system_index < site < self.L - 1:
                post_swap.append(get_swap_mpo(system_shape[0], site_shape[0]))
                pre_swap.append(get_swap_mpo(site_shape[0], system_shape[0]))
            else:
                # Extend arrays to match length of propagator (exp^(iHt)-mpo) array
                post_swap.append(None)
                pre_swap.append(None)
        return post_swap, pre_swap

    def _build_trotter_exponentials(self, hi_list):
        """
            Builds list, which contains all trotterized bond-local exponentials (exp(-1j*\tau*hi) in mpo form.
            At the system index, the list contains only a site local operator. Everywhere else we have operators, which
            act on one bond in the chain (assuming the system is right next to them)
        :return: List of all trotterized bond-local exponentials (exp(-1j*\tau*hi) in mpo form.
        """
        propagator_mpos = []
        if self.system_index == 0:
            for site, hi in enumerate(hi_list):
                if site == 0:
                    # system site
                    mpo = self._system_site_mpo(hi, self.tau/2)
                elif 0 < site < self.L-1:
                    # Couple from system to site
                    mpo = self._mpo_from_hi(hi, self.tau/2, lbond=self.system_index, rbond=site)
                else:
                    # final site
                    mpo = self._mpo_from_hi(hi, self.tau, lbond=self.system_index, rbond=self.L-1)
                propagator_mpos.append(mpo)
        else:
            for site, hi in enumerate(hi_list):
                if site == 0:
                    mpo = self._mpo_from_hi(hi, self.tau, lbond=0, rbond=self.system_index)
                elif 0 < site < self.system_index:
                    # Couple from site to system
                    mpo = self._mpo_from_hi(hi, self.tau/2, lbond=site, rbond=self.system_index)
                elif site == self.system_index:
                    # system site mpo
                    mpo = self._system_site_mpo(hi, self.tau/2)
                elif self.system_index < site < self.L-1:
                    # Couple from system to site
                    mpo = self._mpo_from_hi(hi, self.tau/2, lbond=self.system_index, rbond=site)
                else:
                    # final site
                    mpo = self._mpo_from_hi(hi, self.tau, lbond=self.system_index, rbond=self.L-1)
                propagator_mpos.append(mpo)
        return propagator_mpos

    def _system_site_mpo(self, h, tau):
        """
        :param h: System site local operator
        :param tau: timestep
        :return: trotterized exponential in mpo form for the system site
        """
        if h is None:
            return mp.eye(1, self.shape[self.system_index][0])
        propagator = expm(-1j * tau * h)
        propagator = propagator.reshape(self.shape[self.system_index][0], self.shape[self.system_index][0])
        mpo = mp.MPArray.from_array_global(propagator, ndims=2)
        return self._compress_mpo(mpo)

    def _mpo_from_hi(self, hi, tau, lbond, rbond):
        """
            Builds e^(-j*\tau*hi) for the physical legs and generates the mpo.
            Compresses, if op_compression_kwargs is not None
        :param hi: Bond operator (tuple of (Eigvals, Eigvecs))
        :param tau: timestep for propagator
        :param lbond: Bond index for the left site in hi
        :param rbond: Bond index for the right site in hi
        :return: e^(-1j*\tau*hi) in mpo form.
        """
        # Generate e^(-j*tau*hi)
        physical_legs_exp = expm(-1j * tau * hi)
        # Tensorial shape of hi for the two physical sites i and i+1 in global form
        physical_legs_tensor_shape = (self.shape[lbond][0], self.shape[rbond][0],
                                      self.shape[lbond][0], self.shape[rbond][0])
        physical_legs_exp = physical_legs_exp.reshape(physical_legs_tensor_shape)
        mpo = mp.MPArray.from_array_global(physical_legs_exp, ndims=2)
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
        # System site is at the start
        right_sweep = list()
        # sweep right
        for site in range(self.system_index+1, self.L-1):
            right_sweep.append((site-1, self._compress_mpo(mp.dot(post_swap[site], trotter_exponentials[site]))))
        # right edge propagation
        right_sweep.append((self.L-2, trotter_exponentials[self.L-1]))
        # sweep back to the start
        for site in range(self.L-2, self.system_index, -1):
            right_sweep.append((site-1, self._compress_mpo(mp.dot(trotter_exponentials[site], pre_swap[site]))))
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
            left_sweep.append((site, self._compress_mpo(mp.dot(post_swap[site], trotter_exponentials[site]))))
        # left edge propagataion
        left_sweep.append((0, trotter_exponentials[0]))
        # sweep back to the start
        for site in range(1, self.system_index):
            left_sweep.append((site, self._compress_mpo(mp.dot(trotter_exponentials[site], pre_swap[site]))))
        return left_sweep

    def _combine_swap_and_timeevo(self, trotter_exponentials, post_swap, pre_swap):
        """
            Builds a list of tuples, which contain all the operators, that are necessary for the complete sweep of
            the system site through the chain and back to its initial position (with local time evolutions
            of the system as their own distinct operators)
            Each entry in the list contains a tuple, the first element of the tuple is the index of the left one
            of the two sites in the chain, for which the above mentioned operator applies. The second element is
            the operator itself.
        :param trotter_exponentials: List of trotterized unitary operators
        :param post_swap: List of swap operators to be applied after the time-evo operators
        :param pre_swap: List of swap operators to be applied before the time-evo operators
        :return: List of tuples with entries as described above
        """
        timeevo_ops = list()
        # first local system operator (start)
        timeevo_ops.append((self.system_index, trotter_exponentials[self.system_index]))
        right_sweep = self._get_right_sweep(trotter_exponentials, post_swap, pre_swap)
        timeevo_ops += right_sweep
        if self.system_index != 0:
            # If system is not leftmost site we are not done yet, need a left sweep
            left_sweep = self._get_left_sweep(trotter_exponentials, post_swap, pre_swap)
            timeevo_ops += left_sweep
        # second local system operator (end)
        timeevo_ops.append((self.system_index, trotter_exponentials[self.system_index]))
        self._save_op_ranks(timeevo_ops)
        return timeevo_ops
