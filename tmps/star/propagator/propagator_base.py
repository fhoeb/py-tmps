"""
    Container object for trotterized time evolution operators
"""
import numpy as np
from itertools import repeat, zip_longest
from tmps.utils.cform import canonicalize_to

# TODO: Include building 20 Trotter operators and assert_ndims functions in heres


class StarMPPropagatorBase:
    def __init__(self, shape, system_index, hi_list, tau=0.01, op_compression_kwargs=None, to_cform=None):
        """
            Base Class for StarMPPropagators
        :param shape: Shape of the state (or chain on which) to propagate (in mparray shape form).
        :param system_index: Index of the system site in the chain (place of the system site operator in the hi_list)
        :param hi_list: List/tuple for all terms in the Hamiltonian H = \sum_i hi
                        Ordered like this:
                        - Sites left of the system site (before system_index) couple (from left to right)
                          the current site to the system site (and contain the site local operators for the
                          current sites only!)
                        - The term for the system site must be present and denotes the local Hamiltonian only!
                          May be None, in which case the local Hamiltonian for the site is assumed to be 0
                        - Sites right of the system site (after system_index) couple (from left to right)
                          the system site to the current site (and contain the site local operators for the
                          current sites only!)
        :param tau: Timestep for each invocation of evolve
        :param op_compression_kwargs: Arguments for operator precompression
        :param to_cform: Force canonical form of the trotter operators (None forces no canonical form, 'left' means
                         left canonical, 'right' means right canonical)
        """
        assert isinstance(hi_list, (list, tuple))
        assert len(shape) == len(hi_list)
        # Number of sites:
        self.L = len(shape)
        # Shape of the chain on which the Hamiltonian is embedded
        self.shape = shape
        assert system_index < self.L-1
        self.system_index = system_index
        # Timestep
        self.tau = tau

        # Ranks during pre-compression of bond operators (se _precompress or details)
        # Contains for each e^(-1j*tau*hi) for all tau and hi the following stats:
        # [i, rank_before, rank_after]
        self.op_ranks = []
        self.to_cform = to_cform
        # Parameters for precompression of operators
        self.op_compression_kwargs = op_compression_kwargs
        # Get swap gates
        post_swap, pre_swap = self._get_swaps()

        # Build List of mpos for Suzuki-Trotter time evolution:
        trotter_timeevo_exponentials = self._build_trotter_exponentials(hi_list)
        self.trotter_steps = self._combine_swap_and_timeevo(trotter_timeevo_exponentials, post_swap, pre_swap)

    def _get_swaps(self):
        """
            Overriden by propagator child-classes
        :return: List of swap mpos to be applied before the time step, List of swap mpos to be applied after the
                 time step
        """
        return None, None

    def _build_trotter_exponentials(self, hi_list):
        """
            Overriden by propagator child-classes
            Builds list, which contains all trotterized bond-local exponentials (exp(-1j*\tau*hi) in mpo form.
            At the system index, the list contains only a site local operator. Everywhere else we have operators, which
            act on one bond in the chain (assuming the system is right next to them)
        :return: List of all trotterized bond-local exponentials (exp(-ij*\tau*hi) in mpo form.
        """
        return [None]

    def _combine_swap_and_timeevo(self, trotter_exponentials, post_swap, pre_swap):
        """
            Overriden by propagator child-classes
            Builds a list of tuples, which contain all the operators, that are necessary for the complete sweep of
            the system site through the chain and back to its initial position (with local time evolutions
            of the system as their own distinct operators)
            Each entry in the list contains a tuple, the first element of the tuple is the index of the left one
            of the two sites in the chain, for which the above mentioned operator applies. The second element is
            the operator itself
        :param trotter_exponentials: List of trotterized unitary operators
        :param post_swap: List of swap operators to be applied after the time-evolution operators
        :param pre_swap: List of swap operators to be applied before the time-evolution operators
        :return: List of tuples with entries as described above
        """
        return []

    def _save_op_ranks(self, trotter_steps):
        """
            Pre-compresses the time evolution mpos in the list of trotter-step-tuples using the parameters in
            the op_compression_kwargs dict if passed and not None. If no pre-compression is selected, the
            op_ranks object variable will contain a list, which shows the uncompressed operator ranks only (as tuples
            of the form: (site_index, rank_before, rank_after).
            If selected, the list will contain both the compressed and uncompressed operator ranks
            (as tuples of the form: (site_index, rank_after)).
            The site index is the bath site index for couplings and the system index for the local system operator.
        :param trotter_steps: List of tuples of (starting_index, trotter step)
        """
        for propagator_tuple in trotter_steps:
            if propagator_tuple[0] >= self.system_index and len(propagator_tuple[1]) > 1:
                # To make it possible to uniquely identify the operators for sites to the right of the system site
                # (as system site and the coupling to the first site to its right share the same start_at index),
                # add one to the starting position in the chain, which pushes all subsequent indices by 1
                self.op_ranks.append((propagator_tuple[0]+1, propagator_tuple[1].ranks))
            else:
                self.op_ranks.append((propagator_tuple[0], propagator_tuple[1].ranks))

    def _compress_mpo(self, mpo):
        """
            Compresses and optionall canonicalizes a propagator mpo
        :param mpo:
        :return:
        """
        if self.op_compression_kwargs is not None:
            mpo.compress(**self.op_compression_kwargs)
            canonicalize_to(mpo, to_cform=self.to_cform)
            return mpo
        else:
            return mpo

    @staticmethod
    def construct_hi(shape, system_index, h_site, h_bond):
        """
            Combine h_site and h_bond into a sequence of hi_operators, which form the Hamiltonian via:
            H = \sum_i H_i
            H_i = h_site[i] + h_bond[i if i < system_index else i-1]
            H_system_index = h_site[system_index]
        :param shape: Shape of the state (or chain on which) to propagate (in mparray shape form).
                      Only  axis 0 legs on each site are relevant.
                      This means, that shapes from mps, pmps and mpos are treated equally.
        :param system_index: Place of the local system operator in the h_site list
        :param h_site: List of local site Hamiltonians.
        :param h_bond: List of bond Hamiltonians.
        :return: List of all hi
        """
        L = len(shape)
        assert system_index < L-1
        hi = []
        last_h_b = None
        try:
            for site, (h, h_b) in enumerate(zip_longest(h_site, h_bond)):
                if site < system_index:
                    # Couples to the right
                    hi.append(np.kron(h, np.eye(shape[system_index][0], dtype=complex)) + h_b)
                elif site == system_index:
                    # System local operator only
                    hi.append(h)
                    if h_b is not None:
                        # Case: h_bond has L-1 elements. Store current bond local operator
                        last_h_b = h_b
                else:
                    # Couples to the left
                    if last_h_b is not None:
                        hi.append(np.kron(np.eye(shape[system_index][0], dtype=complex), h) + last_h_b)
                        last_h_b = h_b
                    else:
                        hi.append(np.kron(np.eye(shape[system_index][0], dtype=complex), h) + h_b)
        except TypeError:
            raise AssertionError('h_site and h_bond must be iterable')
        except ValueError:
            raise AssertionError('Dimensions of bond operators must match with system/site dimensions')
        return hi

    @staticmethod
    def _init_hamiltonian_arrays(shape, h_site, h_bond):
        """
            Checks (if site Hamiltonians were passed as lists) correct dimension, if only 1 element was passed
            construct a suitable iterator, which broadcasts this element onto all sites.
        :param shape: Shape of the chain in mparray form
        :param h_site: Iterator/list over local site Hamiltonians
        :param h_bond: Iterator/List over coupling Hamiltonians
        :return: h_site, h_bond
        """
        if isinstance(h_site, list):
            assert len(shape) == len(h_site)
        elif isinstance(h_site, np.ndarray):
            h_site = repeat(h_site, len(shape))
        if isinstance(h_bond, list):
            assert len(shape) - 1 == len(h_bond)
        elif isinstance(h_bond, np.ndarray):
            h_bond = repeat(h_bond, len(shape) - 1)
        return h_site, h_bond

    @classmethod
    def from_hamiltonian(cls, shape, system_index, h_site, h_bond, tau=0.01, op_compression_kwargs=None, to_cform=None):
        """
            Constructor for StarMPPropagators, which takes local and coupling operators seperately.
        :param shape: Shape of the state (or chain on which) to propagate (in mparray shape form).
        :param system_index: Index of the system site in the chain (place of the system site operator in the h_site list)
        :param h_site: Iterator for all local terms in the Hamiltonian for all sites in the chain
        :param h_bond: Iterator for all System-Bath coupling terms in the Hamiltonian for all sites in the chain,
                        Ordered like this:
                        - Sites left of the system site (denoted by system index) couple (from left to right)
                          the current site to the system site
                        - Sites right of the system site (denoted by system index) couple (from left to right)
                          the system site to the current site
                        (list is shorter by one element compared to the h_site list)
        :param tau: Timestep for each invocation of evolve
        :param op_compression_kwargs: Arguments for operator precompression
        :param to_cform: Force canonical form of the trotter operators (None forces no canonical form, 'left' means
                         left canonical, 'right' means right canonical)
        """
        h_site, h_bond = cls._init_hamiltonian_arrays(shape, h_site, h_bond)
        # Divide h_site operators on h_bond operators
        hi_list = cls.construct_hi(shape, system_index, h_site, h_bond)
        return cls(shape, system_index, hi_list, tau=tau, op_compression_kwargs=op_compression_kwargs,
                   to_cform=to_cform)
