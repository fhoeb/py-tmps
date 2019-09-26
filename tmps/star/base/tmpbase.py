"""
    Base class for TMP objects (TMPS/TMPO/TPMPS) for star evolution
"""
from tmps.star.propagator.factory import get as get_propagator, get_from_hamiltonian as get_propagator_from_hamiltonian
from tmps.utils.cform import canonicalize
from tmps.utils.local_compression import LocalCompression


class StarTMPBase:

    def __init__(self, psi_0, propagator, state_compression_kwargs=None, t0=0, psi_0_compression_kwargs=None):
        """
            Constructor for the TMPBase class.
            Superclass for actual evolution clas StarTMPS
            Contains state, mpa type, propagator object and state compression arguments
            and allows access to propagator step tau and op_ranks
        :param psi_0: Initial state as MPArray
        :param propagator: Pre-constructed MPPropagator object.
        :param state_compression_kwargs: Optional compression kwargs
        :param t0: Initial time
        """
        # Check compatible shapes:
        assert propagator.shape == psi_0.shape
        self.mpa_type = None
        self.psi_t = psi_0
        self.propagator = propagator
        self.stepno = 0
        self.t0 = t0
        # Iterator variables
        self.it_max = None
        self.it_get_time = False
        self.psi_0_compression_kwargs = psi_0_compression_kwargs
        # Parameters for compression of time evolved MPS
        self.state_compression_kwargs = {'method': 'svd', 'relerr': 1e-10}
        self.canonicalize_every_step = True
        self.full_compression = False
        self.reduced_canonicalization = True
        self.final_compression = True
        self.compress_sites_step = 3
        self.compress_sites_relerr = 1e-12
        self.compress_sites_rank = None
        self.compress_sites_stable = False
        self.lc = LocalCompression(canonicalize=self.canonicalize_every_step,
                                   reduced_canonicalization=self.reduced_canonicalization,
                                   **self.state_compression_kwargs)
        if state_compression_kwargs is not None:
            self.update_compression(state_compression_kwargs)

    def __iter__(self):
        return self

    def __next__(self):
        if self.it_max is not None:
            if self.stepno >= self.it_max:
                self.it_max = None
                raise StopIteration
        self.evolve()
        if not self.it_get_time:
            return self.psi_t
        else:
            return self.psi_t, self.t

    def __call__(self, nof_steps=None, get_time=False):
        """
            Optional initialization for the iterator
        :param nof_steps: Number of steps after which the iterator should stop
        :param get_time: Return the current time and the state instead of just the state
        """
        if nof_steps is not None:
            self.it_max = self.stepno + nof_steps
        else:
            self.it_max = None
        self.it_get_time = get_time
        return self

    @property
    def op_ranks(self):
        """
            Returns ranks (and site indices) of the time evolution operators in the order in which they are
            applied to the state
        """
        return self.propagator.op_ranks

    @property
    def tau(self):
        """
            Returns timestep
        """
        return self.propagator.tau

    @property
    def t(self):
        """
            Returns current propagation time
        """
        return self.t0 + self.propagator.tau*self.stepno

    @property
    def step(self):
        """
            Returns current propagation step
        """
        return self.stepno

    def evolve(self):
        """
            Base evolve method which propagates the state
        """
        pass

    def fast_evolve(self, end=True):
        """
            There are currently no implemented optimizations for the star geometry beyond those already present in the
            standard evolve-function
        """
        self.evolve()

    def update_compression(self, state_compression_kwargs):
        """
            Interface to change compression on the fly
        :param state_compression_kwargs: Optional compression kwargs for the time evolution
        """
        if state_compression_kwargs is None:
            self.state_compression_kwargs = {'method': 'svd', 'relerr': 1e-10}
            self.canonicalize_every_step = True
            self.full_compression = False
            self.reduced_canonicalization = True
            self.final_compression = True
            self.compress_sites_step = 3
            self.compress_sites_relerr = 1e-10
            self.compress_sites_rank = None
            self.compress_sites_stable = False
        else:
            try:
                self.canonicalize_every_step = state_compression_kwargs.pop('canonicalize_every_step')
            except KeyError:
                self.canonicalize_every_step = True
            try:
                self.full_compression = state_compression_kwargs.pop('full_compression')
            except KeyError:
                self.full_compression = False
            try:
                self.reduced_canonicalization = state_compression_kwargs.pop('reduced_canonicalization')
            except KeyError:
                self.reduced_canonicalization = True
            try:
                self.final_compression = state_compression_kwargs.pop('final_compression')
            except KeyError:
                self.final_compression = True
            try:
                self.compress_sites_step = state_compression_kwargs.pop('sites_step')
            except KeyError:
                self.compress_sites_step = 3
            try:
                self.compress_sites_relerr = state_compression_kwargs.pop('sites_relerr')
            except KeyError:
                self.compress_sites_relerr = 1e-10
            try:
                self.compress_sites_rank = state_compression_kwargs.pop('sites_rank')
            except KeyError:
                self.compress_sites_rank = None
            try:
                self.compress_sites_stable = state_compression_kwargs.pop('sites_stable')
            except KeyError:
                self.compress_sites_stable = False
            self.state_compression_kwargs = state_compression_kwargs
        self.lc.update(canonicalize=self.canonicalize_every_step, reduced_canonicalization=self.reduced_canonicalization,
                       **self.state_compression_kwargs)

    def reset(self, psi_0=None, state_compression_kwargs=None):
        """
            Resets selected properties of the TMP object

            If psi_0 is not None, we set the new initial state to psi_0, otherwise we keep the current state.
            If psi_0 is to be changed, we reset overlap and trotter error tracking, thus it
            is advisable to check the info object before the reset in this case.
            Checks if shape of passed psi_0 is compatible with the shape of the current propagator object

        :param psi_0: Optional new initial state, need not be normalized. Is normalized before propagation
        :param state_compression_kwargs: Optional. If not None, chamge the current parameters for
                                         state compression. If set None old compression parameters are kept,
                                         if empty, default compression is used.
        :return:
        """
        if psi_0 is not None:
            # Check compatible shapes:
            assert self.propagator.shape == psi_0.shape
            self.psi_t = psi_0
        if state_compression_kwargs is not None:
            if not state_compression_kwargs:
                # reset to default compression
                self.update_compression(None)
            else:
                self.update_compression(state_compression_kwargs)

    @classmethod
    def from_hamiltonian(cls, psi_0, ancilla_sites, build_adj, system_index, h_site, h_bond, tau=0.01,
                         state_compression_kwargs=None, op_compression_kwargs=None, t0=0, second_order_trotter=True,
                         force_op_cform=False, psi_0_compression_kwargs=None):
        """
            Constructor which constructs a TMPS object from a hamiltonian given as site local and bond operators
            Uses:
            - either fourth order trotter decomposition of the form:
              U(tau_1)*U(tau_1)*U(tau_2)*U(tau_1)*U(tau_1)
              where each U(tau_i) is itself a second order trotter step, which alternately propagates
              odd and even bonds
            - or second order trotter:
              U(tau) = e^(-j*H_i*tau/2), e^(-j*H_i*tau/2)
              (2 sweeps, back and forth)
            Default is fourth order
    :param psi_0: Initial state as MPArray. Need not be normalized, as it is normalized before propagation
    :param ancilla_sites: Does the propagator have ancilla sites (True for pmps evolution only)
    :param system_index: Index of the system site in the chain (place of the system site operator in h_site)
    :param build_adj: If the adjoint versions of the trotter decomposition should be pre-built as well.
                      (True for mpo evolution only)
    :param h_site: Iterator over local site Hamiltonians. If a a single numpy ndarray is passed
                   this element is broadcast over all sites
    :param h_bond: Iterator over coupling Hamiltonians.
                    Ordered like this:
                        - Sites left of the system site (before system_index) couple (from left to right)
                          the current site to the system site AS IF they were directly adjacent
                        - Sites right of the system site (after system_index) couple (from left to right)
                          the system site to the current site AS IF they were directly adjacent
                    At system_index, the list/iterator may contain either None or the first coupling for the
                    site immediately to the right of the system. If a a single numpy ndarray is passed
                    this element is broadcast over all sites
    :param tau: Timestep for each invocation of evolve/fast_evolve
    :param state_compression_kwargs: Arguments for mps compression after each dot product
                                     (see real time evolution factory function for details)
    :param op_compression_kwargs: Arguments for trotter step operator precompression (see real time evolution
                                  factory function for details)
    :param second_order_trotter: Switch to use second order instead of fourth order trotter if desired
                                 By default fourth order Trotter is used
    :param t0: Initial time
    :param force_op_cform: Force canonical form of time evolution operators to match state canonical form.
                           If not set True, a default is used, depending on the state canonical form
    :param psi_0_compression_kwargs: Optional compresion kwargs for the initial state
    :return: Constructed TMPS object
        """
        try:
            canonicalize_every_step = state_compression_kwargs['canonicalize_every_step']
        except KeyError:
            canonicalize_every_step = True
        if not canonicalize_every_step or force_op_cform:
            cform = canonicalize(psi_0)
        else:
            cform = None
        propagator = get_propagator_from_hamiltonian(psi_0.shape, system_index, h_site, h_bond, tau=tau,
                                                     ancilla_sites=ancilla_sites, build_adj=build_adj,
                                                     op_compression_kwargs=op_compression_kwargs,
                                                     second_order_trotter=second_order_trotter, to_cform=cform)
        return cls(psi_0, propagator=propagator, state_compression_kwargs=state_compression_kwargs, t0=t0,
                   psi_0_compression_kwargs=psi_0_compression_kwargs)

    @classmethod
    def from_hi(cls, psi_0, ancilla_sites, build_adj, system_index, hi_list, tau=0.01, state_compression_kwargs=None,
                op_compression_kwargs=None, t0=0, second_order_trotter=True, force_op_cform=False,
                psi_0_compression_kwargs=None):
        """
            Constructor which constructs a StarTMPS object from a hamiltonian given as bond operators hi
            See from_hamiltonian for further details
        :param psi_0: Initial state as MPArray. Need not be normalized, as it is normalized before propagation
        :param ancilla_sites: Does the propagator have ancilla sites (True for pmps evolution only)
        :param build_adj: If the adjoint versions of the trotter decomposition should be pre-built as well.
                         (True for mpo evolution only)
        :param system_index: Index of the system site in the chain (place of the system site operator in the hi_list)
        :param hi_list: List/tuple for all terms in the Hamiltonian H = sum_i hi
                        Ordered like this:
                        - Sites left of the system site (before system_index) couple (from left to right)
                          the current site to the system site (and contain the site local operators)
                        - The term for the system site must be present and contains the local Hamiltonian only!
                          May be None, in which case the local Hamiltonian for the site is assumed to be 0
                        - Sites right of the system site (after system_index) couple (from left to right)
                          the system site to the current site (and contain the site local operators)
        :param tau: Timestep for each invocation of evolve
        :param state_compression_kwargs: Arguments for mps compression after each second order trotter step U(tau_i)
        :param op_compression_kwargs: Arguments for trotter step operator precompression
        :param t0: Initial time
        :param second_order_trotter: If second or fourth order trotter decomposition should be used for the evolution
        :param force_op_cform: Force canonical form of time evolution operators to match state canonical form.
                               If not set True, a default is used, depending on the state compression
        :param psi_0_compression_kwargs: Optional compresion kwargs for the initial state
        :return: Constructed TMPS object
        """
        try:
            canonicalize_every_step = state_compression_kwargs['canonicalize_every_step']
        except KeyError:
            canonicalize_every_step = True
        if not canonicalize_every_step or force_op_cform:
            cform = canonicalize(psi_0)
        else:
            cform = None
        propagator = get_propagator(psi_0.shape, system_index, hi_list, tau=tau, build_adj=build_adj,
                                    ancilla_sites=ancilla_sites, op_compression_kwargs=op_compression_kwargs,
                                    second_order_trotter=second_order_trotter, to_cform=cform)
        return cls(psi_0, propagator=propagator, state_compression_kwargs=state_compression_kwargs, t0=t0,
                   psi_0_compression_kwargs=psi_0_compression_kwargs)
