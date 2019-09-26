"""
    Base class for TMP objects (TMPS/TMPO/TPMPS) for chain evolution
"""
from tmps.chain.propagator.mppropagator import MPPropagator
from tmps.utils.cform import canonicalize


class TMPBase:

    def __init__(self, psi_0, propagator, state_compression_kwargs=None, t0=0, psi_0_compression_kwargs=None):
        """
            Constructor for the TMPBase class.
            Contains state, mpa type, propagator object and state compression arguments
            and allows access to propagator step tau and op_ranks

        :param psi_0: Initial state as MPArray
        :param propagator: Pre-constructed MPPropagator object.
        :param state_compression_kwargs: Optional compression kwargs for the time evolution
        :param t0: Initial time
        :param psi_0_compression_kwargs: Optional compresion kwargs for the initial state
        """
        # Check compatible shapes:
        assert propagator.shape == psi_0.shape
        self.psi_t = psi_0
        self.propagator = propagator
        self.stepno = 0
        self.t0 = t0
        self.mpa_type = None
        # Iterator variables
        self.it_max = None
        self.it_get_time = False
        self.psi_0_compression_kwargs = psi_0_compression_kwargs
        # Default compression parameters
        self.state_compression_kwargs = {'method': 'svd', 'relerr': 1e-10}
        self.state_compression_kwargs_noc = {'method': 'svd', 'relerr': 1e-10, 'canonicalize': False}
        self.canonicalize_every_step = True
        self.canonicalize_last_step = True
        self.compress_sites_step = 3
        self.compress_sites_relerr = 1e-12
        self.compress_sites_rank = None
        self.compress_sites_stable = False
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
                         like so: for psi_t, t in propagator(nof_steps=x, get_time=True)
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
            Returns ranks of the time evolution operators in the order in which they are applied to the state
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
            Optional fast_evolve, which can do some optimizations if the state after the evolution is not
            immediately required
        :param end: If set False, the state after the evolution is not fully propagated. If set True, the state is
                    fully propagated.
        """
        self.evolve()

    def update_compression(self, state_compression_kwargs):
        """
            Interface to change compression on the fly
        :param state_compression_kwargs: Optional compression kwargs for the time evolution
        """
        # Parameters for compression of time evolved MPS
        # Noc implements choice to canonicalize mps only at certain points
        if state_compression_kwargs is None:
            self.state_compression_kwargs = {'method': 'svd', 'relerr': 1e-10}
            self.state_compression_kwargs_noc = {'method': 'svd', 'relerr': 1e-10, 'canonicalize': False}
            self.canonicalize_every_step = True
            self.canonicalize_last_step = True
            self.compress_sites_step = 3
            self.compress_sites_relerr = 1e-12
            self.compress_sites_rank = None
            self.compress_sites_stable = False
        else:
            try:
                self.canonicalize_every_step = state_compression_kwargs.pop('canonicalize_every_step')
            except KeyError:
                self.canonicalize_every_step = True
            try:
                self.canonicalize_last_step = state_compression_kwargs.pop('canonicalize_last_step')
            except KeyError:
                self.canonicalize_last_step = True
            try:
                self.compress_sites_step = state_compression_kwargs.pop('sites_step')
            except KeyError:
                self.compress_sites_step = 2
            try:
                self.compress_sites_relerr = state_compression_kwargs.pop('sites_relerr')
            except KeyError:
                self.compress_sites_relerr = 1e-12
            try:
                self.compress_sites_rank = state_compression_kwargs.pop('sites_rank')
            except KeyError:
                self.compress_sites_rank = None
            try:
                self.compress_sites_stable = state_compression_kwargs.pop('sites_stable')
            except KeyError:
                self.compress_sites_stable = False
            self.state_compression_kwargs = state_compression_kwargs
            state_compression_kwargs_noc = state_compression_kwargs.copy()
            state_compression_kwargs_noc['canonicalize'] = False
            self.state_compression_kwargs_noc = state_compression_kwargs_noc

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
    def from_hamiltonian(cls, psi_0, ancilla_sites, build_adj, h_site, h_bond, tau=0.01, state_compression_kwargs=None,
                         op_compression_kwargs=None, second_order_trotter=False, t0=0, force_op_cform=False,
                         psi_0_compression_kwargs=None):
        """
            Constructor which constructs a TMP object from a hamiltonian given as site local and bond operators
            Uses:
            - either fourth order trotter decomposition of the form:
              U(\tau_1)*U(\tau_1)*U(\tau_2)*U(\tau_1)*U(\tau_1)
              where each U(\tau_i) is itself a second order trotter step, which alternately propagates
              odd and even bonds
            - or second order trotter:
              U(\tau) = e^(-j*H_{odd}*\tau/2), e^(-j*H_{even}*\tau), e^(-j*H_{odd}*\tau/2)
            Default is fourth order

        :param psi_0: Initial state as MPArray. Need not be normalized, as it is normalized before propagation
        :param ancilla_sites: Does the propagator have ancilla sites (True for pmps evolution only)
        :param build_adj: If the adjoint versions of the trotter decomposition should be pre-built as well.
                          (True for mpo evolution only)
        :param h_site: Iterator over local site Hamiltonians. If a single numpy ndarray is passed
                       this element is broadcast over all sites
        :param h_bond: Iterator over coupling Hamiltonians. If a single numpy ndarray is passed
                       this element is broadcast over all bonds
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
        :param psi_0_compression_kwargs: Optional compression kwargs for the initial state
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
        propagator = MPPropagator.from_hamiltonian(psi_0.shape, h_site, h_bond, tau=tau,
                                                   ancilla_sites=ancilla_sites,
                                                   op_compression_kwargs=op_compression_kwargs,
                                                   second_order_trotter=second_order_trotter, to_cform=cform,
                                                   build_adj=build_adj)
        return cls(psi_0, propagator=propagator, state_compression_kwargs=state_compression_kwargs,
                   t0=t0, psi_0_compression_kwargs=psi_0_compression_kwargs)

    @classmethod
    def from_hi(cls, psi_0, ancilla_sites, build_adj, hi_list, tau=0.01, state_compression_kwargs=None,
                op_compression_kwargs=None, second_order_trotter=False, t0=0, force_op_cform=False,
                psi_0_compression_kwargs=None):
        """
            Constructor which constructs a TMP object from a hamiltonian given as bond operators hi.

        :param psi_0: Initial state as MPArray. Need not be normalized, as it is normalized before propagation
        :param ancilla_sites: Does the propagator have ancilla sites (True for pmps evolution only)
        :param build_adj: If the adjoint versions of the trotter decomposition should be pre-built as well.
                         (True for mpo evolution only)
        :param hi_list: List/Tuple of all terms in the Hamiltonian H = sum_i hi, where hi is local to one bond
        :param tau: Timestep for each invocation of evolve
        :param state_compression_kwargs: Arguments for mps compression after each dot product
                                         (see real time evolution factory function for details)
        :param op_compression_kwargs: Arguments for trotter step operator precompression (see real time evolution
                                      factory function for details)
        :param second_order_trotter: Switch to use second order instead of fourth order trotter if desired
                                     By default fourth order Trotter is used
        :param t0: Initial time
        :param force_op_cform: Force canonical form of time evolution operators to match state canonical form.
                               If not set True, a default is used, depending on the state compression
        :param psi_0_compression_kwargs: Optional compression kwargs for the initial state
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
        propagator = MPPropagator(psi_0.shape, hi_list, tau=tau,
                                  ancilla_sites=ancilla_sites,
                                  op_compression_kwargs=op_compression_kwargs,
                                  second_order_trotter=second_order_trotter, to_cform=cform, build_adj=build_adj)
        return cls(psi_0, propagator=propagator, state_compression_kwargs=state_compression_kwargs,
                   t0=t0, psi_0_compression_kwargs=psi_0_compression_kwargs)
