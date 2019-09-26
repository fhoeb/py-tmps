"""
    TMPO class, which contains all the neccessary parts for performing time evolution for states represented as
    mpos.
"""

import mpnum as mp
from tmps.chain.base.tmpbase import TMPBase
from tmps.utils.cform import canonicalize_to
from tmps.utils.compress import compress_mpa


class TMPO(TMPBase):
    def __init__(self, psi_0, propagator, state_compression_kwargs=None, t0=0, psi_0_compression_kwargs=None):
        """
            Constructor for the TMPO class. Time evolution for quantum states represented by mpo.
            Does required setup steps for evolve to work.
            Uses a pre-constructed MPPropagator object

        :param psi_0: Initial state as MPArray. More precisely as an MPArray with two physical legs on each site,
                      which have the same dimension (axis 0 = axis 1). Will be normalized before propagation
        :param propagator: Pre-constructed compatible MPPropagator object. (Compatibility means, that the
                           chain shape with which it was constructed matches the shape of psi_0 and that
                           it was constructed without ancilla sites, but with pre-calculated adjoints)
        :param state_compression_kwargs: Optional compression kwargs for the state (see real time evolution
                                         factory function for details)
        :param t0: Initial time of the propagation
        :param psi_0_compression_kwargs: Optional compression kwargs for the initial state (see real time evolution
                                         factory function for details)
        """
        super().__init__(psi_0, propagator, state_compression_kwargs=state_compression_kwargs, t0=t0,
                         psi_0_compression_kwargs=psi_0_compression_kwargs)
        self.mpa_type = 'mpo'
        # Propagator should not have ancilla sites here
        assert not self.propagator.ancilla_sites
        assert self.propagator.build_adj
        self.to_cform = propagator.to_cform

        if psi_0_compression_kwargs is None:
            canonicalize_to(self.psi_t, to_cform=self.to_cform)
        else:
            compress_mpa(self.psi_t, to_cform=self.to_cform, **psi_0_compression_kwargs)

        # trace normalize current initial psi_t (=psi_0)
        self.normalize_state()

        # Axes over which MPO-MPA-products happen
        self.axes = (-1, 0)

        # If we deal with an mpo state we want to track the L2-norm of the state.
        # It should be approximately conserved by the trotterized evolution assuming no significant compression losses.
        self.init_state_norm = mp.norm(psi_0)
        self.last_state_norm = self.init_state_norm
        self.curr_state_norm = self.init_state_norm

        self.len_trotter_steps = len(self.propagator.trotter_steps)
        self.len_start_trotter_steps = len(self.propagator.start_trotter_steps)
        self.len_step_trotter_steps = len(self.propagator.step_trotter_steps)
        self.len_end_trotter_steps = len(self.propagator.end_trotter_steps)

        # Contains accumulated trotter errors for each timestep
        self.trotter_error = 0

        # Flag for the fast_evolve propagation type:
        self.start = True

    @property
    def last_normdiff(self):
        """
        :return: Difference in l2 norm between the state before the last propagation step and the current state
        """
        return abs(self.last_state_norm - self.curr_state_norm)

    @property
    def cumulative_normdiff(self):
        """
        :return: Difference in l2 norm between the current and initial state
        """
        return abs(self.init_state_norm - self.curr_state_norm)

    def evolve(self):
        """
            Perform a Trotterized time evolution step by tau. Compress after each dot product. Renormalizes state after
            one full timestep due to compression.
        :return:
        """
        self.last_state_norm = self.curr_state_norm
        # Can use the symmetry of the Suzuki-Trotter decomposition here to do it in one loop and
        # the fact, that for hermitian hi we have U = e^(-i*tau*hi), U^dag = U*
        for index, (trotter_ui, trotter_ui_adj) in enumerate(self.propagator.trotter_steps):
            self.psi_t = mp.dot(trotter_ui, self.psi_t, self.axes)
            if self.canonicalize_every_step:
                self.psi_t.compress(**self.state_compression_kwargs)
            else:
                if index < self.len_trotter_steps-1:
                    self.psi_t.compress(**self.state_compression_kwargs_noc)
                else:
                    if self.canonicalize_last_step:
                        self.psi_t.compress(**self.state_compression_kwargs)
                    else:
                        self.psi_t.compress(**self.state_compression_kwargs_noc)
            self.psi_t = mp.dot(self.psi_t, trotter_ui_adj, self.axes)
            if self.canonicalize_every_step:
                self.psi_t.compress(**self.state_compression_kwargs)
            else:
                if index < self.len_trotter_steps-1:
                    self.psi_t.compress(**self.state_compression_kwargs_noc)
                else:
                    if self.canonicalize_last_step:
                        self.psi_t.compress(**self.state_compression_kwargs)
                    else:
                        self.psi_t.compress(**self.state_compression_kwargs_noc)
        self._normalize_state()
        self.curr_state_norm = mp.norm(self.psi_t)
        self.trotter_error += 2 * self.propagator.step_trotter_error
        self.stepno += 1

    def fast_evolve(self, end=True):
        """
            Perform a Trotterized time evolution step by tau. Compress after each dot product. Renormalizes state after
            one full Trotter-step due to compression.
            Without setting the end flag, the evolution does not do a complete trotter step, and the resulting state is
            therefore not immediately useable. Not setting the end flag allows for leaving out some unnecessary
            double steps and can therefore be used for propagating intermediate steps
        :parameter end: Flag, which can be set to false, if the current propagation only needs to produce an
                        intermediate step, which is not used otherwise.
        :return:
        """
        self.last_state_norm = self.curr_state_norm
        if self.start and end:
            trotter_steps = self.propagator.trotter_steps
        elif self.start and not end:
            trotter_steps = self.propagator.start_trotter_steps
            self.start = False
        elif not self.start and end:
            trotter_steps = self.propagator.end_trotter_steps
            self.start = True
        else:
            trotter_steps = self.propagator.step_trotter_steps
        if not self.canonicalize_every_step:
            canonicalize_to(self.psi_t, to_cform=self.to_cform)
        for index, (trotter_ui, trotter_ui_adj) in enumerate(trotter_steps):
            self.psi_t = mp.dot(trotter_ui, self.psi_t, self.axes)
            if self.canonicalize_every_step:
                self.psi_t.compress(**self.state_compression_kwargs)
            else:
                if index < self.len_trotter_steps-1:
                    self.psi_t.compress(**self.state_compression_kwargs_noc)
                else:
                    if self.canonicalize_last_step:
                        self.psi_t.compress(**self.state_compression_kwargs)
                    else:
                        self.psi_t.compress(**self.state_compression_kwargs_noc)
            self.psi_t = mp.dot(self.psi_t, trotter_ui.adj(), self.axes)
            if self.canonicalize_every_step:
                self.psi_t.compress(**self.state_compression_kwargs)
            else:
                if index < self.len_trotter_steps-1:
                    self.psi_t.compress(**self.state_compression_kwargs_noc)
                else:
                    if self.canonicalize_last_step:
                        self.psi_t.compress(**self.state_compression_kwargs)
                    else:
                        self.psi_t.compress(**self.state_compression_kwargs_noc)
        self._normalize_state()
        self.curr_state_norm = mp.norm(self.psi_t)
        self.trotter_error += 2 * self.propagator.step_trotter_error
        self.stepno += 1
        if self.start and not end:
            self.start = False
        elif not self.start and end:
            self.start = True

    def _normalize_state(self):
        """
            Normalizes psi_t by dividing by the trace. Internal normalization. Can be overloaded for
            tracking purposes. Is only called during the propagation
        :return:
        """
        self.psi_t /= mp.trace(self.psi_t)

    def normalize_state(self):
        """
            Normalizes psi_t by dividing by the trace
        :return:
        """
        self.psi_t /= mp.trace(self.psi_t)

    def info(self):
        """
            Returns an info dict which contains information about the current state of the propagation.
            Designed to be returned at the end of a propagation.
            The info dict contains the following keys:
            'ranks'         (ranks of psi at the end of the propagation)
            'normdiff'      (absolute difference in l2 norms acquired since the beginning of the propagation),
            'last_normdiff' (absolute difference in l2 norms acquired in the last propagation steps)
            'start_norm'    (l2 norm of the initial state)
            'norm'          (l2 norm of the current state)
            'trotter_error' (estimate of the last cumulative trotter error),
            'op_ranks'      (list which contains the bond dimensions of the trotter operators used for the
                             time evolution)
        :return: info dict
        """
        info = dict()
        info['ranks'] = self.psi_t.ranks
        info['size'] = self.psi_t.size
        info['normdiff'] = self.cumulative_normdiff
        info['last_normdiff'] = self.last_normdiff
        info['norm'] = self.curr_state_norm
        info['start_norm'] = self.init_state_norm
        info['trotter_error'] = self.trotter_error
        info['op_ranks'] = self.op_ranks
        return info

    def reset(self, psi_0=None, state_compression_kwargs=None, psi_0_compression_kwargs=None):
        """
            Resets selected properties of the TMPO object

            If psi_0 is not None, we set the new initial state to psi_0, otherwise we keep the current state.
            If psi_0 is to be changed, reset overlap and trotter error tracking.
            Checks if shape of passed psi_0 is compatible with the shape of the current propagator object

        :param psi_0: Optional new initial state, need not be normalized. Is normalized before propagation
        :param state_compression_kwargs: Optional. If not None, chamge the current parameters for
                                         state compression. If set None old compression parameters are kept,
                                         if empty, default compression is used. (see real time evolution
                                         factory function for details)
        :param psi_0_compression_kwargs: Optional compression kwargs for the initial state (see real time evolution
                                         factory function for details)
        :return:
        """
        if psi_0 is not None:
            if psi_0_compression_kwargs is not None:
                compress_mpa(psi_0, to_cform=self.to_cform, **psi_0_compression_kwargs)
            else:
                canonicalize_to(psi_0, to_cform=self.to_cform)
            # Normalize current initial psi_t (=psi_0)
            self.normalize_state()
            self.init_state_norm = mp.norm(psi_0)
            self.last_state_norm = self.init_state_norm
            self.curr_state_norm = self.init_state_norm
            self.trotter_error = 0
            self.stepno = 0
        super().reset(psi_0=psi_0, state_compression_kwargs=state_compression_kwargs)
