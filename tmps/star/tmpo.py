"""
    TMPO class, which contains all the neccessary parts for performing time evolution for states represented as
    mpos.
"""

import mpnum as mp
from tmps.star.base.tmpbase import StarTMPBase
from tmps.utils.cform import canonicalize_to
from tmps.utils.compress import compress_mpa


class StarTMPO(StarTMPBase):

    def __init__(self, psi_0, propagator, state_compression_kwargs=None, t0=0, psi_0_compression_kwargs=None):
        """
            Constructor for the StarTMPO class. Time evolution for quantum states represented by mpo for a
            star geometry. Does required setup steps for evolve to work.
            Uses a pre-constructed MPPropagator object

        :param psi_0: Initial state as MPArray. More precisely as an MPArray with two physical legs on each site,
                      which have the same dimension (axis 0 = axis 1). Will be normalized before propagation
        :param propagator: Pre-constructed compatible StarMPPropagator object. (Compatibility means, that the
                           chain shape with which it was constructed matches the shape of psi_0 and that
                           it was constructed without ancilla sites, but with pre-calculated adjoints)
        :param state_compression_kwargs: Optional compression kwargs for the state (see real time evolution
                                         factory function for details)
        :param t0: Initial time of the propagation
        :param psi_0_compression_kwargs: Optional compression kwargs for the initial state (see real time evolution
                                         factory function for details)
        """
        super().__init__(psi_0, propagator, state_compression_kwargs=state_compression_kwargs, t0=t0)
        self.mpa_type = 'mpo'
        self.system_index = propagator.system_index

        # Check if propagator is valid for the selected state mpa_type
        assert not self.propagator.ancilla_sites
        assert self.propagator.build_adj
        self.axes = (-1, 0)

        self.to_cform = propagator.to_cform

        if psi_0_compression_kwargs is None:
            canonicalize_to(self.psi_t, to_cform=self.to_cform)
        else:
            compress_mpa(self.psi_t, to_cform=self.to_cform, **psi_0_compression_kwargs)

        # trace normalize current initial psi_t (=psi_0)
        self.normalize_state()

        # If we deal with an mpo state we want to track the L2-norm of the state.
        # It should be approximately conserved by the trotterized evolution assuming no significant compression losses.
        self.init_state_norm = mp.norm(psi_0)
        self.last_state_norm = self.init_state_norm
        self.curr_state_norm = self.init_state_norm

        # Contains accumulated trotter errors for each timestep
        self.trotter_error = 0

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
            Perform a Trotterized time evolution step by tau. Compress after each dot product or after each sweep
            through the chain. Renormalizes state after one full timestep due to compression.
        :return:
        """
        self.last_state_norm = self.curr_state_norm
        last_start_at = self.system_index
        if not self.canonicalize_every_step:
            canonicalize_to(self.psi_t, to_cform=self.to_cform)
        for (start_at, trotter_mpo, trotter_mpo_adj) in self.propagator.trotter_steps:
            self.psi_t = mp.partialdot(trotter_mpo, self.psi_t, start_at=start_at, axes=self.axes)
            if not self.full_compression:
                if len(trotter_mpo) > 1:
                    self.lc.compress(self.psi_t, start_at)
            self.psi_t = mp.partialdot(self.psi_t, trotter_mpo_adj, start_at=start_at, axes=self.axes)
            if self.full_compression:
                if start_at == self.propagator.L - 2 or start_at == 0:
                    # Compress after a full sweep
                    if start_at != last_start_at:
                        self.psi_t.compress(**self.state_compression_kwargs)
            else:
                if len(trotter_mpo) > 1:
                    self.lc.compress(self.psi_t, start_at)
        if self.final_compression:
            self.psi_t.compress(**self.state_compression_kwargs)
        else:
            if not self.canonicalize_every_step:
                canonicalize_to(self.psi_t, to_cform=self.to_cform)
        self._normalize_state()
        self.curr_state_norm = mp.norm(self.psi_t)
        self.trotter_error += self.propagator.step_trotter_error
        self.stepno += 1

    def _normalize_state(self):
        """
            Normalizes psi_t by dividing by its trace. Internal normalization. Can be overloaded for
            tracking purposes. Is only called during the propagation
        :return:
        """
        self.psi_t /= mp.trace(self.psi_t)

    def normalize_state(self):
        """
            Normalizes psi_t by dividing by its trace
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
                             time evolution and their respective positions in the chain (first element of each tuple))
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
