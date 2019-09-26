"""
    Imaginary time evolution class. Essentially the same as TMPO class except for the lack of l2-norm tracking.
"""

import numpy as np
import mpnum as mp
from tmps.chain.tmpo import TMPO


class ITMPO(TMPO):
    def __init__(self, psi_0, propagator, state_compression_kwargs=None, t0=0, psi_0_compression_kwargs=None,
                 track_trace=False):
        """
            Same as real time evolution tmp class constructor but imaginary time should be passed here
        """
        assert np.imag(propagator.tau) != 0 and np.real(propagator.tau) == 0
        super().__init__(psi_0, propagator, state_compression_kwargs=state_compression_kwargs, t0=t0,
                         psi_0_compression_kwargs=psi_0_compression_kwargs)
        self.trace_list = []
        self.trace_track = track_trace

    @property
    def state_trace(self):
        """
        :return: Trace of the current state
        """
        try:
            return np.prod(self.trace_list)
        except OverflowError:
            print('State trace too large, could not be computed')
            return None

    @property
    def t(self):
        """
            Returns current equivalent real time
        """
        return self.t0 + np.abs(np.imag(self.propagator.tau))*self.stepno

    @property
    def real_trotter_error(self):
        """
            Absolute value of the otherwise complex trotter error
        """
        return np.abs(self.trotter_error)

    @property
    def tau(self):
        """
            Returns absolute value of the imaginary part of the timestep
        :return:
        """
        return np.abs(np.imag(self.propagator.tau))

    def _normalize_state(self):
        """
            Normalizes psi_t by dividing by the trace. Is only called during the propagation.
            Tracks the trace of the density matrix during propagation
        :return:
        """
        curr_trace = mp.trace(self.psi_t)
        if self.trace_track:
            self.trace_list.append(np.abs(curr_trace))
        self.psi_t /= curr_trace

    def info(self):
        """
            Returns an info dict which contains information about the current state of the propagation.
            Designed to be returned at the end of a propagation.
            The info dict contains the following keys:
            'size'          (number of floating points used for psi_t at the end of the propagation)
            'ranks'         (bond dimensions of psi_t at the end of the propagation)
            'trotter_error' (last cumulative trotter error estimate),
            'op_ranks'      (list which contains the bond dimensions of the trotter operators used for the
                             time evolution)
        :return: info dict
        """
        info = dict()
        info['size'] = self.psi_t.size
        info['ranks'] = self.psi_t.ranks
        info['trotter_error'] = self.real_trotter_error
        info['op_ranks'] = self.op_ranks
        info['trace_list'] = self.trace_list if self.trace_track else None
        info['state_trace'] = self.state_trace if self.trace_track else None
        return info

    @classmethod
    def from_hamiltonian(cls, psi_0, ancilla_sites, build_adj, h_site, h_bond, tau=0.01, state_compression_kwargs=None,
                         op_compression_kwargs=None, second_order_trotter=False, t0=0, force_op_cform=False,
                         psi_0_compression_kwargs=None, track_trace=False):
        """
            Same general form as the parent class constructor, with an additional option to track the trace
        """
        tmp = super(TMPO, cls).from_hamiltonian(psi_0, ancilla_sites, build_adj, h_site, h_bond, tau=tau,
                                                state_compression_kwargs=state_compression_kwargs,
                                                op_compression_kwargs=op_compression_kwargs,
                                                second_order_trotter=second_order_trotter,
                                                t0=t0, force_op_cform=force_op_cform,
                                                psi_0_compression_kwargs=psi_0_compression_kwargs)
        tmp.trace_track = track_trace
        return tmp

    @classmethod
    def from_hi(cls, psi_0, ancilla_sites, build_adj, hi_list, tau=0.01, state_compression_kwargs=None,
                op_compression_kwargs=None, second_order_trotter=False, t0=0, force_op_cform=False,
                psi_0_compression_kwargs=None, track_trace=False):
        """
            Same general form as the parent class constructor, with an additional option to track the trace
        """
        tmp = super(TMPO, cls).from_hi(psi_0, ancilla_sites, build_adj, hi_list, tau=tau,
                                       state_compression_kwargs=state_compression_kwargs,
                                       op_compression_kwargs=op_compression_kwargs,
                                       second_order_trotter=second_order_trotter,
                                       t0=t0, force_op_cform=force_op_cform,
                                       psi_0_compression_kwargs=psi_0_compression_kwargs)
        tmp.trace_track = track_trace
        return tmp
