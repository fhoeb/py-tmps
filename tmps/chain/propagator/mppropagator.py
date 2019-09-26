"""
    Container object for trotterized time evolution operators
"""
import mpnum as mp
import numpy as np
from itertools import repeat, chain
from scipy.linalg import expm
from tmps.utils.cform import canonicalize_to


class MPPropagator:
    def __init__(self, shape, hi_list, tau=0.01, ancilla_sites=False, build_adj=False, op_compression_kwargs=None,
                 second_order_trotter=False, to_cform=None):
        """
            Constructor for the MPPropagator class. Contructs propagation operators which correspond to a particular
            shape of the chain for which we wish to propagate a state (mpo, pmps or mps shape).
            Uses:
            - either fourth order trotter decomposition of the form:
              U(tau_1)*U(tau_1)*U(tau_2)*U(tau_1)*(Utau_1)
              where each U(tau_i) is itself a second order trotter step, which alternately propagates
              odd and even bonds
            - or second order trotter:
              U(tau) = e^(-j*H_odd*tau/2), e^(-j*H_even*tau), e^(-j*H_odd*tau/2)
            Default is fourth order
        :param shape: Shape of the state (or chain on which) to propagate (in mparray shape form).
                      If ancilla_sites is not set True only axis 0 legs are taken into account for the construction of
                      the propagator, which suffices for mps and mpos
        :param hi_list: List/tuple for all (bond local) terms in the Hamiltonian H = sum_i hi
        :param tau: Timestep for each invocation of evolve
        :param ancilla_sites: If axis 1 legs on the chain are to be interpreted as pyhsical legs (for mpos) or
                              as ancilla legs untouched by time evolution as in the case of pmps.
        :param build_adj: If the adjoint trotter-operators should be pre-built as well (for mpo evolution)
        :param op_compression_kwargs: Arguments for operator precompression
        :param second_order_trotter: Switch to use second order instead of fourth order trotter if desired
                                     By default fourth order Trotter is used
        :param to_cform: Force canonical form of the trotter operators (None forces no canonical form, 'left' means
                         left canonical, 'right' means right canonical)
        """
        assert isinstance(hi_list, (list, tuple))
        assert len(shape)-1 == len(hi_list)
        self.ancilla_sites = ancilla_sites
        self.second_order_trotter = second_order_trotter

        # Number of sites:
        self.L = len(shape)

        self.build_adj = build_adj

        if self.build_adj and self.ancilla_sites:
            raise AssertionError('No propagator with ancilla sites and pre-built adjoint operators implemented')
        self._assert_ndims(shape)
        # Shape of the chain on which the Hamiltonian is embedded
        self.shape = shape

        # Timestep
        self.tau = tau

        # Parameters for precompression of local site operators.
        self.op_compression_kwargs = op_compression_kwargs

        self.to_cform = to_cform

        # Ranks during pre-compression of bond operators
        # Contains for each e^(-1j*tau*hi) for all tau and hi the following stats:
        # [i, rank_before, rank_after]
        self.op_ranks = []

        # Build iterator for Suzuki-Trotter time evolution
        # step_trotter_error is the trotter error accumulated in each full timestep
        # trotter_steps contains all individual trotter exponentials ui_(odd/even):
        if not self.second_order_trotter:
            # U(tau_1) U(tau_1) U(tau_2) U(tau_1) U(tau_1)
            # and each U(tau_i) = ui_odd(tau_i/2) * ui_even(tau_i) * ui_odd(tau_i/2)
            self.trotter_steps, self.start_trotter_steps, self.step_trotter_steps, self.end_trotter_steps = \
                self._build_4o_trotter_steps(hi_list, build_adj)
            self.step_trotter_error = tau ** 5
        else:
            # U(tau) = ui_odd(tau/2) * ui_even(tau) * ui_odd(tau/2)
            self.trotter_steps, self.start_trotter_steps, self.step_trotter_steps, self.end_trotter_steps = \
                self._build_2o_trotter_steps(hi_list, build_adj)
            self.step_trotter_error = tau ** 3

    def _assert_ndims(self, shape):
        """
            Checks if ndims per site are all the same, and if they are smaller or equal to 2.
            For physical legs only with two legs per site also check if the leg dimensions agree
        :param shape: state/chain shape to test for ndims
        :return:
        """
        init_site_legs = len(shape[0])
        assert init_site_legs <= 2
        if not self.ancilla_sites and init_site_legs == 2:
            # For mpos/operators we have 2 physical legs per site, check that their dimensions agree
            for site_shape in shape:
                assert init_site_legs == len(site_shape)
                assert site_shape[0] == site_shape[1]
        else:
            for site_shape in shape:
                assert init_site_legs == len(site_shape)

    def _build_even_propagator(self, hi_list, tau, id0, idL):
        """
            Builds propagator for even bonds (even i in hi): e^(-j*H_{even}*tau)
            For an even number of sites we append identities at the start and at the end.
            For an odd number of sites we append an identity at the start.
            Even/odd assumes indices start at 1, so we start with the second hi (python index 1)
        :param hi_list: List/Tuple of all terms in the Hamiltonian H = sum_i hi, where hi is local to one bond
        :param tau: timestep for propagator
        :param id0: identity mpo for the first site
        :param idL: identity mpo for the last site
        :return: e^(-j*H_{even}*\tau) as MPArray (mpo)
        """
        even_generator = self._generate_bond_propagator_mpos(hi_list, tau, 1)
        if self.L % 2 == 0:
            # Even number of sites -> need identity at last site!
            return mp.chain(chain([id0], even_generator, [idL]))
        else:
            return mp.chain(chain([id0], even_generator))

    def _build_odd_propagator(self, hi_list, tau, idL):
        """
            Builds propagator for odd bonds (even i in hi): e^(-j*H_{odd}*\tau)
            For an odd number of sites we append an identity at the end.
            Even/odd assumes indices start at 1, so we start with the first hi (python index 0)
        :param hi_list: List/Tuple of all terms in the Hamiltonian H = sum_i hi, where hi is local to one bond
        :param tau: timestep for propagator
        :param idL: Identity mpo for the last site
        :return: e^(-j*H_{odd}*\tau) as MPArray (mpo)
        """
        odd_generator = self._generate_bond_propagator_mpos(hi_list, tau, 0)
        if self.L % 2 == 0:
            return mp.chain(odd_generator)
        else:
            # Odd number of sites -> need identity at last site!
            return mp.chain(chain(odd_generator, [idL]))

    def _canonicalize(self, propagators):
        """
            Canonicalizes trotter operators (depending on to_cform)
        :param propagators: full trotter operators
        """
        for key, prop in propagators.items():
            canonicalize_to(prop, to_cform=self.to_cform)

    def _save_op_ranks(self, propagators):
        """
            Save bond dimensions of the operators to be retrieved by op_ranks property or info dict.
        :param propagators: full trotter operators
        :return:
        """
        for key, prop in propagators.items():
            self.op_ranks.append((key, prop.ranks))

    def _generate_bond_propagator_mpos(self, hi_list, tau, it_start):
        """
            Builds iterator, which successively yields all e^(-j*\tau*hi) from
            i in [it_start:L-1:it_step]. Compresses e^(-j*tau*hi) if requested (op_compression_kwargs not None)
        :param hi_list: List/Tuple of all terms in the Hamiltonian H = sum_i hi, where hi is local to one bond
        :param tau: timestep for propagator
        :param it_start: starting index in chain
        :return:
        """
        bond = it_start
        # step between indices in chain is 2 for nn-Interactions
        it_step = 2
        for hi in hi_list[it_start::it_step]:
            mpo = self._mpo_from_hi(hi, tau, bond)
            yield mpo
            bond += it_step

    def _mpo_from_hi(self, hi, tau, bond):
        """
            Convenience function, which allows to split mpo generation for 0T and finite T.
            Generates mpos for e^(-j*\tau*hi) as matrix exponential (uses scipy expm)
            For mps and mpo we just build e^(-j*\tau*hi) for the physical legs and generate the mpo.
            For pmps we first generate
            U^((s1, s2, s3), (s1', s2', s3')) = U^((s1, s3), (s1', s3')) * delta^(s2, s2')
            for each hi with U^((s1, s3), (s1', s3')) = e^(-j*tau*hi)
            with s2/s2' ancilla site. And then append a delta^(s4, s4') at the end for the second ancilla.
        :param hi: Bond operator to generate matrix exponential from
        :param tau: timestep for propagator
        :param bond: Bond index (i) for hi
        :return: e^(-j*tau*hi) in mpo form. For mps/mpo propagation is a two site two legs each operator
                 for pmps propagation is a two site four legs (two physical, two ancilla on each site)
                 ready for application to a state.
                 Pre-compresses the operator if op_compression_kwargs not None
        """
        # Generate e^(-j*tau*hi)
        physical_legs_exp = expm(-1j * tau * hi)
        # Tensorial shape of hi for the two physical sites i and i+1 in global form
        physical_legs_tensor_shape = (self.shape[bond][0], self.shape[bond + 1][0],
                                      self.shape[bond][0], self.shape[bond + 1][0])
        physical_legs_exp = physical_legs_exp.reshape(physical_legs_tensor_shape)
        if not self.ancilla_sites:
            # Need only consider physical legs here
            mpo = mp.MPArray.from_array_global(physical_legs_exp, ndims=2)
        else:
            # Here we need to consider that there is an ancilla between the physical sites for which
            # physical_legs_exp was constructed.
            ldim_first_ancilla = self.shape[bond][1]
            ldim_second_ancilla = self.shape[bond + 1][1]
            # U^((s1, s3), (s1', s3')) * delta^(s2, s2')
            physical_and_first_ancilla = np.tensordot(physical_legs_exp, np.eye(ldim_first_ancilla), axes=0)
            # Slide indices s2 and s2' between s1 and s3/1' and s3' respectively
            physical_and_first_ancilla = np.moveaxis(physical_and_first_ancilla, [-2, -1], [1, 4])
            # Add identity for second ancilla bond
            mpo = mp.chain([mp.MPArray.from_array_global(physical_and_first_ancilla, ndims=2),
                            mp.eye(1, ldim_second_ancilla)]).group_sites(2)
        if self.op_compression_kwargs is not None:
            mpo.compress(**self.op_compression_kwargs)
        return mpo

    def _build_identities(self):
        """
            Helper function, that builds suitable identity mpos for the first and the last site of the chain.
            For mpo and mps evolution we may just build them from the shape. For pmps evolution
            we have to group the legs of the two site identities to a single site
        :return: id0, idL (identity on the first site and identity for the last site respectively)
        """
        if self.ancilla_sites:
            id0 = mp.eye(2, self.shape[0])
            # Group together to a single first site with two legs
            id0 = id0.group_sites(2)

            idL = mp.eye(len(self.shape[self.L - 1]), self.shape[self.L - 1])
            # Group together to a single last site with two legs
            idL = idL.group_sites(2)
        else:
            id0 = mp.eye(1, self.shape[0][0])

            idL = mp.eye(1, self.shape[self.L - 1][0])
        return id0, idL

    def _build_4o_trotter_exponentials(self, hi_list):
        """
            Fills a dict with Suzuki-Trotter matrix-exponentials for even/odd bonds in MPO form
            Keys: odd_tau_1, odd_2*tau_1, even_tau_1, odd_tau_1+tau_2, even_tau_2
            for e^(-j*H_{odd}*\tau_1/2), e^(-j*H_{odd}*\tau_1), e^(-j*H_{even}*\tau_1),
                e^(-j*H_{odd}*(\tau_1 + \tau_2)/2), e^(-j*H_{even}*\tau_2)
        :param hi_list: List/Tuple of all terms in the Hamiltonian H = sum_i hi, where hi is local to one bond
        :return: list of the four needed trotter exponentials
        """

        # Trotter timesteps
        tau_1 = 1.0/(4 - 4**(1.0/3.0)) * self.tau
        tau_2 = -self.tau*(4**(1/3) / (4 - 4**(1/3)))
        # Identity mpos for first and last site of chain:
        id0, idL = self._build_identities()

        trotter_exponentials = {'odd_tau_1': self._build_odd_propagator(hi_list, tau_1/2, idL),
                                'odd_2*tau_1': self._build_odd_propagator(hi_list, tau_1, idL),
                                'even_tau_1': self._build_even_propagator(hi_list, tau_1, id0, idL),
                                'odd_tau_1+tau_2': self._build_odd_propagator(hi_list, (tau_1 + tau_2)/2, idL),
                                'even_tau_2': self._build_even_propagator(hi_list, tau_2, id0, idL)}
        return trotter_exponentials

    def _build_4o_trotter_steps(self, hi_list, build_adj):
        """
            List for matrix-exponentials for one full Suzuki-Trotter step.
            For fourth order ST: U(tau_1)*U(tau_1)*U(tau_2)*U(tau_1)*(Utau_1)
            where each U(tau_i) consists of 3 Matrix exponentials
        :param hi_list: List/Tuple of all terms in the Hamiltonian H = sum_i hi, where hi is local to one bond
        :param build_adj: If set True, the adjoint operators are built as well. The lists then contain two operators
                          for each element (U, U^dag).
        :return: Lists which contains the sequence of all exponentials for the full 4o Suzuki-Trotter step.
                 The first list of operators is used for evolve. The other three are used for fast_evolve
        """
        trotter_exponentials = self._build_4o_trotter_exponentials(hi_list)
        self._canonicalize(trotter_exponentials)
        self._save_op_ranks(trotter_exponentials)
        start_full = [trotter_exponentials['odd_tau_1'],
                      trotter_exponentials['even_tau_1']]
        start = [trotter_exponentials['odd_2*tau_1'],
                 trotter_exponentials['even_tau_1']]
        end = [trotter_exponentials['even_tau_1']]
        step = [trotter_exponentials['odd_2*tau_1'],
                trotter_exponentials['even_tau_1'],
                trotter_exponentials['odd_tau_1+tau_2']]
        U_full = start_full + step + [trotter_exponentials['even_tau_2']] + step[::-1] + start_full[::-1]
        U_start = start_full + step + [trotter_exponentials['even_tau_2']] + step[::-1] + end
        U_step = start + step + [trotter_exponentials['even_tau_2']] + step[::-1] + end
        U_end = start + step + [trotter_exponentials['even_tau_2']] + step[::-1] + start_full[::-1]
        if build_adj:
            trotter_exponentials_adj = {}
            for key, val in trotter_exponentials.items():
                trotter_exponentials_adj[key] = val.adj()
            start_full = [trotter_exponentials_adj['odd_tau_1'],
                          trotter_exponentials_adj['even_tau_1']]
            start = [trotter_exponentials_adj['odd_2*tau_1'],
                     trotter_exponentials_adj['even_tau_1']]
            end = [trotter_exponentials_adj['even_tau_1']]
            step = [trotter_exponentials_adj['odd_2*tau_1'],
                    trotter_exponentials_adj['even_tau_1'],
                    trotter_exponentials_adj['odd_tau_1+tau_2']]
            U_full_adj = start_full + step + [trotter_exponentials_adj['even_tau_2']] + step[::-1] + start_full[::-1]
            U_start_adj = start_full + step + [trotter_exponentials_adj['even_tau_2']] + step[::-1] + end
            U_step_adj = start + step + [trotter_exponentials_adj['even_tau_2']] + step[::-1] + end
            U_end_adj = start + step + [trotter_exponentials_adj['even_tau_2']] + step[::-1] + start_full[::-1]
            return list(zip(U_full, U_full_adj)), list(zip(U_start, U_start_adj)), \
                   list(zip(U_step, U_step_adj)), list(zip(U_end, U_end_adj))
        else:
            return U_full, U_start, U_step, U_end

    def _build_2o_trotter_exponentials(self, hi_list):
        """
            Fills a dict with Suzuki-Trotter matrix-exponentials for even/odd bonds in MPO form
            Keys: odd, even
            for e^(-j*H_odd*tau/2), e^(-j*H_even*tau)
        :return: list of the two needed trotter exponentials
        """
        # Identity mpos for first and last site of chain:
        id0, idL = self._build_identities()
        trotter_exponentials = {'odd': self._build_odd_propagator(hi_list, self.tau/2, idL),
                                'even': self._build_even_propagator(hi_list, self.tau, id0, idL),
                                'odd_2*tau': self._build_odd_propagator(hi_list, self.tau, idL)}
        return trotter_exponentials

    def _build_2o_trotter_steps(self, hi_list, build_adj):
        """
            List for matrix-exponentials for one full Suzuki-Trotter step.
            For second order Trotter: U(\tau) = e^(-j*H_{odd}*\tau/2), e^(-j*H_{even}*\tau), e^(-j*H_{odd}*\tau/2)
        :param hi_list: List/Tuple of all terms in the Hamiltonian H = sum_i hi, where hi is local to one bond
        :param build_adj: If set True, the adjoint operators are built as well. The lists then contain two operators
                          for each element (U, U^dag).
        :return: Lists which contains the sequence of all exponentials for the full 2o Trotter step.
                 The first list of operators is used for evolve. The other three are used for fast_evolve
        """
        trotter_exponentials = self._build_2o_trotter_exponentials(hi_list)
        self._canonicalize(trotter_exponentials)
        self._save_op_ranks(trotter_exponentials)
        U_full = [trotter_exponentials['odd'],
                  trotter_exponentials['even'],
                  trotter_exponentials['odd']]
        U_start = [trotter_exponentials['odd'],
                   trotter_exponentials['even']]
        U_step = [trotter_exponentials['odd_2*tau'],
                  trotter_exponentials['even']]
        U_end = [trotter_exponentials['odd_2*tau'],
                 trotter_exponentials['even'],
                 trotter_exponentials['odd']]
        if build_adj:
            trotter_exponentials_adj = {}
            for key, val in trotter_exponentials.items():
                trotter_exponentials_adj[key] = val.adj()
            U_full_adj = [trotter_exponentials_adj['odd'],
                          trotter_exponentials_adj['even'],
                          trotter_exponentials_adj['odd']]
            U_start_adj = [trotter_exponentials_adj['odd'],
                           trotter_exponentials_adj['even']]
            U_step_adj = [trotter_exponentials_adj['odd_2*tau'],
                          trotter_exponentials_adj['even']]
            U_end_adj = [trotter_exponentials_adj['odd_2*tau'],
                         trotter_exponentials_adj['even'],
                         trotter_exponentials_adj['odd']]
            return list(zip(U_full, U_full_adj)), list(zip(U_start, U_start_adj)), \
                   list(zip(U_step, U_step_adj)), list(zip(U_end, U_end_adj))
        else:
            return U_full, U_start, U_step, U_end

    @staticmethod
    def _init_hamiltonian_arrays(shape, h_site, h_bond):
        """
            Checks (if site/bond hamiltonians were passed as lists) correct dimension, if only 1 element was passed
            construct a suitable iterator, which broadcasts this element onto all sites/bonds.
        :param shape: Shape of the chain in mparray form (as returned by mpa.shape)
        :param h_site: Iterator/list over local site Hamiltonians
        :param h_bond: Iterator/list over bond Hamiltonians.
        :return: h_site, h_bond
        """
        if isinstance(h_site, list):
            assert len(shape) == len(h_site)
        elif isinstance(h_site, np.ndarray):
            h_site = repeat(h_site, len(shape))
        if isinstance(h_bond, list):
            assert len(shape) - 1 == len(h_bond)
        elif isinstance(h_bond, np.ndarray):
            h_bond = repeat(h_bond, len(shape)-1)
        return h_site, h_bond

    @staticmethod
    def construct_hi(shape, h_site, h_bond):
        """
            Splits site local contributions of hamiltonian up in two parts and constructs:
            hi_site = h_loc_i/2 + h_bond_i,i+1 + h_loc_i+1/2
            except at start and end of the chain, where the local operators contribute fully.
        :param shape: Shape of the state (or chain on which) to propagate (in mparray shape form).
                      Only the only axis 0 legs on each site are relevant.
                      This means, that shapes from mps, pmps and mpos are treated equally here.
        :param h_site: Iterator over local site Hamiltonians.
        :param h_bond: Iterator over bond Hamiltonians.
        :return: List of all hi
        """
        L = len(shape)
        # contains all split up site contributions to hi
        hi_site = []
        # split up site local contributions and divide them among the bonds
        try:
            for site, h in enumerate(h_site):
                if site == 0:
                    hi_site.append(np.kron(h, np.eye(shape[site + 1][0], dtype=complex)))
                elif site == L-1:
                    hi_site[site - 1] += np.kron(np.eye(shape[site - 1][0], dtype=complex), h)
                else:
                    hi_site.append(np.kron(h / 2, np.eye(shape[site + 1][0], dtype=complex)))
                    hi_site[site - 1] += np.kron(np.eye(shape[site - 1][0], dtype=complex), h / 2)
        except TypeError:
            raise AssertionError('h_site must be iterable')
        # add the bond operators to the site local ones
        try:
            return [site_op + bond_op for site_op, bond_op in zip(hi_site, h_bond)]
        except TypeError:
            raise AssertionError('h_bond must be iterable')
        except ValueError:
            raise AssertionError('Dimensions of bond operators must match with site dimensions')

    @classmethod
    def from_hamiltonian(cls, shape, h_site, h_bond, tau=0.01, ancilla_sites=False, op_compression_kwargs=None,
                         second_order_trotter=False, to_cform=None, build_adj=False):
        """
            Constructor for the MPPropagator class. Contructs propagation operators which correspond to a particular
            shape of the chain for which we wish to propagate a state (mpo, pmps or mps shape).
            Uses:
            - either fourth order trotter decomposition of the form:
              U(tau_1)*U(tau_1)*U(tau_2)*U(tau_1)*U(tau_1)
              where each U(tau_i) is itself a second order trotter step, which alternately propagates
              odd and even bonds
            - or second order trotter:
              U(tau) = e^(-j*H_odd*tau/2), e^(-j*H_even*tau), e^(-j*H_odd*tau/2)
            Default is fourth order
        :param shape: Shape (physical dimensions, as returned by mpnum arrays using mps.shape) of the chain to propagate
                      If ancilla_sites is not set True only axis 0 legs are taken into account for the construction of
                      the propagator, which suffices for mps and mpos
        :param h_site: Iterator over local site Hamiltonians. If a list with only 1 element is passed
                       this element is broadcast over all sites
        :param h_bond: Iterator over bond Hamiltonians. If a list with only 1 element is passed
                       this element is broadcast over all bonds
        :param tau: Timestep
        :param ancilla_sites: If the chain has ancilla sites (for pmps evolution)
        :param op_compression_kwargs: Arguments for trotter step operator pre-compression (see real time evolution
                                      factory function for details)
        :param second_order_trotter: Switch to use second order instead of fourth order trotter if desired
                                     By default fourth order Trotter is used
        :param to_cform: Force canonical form of the trotter operators (None forces no canonical form, 'left' means
                         left canonical, 'right' means right canonical)
        :param build_adj: If the adjoint trotter-operators should be pre-built as well (for mpo evolution)
        """
        # Initialize site/bond arrays
        h_site, h_bond = cls._init_hamiltonian_arrays(shape, h_site, h_bond)
        # Divide h_site operators on h_bond operators
        hi_list = cls.construct_hi(shape, h_site, h_bond)
        return cls(shape, hi_list, tau=tau, ancilla_sites=ancilla_sites, op_compression_kwargs=op_compression_kwargs,
                   second_order_trotter=second_order_trotter, to_cform=to_cform, build_adj=build_adj)
