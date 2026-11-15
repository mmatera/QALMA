"""
RPAGibbsDensityOperator.

A sub-class of a GibbsDensityOperator that estimates partial traces
using RPA-like approximations.
"""

import logging
from typing import Optional

import numpy as np

from qalma.meanfield.gibbs_partial_trace import (
    gibbs_meanfield_partial_trace,
)
from qalma.meanfield.variational import variational_quadratic_mfa
from qalma.model import SystemDescriptor
from qalma.operators import Operator
from qalma.operators.basic import LocalOperator
from qalma.operators.quadratic import QuadraticFormOperator
from qalma.operators.states.gibbs import GibbsDensityOperator
from qalma.operators.states.register_ops._gibbs import _inherit_gibbs_arithmetic

from .dynamic import _dynamic_correction
from .environment import rpa_zero_modes
from .functions import rpa_entanglement_entropy, rpa_negativity
from .rpa_structure import RPAStructure, build_rpa_structure, rpa_coupling_kernel
from .utils import _STATIC_GAP, _ZERO_MODE_RTOL

LOGGER = logging.getLogger(__name__)


def rpa_generator_correction(
    generator: Operator,
    k_mf: Operator,
    sites: frozenset,
    sigma_mf,
    dynamic: bool = False,
    n_tau: int = 32,
    static_gap: float = _STATIC_GAP,
    structure: Optional[RPAStructure] = None,
) -> Optional[Operator]:
    r"""RPA correction :math:`K_{A,RPA}` to the mean-field reduced generator.

    The reduced generator is :math:`K_A' = K_{A,MF} + K_{A,RPA}`.

    Parameters
    ----------
    generator : Operator
        Generator ``K`` of the whole Gibbs state.
    k_mf : Operator
        Mean-field reduced generator :math:`K_{A,MF}` of the subsystem
        (the terms of ``K`` inside ``sites`` plus the projected boundary
        terms). It is only used by the dynamical correction.
    sites : frozenset
        Sites of the subsystem ``A``.
    sigma_mf : ProductDensityOperator
        Mean-field reference state.
    dynamic : bool, optional
        If ``True``, use the full imaginary-time kernel (dynamical
        correction, computed with the bosonized RPA) instead of the static
        approximation. This builds dense matrices of the size of the Hilbert
        space of ``A``. The default is False.
    n_tau : int, optional
        Number of imaginary-time quadrature nodes (dynamic only).
    static_gap : float, optional
        Frequency of the static bosons. Ignored if ``structure`` is given.
    structure : RPAStructure, optional
        Precomputed subsystem-independent data. If ``None``, it is built
        from ``generator`` and ``sigma_mf``.

    Returns
    -------
    Operator or None
        The correction :math:`K_{A,RPA}`, or ``None`` if it vanishes
        (for instance, if ``A`` is not coupled to the environment).

    Raises
    ------
    ValueError
        If the RPA is unstable.
    """
    if structure is None:
        structure = build_rpa_structure(generator, sigma_mf, static_gap)
    system = structure.system
    if dynamic:
        return _dynamic_correction(k_mf, structure, sites, n_tau)

    sigma, local_basis, sites_a = rpa_coupling_kernel(structure, sites)
    if sigma.size == 0 or not np.any(np.abs(sigma) > 1e-14):
        return None
    d_ops = [
        LocalOperator(site, e, system) for site in sites_a for e in local_basis[site]
    ]
    lam, vec = np.linalg.eigh(sigma)
    basis, weights = [], []
    for l_a, v_a in zip(lam, vec.T):
        if abs(l_a) < 1e-14:
            continue
        q_a = sum(c * op for c, op in zip(v_a, d_ops) if abs(c) > 1e-14)
        basis.append(q_a.simplify())
        weights.append(-0.5 * l_a)
    return QuadraticFormOperator(tuple(basis), tuple(weights), system)


# ### The RPA Gibbs state ###


class RPAGibbsDensityOperator(GibbsDensityOperator):
    r"""Gibbs state whose reductions include the RPA correction.

    It behaves as a :class:`~qalma.operators.states.gibbs.GibbsDensityOperator`,
    but :meth:`partial_trace` and :meth:`reduce` return by default the
    mean-field plus RPA approximation of the reduced state
    (:math:`K_A' = K_{A,MF} + K_{A,RPA}`, see :mod:`qalma.meanfield.rpa`).

    The data used by the RPA that do not depend on the subsystem (the
    mean-field state, the centred coupling matrix, the local bases and the
    single-site susceptibilities and bosonic modes) are computed once, on
    demand, and stored in :attr:`rpa_structure`, so they are reused by
    successive partial traces.

    Multiplying or dividing the state by a positive scalar returns an
    ``RPAGibbsDensityOperator`` that shares the stored structure. Other
    arithmetic operations give the standard operator types.

    Parameters
    ----------
    k : Operator
        The generator of the state.
    system : SystemDescriptor, optional
        The system on which the state acts.
    prefactor : float, optional
        Normalization prefactor. The default is 1.
    normalized : bool, optional
        Whether ``k`` is already normalized.
    meanfield : ProductDensityOperator, optional
        Mean-field state. If ``None``, it is computed on demand.
    symmetry_projections : tuple, optional
        Symmetry projections applied to the reduced states.
    dynamic : bool, optional
        Use by default the dynamical RPA correction instead of the static
        one. The default is False.
    n_tau : int, optional
        Number of imaginary-time quadrature nodes of the dynamical
        correction. The default is 32.
    static_gap : float, optional
        Frequency of the static bosons. The default is 1e-6.
    hybrid : bool, optional
        Use by default the non perturbative hybrid reduction, with the
        subsystem exact and the RPA only on the environment (see
        :func:`rpa_hybrid_state`). The default is False.
    hybrid_options : dict, optional
        Options of :func:`rpa_hybrid_state`.
    zero_rtol : float, optional
        Relative tolerance to identify the zero modes of the bosonic RPA
        problems, which are removed (see :func:`_bogoliubov_reduced`). The
        broken symmetry associated with them must be restored with the
        ``symmetry_projections`` of the state. The default is 1e-5.
    """

    def __init__(
        self,
        k: Operator,
        system: Optional[SystemDescriptor] = None,
        prefactor: float = 1.0,
        normalized: bool = False,
        meanfield=None,
        symmetry_projections: tuple = (),
        dynamic: bool = False,
        n_tau: int = 32,
        static_gap: float = _STATIC_GAP,
        hybrid: bool = False,
        hybrid_options: Optional[dict] = None,
        zero_rtol: float = _ZERO_MODE_RTOL,
    ):
        super().__init__(
            k,
            system=system,
            prefactor=prefactor,
            normalized=normalized,
            meanfield=meanfield,
            symmetry_projections=symmetry_projections,
        )
        self.dynamic = dynamic
        self.n_tau = n_tau
        self.static_gap = static_gap
        self.hybrid = hybrid
        self.hybrid_options = hybrid_options
        self.zero_rtol = zero_rtol
        self._rpa_structure: Optional[RPAStructure] = None

    @property
    def meanfield(self):
        """The self-consistent mean-field state (computed on demand)."""
        if self._meanfield is None:
            self._meanfield = variational_quadratic_mfa(self.k.flat())
        return self._meanfield

    @property
    def rpa_structure(self) -> RPAStructure:
        """The subsystem-independent data of the RPA (computed on demand)."""
        if self._rpa_structure is None:
            self._rpa_structure = build_rpa_structure(
                self.k, self.meanfield, self.static_gap, self.zero_rtol
            )
        return self._rpa_structure

    def partial_trace(
        self,
        sites: frozenset | SystemDescriptor,
        rpa: bool = True,
        dynamic: Optional[bool] = None,
        exact: bool = True,
        hybrid: Optional[bool] = None,
        hybrid_options: Optional[dict] = None,
    ):
        """Compute the partial trace over the complement of ``sites``.

        Parameters
        ----------
        sites : frozenset[str] or SystemDescriptor
            Sites to *keep*. All other sites are traced out.
        rpa : bool, optional
            If ``True`` (default), add the RPA correction to the mean-field
            generator of the reduced state.
        dynamic : bool, optional
            Use the dynamical RPA correction. If ``None`` (default), use
            the value given when the state was built.
        exact : bool, optional
            If ``True`` (default), small systems are reduced exactly, as in
            :class:`~qalma.operators.states.gibbs.GibbsDensityOperator`.
            If ``False``, the mean-field plus RPA reduction is always used.
        hybrid : bool, optional
            Use the non perturbative hybrid reduction. If ``None``
            (default), use the value given when the state was built.
        hybrid_options : dict, optional
            Options of :func:`rpa_hybrid_state`. If ``None``, use the ones
            given when the state was built.

        Returns
        -------
        Operator
            The reduced density operator.
        """
        if isinstance(sites, SystemDescriptor):
            sites = frozenset(sites.sites)
        return gibbs_meanfield_partial_trace(
            self,
            sites,
            rpa=rpa,
            dynamic=self.dynamic if dynamic is None else dynamic,
            n_tau=self.n_tau,
            exact=exact,
            hybrid=self.hybrid if hybrid is None else hybrid,
            hybrid_options=(
                self.hybrid_options if hybrid_options is None else hybrid_options
            ),
        )

    def reduce(self, sites, state=None, **kwargs):
        """Alias of :meth:`partial_trace`. ``state`` is ignored."""
        return self.partial_trace(sites, **kwargs)

    def rpa_zero_modes(self, sites: Optional[frozenset] = None) -> int:
        """Number of zero modes removed from the bosonic RPA problem.

        See :func:`rpa_zero_modes`.
        """
        return rpa_zero_modes(
            self.rpa_structure, None if sites is None else frozenset(sites)
        )

    def rpa_entropy(self, sites: frozenset) -> float:
        """Bosonic RPA entropy of the subsystem ``sites``, in bits.

        See :func:`rpa_entanglement_entropy`.
        """
        return rpa_entanglement_entropy(self.rpa_structure, frozenset(sites))

    def rpa_negativity(
        self, sites_b: frozenset, sites_c: Optional[frozenset] = None
    ) -> float:
        """Bosonic RPA negativity between ``sites_b`` and ``sites_c``.

        See :func:`rpa_negativity`.
        """
        return rpa_negativity(
            self.rpa_structure,
            frozenset(sites_b),
            None if sites_c is None else frozenset(sites_c),
        )

    def _with_prefactor(self, prefactor: complex) -> "RPAGibbsDensityOperator":
        """Copy of the state with a different prefactor, sharing the structure."""
        assert isinstance(prefactor, float), "prefactor must be a float."
        new_state = type(self)(
            self.k,
            system=self.system,
            prefactor=prefactor,
            normalized=self.normalized,
            meanfield=self._meanfield,
            symmetry_projections=self.symmetry_projections,
            dynamic=self.dynamic,
            n_tau=self.n_tau,
            static_gap=self.static_gap,
            hybrid=self.hybrid,
            hybrid_options=self.hybrid_options,
            zero_rtol=self.zero_rtol,
        )
        new_state._rpa_structure = self._rpa_structure
        return new_state

    def __truediv__(self, operand):
        if isinstance(operand, (int, float)) and operand > 0:
            return self._with_prefactor(self.prefactor / operand)
        return super().__truediv__(operand)


_inherit_gibbs_arithmetic(RPAGibbsDensityOperator)
