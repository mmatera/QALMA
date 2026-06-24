r"""Classes to represent density operators as Gibbs states.

.. math::

    \rho = \frac{e^{-K}}{\mathrm{Tr}(e^{-K})}

"""

from typing import Callable, Dict, Iterable, Optional, Tuple, Union, cast

import numpy as np
from qutip import Qobj

from qalma.model import SystemDescriptor
from qalma.operators.arithmetic import OneBodyOperator
from qalma.operators.basic import (
    LocalOperator,
    Operator,
)
from qalma.operators.product import (
    ScalarOperator,
)
from qalma.operators.states.basic import (
    DensityOperatorMixin,
    DensityOperatorProtocol,
)
from qalma.operators.states.product import ProductDensityOperator
from qalma.operators.states.utils import k_by_site_from_operator
from qalma.qutip_tools.tools import is_diagonal_op, safe_exp_and_normalize

__all__ = ["GibbsDensityOperator", "GibbsProductDensityOperator"]


class GibbsDensityOperator(DensityOperatorMixin, Operator):
    r"""Density operator of the form :math:`\rho = \lambda\, e^{-K} / \mathrm{Tr}(e^{-K})`.

    Stores the operator implicitly through its generator :math:`K` rather
    than as an explicit matrix, enabling efficient representation of
    many-body Gibbs states. The full QuTiP matrix is computed on demand via
    :meth:`~qalma.operators.states.gibbs.GibbsDensityOperator.to_qutip`
    and the normalization is performed lazily on the first call.

    Parameters
    ----------
    k : Operator
        The generator operator :math:`K`. The Gibbs state is
        :math:`\rho \propto e^{-K}`.
    system : SystemDescriptor or None, optional
        Descriptor of the full lattice system. Defaults to ``k.system``.
    prefactor : float, optional
        Positive scalar weight :math:`\lambda`. Default is ``1.0``.
    normalized : bool, optional
        If ``True``, assumes :math:`K` is already normalized so that
        :math:`\mathrm{Tr}(e^{-K}) = 1`. Default is ``False``.
    meanfield : Operator or None, optional
        Pre-computed mean-field approximation. Used internally to cache
        the mean-field state and avoid redundant computation.
    symmetry_projections : tuple[Callable, ...], optional
        Sequence of projection functions to enforce symmetries after
        normalization. Default is an empty tuple (no projections).

    Attributes
    ----------
    k : Operator
        The generator :math:`K`. May be shifted during normalization.
    prefactor : complex
        Scalar weight :math:`\lambda`.
    normalized : bool
        Whether the generator has been normalized.

    """

    _free_energy: float
    normalized: bool
    prefactor: complex
    k: Operator

    def __init__(
        self,
        k: Operator,
        system: Optional[SystemDescriptor] = None,
        prefactor=1.0,
        normalized=False,
        meanfield=None,
        symmetry_projections: Tuple[Callable, ...] = tuple(),
    ):
        self.symmetry_projections = symmetry_projections

        if prefactor == 0:
            self.k = ScalarOperator(0, k.system)
            self.f_global = 0.0
            self._free_energy = 0.0
            self.normalized = normalized
            self.prefactor = 0
            self.normalized = normalized
            self.system = system if system is not None else k.system
            self._meanfield = ProductDensityOperator({}, system=self.system, weight=0)
            return

        assert prefactor > 0
        self.k = k
        assert isinstance(k, Operator)
        self.f_global = 0.0
        self._free_energy = 0.0
        self.prefactor = prefactor
        self.normalized = normalized
        self.system = system if system is not None else k.system
        self._meanfield = meanfield

    def __neg__(self):
        """Return the negation of the operator's QuTiP representation."""
        return -(self.to_qutip_operator())

    def __repr__(self):
        """Build the repr str."""
        tqo = self.to_qutip_operator()
        result = "Gibbs operator"
        result += f"\n->as Qutip Operator {type(tqo)}\n"
        result += repr(tqo)
        return result

    def __truediv__(self, operand):
        """Divide the operator by a scalar or another operator.

        Parameters
        ----------
        operand : int, float, complex, or Operator
            The divisor.

        Returns
        -------
        GibbsDensityOperator or Operator
            A new :class:`GibbsDensityOperator` with scaled prefactor if
            ``operand`` is scalar, or ``self * operand.inv()`` otherwise.

        Raises
        ------
        ValueError
            If ``operand`` is neither a scalar nor an :class:`Operator`.

        """
        if isinstance(operand, (int, float, complex)):
            return GibbsDensityOperator(
                self.k,
                self.system,
                self.prefactor / operand,
                normalized=self.normalized,
            )
        if isinstance(operand, Operator):
            return self * operand.inv()
        raise ValueError("Division of an operator by ", type(operand), " not defined.")

    def acts_over(self) -> frozenset:
        """Return the set of sites on which the operator acts non-trivially.

        Returns
        -------
        frozenset[str]
            Intersection of the sites of :math:`K` and the system sites.

        """
        return self.k.acts_over().intersection(self.system.sites)

    @property
    def free_energy(self):
        r"""Free energy :math:`F = -\log Z` where :math:`Z = \mathrm{Tr}(e^{-K})`.

        Triggers normalization on first access if not yet normalized.

        Returns
        -------
        float
            The free energy of the Gibbs state.

        """
        if not self.normalized:
            self.normalize()
        return self._free_energy

    @free_energy.setter
    def free_energy(self, value):
        """Set the free energy directly.

        Parameters
        ----------
        value : float
            The free energy value to assign.

        """
        self._free_energy = value

    def logm(self) -> Operator:
        r"""Return the matrix logarithm :math:`\log\rho = -K`.

        Normalizes the operator first to ensure :math:`\mathrm{Tr}(e^{-K})=1`.

        Returns
        -------
        Operator
            The operator :math:`-K`.

        """
        self.normalize()
        k = self.k
        return -k

    def variational_free_energy(self, ham: Operator) -> float:
        r"""Compute the variational free energy of ``ham`` under this state.

        Overrides the base-class implementation with an analytic shortcut:
        after normalization :math:`\mathrm{Tr}(e^{-K}) = 1`, so
        :math:`\log\rho = -K` exactly and

        .. math::

            F_{\rm var}[\rho, H]
                = \mathrm{Tr}[\rho\,(H + \log\rho)]
                = \mathrm{Tr}[\rho\,(H - K)].

        This avoids building the explicit :math:`H + \log\rho` operator.

        Parameters
        ----------
        ham : Operator
            The generator :math:`H` of the target Gibbs state.

        Returns
        -------
        float
            :math:`\mathrm{Tr}[\rho\,(H - K)]`.

        """
        self.normalize()
        return float(np.real(cast(complex, self.expect(ham - self.k))))

    def normalize(self) -> Operator:
        r"""Normalize :math:`K` so that :math:`\mathrm{Tr}(e^{-K}) = 1`.

        The normalization shifts :math:`K` by :math:`\log Z` and stores
        :math:`F = -\log Z` as the free energy. This is a no-op if the
        operator is already normalized.

        Returns
        -------
        Operator
            ``self``, normalized in-place.

        """
        if not self.normalized:
            self.to_qutip(cast(Tuple[str], tuple()))

        return self

    def partial_trace(self, sites: Union[frozenset, SystemDescriptor]):
        """Compute the partial trace over the complement of ``sites``.

        Uses the mean-field Gibbs partial trace, which approximates the
        reduced state as a product of local Gibbs states.

        Parameters
        ----------
        sites : frozenset[str] or SystemDescriptor
            Sites to *keep*. All other sites are traced out.

        Returns
        -------
        Operator
            The reduced density operator on the subsystem defined by
            ``sites``.

        """
        # pylint: disable=import-outside-toplevel
        from qalma.meanfield.gibbs_partial_trace import (
            gibbs_meanfield_partial_trace,
        )

        if isinstance(sites, SystemDescriptor):
            sites = frozenset(sites.sites)
        return gibbs_meanfield_partial_trace(self, sites)

    def reduce(self, sites, state=None):
        """Alias of :meth:`~qalma.operators.states.gibbs.GibbsDensityOperator.partial_trace`.

        Parameters
        ----------
        sites : frozenset[str] or SystemDescriptor
            Sites to *keep*.
        state : optional
            Ignored. Included for interface compatibility.

        Returns
        -------
        Operator
            The reduced density operator on the subsystem defined by
            ``sites``.

        """
        return self.partial_trace(sites)

    def to_qutip_operator(self):
        r"""Return a :class:`~qalma.operators.states.qutip.QutipDensityOperator` representation.

        Computes the full matrix :math:`\rho = e^{-K}/Z` as a
        ``qutip.Qobj`` and wraps it in a
        :class:`~qalma.operators.states.qutip.QutipDensityOperator`.

        Returns
        -------
        QutipDensityOperator
            The explicit matrix representation of the Gibbs state.

        """
        # pylint: disable=import-outside-toplevel
        from qalma.operators.states import QutipDensityOperator

        block = tuple(sorted(self.system.sites))
        names = {name: pos for pos, name in enumerate(block)}
        rho_qutip = self.to_qutip(block) if block else 1
        prefactor = getattr(self, "prefactor", 1.0)
        return QutipDensityOperator(
            rho_qutip, names=names, system=self.system, prefactor=prefactor
        )

    def to_qutip(self, block: Optional[Tuple[str, ...]] = None):
        """Return the QuTiP matrix representation of the Gibbs state.

        If not yet normalized, computes :math:`e^{-K}`, normalizes it, and
        stores the free energy. Subsequent calls use the cached normalized
        generator. If ``block`` is a proper subset of all system sites, the
        result is a partial trace over the missing sites.

        Parameters
        ----------
        block : tuple[str, ...] or None, optional
            Ordered list of site names for the tensor-product structure of
            the returned :class:`qutip.Qobj`. Defaults to all sites in
            lexicographical order.

        Returns
        -------
        qutip.Qobj or float
            The density matrix restricted to ``block``. Returns ``1.0`` if
            :math:`K` evaluates to a scalar.

        """
        system = self.system
        all_sites = tuple(system.sites)
        if block is None:
            block = tuple(sorted(all_sites))
        elif len(block) < len(all_sites):
            assert all(
                site in all_sites for site in block
            ), "sites must be in the (sub)system"
            block = block + tuple(
                sorted((site for site in all_sites if site not in block))
            )

        if self.normalized:
            result = (-self.k).to_qutip(block).expm()
        else:
            k_qutip = self.k.to_qutip(block)
            if not isinstance(k_qutip, Qobj):
                return 1.0
            result, log_prefactor = safe_exp_and_normalize(-k_qutip)
            self.k = self.k + log_prefactor
            self._free_energy = -log_prefactor
            self.normalized = True
            if block:
                result = result.permute(tuple(all_sites.index(site) for site in block))
            result = result.ptrace(tuple(range(len(block))))
        return result


class GibbsProductDensityOperator(DensityOperatorMixin, Operator):
    r"""Product Gibbs state :math:`\rho = \lambda \bigotimes_i \rho_i`.

    Represents a density operator that factorizes over sites:

    .. math::

        \rho = \lambda \bigotimes_i \frac{e^{-K_i}}{\mathrm{Tr}(e^{-K_i})}

    where each :math:`K_i` is a local operator on site :math:`i`. This is
    the mean-field (product state) approximation to a full Gibbs state.

    Parameters
    ----------
    k : Operator or dict[str, Qobj]
        Either a many-body operator whose one-body terms define the local
        generators :math:`K_i`, or a dict mapping site names to local
        :class:`qutip.Qobj` generators directly.
    system : SystemDescriptor or None, optional
        Descriptor of the full lattice system. Required when ``k`` is a
        dict. Defaults to ``k.system`` when ``k`` is an :class:`Operator`.
    prefactor : complex, optional
        Positive real scalar weight :math:`\lambda`. Default is ``1``.
    normalized : bool, optional
        If ``True``, assumes each local :math:`K_i` is already normalized.
        Default is ``False``.

    Attributes
    ----------
    k_by_site : dict[str, Qobj]
        Local generators :math:`K_i` stored as :class:`qutip.Qobj`, one
        per site, normalized so that :math:`\mathrm{Tr}(e^{-K_i}) = 1`.
    prefactor : complex
        Scalar weight :math:`\lambda`.
    free_energies : dict[str, float]
        Local free energies :math:`F_i = -\log Z_i` for each site.
    isherm : bool
        Always ``True`` — Gibbs states are Hermitian by construction.

    """

    k_by_site: Dict[str, Operator]
    prefactor: complex
    free_energies: Dict[str, float]
    isherm: bool = True

    def __init__(
        self,
        k: Union[Operator, Dict[str, Operator]],
        system: Optional[SystemDescriptor] = None,
        prefactor: complex = 1,
        normalized: bool = False,
    ):
        self_system: SystemDescriptor
        k_by_site: Dict[str, Operator]
        f_locals: Dict[str, float]

        assert abs(np.imag(prefactor)) == 0 and np.real(prefactor) > 0

        self.prefactor = prefactor
        if isinstance(k, dict):
            assert system is not None
            self_system = self.system = cast(SystemDescriptor, system)
            k_by_site = k
            assert all(isinstance(k_loc, Qobj) for k_loc in k_by_site.values())
        else:
            k_operator: Operator = cast(Operator, k)
            k_operator = k_operator.simplify()
            system = k_operator.system.union(system)
            self_system = self.system = system
            k_by_site = k_by_site_from_operator(k_operator)
            assert all(isinstance(k_loc, Qobj) for k_loc in k_by_site.values())

        if normalized:
            f_locals = {site: 0.0 for site in k_by_site}
        else:

            def safe_local_f(op_loc):
                assert isinstance(op_loc, Qobj)
                spectrum = (-op_loc).eigenenergies()
                f0 = max(spectrum)
                spectrum = spectrum - f0
                return -np.log(sum(np.exp(spectrum))) - f0

            f_locals = {site: safe_local_f(l_op) for site, l_op in k_by_site.items()}

            for site in k_by_site:
                k_by_site[site] = k_by_site[site] - f_locals[site]

        # Add missing terms
        for site in self_system.sites:
            if site in k_by_site:
                continue
            f_local = np.log(self_system.dimensions[site])
            f_locals[site] = -f_local
            k_by_site[site] = self_system.site_identity(site) * f_local

        self.free_energies = f_locals
        self.k_by_site = k_by_site

    def __neg__(self):
        """Return the negation of the product state representation."""
        return -(self.to_product_state())

    def __repr__(self):
        """Build the repr str."""
        result = "Gibbs Product density operator:\n"
        result += "\n".join(
            f"{site}:exp(-1*{op})" for site, op in self.k_by_site.items()
        )
        result += f"\n free energies:{self.free_energies}"
        return result

    def acts_over(self) -> frozenset:
        """Return the set of sites on which the operator acts non-trivially.

        Returns
        -------
        frozenset[str]
            Names of all sites with a non-trivial local generator
            :math:`K_i`.

        """
        return frozenset(site for site in self.k_by_site)

    def expect(
        self,
        obs_objs: Union[Operator, Iterable],
        _local_states: Optional[Dict[frozenset, DensityOperatorProtocol]] = None,
    ) -> Union[np.ndarray, dict, complex]:
        r"""Compute the expectation value :math:`\langle O \rangle_\rho`.

        Delegates to the :class:`ProductDensityOperator` representation,
        which computes expectation values efficiently using the product
        structure.

        Parameters
        ----------
        obs_objs : Operator or Iterable[Operator]
            Observable or collection of observables whose expectation values
            are computed.
        _local_states : dict[frozenset, DensityOperatorProtocol] or None, optional
            Pre-computed local states for subsystems. Used internally to
            avoid redundant partial traces.

        Returns
        -------
        complex or np.ndarray or dict
            Expectation value(s) of the observable(s).

        """
        return (self.to_product_state()).expect(obs_objs, _local_states=_local_states)

    @property
    def isdiagonal(self) -> bool:
        """``True`` if all local generators are diagonal.

        Returns
        -------
        bool
            ``True`` only when every :math:`K_i` is a diagonal matrix.

        """
        for operator in self.k_by_site.values():
            if not is_diagonal_op(operator):
                return False
        return True

    def logm(self) -> Operator:
        r"""Return the matrix logarithm :math:`\log\rho = -\sum_i K_i`.

        Exploits the product structure: since :math:`\rho = \bigotimes_i
        e^{-K_i}`, we have :math:`\log\rho = -\sum_i K_i`.

        Returns
        -------
        OneBodyOperator
            The sum :math:`-\sum_i K_i` as a one-body operator.

        """
        terms = tuple(
            LocalOperator(site, -loc_op, self.system)
            for site, loc_op in self.k_by_site.items()
        )
        return OneBodyOperator(terms, self.system, False)

    def partial_trace(self, sites: Union[frozenset, SystemDescriptor]):
        """Compute the partial trace over the complement of ``sites``.

        For a product state, the partial trace simply discards the local
        factors of the traced-out sites.

        Parameters
        ----------
        sites : frozenset[str] or SystemDescriptor
            Sites to *keep*.

        Returns
        -------
        GibbsProductDensityOperator
            The reduced product Gibbs state on the subsystem defined by
            ``sites``.

        """
        if isinstance(sites, SystemDescriptor):
            subsystem = sites
            sites = frozenset(
                (site for site in subsystem.sites if site in self.system.dimensions)
            )
        else:
            subsystem = self.system.subsystem(sites)

        k_by_site = {
            site: localstate
            for site, localstate in self.k_by_site.items()
            if site in sites
        }
        return GibbsProductDensityOperator(
            k_by_site,
            subsystem,
            self.prefactor,
            True,
        )

    def reduce(self, sites, state=None):
        """Alias of :meth:`~qalma.operators.states.gibbs.GibbsProductDensityOperator.partial_trace`.

        Parameters
        ----------
        sites : frozenset[str] or SystemDescriptor
            Sites to *keep*.
        state : optional
            Ignored. Included for interface compatibility.

        Returns
        -------
        GibbsProductDensityOperator
            The reduced product Gibbs state on the subsystem defined by
            ``sites``.

        """
        return self.partial_trace(sites)

    def to_product_state(self):
        r"""Convert to an explicit :class:`ProductDensityOperator`.

        Computes each local density matrix :math:`\rho_i = e^{-K_i}` and
        assembles them into a :class:`ProductDensityOperator`.

        Returns
        -------
        ProductDensityOperator
            The product state with explicit local density matrices.

        """
        local_states = {
            site: (-local_k).expm() for site, local_k in self.k_by_site.items()
        }
        return ProductDensityOperator(
            local_states,
            self.prefactor,
            system=self.system,
            normalized=False,
        )

    def to_qutip(self, block: Optional[Tuple[str, ...]] = None):
        r"""Return the QuTiP matrix representation of the product Gibbs state.

        Delegates to :meth:`~qalma.operators.states.gibbs.GibbsProductDensityOperator.to_product_state`
        and then calls its
        :meth:`~qalma.operators.states.product.ProductDensityOperator.to_qutip` method.

        Parameters
        ----------
        block : tuple[str, ...] or None, optional
            Ordered list of site names defining the tensor-product structure.
            Defaults to all sites in lexicographical order.

        Returns
        -------
        qutip.Qobj
            The full density matrix :math:`\bigotimes_i \rho_i` over
            ``block``.

        """
        return self.to_product_state().to_qutip(block)
