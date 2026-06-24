"""Different representations for operators."""

import logging
import warnings
from functools import cached_property, reduce

# from types import MappingProxyType
from typing import Dict, Iterable, Optional, Tuple, Union

import numpy as np
from qutip import Qobj as _Qobj
from scipy.linalg import expm as _scp_expm, logm as _scp_logm
from scipy.linalg._matfuncs_inv_ssq import LogmExactlySingularWarning

from qalma.model import SystemDescriptor
from qalma.qutip_tools.tools import (
    _to_array,
    fast_tensor,
    is_diagonal_op,
    is_empty_op,
    is_scalar_op,
    ishermitian,
    norm,
    to_qobj,
)
from qalma.settings import (
    QALMA_TOLERANCE,
)

from .basic import LocalOperator, Operator

# from scipy.linalg import ishermitian


__all__ = ["ProductOperator", "ScalarOperator"]


class ProductOperator(Operator):
    r"""Tensor product of local operators acting on different sites.

    Represents an operator of the form

    .. math::

        \lambda \bigotimes_{i \in S} O_i

    where :math:`\lambda` is a scalar prefactor, :math:`S` is a subset of
    lattice sites, and each :math:`O_i` is a local operator acting on site
    :math:`i`. Sites not in :math:`S` are implicitly acted on by the identity.

    Attributes
    ----------
    prefactor : complex
        Global scalar prefactor :math:`\lambda`.
    site_factors : dict[str, np.ndarray]
        Mapping from site name to local operator matrix (numpy array).
    system : SystemDescriptor
        Descriptor of the full lattice system.

    """

    _to_qutip_cache: Dict[Optional[Tuple[str, ...]], _Qobj]
    prefactor: complex
    site_factors: Dict[str, np.ndarray]
    system: SystemDescriptor

    def __init__(
        self,
        sites_operators: dict,
        prefactor: complex = 1.0,
        system: Optional[SystemDescriptor] = None,
        _qutip_factors: Optional[Dict[str, _Qobj]] = None,
    ):
        """Initialize a ProductOperator.

        Parameters
        ----------
        sites_operators : dict[str, np.ndarray or Qobj or scalar]
            Mapping from site name to local operator. Scalar values are
            absorbed into ``prefactor`` and removed from the dict.
        prefactor : complex, optional
            Global scalar prefactor. Default is 1.0.
        system : SystemDescriptor
            Descriptor of the full lattice system. Must not be ``None``.
        _qutip_factors : dict[str, Qobj], optional
            Pre-computed QuTiP representations of the local factors.
            Used internally to avoid redundant conversions.

        """
        assert system is not None
        remove_numbers = False
        for site, local_op in sites_operators.items():
            if isinstance(local_op, (int, float, complex)):
                prefactor *= local_op
                remove_numbers = True

        if remove_numbers:
            sites_operators = {
                s: local_op
                for s, local_op in sites_operators.items()
                if not isinstance(local_op, (int, float, complex))
            }

        if all(isinstance(value, _Qobj) for value in sites_operators.values()):
            self.__dict__["site_factors_qutip"] = sites_operators
        elif _qutip_factors is not None:
            self.__dict__["site_factors_qutip"] = _qutip_factors

        sites_operators = {key: _to_array(op) for key, op in sites_operators.items()}
        self.site_factors = sites_operators
        if any(is_empty_op(op) for op in sites_operators.values()):
            prefactor = 0
            self.site_factors = {}
        self.prefactor = prefactor
        assert isinstance(prefactor, (int, float, complex)), f"{type(prefactor)}"
        self.system = system
        if system is not None:
            self.size = len(system.sites)
            self.dimensions = {
                name: site["dimension"] for name, site in system.sites.items()
            }
        self._to_qutip_cache = {}

    @cached_property
    def site_factors_qutip(self) -> Dict[str, _Qobj]:
        """Qutip representations of the local site factors.

        Returns
        -------
        dict[str, Qobj]
            Mapping from site name to the corresponding :class:`qutip.Qobj`
            local operator. Computed once and cached.

        """
        return {key: to_qobj(op.copy()) for key, op in self.site_factors.items()}

    @cached_property
    def _dense_tensor(self):
        """Build the stacked dense representation for homogeneous systems.

        Only valid where every site has the same local Hilbert-space dimension d.

        Returns ``(sites, tensor)`` where:
          - ``sites``  is a sorted tuple of site names matching axis-0, and
          - ``tensor`` is a complex128 ndarray of shape ``(N, d, d)``.

        Raises ``ValueError`` for heterogeneous systems; callers should
        catch it and fall back to iterating over ``_dense``.
        """
        if not self.site_factors:
            return (), np.empty((0,), dtype=np.complex128)
        dense = self.site_factors
        sites = tuple(sorted(dense))
        shapes = {dense[s].shape for s in sites}
        if len(shapes) > 1:
            raise ValueError(
                "ProductOperator._dense_tensor: heterogeneous site dimensions "
                f"{shapes}; use _dense instead."
            )
        return sites, np.stack([dense[s] for s in sites])  # (N, d, d)

    @staticmethod
    def _trace2(a: np.ndarray, b: np.ndarray) -> complex:
        """Tr(a @ b) without allocating an intermediate matrix.

        Equivalent to ``np.einsum('ij,ji->', a, b)`` but avoids einsum's
        fixed Python overhead, which dominates for the small matrices
        (d=2,3,4) typical in spin/boson lattice models.
        """
        return complex((a * b.T).sum())

    def __bool__(self):
        """Return ``True`` if the operator is non-zero.

        An operator is considered zero if its prefactor is zero or if
        any local factor is the zero matrix.
        """
        return bool(self.prefactor) and all(
            factor.any() for factor in self.site_factors.values()
        )

    def __neg__(self):
        """Return the negation of the operator."""
        return ProductOperator(self.site_factors, -self.prefactor, self.system)

    def __pow__(self, exp):
        r"""Return the operator raised to the power ``exp``.

        Each local factor is raised independently to ``exp``, and the
        prefactor is raised to ``exp`` as well.

        Parameters
        ----------
        exp : int or float
            The exponent.

        Returns
        -------
        ProductOperator
            The operator :math:`(\lambda \bigotimes O_i)^{\text{exp}}`.

        """
        return ProductOperator(
            {s: op**exp for s, op in self.site_factors_qutip.items()},
            self.prefactor**exp,
            self.system,
        )

    def __repr__(self):
        """Return a human-readable string representation of the product operator."""
        result = "  " + str(self.prefactor) + " * (\n  "
        result += "  (x)\n  ".join(
            f"({item[1].full()} <-  {item[0]})"
            for item in sorted(self.site_factors_qutip.items(), key=lambda x: x[0])
        )
        result += "\n   )"
        return result

    def _repr_latex(self):
        """Latex representation."""
        factors_latex = []
        for site, qutip_op in self.site_factors_qutip.items():
            # pylint: disable=protected-access
            tex = qutip_op._repr_latex_().replace("$$", "$")
            parts = tex.split("$")
            if len(parts) == 3:
                tex = parts[1]
            else:
                tex = "-?-"

            prefactor = self.prefactor
            if prefactor == 1:
                factors_latex.append(tex + "_{" + site + "}")
            elif prefactor < 0:
                factors_latex.append(f"({prefactor}) *" + tex + "_{" + site + "}")
            else:
                factors_latex.append(f"{prefactor} *" + tex + "_{" + site + "}")
        return "$" + "\\otimes".join(factors_latex) + "$"

    def acts_over(self) -> frozenset:
        """Return the set of sites on which this operator acts non-trivially.

        Returns
        -------
        frozenset[str]
            Site names with a non-identity local factor.

        """
        return frozenset(site for site in self.site_factors)

    def dag(self):
        r"""Return the adjoint (Hermitian conjugate) of the operator.

        Each local factor is conjugate-transposed and the prefactor is
        complex-conjugated.

        Returns
        -------
        ProductOperator
            The operator :math:`(\lambda \bigotimes O_i)^\dagger`.

        """
        sites_op_dag = {key: op.T.conj() for key, op in self.site_factors.items()}
        prefactor = self.prefactor
        if isinstance(prefactor, complex):
            prefactor = prefactor.conjugate()
        return ProductOperator(sites_op_dag, prefactor, self.system)

    def expm(self):
        r"""Return the matrix exponential :math:`e^{\lambda O}`.

        For single-site operators the exponential is computed exactly via
        ``scipy.linalg.expm``. For multi-site operators falls back to
        the base-class implementation.

        Returns
        -------
        Operator
            The matrix exponential of the operator.

        """
        sites_op = self.site_factors
        n_ops = len(sites_op)
        if n_ops == 0:
            return ScalarOperator(np.exp(self.prefactor), self.system)

        if n_ops == 1:
            site, operator = next(iter(sites_op.items()))

            result = LocalOperator(
                site, _scp_expm(self.prefactor * operator), self.system
            )
            return result
        result = super().expm()
        return result

    def flat(self):
        """Reduce the operator to the simplest equivalent type.

        Returns
        -------
        ScalarOperator
            If the operator has no site factors.
        LocalOperator
            If the operator acts on exactly one site.
        ProductOperator
            Otherwise, returns ``self`` unchanged.

        """
        nfactors = len(self.site_factors)
        if nfactors == 0:
            return ScalarOperator(self.prefactor, self.system)
        if nfactors == 1:
            name, op_factor = list(self.site_factors_qutip.items())[0]
            return LocalOperator(name, self.prefactor * op_factor, self.system)
        return self

    def hermitian_part(self):
        r"""Return the Hermitian part of the operator, ``(O + O†) / 2``.

        Returns
        -------
        Operator
            The Hermitian part. Returns ``self`` if the operator is already
            Hermitian.

        """
        from qalma.operators import SumOperator

        if self.isherm:
            return self
        if all(ishermitian(op) for op in self.site_factors.values()):
            return ProductOperator(
                self.site_factors, np.real(self.prefactor), self.system
            )
        half_self = self * 0.5
        return SumOperator(
            (half_self, half_self.dag()), system=self.system, isherm=True
        )

    def inv(self):
        """Return the inverse operator :math:`O^{-1}`.

        Each local factor is inverted independently and the prefactor is
        reciprocated.

        Returns
        -------
        Operator
            The inverse of the operator.

        """
        sites_op = self.site_factors_qutip
        system = self.system
        prefactor = self.prefactor

        n_ops = len(sites_op)
        sites_op = {site: op_local.inv() for site, op_local in sites_op.items()}
        if n_ops == 1:
            site, op_local = next(iter(sites_op.items()))
            return LocalOperator(site, op_local / prefactor, system)
        return ProductOperator(sites_op, 1 / prefactor, system)

    @cached_property
    def isherm(self) -> bool:
        """``True`` if the operator is Hermitian.

        An operator is Hermitian if all local factors are Hermitian and
        the prefactor is real (up to ``QALMA_TOLERANCE``).
        """
        # TODO: check if it worth to check that factors are not hermitian
        # up to a phase factor.
        if not all(ishermitian(loc_op) for loc_op in self.site_factors.values()):
            return False
        prefactor = self.prefactor
        if isinstance(prefactor, (int, float, np.float64)):
            return True
        if isinstance(prefactor, (complex, np.complex128)):
            return abs(prefactor.imag) < QALMA_TOLERANCE
        return False

    @cached_property
    def isdiagonal(self) -> bool:
        """``True`` if the operator is diagonal in the site-local basis.

        Returns ``True`` only when every local factor is a diagonal
        matrix.
        """
        for factor_op in self.site_factors.values():
            if not is_diagonal_op(factor_op):
                return False
        return True

    def logm(self):
        r"""Return the matrix logarithm of the operator.

        Uses the identity :math:`\log(\lambda \bigotimes_i O_i) =
        \log\lambda + \sum_i \log O_i` valid when the local factors
        commute. Each local logarithm is computed via
        ``scipy.linalg.logm``.

        Returns
        -------
        Operator
            The matrix logarithm as a :class:`OneBodyOperator` plus a
            scalar term.

        """
        # pylint: disable=import-outside-toplevel
        from qalma.operators.arithmetic import OneBodyOperator

        system = self.system
        warnings.filterwarnings("ignore", category=LogmExactlySingularWarning)
        terms = tuple(
            LocalOperator(site, _scp_logm(loc_op), system)
            for site, loc_op in self.site_factors.items()
        )
        warnings.resetwarnings()
        result = OneBodyOperator(terms, system, False)
        result = result + ScalarOperator(np.log(self.prefactor), system)
        return result

    def norm(self, ord=None):
        """Return the norm of the operator.

        The norm is computed as the product of the prefactor and the local
        norms. For Frobenius and nuclear norms, an additional factor
        accounting for the identity contribution from sites not in
        ``site_factors`` is included.

        Parameters
        ----------
        ord : str or None, optional
            Order of the norm. Supported values are ``None`` (operator norm),
            ``'fro'`` (Frobenius), and ``'nuc'`` (nuclear). Default is
            ``None``.

        Returns
        -------
        float
            The norm of the operator.

        """
        result = self.prefactor
        for op_loc in self.site_factors_qutip.values():
            result *= norm(op_loc, ord)

        if ord in ("fro", "nuc"):
            dim_factor = 1.0
            for dim in (
                dim
                for site, dim in self.system.dimensions.items()
                if site not in self.site_factors
            ):
                dim_factor *= dim
            if ord == "fro":
                result *= dim_factor**0.5
            else:
                result *= dim_factor

        return result

    def partial_trace(self, sites: Union[frozenset, SystemDescriptor]):
        """Compute the partial trace over the complement of ``sites``.

        Parameters
        ----------
        sites : frozenset[str] or SystemDescriptor
            Sites to *keep*. All other sites are traced out.

        Returns
        -------
        Operator
            The reduced operator acting on the subsystem defined by ``sites``.
            Returns a :class:`ScalarOperator` if all sites are traced out.

        """
        full_system_sites = self.system.sites
        dimensions = self.dimensions

        if isinstance(sites, SystemDescriptor):
            subsystem = sites
            sites = frozenset(sites.sites.keys())
        else:
            subsystem = self.system.subsystem(sites)

        sites_out = tuple(s for s in full_system_sites if s not in sites)
        sites_op = self.site_factors
        prefactors = [
            sites_op[s].trace() if s in sites_op else dimensions[s] for s in sites_out
        ]

        sites_op = {s: o for s, o in sites_op.items() if s in sites}
        prefactor = self.prefactor
        for factor in prefactors:
            if factor == 0:
                return ScalarOperator(factor, subsystem)
            prefactor *= factor

        if len(sites_op) == 0:
            return ScalarOperator(prefactor, subsystem)
        qutip_factors = self.__dict__.get("site_factors_qutip", None)
        if qutip_factors is not None:
            qutip_factors = {
                site: qutip_factors[site] for site in sites if site in qutip_factors
            }
        return ProductOperator(
            sites_op, prefactor, subsystem, _qutip_factors=qutip_factors
        )

    def reduce(self, sites: Iterable, state=None) -> Operator:
        """Compute the partial trace over sites not listed in ``sites``.

        If the state is not provided, the result is the partial trace divided
        by the dimension of the subsystem traced out.

        Parameters
        ----------
        sites : Iterable[str]
            Sites to *keep* after the reduction.
        state : DensityOperator or None, optional
            State relative to which the reduction is performed. If ``None``,
            the reduction is the partial trace normalized by the dimension of
            the traced-out subsystem.

        Returns
        -------
        Operator
            The reduced operator acting on the subsystem defined by ``sites``.

        """
        acts_over = self.acts_over()
        prefactor = self.prefactor
        sites = acts_over.intersection(sites)
        environment = acts_over - sites
        if not environment:
            return self
        system = self.system
        if not sites:
            if state is None:
                value = self.tr()
                dimensions = system.dimensions
                value /= reduce(
                    lambda x, y: x * y, (dimensions[site] for site in acts_over)
                )
                return ScalarOperator(value, system)
            return ScalarOperator(state.expect(self), system)

        system = self.system
        # Special cases:
        if state is None:
            dimensions = self.system.dimensions
            sites_op = self.site_factors

            for site in environment:
                prefactor *= sites_op[site].trace() / dimensions[site]
            return ProductOperator(
                {site: sites_op[site] for site in sites}, prefactor, system
            )
        # ProductDensityOperator:
        if hasattr(state, "terms"):
            return self.to_qutip_operator().reduce(sites, state)

        if hasattr(state, "to_product_state"):
            state = state.to_product_state()
        if isinstance(state, ProductOperator):
            state_by_site = state.site_factors
            sites_op = self.site_factors
            for site in environment:
                prefactor *= (sites_op[site] @ state_by_site[site]).trace()
            result = ProductOperator(
                {site: sites_op[site] for site in sites}, prefactor, system
            )
        else:
            # General case:
            env_tuple = tuple(environment)
            state = state.partial_trace(environment).to_qutip(env_tuple)
            sites_ops = self.site_factors
            # TODO: check if we can do more using numpy
            prefactor *= (
                state
                * fast_tensor(*(self.site_factors_qutip[site] for site in env_tuple))
            ).tr()
            sites_op = {site: op_q for site, op_q in sites_ops.items() if site in sites}
            result = ProductOperator(sites_op, prefactor, system)
        return result

    def simplify(self) -> Operator:
        """Simplifies a product operator.

        - first, collect all the scalar factors and
          absorbe them in the prefactor.
        - If the prefactor vanishes, or all the factors are scalars,
          return a ScalarOperator.
        - If there is just one nontrivial factor, return a LocalOperator.
        - If no reduction is possible, return self.
        """
        # Remove multiples of the identity
        nontrivial_factors = {}
        prefactor = self.prefactor
        if prefactor == 0:
            return ScalarOperator(0, self.system)
        for site, op_factor in self.site_factors.items():
            if is_scalar_op(op_factor):
                prefactor *= op_factor[0, 0]
                assert isinstance(
                    prefactor, (int, float, complex)
                ), f"{type(prefactor)}:{prefactor}"
                if not prefactor:
                    return ScalarOperator(0, self.system)
            else:
                nontrivial_factors[site] = op_factor
        nops = len(nontrivial_factors)
        if nops == 0:
            return ScalarOperator(prefactor, self.system)
        if nops == 1:
            site, op_local = next(iter(nontrivial_factors.items()))
            return LocalOperator(site, to_qobj(op_local * prefactor), self.system)
        if nops != len(self.site_factors):
            return ProductOperator(nontrivial_factors, prefactor, self.system)
        return self

    def to_qutip(self, block: Optional[Tuple[str, ...]] = None):
        """Return a QuTiP object acting over the sites listed in ``block``.

        By default (``block=None``) returns a :class:`qutip.Qobj`
        acting over all the sites, in lexicographical order.

        Parameters
        ----------
        block : tuple[str, ...] or None, optional
            Ordered list of site names defining the tensor-product structure
            of the returned object. Sites not present in ``site_factors``
            contribute an identity factor. Default is ``None``.

        Returns
        -------
        qutip.Qobj
            Full tensor-product operator over ``block``.

        """
        cached = self._to_qutip_cache.get(block, None)
        if cached is not None:
            return cached

        sites_op = self.site_factors_qutip
        system = self.system
        sites = system.sites if system else {}
        # Ensure that block has the sites in the operator.
        orig_block = block
        if block is None:
            if system is not None:
                block = tuple(sorted(sites))
            else:
                block = tuple(sorted(self.acts_over()))

            if len(block) > 8:
                logging.warning(
                    "Asking for a qutip representation of an operator over the full system"
                )

        else:
            block = tuple((site for site in block if site in sites)) + tuple(
                sorted(site for site in sites_op if site not in block)
            )
        if len(block) == 0:
            return self.prefactor

        factors = (
            (sites_op[site] if site in sites_op else sites[site]["identity"])
            for site in block
        )
        self._to_qutip_cache[orig_block] = result = self.prefactor * fast_tensor(
            *factors
        )
        return result

    def to_qutip_operator(self) -> Operator:
        """Return a :class:`QutipOperator` representation of this operator.

        Returns
        -------
        Operator
            A :class:`ScalarOperator` if the prefactor or site factors are
            zero, otherwise a :class:`QutipOperator`.

        """
        prefactor = self.prefactor
        if not (prefactor and self.site_factors_qutip):
            return ScalarOperator(prefactor, self.system)
        return super().to_qutip_operator()

    def tidyup(self, atol=None):
        """Return a copy of the operator with small matrix elements zeroed out.

        Parameters
        ----------
        atol : float or None, optional
            Absolute tolerance below which elements are set to zero.
            Passed directly to :meth:`qutip.Qobj.tidyup`. Default is
            QuTiP's internal tolerance.

        Returns
        -------
        ProductOperator
            Cleaned-up operator.

        """
        tidy_site_operators = {
            name: op_s.tidyup(atol) for name, op_s in self.site_factors_qutip.items()
        }
        return ProductOperator(tidy_site_operators, self.prefactor, self.system)


class ScalarOperator(ProductOperator):
    r"""A product operator that acts as a scalar multiple of the identity.

    Represents :math:`\lambda \, \mathbb{I}` where :math:`\lambda` is a
    complex scalar and :math:`\mathbb{I}` is the identity on the full system.
    This is a special case of :class:`ProductOperator` with no non-trivial
    site factors.

    Parameters
    ----------
    prefactor : complex
        The scalar value :math:`\lambda`.
    system : SystemDescriptor
        Descriptor of the full lattice system.

    """

    def __init__(self, prefactor, system):
        assert system is not None
        super().__init__({}, prefactor, system)

    def __bool__(self):
        """Return ``True`` if the scalar is non-zero."""
        return bool(self.prefactor)

    def __neg__(self):
        """Return the negation of the scalar operator."""
        return ScalarOperator(-self.prefactor, self.system)

    def __repr__(self):
        """Return a human-readable string representation of the scalar operator."""
        result = (
            str(self.prefactor) + " * Identity_{" + ",".join(self.system.sites) + "} "
        )

        return result

    def _repr_latex_(self):

        return (
            "$\\left("
            + str(self.prefactor)
            + " \\times \\mathbb{I}\\right)_{"
            + ",".join(self.system.sites)
            + "}$"
        )

    def acts_over(self) -> frozenset:
        """Return the empty frozenset — a scalar operator acts trivially on all sites.

        Returns
        -------
        frozenset
            Always the empty frozenset.

        """
        return frozenset()

    def dag(self):
        """Return the adjoint of the scalar operator.

        Returns
        -------
        ScalarOperator
            A scalar operator with the complex-conjugated prefactor.
            Returns ``self`` if the prefactor is real.

        """
        if isinstance(self.prefactor, complex):
            return ScalarOperator(self.prefactor.conjugate(), self.system)
        return self

    def hermitian_part(self):
        """Return the Hermitian part of the scalar operator.

        Returns
        -------
        ScalarOperator
            A scalar operator with the real part of the prefactor.
            Returns ``self`` if already Hermitian.

        """
        if self.isherm:
            return self
        return ScalarOperator(np.real(self.prefactor), self.system)

    @property
    def isherm(self):
        """Return ``True`` if the scalar prefactor is real (up to ``QALMA_TOLERANCE``).

        Returns
        -------
        bool
            Whether the prefactor has negligible imaginary part.

        """
        prefactor = self.prefactor
        return not (
            isinstance(prefactor, complex) and abs(prefactor.imag) > QALMA_TOLERANCE
        )

    @property
    def isdiagonal(self) -> bool:
        """``True`` always — the identity is diagonal in any basis."""
        return True

    def logm(self):
        r"""Return the matrix logarithm of the scalar operator.

        Returns
        -------
        ScalarOperator
            A scalar operator with prefactor :math:`\ln(\lambda)`.

        """
        return ScalarOperator(np.log(self.prefactor), self.system)

    def norm(self, ord=None):
        r"""Return the norm of the scalar operator.

        Parameters
        ----------
        ord : str or None, optional
            Order of the norm. Supported values are ``None`` (operator norm),
            ``'fro'`` (Frobenius), and ``'nuc'`` (nuclear). Default is
            ``None``.

        Returns
        -------
        float
            The norm :math:`|\lambda| \cdot \|\mathbb{I}\|_{\text{ord}}`.

        """
        result = self.prefactor
        if ord in ("fro", "nuc"):
            dim_factor = 1.0
            for dim in (dim for site, dim in self.system.dimensions.items()):
                dim_factor *= dim
            if ord == "fro":
                result *= dim_factor**0.5
            else:
                result *= dim_factor

        return result

    def reduce(self, sites: Iterable, state=None) -> Operator:
        """Return ``self`` unchanged — reducing a scalar leaves it invariant.

        Parameters
        ----------
        sites : Iterable[str]
            Ignored. Included for interface compatibility.
        state : optional
            Ignored. Included for interface compatibility.

        Returns
        -------
        ScalarOperator
            ``self``.

        """
        return self

    def simplify(self):
        """Return ``self`` — a scalar operator is already in its simplest form.

        Returns
        -------
        ScalarOperator
            ``self``.

        """
        return self

    def tidyup(self, atol=None):
        """Return a zeroed scalar if the prefactor is below tolerance.

        Parameters
        ----------
        atol : float or None, optional
            Absolute tolerance. Defaults to ``QALMA_TOLERANCE``.

        Returns
        -------
        ScalarOperator
            A zero scalar operator if ``|prefactor| < atol``, otherwise
            ``self``.

        """
        if atol is None:
            atol = QALMA_TOLERANCE
        if abs(self.prefactor) < atol:
            return ScalarOperator(0, self.system)
        return self

    def to_qutip(self, block: Optional[Tuple[str, ...]] = None):
        r"""Return a QuTiP identity operator scaled by the prefactor.

        Parameters
        ----------
        block : tuple[str, ...] or None, optional
            Ordered list of site names defining the tensor-product structure.
            Default is all sites in lexicographical order.

        Returns
        -------
        qutip.Qobj or complex
            :math:`\lambda \, \mathbb{I}_{\text{block}}`. Returns the
            bare scalar if ``block`` is an empty tuple.

        """
        system = self.system
        sites = system.sites
        if block is None:
            block = tuple(sorted(sites))
        elif len(block) == 0:
            return self.prefactor

        factors = (sites[site]["identity"] for site in block)
        return self.prefactor * fast_tensor(*factors)

    def to_qutip_operator(self):
        """Return ``self`` — a scalar operator is its own QuTiP representation.

        Returns
        -------
        ScalarOperator
            ``self``.

        """
        return self
