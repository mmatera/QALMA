# -*- coding: utf-8 -*-
# pylint: disable=invalid-name
"""Classes and functions for operator arithmetic."""

from typing import Iterable, Optional, Set, Tuple, Union

import numpy as np
from scipy.linalg import eigvals as _eigvals, expm as _scp_expm

from qalma.model import SystemDescriptor
from qalma.operators.basic import (
    LocalOperator,
    Operator,
)
from qalma.operators.product import (
    ProductOperator,
    ScalarOperator,
)
from qalma.operators.qutip import QutipOperator
from qalma.settings import QALMA_TOLERANCE

__all__ = ["SumOperator", "OneBodyOperator", "iterable_to_operator", "NBodyOperator"]


class SumOperator(Operator):
    r"""Linear combination of operators.

    Represents a sum of the form

    .. math::

        O = \sum_k O_k

    where each term :math:`O_k` is an arbitrary :class:`Operator`. Terms may
    act on overlapping or disjoint subsets of sites.

    Parameters
    ----------
    term_tuple : tuple[Operator, ...]
        Non-empty tuple of operators to be summed.
    system : SystemDescriptor
        Descriptor of the full lattice system. Must not be ``None``.
    isherm : bool or None, optional
        If known, whether the sum is Hermitian. ``None`` defers the check.
    isdiag : bool or None, optional
        If known, whether the sum is diagonal. ``None`` defers the check.
    simplified : bool, optional
        If ``True``, marks the operator as already simplified, skipping
        redundant simplification calls. Default is ``False``.

    Attributes
    ----------
    terms : tuple[Operator, ...]
        The individual summands.
    system : SystemDescriptor
        The full lattice system.

    """

    terms: Tuple[Operator]

    def __init__(
        self,
        term_tuple: tuple,
        system=None,
        isherm: Optional[bool] = None,
        isdiag: Optional[bool] = None,
        simplified: Optional[bool] = False,
    ):
        assert system is not None
        assert isinstance(term_tuple, tuple)
        assert len(term_tuple) > 0
        assert self not in term_tuple, "cannot be a term of myself."
        self.terms = term_tuple

        if system is None and term_tuple:
            for term in term_tuple:
                if system is None:
                    system = term.system
                else:
                    system = system.union(term.system)

        self.system = system
        self._isherm = isherm
        self._isdiagonal = isdiag
        self._simplified = simplified

    def __bool__(self):
        """Return ``True`` if at least one term is non-zero.

        Returns
        -------
        bool
            ``False`` only if the term list is empty or every term is zero.

        """
        if len(self.terms) == 0:
            return False

        if any(bool(t) for t in self.terms):
            return True
        return False

    def __pow__(self, exp):
        r"""Return the operator raised to a non-negative integer power.

        Computed by repeated multiplication. Negative or non-integer
        exponents raise :class:`TypeError`.

        Parameters
        ----------
        exp : int
            Non-negative integer exponent.

        Returns
        -------
        Operator
            The operator :math:`O^{\text{exp}}`. Returns the scalar ``1``
            for ``exp == 0``.

        Raises
        ------
        TypeError
            If ``exp`` is negative or not an integer.

        """
        isherm = self._isherm
        if isinstance(exp, int):
            if exp == 0:
                return 1
            if exp == 1:
                return self
            if exp > 1:
                result = self
                exp -= 1
                while exp:
                    exp -= 1
                    result = result * self
                if isherm:
                    result = SumOperator(result.terms, self.system, True)
                return result

            raise TypeError("SumOperator does not support negative powers")
        raise TypeError(
            (
                f"unsupported operand type(s) for ** or pow(): "
                f"'SumOperator' and '{type(exp).__name__}'"
            )
        )

    def __neg__(self):
        """Multiply the operator by -1."""
        return SumOperator(tuple(-t for t in self.terms), self.system, self._isherm)

    def __repr__(self):
        """Build the repr str."""
        return "(\n" + "\n  +".join(repr(t) for t in self.terms) + "\n)"

    def _repr_latex_(self):
        """Build a LaTeX representation."""
        # pylint: disable=protected-access
        terms = self.terms
        if len(terms) > 6:
            result = " + ".join(term._repr_latex_()[1:-1] for term in terms[:3])
            result += f" + \\ldots ({len(terms)-6} terms) \\ldots + "
            result += " + ".join(term._repr_latex_()[1:-1] for term in terms[-3:])
        else:
            result = " + ".join(term._repr_latex_()[1:-1] for term in terms)
        return f"${result}$"

    def _set_system_(self, system=None):
        """Set the system descriptor for this operator and all its terms.

        Parameters
        ----------
        system : SystemDescriptor or None, optional
            The system descriptor to assign.

        Returns
        -------
        SumOperator
            ``self``, with the system updated in-place.

        """
        self.system = system
        for term in self.terms:
            # pylint: disable=protected-access
            term._set_system_(system)
        return self

    def acts_over(self) -> frozenset:
        """Return the union of sites over which any term acts non-trivially.

        Returns
        -------
        frozenset[str]
            All sites with at least one non-identity local factor across
            all terms.

        """
        result: Set[str] = set()
        system_size = len(self.system.sites)
        for term in self.terms:
            term_acts_over = term.acts_over()
            result = result.union(term_acts_over)
            if len(result) >= system_size:
                break
        return frozenset(result)

    def dag(self):
        r"""Return the adjoint operator :math:`O^{\dagger}`.

        Returns
        -------
        SumOperator
            Sum of the adjoints of each term. Returns ``self`` if the
            operator is already marked as Hermitian.

        """
        if self._isherm:
            return self
        return SumOperator(tuple(term.dag() for term in self.terms), self.system)

    def flat(self):
        """Flatten nested sums using associativity.

        Any term that is itself a :class:`SumOperator` is expanded in-place,
        producing a single-level sum of non-sum terms.

        Returns
        -------
        SumOperator
            A flat sum with no :class:`SumOperator` terms, or ``self`` if
            no flattening was needed.

        """
        terms = []
        changed = False
        for term in self.terms:
            if isinstance(term, SumOperator):
                term_flat = term.flat()
                if hasattr(term_flat, "terms"):
                    terms.extend(term_flat.terms)
                else:
                    terms.append(term_flat)
                changed = True
            else:
                new_term = term.flat()
                assert isinstance(
                    new_term, Operator
                ), f"{type(term)} produces type({new_term})"
                terms.append(new_term)
                if term is not new_term:
                    changed = True
        if changed:
            return SumOperator(tuple(terms), self.system, isherm=self._isherm)
        return self

    def hermitian_part(self):
        r"""Return the Hermitian part :math:`\frac{O + O^{\dagger}}{2}`.

        Returns
        -------
        SumOperator
            A sum of the Hermitian parts of each term, marked as Hermitian.
            Returns ``self`` if already marked as Hermitian.

            Subclasses automatically get the correct return type
            because ``type(self)`` is used as the constructor.

        """
        if self._isherm is True:
            return self
        return type(self)(
            tuple(t.hermitian_part() for t in self.terms),
            system=self.system,
            isherm=True,
        ).simplify()

    @property
    def isherm(self) -> bool:
        """``True`` if the operator is Hermitian.

        First checks each term individually. If all terms are Hermitian,
        returns ``True``. Otherwise applies a more aggressive test:
        simplifies the anti-Hermitian part and checks whether its
        Frobenius norm vanishes (up to ``QALMA_TOLERANCE``). The result
        is cached in ``_isherm``.
        """
        isherm = self._isherm

        def aggresive_hermitian_test(non_hermitian_tuple: Tuple[Operator, ...]):
            """Determine if the antihermitian part is zero."""
            # Here we assume that after simplify, the operator is a single term
            # (not a SumOperator), a OneBodyOperator, or a sum of a one-body operator
            # and terms acting over an specific block.
            nh_sum = SumOperator(non_hermitian_tuple, self.system).simplify()
            if not hasattr(nh_sum, "terms"):
                self._isherm = nh_sum.isherm
                return self._isherm
            # Hermitian until the opposite is shown:
            isherm = True
            for term in nh_sum.terms:
                term_isherm = term.isherm
                # if term_isherm could not determine by itself if the
                # term is hermitian, try harder looking at the frobenious norm
                # of its anti-hermitian part. This step can be very costly...
                if term_isherm is None:
                    # Last resource:
                    ah_part = term - term.dag()
                    term_isherm = abs((ah_part * ah_part).tr()) < QALMA_TOLERANCE
                if not term_isherm:
                    isherm = False
                    break
            self._isherm = isherm
            return isherm

        if isherm is None:
            # First, collect the non-hermitian terms
            non_hermitian = tuple((term for term in self.terms if not term.isherm))
            # If there are non-hermitian terms, try the more aggressive strategy
            # over these terms.
            if non_hermitian:
                return aggresive_hermitian_test(non_hermitian)

            self._isherm = True
            return True

        return bool(self._isherm)

    @property
    def isdiagonal(self) -> bool:
        """``True`` if all terms are diagonal in the site-local basis.

        Simplifies the operator first if not already simplified, then
        checks each term. The result is cached in ``_isdiagonal``.
        """
        if self._isdiagonal is None:
            simplified = self if self._simplified else self.simplify()
            try:
                self._isdiagonal = all(term.isdiagonal for term in simplified.terms)
            except AttributeError:
                self._isdiagonal = simplified.isdiagonal
        return self._isdiagonal

    @property
    def is_zero(self) -> bool:
        """``True`` if the operator simplifies to zero.

        Simplifies the operator and checks whether all resulting terms are
        zero. Sets ``_isherm = True`` if the operator is confirmed zero.
        """
        simplify_self = self if self._simplified else self.simplify()
        if hasattr(simplify_self, "terms"):
            result = all(term.is_zero for term in simplify_self.terms)
        else:
            result = simplify_self.is_zero
        if result:
            self._isherm = True
        return result

    def n_body_sector(self) -> int:
        """Return the maximum n-body sector among all terms.

        Returns
        -------
        int
            The largest ``n`` such that some term acts non-trivially on
            exactly ``n`` sites.

        """
        return max(term.n_body_sector() for term in self.terms)

    def num_terms(self) -> int:
        """Return the number of terms in the sum.

        Returns
        -------
        int
            Length of ``self.terms``.

        """
        return len(self.terms)

    def partial_trace(self, sites: Union[frozenset, SystemDescriptor]):
        """Compute the partial trace over the complement of ``sites``.

        Parameters
        ----------
        sites : frozenset[str] or SystemDescriptor
            Sites to *keep*. All other sites are traced out.

        Returns
        -------
        Operator
            The reduced operator acting on the subsystem defined by
            ``sites``. Zero terms are dropped before returning.

        """
        if not isinstance(sites, SystemDescriptor):
            sites = self.system.subsystem(sites)
        new_terms = tuple((term.partial_trace(sites) for term in self.terms))
        subsystem = new_terms[0].system
        new_terms = tuple((term for term in new_terms if term))
        return iterable_to_operator(
            new_terms,
            subsystem,
            isherm=self._isherm,
            isdiag=self._isdiagonal,
            simplified=self._simplified,
        )

    def reduce(self, sites: Iterable, state=None):
        """Reduce the operator to a subsystem, optionally weighted by a state.

        Applies ``reduce`` to each term and assembles the result. If
        ``state`` is ``None``, the reduction is a partial trace normalized
        by the dimension of the traced-out subsystem.

        Parameters
        ----------
        sites : Iterable[str]
            Sites to *keep* after the reduction.
        state : DensityOperator or None, optional
            State relative to which the reduction is performed.

        Returns
        -------
        Operator
            The reduced operator acting on the subsystem defined by ``sites``.

        """
        new_terms = (term.reduce(sites, state) for term in self.terms)
        return iterable_to_operator(new_terms, self.system, isherm=self._isherm)

    def simplify(self):
        """Simplify the operator by grouping and combining like terms.

        Groups terms by the block of sites they act on. Terms acting on the
        same block are added together. Returns a simpler operator type when
        possible (e.g. a single term becomes that term directly).

        Returns
        -------
        Operator
            A simplified equivalent operator.

        """
        if self._simplified:
            return self
        if len(self.terms) == 1:
            return self.terms[0].simplify()

        # pylint: disable=import-outside-toplevel
        from qalma.operators.simplify import group_terms_by_blocks

        return group_terms_by_blocks(self.flat())

    def to_qutip(self, block: Optional[Tuple[str, ...]] = None):
        """Return the QuTiP representation of the sum.

        Parameters
        ----------
        block : tuple[str, ...] or None, optional
            Ordered list of site names defining the tensor-product structure
            of the returned :class:`qutip.Qobj`. Defaults to all system
            sites in lexicographical order.

        Returns
        -------
        qutip.Qobj
            The sum of the QuTiP representations of all terms.

        """
        terms = self.terms
        system = self.system
        assert all(system.contains(term.system) for term in terms)
        if block is None:
            block = tuple(sorted(self.acts_over() if system is None else system.sites))
        else:
            block = block + tuple(
                sorted(site for site in self.acts_over() if site not in block)
            )
        if len(self.terms) == 0:
            return ScalarOperator(0, self.system).to_qutip(block)

        qutip_terms = (t.to_qutip(block) for t in terms)
        result = sum(qutip_terms)
        return result

    def tr(self):
        """Return the trace of the operator over the full system.

        Returns
        -------
        complex
            The sum of the traces of all terms.

        """
        return sum(t.tr() for t in self.terms)

    def tidyup(self, atol=None):
        """Return a copy with small matrix elements zeroed out.

        Applies ``tidyup`` to each term and drops zero terms.

        Parameters
        ----------
        atol : float or None, optional
            Absolute tolerance passed to each term's ``tidyup``. Defaults
            to QuTiP's internal tolerance.

        Returns
        -------
        Operator
            Cleaned-up operator with zero terms removed.

        """
        tidy_terms = [term.tidyup(atol) for term in self.terms]
        tidy_terms = tuple((term for term in tidy_terms if term))
        return (
            type(self)(
                tidy_terms,
                self.system,
                isherm=self._isherm,
                isdiag=getattr(self, "_isdiagonal", None),
            )
            if tidy_terms
            else iterable_to_operator(tidy_terms, self.system)
        )


NBodyOperator = SumOperator


class OneBodyOperator(SumOperator):
    r"""Linear combination of local (single-site) operators.

    A special case of :class:`SumOperator` restricted to terms that each
    act on at most one site. Represents operators of the form

    .. math::

        O = \lambda_0 \mathbb{I} + \sum_i O_i

    where each :math:`O_i` is a :class:`LocalOperator` acting on site
    :math:`i` and :math:`\lambda_0` is an optional scalar term.

    During construction, terms are automatically simplified and grouped by
    site: multiple local operators on the same site are added together into
    a single :class:`LocalOperator`.

    Parameters
    ----------
    terms : tuple[Operator, ...]
        Tuple of local operators and optional scalar operators.
    system : SystemDescriptor
        Descriptor of the full lattice system. Must not be ``None``.
    check_and_convert : bool, optional
        If ``True`` (default), validates and simplifies terms during
        construction. Set to ``False`` only when terms are already
        guaranteed to be simplified local operators.
    isherm : bool or None, optional
        Whether the operator is Hermitian. ``None`` defers the check.
    isdiag : bool or None, optional
        Whether the operator is diagonal. ``None`` defers the check.
    simplified : bool, optional
        Whether the operator is already simplified. Default is ``False``.

    """

    def __init__(
        self,
        terms,
        system=None,
        check_and_convert=True,
        isherm: Optional[bool] = None,
        isdiag: Optional[bool] = None,
        simplified: Optional[bool] = False,
    ):
        assert isinstance(terms, tuple)
        assert system is not None

        def collect_systems(terms, system):
            for term in terms:
                if not hasattr(term, "system"):
                    continue
                term_system = term.system
                if term_system is None:
                    continue
                if system is None:
                    system = term.system
                else:
                    system = system.union(term_system)
            return system

        if check_and_convert:
            system = collect_systems(terms, system)
            # Ensure that all the terms are operators.
            terms = [
                term if isinstance(term, Operator) else ScalarOperator(term, system)
                for term in terms
            ]
            if isherm:
                terms = [
                    term if term.isherm else (term + term.dag()) * 0.5 for term in terms
                ]
            terms, system = self._simplify_terms(terms, system)
            simplified = True
            if len(terms) == 0:
                terms = tuple((ScalarOperator(0.0, system),))

        super().__init__(
            terms, system=system, isherm=isherm, isdiag=isdiag, simplified=simplified
        )

    def __repr__(self):
        """Build the repr str."""
        return "  " + "\n  +".join("(" + repr(term) + ")" for term in self.terms)

    def __neg__(self):
        """Multiply the operator by -1."""
        return OneBodyOperator(tuple(-term for term in self.terms), self.system)

    def dag(self):
        r"""Return the adjoint :math:`O^{\dagger}`.

        Returns
        -------
        OneBodyOperator
            Sum of the adjoints of each local term.

        """
        return OneBodyOperator(
            tuple(term.dag() for term in self.terms),
            system=self.system,
            check_and_convert=False,
        )

    def expm(self):
        r"""Return the matrix exponential :math:`e^O`.

        Exploits the fact that local operators on different sites commute:

        .. math::

            e^{\lambda_0 + \sum_i O_i} = e^{\lambda_0} \bigotimes_i e^{O_i}

        Each local exponential is computed via ``scipy.linalg.expm``.
        The diagonal of each local operator is shifted to avoid numerical
        overflow before exponentiation.

        Returns
        -------
        ProductOperator
            The matrix exponential as a product of local exponentials.

        """
        sites_op = {}
        ln_prefactor = 0
        for term in self.simplify().terms:
            if not bool(term):
                assert False, "No empty terms should reach here"
                continue
            if isinstance(term, ScalarOperator):
                ln_prefactor += term.prefactor
                continue
            operator = term.operator
            k_0 = max(np.real(_eigvals(operator)))

            operator = operator.copy()
            np.fill_diagonal(operator, operator.diagonal() - k_0)
            ln_prefactor += k_0
            sites_op[term.site] = _scp_expm(operator)

        prefactor = np.exp(ln_prefactor)
        return ProductOperator(sites_op, prefactor=prefactor, system=self.system)

    def simplify(self):
        """Simplify by grouping local operators acting on the same site.

        Returns
        -------
        Operator
            A simplified :class:`OneBodyOperator`, or a single
            :class:`ScalarOperator` / :class:`LocalOperator` if only one
            term remains after grouping.

        """
        if self._simplified:
            return self
        terms = self.terms
        if self._isherm:
            terms = (
                term if term.isherm else (term + term.dag()) * 0.5 for term in terms
            )
        terms, system = self._simplify_terms(terms, self.system)
        num_terms = len(terms)
        if num_terms == 0:
            return ScalarOperator(0, system)
        if num_terms == 1:
            return terms[0]
        return OneBodyOperator(
            terms, system, isherm=self._isherm, isdiag=self._isdiagonal, simplified=True
        )

    @staticmethod
    def _simplify_terms(terms, system):
        """Simplify terms (internal).

        Group terms by subsystem and combine local operators on the same
        site.

        Scalar terms are summed into a single :class:`ScalarOperator`.
        :class:`LocalOperator` terms on the same site are added together.
        :class:`QutipOperator` terms are converted to :class:`LocalOperator`
        before grouping.

        Parameters
        ----------
        terms : Iterable[Operator]
            Input terms to simplify and group.
        system : SystemDescriptor
            The full lattice system.

        Returns
        -------
        tuple[Operator, ...]
            Simplified and grouped terms.
        SystemDescriptor
            The (possibly updated) system descriptor.

        Raises
        ------
        ValueError
            If any term is not a scalar, local, or QuTiP operator.

        """
        simply_terms = [term.simplify() for term in terms]
        terms = []
        terms_by_subsystem = {}
        scalar_term_value = 0
        scalar_term = None

        for term in simply_terms:
            if isinstance(term, SumOperator):
                terms.extend(term.terms)
            elif isinstance(term, (ScalarOperator, LocalOperator)):
                terms.append(term)
            elif isinstance(term, QutipOperator):
                terms.append(
                    LocalOperator(
                        tuple(term.acts_over())[0],
                        term.operator * term.prefactor,
                        system=term.system,
                    )
                )
            else:
                raise ValueError(
                    f"A OneBodyOperator can not have {type(term)} as a term."
                )
        # Now terms are just scalars and local operators.

        for term in terms:
            if isinstance(term, ScalarOperator):
                scalar_term = term
                scalar_term_value += term.prefactor
            elif isinstance(term, LocalOperator):
                terms_by_subsystem.setdefault(term.site, []).append(term)

        if scalar_term is None:
            terms = []
        elif scalar_term_value == scalar_term.prefactor:
            terms = [scalar_term]
        else:
            terms = [ScalarOperator(scalar_term_value, system)]

        # Reduce the local terms
        for _, local_terms in terms_by_subsystem.items():
            if len(local_terms) > 1:
                terms.append(sum(local_terms))
            else:
                terms.extend(local_terms)

        return tuple(terms), system


def iterable_to_operator(terms: Iterable[Operator], system, **kwargs) -> Operator:
    """Convert an iterable of operators into a single operator.

    Returns the simplest possible type: a :class:`ScalarOperator` for an
    empty iterable, the single term directly for a one-element iterable,
    or a :class:`SumOperator` otherwise.

    Parameters
    ----------
    terms : Iterable[Operator]
        Operators to combine.
    system : SystemDescriptor
        The full lattice system, passed to :class:`SumOperator` or
        :class:`ScalarOperator` if needed.
    **kwargs
        Additional keyword arguments forwarded to :class:`SumOperator`
        (e.g. ``isherm``, ``isdiag``, ``simplified``).

    Returns
    -------
    Operator
        A :class:`ScalarOperator` (zero) if ``terms`` is empty, the single
        element if ``terms`` has one item, or a :class:`SumOperator`
        otherwise.

    """
    terms_tuple = tuple(terms)
    if len(terms_tuple) == 0:
        return ScalarOperator(0, system)
    if len(terms_tuple) == 1:
        return terms_tuple[0]
    return SumOperator(terms_tuple, system, **kwargs)
