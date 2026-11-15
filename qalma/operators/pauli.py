r"""Pauli strings: products of Pauli matrices on two-level sites.

A :class:`PauliStringOperator` represents

.. math::

    \lambda\, \sigma^{a_1}_{1} \otimes \sigma^{a_2}_{2} \otimes \cdots
    \otimes \sigma^{a_n}_{n},\qquad a_k \in \{0, x, y, z\},

with :math:`\sigma^0 = \mathbb{1}`. The string is stored as a single Python
integer of :math:`2n` bits: the pair of bits ``(2k+1, 2k)`` holds the code of
the Pauli matrix on the ``k``-th site of the system,

====== ==== ===============
code   bits operator
====== ==== ===============
0      00   identity
1      01   :math:`\sigma^x`
2      10   :math:`\sigma^y`
3      11   :math:`\sigma^z`
====== ==== ===============

The position ``k`` of each site is its position in ``system.sites`` (the
order of the nodes of the graph). That order is also the one used to build
Jordan-Wigner strings in :func:`fermion_annihilation` and related helpers.

With this encoding the label of a product is the bitwise XOR of the labels
of the factors, and the phase follows from population counts, so products
and commutators cost :math:`O(n/64)` word operations and are exact: phases
are always one of :math:`\pm 1, \pm i`.

The phase is computed in the symplectic representation: writing
:math:`\sigma(x, z) = i^{x\cdot z} X^x Z^z` (so that :math:`Y = iXZ`), with
``x = hi ^ lo`` and ``z = hi`` for each pair of bits,

.. math::

    \sigma(x_1,z_1)\,\sigma(x_2,z_2) =
    i^{\,x_1\cdot z_1 + x_2\cdot z_2 - x_3\cdot z_3 + 2 z_1\cdot x_2}\,
    \sigma(x_3, z_3),

where :math:`x_3 = x_1 \oplus x_2`, :math:`z_3 = z_1 \oplus z_2`, and two
strings commute iff :math:`x_1\cdot z_2 + z_1\cdot x_2` is even.
"""

from functools import cached_property
from typing import Dict, Iterable, List, Mapping, Optional, Tuple, Union

import numpy as np

from qalma.model import SystemDescriptor
from qalma.settings import QALMA_TOLERANCE

from .basic import LocalOperator, Operator
from .product import ProductOperator, ScalarOperator

__all__ = [
    "PauliStringOperator",
    "anticommutator_pauli",
    "commutator_pauli",
    "fermion_annihilation",
    "fermion_creation",
    "fermion_number",
    "is_pauli_sum",
    "majorana_operators",
    "pauli_terms",
    "to_pauli_sum",
]


PAULI_LABELS = "IXYZ"
_LABEL_TO_CODE = {
    "I": 0,
    "0": 0,
    "X": 1,
    "x": 1,
    "Y": 2,
    "y": 2,
    "Z": 3,
    "z": 3,
    "SX": 1,
    "SY": 2,
    "SZ": 3,
    "sx": 1,
    "sy": 2,
    "sz": 3,
}
PAULI_MATRICES = (
    np.array([[1, 0], [0, 1]], dtype=complex),
    np.array([[0, 1], [1, 0]], dtype=complex),
    np.array([[0, -1j], [1j, 0]], dtype=complex),
    np.array([[1, 0], [0, -1]], dtype=complex),
)
# i**k, exact.
_I_POWERS = (1.0 + 0j, 1j, -1.0 + 0j, -1j)


# ---------------------------------------------------------------------------
# Site ordering
# ---------------------------------------------------------------------------


class _SiteIndex:
    """Position of each site in the bit string, and the mask ``0b0101...``."""

    __slots__ = ("order", "position", "mask")

    def __init__(self, system: SystemDescriptor):
        self.order: Tuple[str, ...] = tuple(system.sites)
        self.position: Dict[str, int] = {s: k for k, s in enumerate(self.order)}
        self.mask: int = int("01" * len(self.order), 2) if self.order else 0


def _site_index(system: SystemDescriptor) -> _SiteIndex:
    """Return the (cached) site index of ``system``.

    ``SystemDescriptor`` is not hashable, so the index is cached as an
    attribute of the system itself. It is rebuilt if the set of sites of the
    system changes.
    """
    index = system.__dict__.get("_pauli_site_index", None)
    if index is None or len(index.order) != len(system.sites):
        index = _SiteIndex(system)
        system.__dict__["_pauli_site_index"] = index
    return index


def _remap_code(code: int, src: SystemDescriptor, dst: SystemDescriptor) -> int:
    """Express ``code``, written in the site order of ``src``, in that of ``dst``."""
    if src is dst or code == 0:
        return code
    src_index, dst_index = _site_index(src), _site_index(dst)
    if src_index.order == dst_index.order:
        return code
    dst_position = dst_index.position
    result = 0
    k = 0
    while code:
        local = code & 3
        if local:
            result |= local << (2 * dst_position[src_index.order[k]])
        code >>= 2
        k += 1
    return result


def _symplectic(code: int, mask: int) -> Tuple[int, int]:
    """Split an interleaved code into the masks ``(x, z)`` (bits at even positions)."""
    high = (code >> 1) & mask
    return (code ^ high) & mask, high


def _product_phase(code_1: int, code_2: int, mask: int) -> Tuple[int, int]:
    """Return ``(k, code_3)`` such that ``P(code_1) P(code_2) = i**k P(code_3)``."""
    x_1, z_1 = _symplectic(code_1, mask)
    x_2, z_2 = _symplectic(code_2, mask)
    code_3 = code_1 ^ code_2
    x_3, z_3 = x_1 ^ x_2, z_1 ^ z_2
    k = (
        (x_1 & z_1).bit_count()
        + (x_2 & z_2).bit_count()
        - (x_3 & z_3).bit_count()
        + 2 * (z_1 & x_2).bit_count()
    ) & 3
    return k, code_3


def _anticommute(code_1: int, code_2: int, mask: int) -> bool:
    """``True`` if the strings anticommute."""
    x_1, z_1 = _symplectic(code_1, mask)
    x_2, z_2 = _symplectic(code_2, mask)
    return bool(((x_1 & z_2).bit_count() + (z_1 & x_2).bit_count()) & 1)


# ---------------------------------------------------------------------------
# The operator class
# ---------------------------------------------------------------------------


class PauliStringOperator(ProductOperator):
    r"""Product of Pauli matrices times a complex prefactor.

    Subclass of :class:`ProductOperator`, so every routine written for
    product operators (expectation values, reductions, conversion to QuTiP,
    scalar products, ...) accepts it. The attribute ``site_factors`` is
    built lazily from the bit string, only when such a routine needs it.

    Products, sums of equal strings, commutators and anticommutators
    between Pauli strings are computed symbolically on the bit strings, and
    the result is again a :class:`PauliStringOperator` (or a
    :class:`ScalarOperator` when the string reduces to the identity).

    Parameters
    ----------
    paulis : dict[str, int | str] or int
        Either a mapping ``site -> Pauli`` (``0..3`` or one of ``"I"``,
        ``"X"``, ``"Y"``, ``"Z"``), or the bit string already encoded in
        the site order of ``system``.
    prefactor : complex, optional
        Global scalar prefactor. Default ``1``.
    system : SystemDescriptor
        The system. Every site the string acts on must have dimension 2.

    Examples
    --------
    >>> system = build_spin_chain(4)
    >>> p = PauliStringOperator({"1[0]": "X", "1[1]": "Z"}, 0.5, system)
    >>> q = PauliStringOperator.from_label("YZII", system)
    >>> p * q          # (-0.5j) Z@1[0]
    """

    code: int

    def __init__(
        self,
        paulis: Union[Mapping[str, Union[int, str]], int],
        prefactor: complex = 1.0,
        system: Optional[SystemDescriptor] = None,
    ):
        # ProductOperator.__init__ is not called on purpose: the dense
        # factors are derived from ``code`` on demand.
        assert system is not None
        index = _site_index(system)
        if isinstance(paulis, (int, np.integer)):
            code = int(paulis)
            if code < 0 or code >> (2 * len(index.order)):
                raise ValueError(f"code {code} does not fit in the system.")
        else:
            code = 0
            position = index.position
            for site, pauli in paulis.items():
                local = (
                    _LABEL_TO_CODE.get(pauli, -1) if isinstance(pauli, str) else pauli
                )
                if not 0 <= local <= 3:
                    raise ValueError(f"invalid Pauli code {pauli!r} at {site}.")
                if local:
                    code |= local << (2 * position[site])
        self._init_fast(code, prefactor, system)
        dimensions = system.dimensions
        for site in self.acts_over():
            if dimensions[site] != 2:
                raise ValueError(
                    f"site {site} has dimension {dimensions[site]}; "
                    "Pauli strings need two-level sites."
                )

    def _init_fast(self, code: int, prefactor: complex, system: SystemDescriptor):
        """Set the attributes without any validation (internal fast path)."""
        if not prefactor:
            code = 0
        self.code = code
        self.prefactor = prefactor
        self.system = system
        self._to_qutip_cache = {}

    @classmethod
    def _new(
        cls, code: int, prefactor: complex, system: SystemDescriptor
    ) -> "PauliStringOperator":
        """Build a string from an already validated code (internal fast path)."""
        result = cls.__new__(cls)
        result._init_fast(code, prefactor, system)
        return result

    @classmethod
    def from_label(
        cls, label: str, system: SystemDescriptor, prefactor: complex = 1.0
    ) -> "PauliStringOperator":
        """Build a string from a label like ``"XIZY"``, one letter per site.

        The k-th letter acts on the k-th site of ``system.sites``.
        """
        order = _site_index(system).order
        if len(label) != len(order):
            raise ValueError(
                f"label has {len(label)} letters, but the system has {len(order)} sites."
            )
        return cls(dict(zip(order, label)), prefactor, system)

    @classmethod
    def from_operator(
        cls, operator: Operator, atol: float = QALMA_TOLERANCE
    ) -> "PauliStringOperator":
        """Convert a product of local operators that is a single Pauli string.

        Raises ``ValueError`` if ``operator`` is not proportional to a Pauli
        string; use :func:`to_pauli_sum` for general operators.
        """
        if isinstance(operator, PauliStringOperator):
            return operator
        result = to_pauli_sum(operator, atol)
        if isinstance(result, ScalarOperator):
            return cls._new(0, result.prefactor, result.system)
        if isinstance(result, PauliStringOperator):
            return result
        raise ValueError("the operator is not proportional to a single Pauli string.")

    # ------------------------------------------------------------------
    # Representation of the string
    # ------------------------------------------------------------------

    @cached_property
    def paulis(self) -> Dict[str, int]:
        """Mapping ``site -> code`` (``1, 2, 3`` for ``X, Y, Z``) of the nontrivial factors."""
        order = _site_index(self.system).order
        result = {}
        code, k = self.code, 0
        while code:
            local = code & 3
            if local:
                result[order[k]] = local
            code >>= 2
            k += 1
        return result

    @property
    def label(self) -> str:
        """The string as a label, one letter per site, in the order of ``system.sites``."""
        code = self.code
        return "".join(
            PAULI_LABELS[(code >> (2 * k)) & 3]
            for k in range(len(_site_index(self.system).order))
        )

    @property
    def weight(self) -> int:
        """Number of sites with a nontrivial factor."""
        code = self.code
        return ((code | (code >> 1)) & _site_index(self.system).mask).bit_count()

    @cached_property
    def site_factors(self) -> Dict[str, np.ndarray]:  # type: ignore[override]
        """Dense local factors, for compatibility with :class:`ProductOperator`."""
        return {site: PAULI_MATRICES[c].copy() for site, c in self.paulis.items()}

    @property
    def size(self) -> int:
        """Number of sites of the system."""
        return len(self.system.sites)

    @property
    def dimensions(self) -> Dict[str, int]:
        """Dimensions of the sites of the system."""
        return self.system.dimensions

    def __repr__(self):
        """Return e.g. ``(0.5+0j) * X@1[0] Z@1[2]``."""
        factors = " ".join(f"{PAULI_LABELS[c]}@{s}" for s, c in self.paulis.items())
        return f"{self.prefactor} * {factors or 'I'}"

    def _repr_latex_(self):
        factors = " ".join(
            f"\\sigma^{{{PAULI_LABELS[c].lower()}}}_{{{s}}}"
            for s, c in self.paulis.items()
        )
        factors = factors or "\\mathbb{1}"
        return f"$({self.prefactor})\\, {factors}$"

    # ------------------------------------------------------------------
    # Algebra
    # ------------------------------------------------------------------

    def code_in(self, system: SystemDescriptor) -> int:
        """Return the bit string expressed in the site order of ``system``."""
        return _remap_code(self.code, self.system, system)

    def commutes_with(self, other: "PauliStringOperator") -> bool:
        """``True`` if the two strings commute (prefactors are ignored)."""
        system = self.system.union(other.system)
        return not _anticommute(
            self.code_in(system), other.code_in(system), _site_index(system).mask
        )

    def __bool__(self):
        return bool(self.prefactor)

    @property
    def is_zero(self) -> bool:
        """``True`` if the prefactor vanishes."""
        return not self.prefactor

    def __neg__(self):
        return self._new(self.code, -self.prefactor, self.system)

    def __pow__(self, exp):
        r"""Integer powers use :math:`P^2 = \mathbb{1}`; other exponents fall back."""
        if isinstance(exp, (int, np.integer)):
            prefactor = self.prefactor**exp
            if exp % 2 == 0 or self.code == 0:
                return ScalarOperator(prefactor, self.system)
            return self._new(self.code, prefactor, self.system)
        return super().__pow__(exp)

    def acts_over(self) -> frozenset:
        """Sites with a nontrivial factor."""
        return frozenset(self.paulis)

    def dag(self):
        """Adjoint: Pauli matrices are hermitian, so only the prefactor changes."""
        prefactor = self.prefactor
        if isinstance(prefactor, complex):
            prefactor = prefactor.conjugate()
        return self._new(self.code, prefactor, self.system)

    def expm(self):
        r"""Return :math:`e^{\lambda P} = \cosh\lambda + \sinh(\lambda) P`."""
        from .arithmetic import SumOperator

        prefactor, system = self.prefactor, self.system
        if self.code == 0:
            return ScalarOperator(np.exp(prefactor), system)
        return SumOperator(
            (
                ScalarOperator(complex(np.cosh(prefactor)), system),
                self._new(self.code, complex(np.sinh(prefactor)), system),
            ),
            system,
        )

    def flat(self):
        """Return a :class:`ScalarOperator` for the identity, ``self`` otherwise."""
        return self.simplify()

    def hermitian_part(self):
        """Hermitian part: the real part of the prefactor."""
        if self.isherm:
            return self
        return self._new(self.code, float(np.real(self.prefactor)), self.system)

    def inv(self):
        r"""Return :math:`(\lambda P)^{-1} = \lambda^{-1} P`, using :math:`P^2 = 1`."""
        return self._new(self.code, 1 / self.prefactor, self.system)

    @property
    def isherm(self) -> bool:  # type: ignore[override]
        """``True`` if the prefactor is real (up to ``QALMA_TOLERANCE``)."""
        return abs(np.imag(self.prefactor)) < QALMA_TOLERANCE

    @property
    def isdiagonal(self) -> bool:  # type: ignore[override]
        """``True`` if the string has only identities and :math:`\\sigma^z` factors."""
        x_mask, _ = _symplectic(self.code, _site_index(self.system).mask)
        return x_mask == 0

    def norm(self, ord=None):
        """Operator (``None``), Frobenius (``'fro'``) or nuclear (``'nuc'``) norm."""
        result = abs(self.prefactor)
        if ord in ("fro", "nuc"):
            dim = 2.0 ** len(self.system.sites)
            result *= dim**0.5 if ord == "fro" else dim
        return result

    def partial_trace(self, sites: Union[frozenset, SystemDescriptor]):
        """Partial trace over the sites not in ``sites``.

        Vanishes if a traced-out site carries a nontrivial factor; otherwise
        each traced-out site contributes a factor 2.
        """
        if isinstance(sites, SystemDescriptor):
            subsystem = sites
            sites = frozenset(sites.sites)
        else:
            subsystem = self.system.subsystem(sites)
        paulis = self.paulis
        if any(site not in sites for site in paulis):
            return ScalarOperator(0, subsystem)
        prefactor = self.prefactor * 2 ** (
            len(self.system.sites) - len(subsystem.sites)
        )
        if not paulis:
            return ScalarOperator(prefactor, subsystem)
        return self._new(self.code_in(subsystem), prefactor, subsystem)

    def reduce(self, sites: Iterable, state=None) -> Operator:
        """Normalized partial trace, or partial expectation value over ``state``."""
        if state is not None:
            return super().reduce(sites, state)
        sites = frozenset(sites)
        if self.acts_over() <= sites:
            return self
        return ScalarOperator(0, self.system)

    def simplify(self) -> Operator:
        """Return a :class:`ScalarOperator` for the identity or zero, ``self`` otherwise."""
        if self.code == 0 or not self.prefactor:
            return ScalarOperator(self.prefactor if self.code == 0 else 0, self.system)
        return self

    def tidyup(self, atol=None):
        """Remove a negligible prefactor, or its negligible real/imaginary part."""
        if atol is None:
            atol = QALMA_TOLERANCE
        prefactor = complex(self.prefactor)
        if abs(prefactor) < atol:
            return ScalarOperator(0, self.system)
        real, imag = prefactor.real, prefactor.imag
        real = 0.0 if abs(real) < atol else real
        imag = 0.0 if abs(imag) < atol else imag
        return self._new(self.code, complex(real, imag) if imag else real, self.system)

    def tr(self) -> complex:
        """Trace: zero unless the string is the identity."""
        if self.code:
            return 0
        return self.prefactor * 2 ** len(self.system.sites)

    def to_product_operator(self) -> ProductOperator:
        """Return the equivalent dense :class:`ProductOperator`."""
        return ProductOperator(dict(self.site_factors), self.prefactor, self.system)

    def __getstate__(self):
        """Pickle the code, the prefactor and the system, not the caches."""
        return {
            "code": self.code,
            "prefactor": self.prefactor,
            "system": self.system,
            "_to_qutip_cache": {},
        }

    def save_hdf5(self, parent, name: str):
        """Store the string as the group ``parent[name]`` (see :mod:`qalma.operators.hdf5`)."""
        from .hdf5 import save_operator

        return save_operator(parent, name, self)

    @classmethod
    def load_hdf5(cls, group, system: Optional[SystemDescriptor] = None):
        """Load a string stored with :meth:`save_hdf5`."""
        from .hdf5 import load_operator

        result = load_operator(group, system)
        if not isinstance(result, cls):
            raise TypeError(f"{group.name} does not hold a Pauli string.")
        return result

    def _set_system_(self, system=None):
        """Change the system, keeping the meaning of the string."""
        if system is not None and self.system is not None:
            self.code = _remap_code(self.code, self.system, system)
        self.__dict__.pop("paulis", None)
        self.__dict__.pop("site_factors", None)
        self.system = system
        return self


# ---------------------------------------------------------------------------
# Symbolic algebra on sums of Pauli strings
# ---------------------------------------------------------------------------


def pauli_product(
    op_1: PauliStringOperator, op_2: PauliStringOperator
) -> Union[PauliStringOperator, ScalarOperator]:
    """Product of two Pauli strings, computed on the bit strings."""
    system = op_1.system.union(op_2.system)
    mask = _site_index(system).mask
    k, code = _product_phase(op_1.code_in(system), op_2.code_in(system), mask)
    prefactor = op_1.prefactor * op_2.prefactor * _I_POWERS[k]
    if code == 0 or not prefactor:
        return ScalarOperator(prefactor if code == 0 else 0, system)
    return PauliStringOperator._new(code, prefactor, system)


def pauli_terms(
    operator: Operator,
) -> Optional[List[Union[PauliStringOperator, ScalarOperator]]]:
    """Return the terms of ``operator`` if it is a sum of Pauli strings and scalars.

    Returns ``None`` otherwise.
    """
    from .arithmetic import SumOperator

    if isinstance(operator, (PauliStringOperator, ScalarOperator)):
        return [operator]
    # PauliSumOperator is a SumOperator whose terms are strings and scalars.
    if isinstance(operator, SumOperator):
        result = []
        for term in operator.terms:
            sub_terms = pauli_terms(term)
            if sub_terms is None:
                return None
            result.extend(sub_terms)
        return result
    return None


def _code_and_prefactor(term, system) -> Tuple[int, complex]:
    if isinstance(term, PauliStringOperator):
        return term.code_in(system), term.prefactor
    return 0, term.prefactor


def pauli_dict_to_operator(
    coefficients: Mapping[int, complex], system: SystemDescriptor
) -> Operator:
    """Build :math:`\\sum_c a_c P_c` from a mapping ``code -> a_c``.

    Returns a :class:`ScalarOperator`, a :class:`PauliStringOperator` or a
    :class:`~qalma.operators.pauli_sum.PauliSumOperator`, depending on the
    number of terms.
    """
    from .pauli_sum import PauliSumOperator

    coefficients = {code: coeff for code, coeff in coefficients.items() if coeff}
    if not coefficients:
        return ScalarOperator(0, system)
    if len(coefficients) == 1:
        ((code, coeff),) = coefficients.items()
        if code == 0:
            return ScalarOperator(coeff, system)
        return PauliStringOperator._new(code, coeff, system)
    return PauliSumOperator.from_dict(coefficients, system).collapse()


def collect_pauli_terms(
    terms: Iterable[Union[PauliStringOperator, ScalarOperator]],
    system: SystemDescriptor,
) -> Dict[int, complex]:
    """Add up the coefficients of equal strings."""
    result: Dict[int, complex] = {}
    for term in terms:
        code, coeff = _code_and_prefactor(term, system)
        result[code] = result.get(code, 0) + coeff
    return result


def _bilinear_pauli(op_1: Operator, op_2: Operator, mode: str) -> Operator:
    """Product, commutator or anticommutator of two sums of Pauli strings."""
    from .pauli_sum import as_pauli_sum

    sum_1, sum_2 = as_pauli_sum(op_1), as_pauli_sum(op_2)
    if sum_1 is None or sum_2 is None:
        raise TypeError("both operands must be sums of Pauli strings.")
    return sum_1._binary(sum_2, mode)


def commutator_pauli(op_1: Operator, op_2: Operator) -> Operator:
    r"""Commutator of two sums of Pauli strings, computed symbolically.

    Uses :math:`[P, Q] = 0` if the strings commute and :math:`2PQ`
    otherwise, for all the pairs of terms at once (see
    :class:`~qalma.operators.pauli_sum.PauliSumOperator`).
    """
    return _bilinear_pauli(op_1, op_2, "commutator")


def anticommutator_pauli(op_1: Operator, op_2: Operator) -> Operator:
    r"""Anticommutator of two sums of Pauli strings, computed symbolically."""
    return _bilinear_pauli(op_1, op_2, "anticommutator")


def is_pauli_sum(operator) -> bool:
    """``True`` if ``operator`` is a (possibly nested) sum of Pauli strings and scalars."""
    from .arithmetic import SumOperator
    from .pauli_sum import PauliSumOperator

    if isinstance(operator, (PauliStringOperator, ScalarOperator, PauliSumOperator)):
        return True
    if isinstance(operator, SumOperator):
        return all(is_pauli_sum(term) for term in operator.terms)
    return False


# ---------------------------------------------------------------------------
# Conversion of dense operators
# ---------------------------------------------------------------------------


def _local_pauli_components(
    matrix: np.ndarray, atol: float
) -> List[Tuple[int, complex]]:
    r"""Coefficients :math:`\mathrm{Tr}(\sigma^a M)/2` of a 2x2 matrix."""
    matrix = np.asarray(matrix)
    if matrix.shape != (2, 2):
        raise ValueError(f"local operator of shape {matrix.shape} is not 2x2.")
    a00, a01, a10, a11 = matrix[0, 0], matrix[0, 1], matrix[1, 0], matrix[1, 1]
    components = (
        (0, (a00 + a11) / 2),
        (1, (a01 + a10) / 2),
        (2, 1j * (a01 - a10) / 2),
        (3, (a00 - a11) / 2),
    )
    return [(c, complex(v)) for c, v in components if abs(v) > atol]


def to_pauli_sum(operator: Operator, atol: float = QALMA_TOLERANCE) -> Operator:
    """Expand ``operator`` as a sum of Pauli strings.

    Supports :class:`ScalarOperator`, :class:`LocalOperator`,
    :class:`ProductOperator` (including Pauli strings) and sums of them. A
    product acting nontrivially on ``k`` sites expands into at most
    ``4**k`` strings. Other operators (e.g. :class:`QutipOperator`) are
    converted first with ``as_sum_of_products``, if that yields a supported
    form.
    """
    from .arithmetic import SumOperator
    from .pauli_sum import PauliSumOperator

    if isinstance(operator, PauliSumOperator):
        return operator.collapse()
    system = operator.system
    coefficients: Dict[int, complex] = {}

    def add_product(site_factors: Mapping[str, np.ndarray], prefactor, sys_):
        index = _site_index(system)
        partial: Dict[int, complex] = {0: prefactor}
        for site, factor in site_factors.items():
            shift = 2 * index.position[site]
            components = _local_pauli_components(factor, atol)
            partial = {
                code | (local << shift): coeff * value
                for code, coeff in partial.items()
                for local, value in components
            }
        for code, coeff in partial.items():
            coefficients[code] = coefficients.get(code, 0) + coeff

    def visit(op: Operator):
        if isinstance(op, (int, float, complex, np.number)):
            coefficients[0] = coefficients.get(0, 0) + op
        elif isinstance(op, PauliStringOperator):
            code = op.code_in(system)
            coefficients[code] = coefficients.get(code, 0) + op.prefactor
        elif isinstance(op, PauliSumOperator):
            for code, coeff in op.to_dict().items():
                code = _remap_code(code, op.system, system)
                coefficients[code] = coefficients.get(code, 0) + coeff
        elif isinstance(op, ProductOperator):
            add_product(op.site_factors, op.prefactor, op.system)
        elif isinstance(op, LocalOperator):
            add_product({op.site: op.operator}, 1, op.system)
        elif isinstance(op, SumOperator):
            for term in op.terms:
                visit(term)
        else:
            expanded = op.as_sum_of_products()
            if expanded is op:
                raise TypeError(f"cannot expand {type(op)} in Pauli strings.")
            visit(expanded)

    visit(operator)
    coefficients = {c: v for c, v in coefficients.items() if abs(v) > atol}
    return pauli_dict_to_operator(coefficients, system)


# ---------------------------------------------------------------------------
# Fermions through the Jordan-Wigner transformation
# ---------------------------------------------------------------------------


def _jordan_wigner_string(system: SystemDescriptor, site: str) -> int:
    """Code of :math:`\\prod_{k < j} \\sigma^z_k`, with ``j`` the position of ``site``."""
    position = _site_index(system).position[site]
    return int("11" * position, 2) if position else 0


def majorana_operators(
    system: SystemDescriptor, site: str
) -> Tuple[PauliStringOperator, PauliStringOperator]:
    r"""Majorana operators of the mode at ``site``, as Pauli strings.

    .. math::

        \gamma_{j,1} = \Big(\prod_{k<j}\sigma^z_k\Big)\sigma^x_j,\qquad
        \gamma_{j,2} = \Big(\prod_{k<j}\sigma^z_k\Big)\sigma^y_j,

    with the sites ordered as in ``system.sites``. They satisfy
    :math:`\{\gamma_a, \gamma_b\} = 2\delta_{ab}`.
    """
    if system.dimensions[site] != 2:
        raise ValueError(f"site {site} is not a two-level site.")
    string = _jordan_wigner_string(system, site)
    shift = 2 * _site_index(system).position[site]
    return (
        PauliStringOperator._new(string | (1 << shift), 1.0, system),
        PauliStringOperator._new(string | (2 << shift), 1.0, system),
    )


def fermion_annihilation(system: SystemDescriptor, site: str) -> Operator:
    r"""Jordan-Wigner annihilation operator :math:`c_j = (\gamma_{j,1} + i\gamma_{j,2})/2`.

    The empty state of each mode is the eigenvector of :math:`\sigma^z` with
    eigenvalue ``+1`` (the first basis state), so that
    :math:`c^\dagger_j c_j = (1 - \sigma^z_j)/2`.
    """
    gamma_1, gamma_2 = majorana_operators(system, site)
    return pauli_dict_to_operator({gamma_1.code: 0.5, gamma_2.code: 0.5j}, system)


def fermion_creation(system: SystemDescriptor, site: str) -> Operator:
    r"""Jordan-Wigner creation operator :math:`c^\dagger_j = (\gamma_{j,1} - i\gamma_{j,2})/2`."""
    gamma_1, gamma_2 = majorana_operators(system, site)
    return pauli_dict_to_operator({gamma_1.code: 0.5, gamma_2.code: -0.5j}, system)


def fermion_number(system: SystemDescriptor, site: str) -> Operator:
    r"""Occupation number :math:`n_j = (1 - \sigma^z_j)/2`."""
    code = 3 << (2 * _site_index(system).position[site])
    return pauli_dict_to_operator({0: 0.5, code: -0.5}, system)
