"""Different representations for operators."""

import logging
from functools import cached_property, reduce
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import qutip  # type: ignore[import-untyped]
from qutip import Qobj

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
    QALMA_ALLOW_OVERWRITE_BINDINGS,
)

from .utils import find_arithmetic_implementation

__all__ = ["Operator", "LocalOperator"]


class Operator:  # pylint: disable=too-many-public-methods
    """Base class for operators."""

    system: SystemDescriptor
    prefactor: complex = 1.0

    # TODO check if it is possible implementing this
    # with multimethods
    __add__dispatch__: Dict[Tuple, Callable] = {}
    __mul__dispatch__: Dict[Tuple, Callable] = {}

    @staticmethod
    def _register_handler(dispatch_table: Dict, key: Tuple | List[Tuple]):
        """Return a decorator that registers a handler in ``dispatch_table``.

        Parameters
        ----------
        dispatch_table:
            One of ``Operator.__add__dispatch__`` or
            ``Operator.__mul__dispatch__``.
        key:
            A single ``(TypeA, TypeB)`` pair, or a list of such pairs, that
            map to the decorated function.
        """

        def register_func(func):
            keys = key if isinstance(key[0], (list, tuple)) else (key,)
            for curr_key in keys:
                if curr_key in dispatch_table:
                    if not QALMA_ALLOW_OVERWRITE_BINDINGS:
                        assert curr_key not in dispatch_table, (
                            f"{curr_key} already registered in "
                            f"{dispatch_table[curr_key].__code__}."
                        )
                dispatch_table[curr_key] = func
            return func

        return register_func

    @staticmethod
    def register_add_handler(key: Tuple | List[Tuple]):
        """Return a decorator that registers an addition handler."""
        return Operator._register_handler(Operator.__add__dispatch__, key)

    @staticmethod
    def register_mul_handler(key: Tuple | List[Tuple]):
        """Return a decorator that registers a multiplication handler."""
        return Operator._register_handler(Operator.__mul__dispatch__, key)

    def __bool__(self):
        """Return False if the operator is zero, True otherwise."""
        return not self.is_zero

    def __add__(self, term):
        """Add ``term`` to this operator using the registered dispatch table."""
        dispatch_table = Operator.__add__dispatch__
        # First try with the cases stored in the dispatch table:
        func = dispatch_table.get((type(self), type(term)), None)
        if func is not None:
            return func(self, term)

        func = dispatch_table.get((type(term), type(self)), None)
        if func is not None:
            return func(term, self)

        # Now, look for cases associated to the class hierarchy
        func = find_arithmetic_implementation(self, term, dispatch_table)
        if func:
            return func(self, term)
        func = find_arithmetic_implementation(term, self, dispatch_table)
        if func:
            return func(term, self)
        try:
            return term.__radd__(self)
        except TypeError as exc:
            raise TypeError(f"{type(self)} cannot be added with  {type(term)}") from exc

    def __mul__(self, factor):
        """Multiply this operator by ``factor`` using the registered dispatch table."""
        # Use multiple dispatch to determine how to multiply
        dispatch_table = Operator.__mul__dispatch__
        # First try with the cases stored in the dispatch table:
        func = dispatch_table.get((type(self), type(factor)), None)
        if func is not None:
            return func(self, factor)
        # Now, look for cases associated to the class hierarchy
        func = find_arithmetic_implementation(self, factor, dispatch_table)
        if func:
            return func(self, factor)

        try:
            return factor.__rmul__(self)
        except TypeError as exc:
            raise TypeError(
                f"{type(self)} cannot be multiplied with  {type(factor)}"
            ) from exc

    def __neg__(self):
        """Return the negation of this operator."""
        return -(self.to_qutip_operator())

    def __sub__(self, operand):
        """Subtract ``operand`` from this operator."""
        from qalma.operators.product import ScalarOperator

        if operand is self:
            return ScalarOperator(0, self.system)
        if operand is None:
            raise ValueError("None can not be an operand")
        neg_op = -operand
        return self + neg_op

    def __radd__(self, term):
        """Add this operator to ``term`` (right-hand addition)."""
        # Use multiple dispatch to determine how to add
        dispatch_table = Operator.__add__dispatch__
        # First try with the cases stored in the dispatch table:
        func = dispatch_table.get(
            (
                type(term),
                type(self),
            ),
            None,
        )
        if func is not None:
            return func(term, self)
        # Now, look for cases associated to the class hierarchy
        func = find_arithmetic_implementation(term, self, dispatch_table)
        if func:
            return func(term, self)

        # Last chance: try in the opposite direction
        func = dispatch_table.get(
            (
                type(self),
                type(term),
            ),
            None,
        )
        if func is not None:
            return func(self, term)
        func = find_arithmetic_implementation(self, term, dispatch_table)
        if func:
            return func(self, term)

        raise TypeError(f"{type(self)} cannot be added with  {type(term)}")

    def __rmul__(self, factor):
        """Multiply ``factor`` by this operator (right-hand multiplication)."""
        # Use __mul__dispatch__ to determine how to evaluate the product

        dispatch_table = Operator.__mul__dispatch__

        # First try with the cases stored in the dispatch table:
        func = dispatch_table.get((type(factor), type(self)), None)
        if func is not None:
            return func(factor, self)
        # Now, look for cases associated to the class hierarchy
        func = find_arithmetic_implementation(factor, self, dispatch_table)
        if func:
            return func(factor, self)

        raise TypeError(f"{type(factor)} cannot be multiplied with  {type(self)}")

    def __rsub__(self, operand):
        """Subtract this operator from ``operand`` (right-hand subtraction)."""
        if operand is None:
            raise ValueError("None can not be an operand")

        neg_self = -self
        return operand + neg_self

    def __pow__(self, exponent):
        """Raise this operator to ``exponent`` via the QuTiP representation."""
        if exponent is None:
            raise ValueError("None can not be an operand")

        return self.to_qutip_operator() ** exponent

    def __truediv__(self, operand):
        """Divide this operator by a scalar or another operator."""
        if isinstance(operand, (int, float, complex)):
            return self * (1.0 / operand)
        if isinstance(operand, Operator):
            return self * operand.inv()
        raise ValueError("Division of an operator by ", type(operand), " not defined.")

    def _repr_latex_(self):
        """Latex representation."""
        acts_over = sorted(self.acts_over())
        if len(acts_over) > 4:
            return repr(self)
        qutip_repr = self.to_qutip(tuple(acts_over))
        if isinstance(qutip_repr, Qobj):
            # pylint: disable=protected-access
            parts = qutip_repr._repr_latex_().replace("$$", "$").split("$")
            if len(parts) != 3:
                tex = "-?-"
            else:
                tex = parts[1]
        else:
            tex = str(qutip_repr)
        result = f"${tex}_" + "{" + ",".join(acts_over) + "}$"
        return result

    def acts_over(self) -> frozenset:
        """Return the list of sites over which the operator acts nontrivially.

        If this cannot be determined, return None.
        """
        raise NotImplementedError

    def as_sum_of_products(self):
        """Decompose an operator as a sum of product operators."""
        return self

    def dag(self):
        """Adjoint operator of quantum object."""
        return self.to_qutip_operator().dag()

    def flat(self):
        """Simplifies sums and products."""
        return self

    def hermitian_part(self):
        r"""Return the Hermitian part of the operator, $\frac{O + O^{\dagger}}{2}$."""
        if self.isherm:
            return self
        return (self + self.dag()) * 0.5

    @property
    def isherm(self) -> bool:
        """Check if the operator is hermitian."""
        return self.to_qutip(tuple()).tidyup().isherm

    @property
    def isdiagonal(self) -> bool:
        """Check if the operator is diagonal."""
        return False

    @property
    def is_zero(self) -> bool:
        """True if self is a null operator."""
        return is_empty_op(self)

    def eigenenergies(self):
        """List of eigenstates of the operator."""
        return self.to_qutip_operator().eigenenergies()

    def eigenstates(self):
        """List of eigenstates of the operator."""
        return self.to_qutip_operator().eigenstates()

    def expm(self) -> "Operator":
        """Compute the exponential of the Qutip representation of the operator."""
        # Import here to avoid circular dependency
        # pylint: disable=import-outside-toplevel
        # type: ignore[import-untyped]
        from scipy.sparse.linalg import ArpackError

        from qalma.operators.functions import eigenvalues
        from qalma.operators.qutip import QutipOperator

        op_qutip = self.to_qutip()
        try:
            max_eval = eigenvalues(op_qutip, sort="high", sparse=True, eigvals=3)[0]
        except ArpackError:
            max_eval = max(op_qutip.diag())

        op_qutip = (op_qutip - max_eval).expm()
        return QutipOperator(op_qutip, self.system, prefactor=np.exp(max_eval))

    def inv(self) -> "Operator":
        """Return the inverse of the operator."""
        return self.to_qutip_operator().inv()

    def logm(self) -> "Operator":
        """Logarithm of the operator."""
        return self.to_qutip_operator().logm()

    def n_body_sector(self) -> int:
        """Return the maximum number of factors of any term in a product state decomposition."""
        return len(self.acts_over())

    def num_terms(self) -> int:
        """Return the number of terms that span the operator."""
        return 1

    def norm(self, ord: Optional[int | str | float] = None):
        """Compute the norm of the operator."""
        return norm(self.to_qutip(), ord)

    def partial_trace(self, sites: Union[frozenset, SystemDescriptor]):
        """Partial trace over sites not listed in `sites`."""
        raise NotImplementedError

    def reduce(self, sites: Iterable, state=None):
        """Compute the partial trace of the product of the operator and the density operator.

        The density operator acts on the subsystem which is traced out. If the
        state is not provided, the result is the partial trace divided by the
        dimension of the subsystem traced out.

        Parameters
        ----------
        sites : Iterable
            Sites to keep.
        state : Optional[DensityOperatorProtocol]
            The state relative to which the reduction is made.

        Returns
        -------
        Operator
            The reduced operator.

        """
        raise NotImplementedError

    def _set_system_(self, system=None):
        """Change the system associated to the operator, and references of other operators inside.

        In a multiprocess context, the `system` attribute of the objects
        generated by the children process lost their identity regarding
        the `system` attribute of the committed object. To get the right
        reference on the returned objects, call this method without
        parameters in the worker, before returning the objects. Then, in
        the main process, set back the original system object.
        """
        self.system = system
        return self

    def simplify(self) -> "Operator":
        """Return a more efficient representation."""
        return self

    def to_qutip(self, block: Optional[Tuple[str, ...]] = None):
        """Convert to a Qutip object."""
        raise NotImplementedError

    def to_qutip_operator(self):
        """Produce a Qutip representation of the operator."""
        # pylint: disable=import-outside-toplevel

        block = tuple(sorted(self.acts_over()))
        if len(block) == 0:
            return self
        site_names = {site: i for i, site in enumerate(block)}
        qobj = self.to_qutip(block)
        if isinstance(qobj, Qobj):
            from .qutip import QutipOperator

            assert qobj.type != "scalar"
            return QutipOperator(qobj, system=self.system, names=site_names)

        from .product import ScalarOperator

        return ScalarOperator(qobj, self.system)

    # pylint: disable=invalid-name
    def tr(self) -> complex:
        r"""Return the trace of the operator over the full system.

        Delegates to ``partial_trace`` with an empty site set, then
        returns the scalar ``prefactor`` of the result.  Subclasses that
        compute ``tr`` via a different code-path should override this.

        Returns
        -------
        complex
            :math:`\mathrm{Tr}(O)`.

        """
        return self.partial_trace(frozenset()).prefactor

    def tidyup(self, _atol=None):
        """Remove tiny elements of the operator."""
        return self


class LocalOperator(Operator):
    """Operator acting over a single site."""

    _to_qutip_cache: Dict[Optional[Tuple[str, ...]], Qobj]
    operator: np.ndarray
    site: str

    def __init__(
        self,
        site: str,
        local_operator,
        system: Optional[SystemDescriptor] = None,
    ):
        """Initialize a LocalOperator.

        Parameters
        ----------
        site : str
            Name of the site on which this operator acts.
        local_operator : np.ndarray, Qobj, or scalar
            The local matrix. Scalars are interpreted as multiples of the
            site identity.
        system : SystemDescriptor
            Descriptor of the full lattice system. Must not be ``None``.

        """
        assert system is not None
        self.site = site
        if isinstance(local_operator, (int, float, complex)):
            local_operator = system.site_identity(site) * local_operator

        self.operator = _to_array(local_operator)
        if isinstance(local_operator, Qobj):
            self.__dict__["operator_qutip"] = local_operator

        self._to_qutip_cache = {}
        self.system = system

    def __bool__(self):
        """Return False if the operator matrix is all zeros, True otherwise."""
        return bool(self.operator.any())

    def __neg__(self):
        """Return the negation of this local operator."""
        return LocalOperator(self.site, -self.operator, self.system)

    def __pow__(self, exp):
        """Raise this local operator to the power ``exp``."""
        operator = self.operator_qutip
        if exp < 0 and hasattr(operator, "inv"):
            operator = operator.inv()
            exp = -exp

        return LocalOperator(self.site, operator**exp, self.system)

    def __repr__(self):
        """Return a string representation of this local operator."""
        return f"Local Operator on site {self.site}:" f"\n {repr(self.operator_qutip)}"

    @cached_property
    def operator_qutip(self) -> Qobj:
        """Return a Qutip representation of the local operator."""
        return to_qobj(self.operator.copy())

    def acts_over(self) -> frozenset:
        """Return the singleton set containing the site of this operator.

        Returns
        -------
        frozenset[str]
            ``frozenset({self.site})``.

        """
        return frozenset((self.site,))

    def dag(self):
        """Return the adjoint operator."""
        operator = self.operator
        if self.isherm:
            return self
        return LocalOperator(self.site, operator.T.conj(), self.system)

    def expm(self):
        """Return the matrix exponential :math:`e^O` of the local operator.

        Returns
        -------
        LocalOperator
            A local operator on the same site with matrix :math:`e^O`.

        """
        return LocalOperator(self.site, self.operator_qutip.expm(), self.system)

    def hermitian_part(self):
        """Return the Hermitian part of the local operator, (O + O†) / 2."""
        op = self.operator
        if self.isherm:
            return self
        op = (op + op.T.conj()) * 0.5
        return LocalOperator(self.site, op, self.system)

    def inv(self):
        """Return the inverse operator :math:`O^{-1}`.

        Returns
        -------
        LocalOperator
            A local operator on the same site with matrix :math:`O^{-1}`.

        """
        operator = self.operator_qutip
        system = self.system
        site = self.site
        return LocalOperator(
            site,
            operator.inv() if hasattr(operator, "inv") else 1 / operator,
            system,
        )

    @cached_property
    def isherm(self) -> bool:
        """``True`` if the local matrix is Hermitian."""
        return ishermitian(self.operator)

    @cached_property
    def isdiagonal(self) -> bool:
        """``True`` if the local matrix is diagonal."""
        return is_diagonal_op(self.operator)

    def logm(self):
        r"""Return the matrix logarithm of the local operator.

        Computed via eigendecomposition. Eigenvalues below ``1e-50`` are
        clamped to avoid numerical divergence in the logarithm.

        Returns
        -------
        LocalOperator
            A local operator on the same site with matrix :math:`\\log O`.

        """

        def log_qutip(loc_op):
            """Compute matrix log via eigendecomposition, clamping near-zero eigenvalues."""
            evals, evecs = loc_op.eigenstates()
            evals[abs(evals) < 1.0e-50] = 1.0e-50
            return sum(
                np.log(e_val) * e_vec * e_vec.dag()
                for e_val, e_vec in zip(evals, evecs)
            )

        return LocalOperator(self.site, log_qutip(self.operator_qutip), self.system)

    def norm(self, ord=None):
        """Compute the norm of the operator."""
        result = norm(self.operator, ord)
        if ord in ("fro", "nuc"):
            dim_factor = 1.0
            for dim in (
                dim for site, dim in self.system.dimensions.items() if site != self.site
            ):
                dim_factor *= dim
            if ord == "fro":
                result *= dim_factor**0.5
            else:
                result *= dim_factor

        return result

    def partial_trace(self, sites: Union[frozenset, SystemDescriptor]):
        r"""Compute the partial trace over the complement of ``sites``.

        If the operator's site is not in ``sites``, returns a
        :class:`~qalma.operators.product.ScalarOperator` with value
        :math:`\\mathrm{Tr}(O) \\cdot \\prod_{j \\notin \\{i\\} \\cup \\text{sites}} d_j`.
        Otherwise returns a :class:`LocalOperator` scaled by the same
        dimensional prefactor.

        Parameters
        ----------
        sites : frozenset[str] or SystemDescriptor
            Sites to *keep*. All other sites are traced out.

        Returns
        -------
        Operator
            The reduced operator on the subsystem defined by ``sites``.

        """
        # pylint: disable=import-outside-toplevel

        system = self.system
        dimensions = system.dimensions
        subsystem = (
            sites if isinstance(sites, SystemDescriptor) else system.subsystem(sites)
        )
        local_sites = subsystem.sites
        site = self.site
        prefactors = [
            d for s, d in dimensions.items() if s != site and s not in local_sites
        ]

        if len(prefactors) > 0:
            prefactor = reduce(lambda x, y: x * y, prefactors)
        else:
            prefactor = 1

        local_op = self.operator
        if site not in local_sites:
            from .product import ScalarOperator

            return ScalarOperator(local_op.trace() * prefactor, subsystem)
        return LocalOperator(site, local_op * prefactor, subsystem)

    def reduce(self, sites: Iterable, state=None) -> Operator:
        """Compute the partial trace of the product of the operator and the density operator.

        The density operator acts on the subsystem which is traced out. If the
        state is not provided, the result is the partial trace divided by the
        dimension of the subsystem traced out.

        Parameters
        ----------
        sites : Iterable
            Sites to keep.
        state : Optional[DensityOperatorProtocol]
            The state relative to which the reduction is made.

        Returns
        -------
        Operator
            The reduced operator.

        """
        # pylint: disable=import-outside-toplevel

        scalar_val: complex
        site = self.site
        if site in sites:
            return self
        system = self.system
        if state is not None:
            scalar_val = state.expect(self)
        else:
            scalar_val = self.operator.trace() / system.dimensions[site]

        from .product import ScalarOperator

        return ScalarOperator(scalar_val, system)

    def simplify(self):
        """Return a simpler equivalent operator if the local matrix is scalar.

        If the local matrix is a multiple of the identity, returns a
        :class:`~qalma.operators.product.ScalarOperator`. Otherwise returns
        ``self`` unchanged.

        Returns
        -------
        Operator
            A :class:`~qalma.operators.product.ScalarOperator` if the matrix
            is proportional to the identity, otherwise ``self``.

        """
        # TODO: reduce multiples of the identity to ScalarOperators
        # pylint: disable=import-outside-toplevel
        operator = self.operator
        if not is_scalar_op(operator):
            return self
        value = operator[0, 0] * self.prefactor

        from .product import ScalarOperator

        return ScalarOperator(value, self.system)

    def to_qutip(self, block: Optional[Tuple[str, ...]] = None):
        """Convert to a Qutip object."""
        cached = self._to_qutip_cache.get(block, None)
        if cached is not None:
            return cached

        site = self.site
        system = self.system
        sites = system.sites
        dimensions = system.dimensions
        operator = self.operator
        # Ensure that block at least contains site
        orig_block = block
        if block is None:
            block = tuple(sorted(sites))
            if len(block) > 8:
                logging.warning(
                    "Asking for a qutip representation of an operator over the full system"
                )
        elif site not in block:
            block = block + (site,)
        # Ensure that operator is a qutip operator
        if isinstance(operator, (int, float, complex)):
            operator = qutip.qeye(dimensions[site]) * operator
        elif isinstance(operator, Operator):
            operator = operator.to_qutip((site,))
        else:
            operator = self.operator_qutip
        # Build factors
        factors_dict = (operator if s == site else sites[s]["identity"] for s in block)
        self._to_qutip_cache[orig_block] = result = fast_tensor(*factors_dict)
        return result

    def tidyup(self, atol=None):
        """Remove tiny elements of the operator."""
        return LocalOperator(self.site, self.operator_qutip.tidyup(atol), self.system)
