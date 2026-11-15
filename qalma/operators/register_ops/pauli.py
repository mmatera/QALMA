"""Arithmetic bindings for PauliStringOperator and PauliSumOperator.

Covers:
  - PauliStringOperator * PauliStringOperator   (symbolic, on the bit strings)
  - PauliStringOperator * Number / ScalarOperator  (and reversed)
  - PauliStringOperator + PauliStringOperator   (equal strings are merged)
  - PauliStringOperator + Number / ScalarOperator  (and reversed)

Any other combination (with LocalOperator, ProductOperator, SumOperator,
QutipOperator, states, ...) uses the handlers registered for
ProductOperator, which work on the dense ``site_factors``.
"""

from qalma.operators.arithmetic import SumOperator
from qalma.operators.basic import Operator
from qalma.operators.pauli import PauliStringOperator, pauli_product
from qalma.operators.pauli_sum import PauliSumOperator, as_pauli_sum
from qalma.operators.product import ProductOperator, ScalarOperator
from qalma.operators.utils import register_dispatch_fallback

from ._common import NUMERIC_TYPES

register_dispatch_fallback(PauliStringOperator, ProductOperator)
register_dispatch_fallback(PauliSumOperator, SumOperator)


@Operator.register_mul_handler((PauliStringOperator, PauliStringOperator))
def _(x_op: PauliStringOperator, y_op: PauliStringOperator):
    return pauli_product(x_op, y_op)


@Operator.register_mul_handler([(PauliStringOperator, t) for t in NUMERIC_TYPES])
def _(x_op: PauliStringOperator, y_value: complex) -> Operator:
    if y_value:
        return PauliStringOperator._new(
            x_op.code, x_op.prefactor * y_value, x_op.system
        )
    return ScalarOperator(0, x_op.system)


@Operator.register_mul_handler([(t, PauliStringOperator) for t in NUMERIC_TYPES])
def _(y_value: complex, x_op: PauliStringOperator) -> Operator:
    if y_value:
        return PauliStringOperator._new(
            x_op.code, x_op.prefactor * y_value, x_op.system
        )
    return ScalarOperator(0, x_op.system)


@Operator.register_mul_handler(
    [(PauliStringOperator, ScalarOperator), (ScalarOperator, PauliStringOperator)]
)
def _(x_op: Operator, y_op: Operator) -> Operator:
    if isinstance(x_op, ScalarOperator):
        x_op, y_op = y_op, x_op
    system = x_op.system.union(y_op.system)
    prefactor = x_op.prefactor * y_op.prefactor
    if prefactor:
        return PauliStringOperator._new(x_op.code_in(system), prefactor, system)
    return ScalarOperator(0, system)


@Operator.register_add_handler((PauliStringOperator, PauliStringOperator))
def _(x_op: PauliStringOperator, y_op: PauliStringOperator) -> Operator:
    system = x_op.system.union(y_op.system)
    code = x_op.code_in(system)
    if code == y_op.code_in(system):
        prefactor = x_op.prefactor + y_op.prefactor
        if code == 0 or not prefactor:
            return ScalarOperator(prefactor if code == 0 else 0, system)
        return PauliStringOperator._new(code, prefactor, system)
    return as_pauli_sum(x_op).add(as_pauli_sum(y_op))


@Operator.register_add_handler([(PauliStringOperator, t) for t in NUMERIC_TYPES])
def _(x_op: PauliStringOperator, y_value: complex) -> Operator:
    return x_op + ScalarOperator(y_value, x_op.system)


@Operator.register_add_handler(
    [(PauliStringOperator, ScalarOperator), (ScalarOperator, PauliStringOperator)]
)
def _(x_op: Operator, y_op: Operator) -> Operator:
    if isinstance(x_op, ScalarOperator):
        x_op, y_op = y_op, x_op
    system = x_op.system.union(y_op.system)
    if x_op.code == 0:
        return ScalarOperator(x_op.prefactor + y_op.prefactor, system)
    if not y_op.prefactor:
        return x_op
    return as_pauli_sum(x_op).add(as_pauli_sum(y_op))


# ---------------------------------------------------------------------------
# PauliSumOperator
# ---------------------------------------------------------------------------

_PAULI_TYPES = (PauliSumOperator, PauliStringOperator, ScalarOperator)
_PAULI_SUM_PAIRS = [
    (a, b) for a in _PAULI_TYPES for b in _PAULI_TYPES if PauliSumOperator in (a, b)
]


@Operator.register_mul_handler(_PAULI_SUM_PAIRS)
def _(x_op: Operator, y_op: Operator) -> Operator:
    return as_pauli_sum(x_op).mul(as_pauli_sum(y_op))


@Operator.register_add_handler(_PAULI_SUM_PAIRS)
def _(x_op: Operator, y_op: Operator) -> Operator:
    return as_pauli_sum(x_op).add(as_pauli_sum(y_op))


@Operator.register_mul_handler(
    [(PauliSumOperator, t) for t in NUMERIC_TYPES]
    + [(t, PauliSumOperator) for t in NUMERIC_TYPES]
)
def _(x_op, y_op) -> Operator:
    if isinstance(x_op, PauliSumOperator):
        return x_op.scale(y_op)
    return y_op.scale(x_op)


@Operator.register_add_handler([(PauliSumOperator, t) for t in NUMERIC_TYPES])
def _(x_op: PauliSumOperator, y_value: complex) -> Operator:
    if not y_value:
        return x_op
    return x_op.add(as_pauli_sum(ScalarOperator(y_value, x_op.system)))
