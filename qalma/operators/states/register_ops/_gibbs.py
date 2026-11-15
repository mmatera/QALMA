"""Arithmetic handlers involving GibbsDensityOperator.

All handlers delegate to `.to_qutip_operator()`, which converts a Gibbs state
into a plain ProductDensityOperator before dispatching to the product handlers.

Registered operations:
  - Gibbs + numbers                     ->  (mul or convert)
  - Gibbs + BasicOperator/SumOp         ->  (convert, then add)
  - Gibbs * numbers                     ->  (mul or convert)
  - Gibbs * NonProductBasicOp/SumOp     ->  (convert, then mul)
  - NonProductBasicOp/SumOp * Gibbs     ->  (convert, then mul)
  - GibbsProduct * Gibbs               ->  (convert both, then mul)
  - Gibbs * GibbsProduct               ->  (convert both, then mul)
  - Gibbs * Gibbs               ->  (convert both, then mul)
  - Gibbs * ScalarOp/LocalOp/ProductOp  ->  (convert, then mul)
  - ScalarOp/LocalOp/ProductOp * Gibbs  ->  (convert, then mul)
"""

from typing import Callable, Tuple

import numpy as np

from qalma.operators.arithmetic import OneBodyOperator, SumOperator
from qalma.operators.basic import LocalOperator, Operator
from qalma.operators.product import ProductOperator, ScalarOperator
from qalma.operators.states.arithmetic import MixtureDensityOperator
from qalma.operators.states.basic import DensityOperatorMixin
from qalma.operators.states.gibbs import (
    GibbsDensityOperator,
    GibbsProductDensityOperator,
)
from qalma.operators.states.product import ProductDensityOperator

from ._types import (
    BASIC_OPERATOR_TYPES,
    COMPLEX_NUMERIC_TYPES,
    NON_PRODUCT_BASIC_OPERATOR_TYPES,
    REAL_NUMERIC_TYPES,
)
from ._wrappers import _wrapper_gibbs as _wrapper, _wrapper_gibbs_product


def _inherit_gibbs_arithmetic(subclass: type) -> None:
    """Register for ``subclass`` the arithmetic handlers of the Gibbs state.

    The dispatch of the arithmetic of the operators is done on the exact
    types of the operands, so a subclass must be registered explicitly.
    Products by positive real scalars are overridden to preserve the type.
    """
    # pylint: disable=import-outside-toplevel
    from qalma.operators.states.register_ops import _gibbs  # noqa: F401

    base = GibbsDensityOperator
    for table in (Operator.__add__dispatch__, Operator.__mul__dispatch__):
        for (left, right), func in list(table.items()):
            if left is base or right is base:
                new_key = (
                    subclass if left is base else left,
                    subclass if right is base else right,
                )
                table.setdefault(new_key, func)
                if left is base and right is base:
                    table.setdefault((subclass, base), func)
                    table.setdefault((base, subclass), func)

    mul_table = Operator.__mul__dispatch__

    def _scaled(fallback_key: Tuple[type, type]) -> Callable:
        fallback = mul_table.get(fallback_key)

        def handler(x_op, y_op):
            state, factor = (x_op, y_op) if isinstance(x_op, subclass) else (y_op, x_op)
            if isinstance(factor, (int, float)) and factor > 0:
                return state._with_prefactor(  # pylint: disable=protected-access
                    state.prefactor * factor
                )
            return fallback(x_op, y_op)

        return handler

    for key in [k for k in mul_table if k[0] is base or k[1] is base]:
        other = key[1] if key[0] is base else key[0]
        if other in (int, float, np.float64):
            new_key = (subclass, other) if key[0] is base else (other, subclass)
            mul_table[new_key] = _scaled(key)


# TODO: consider a shortcut


@Operator.register_add_handler(
    [(GibbsDensityOperator, type_1) for type_1 in REAL_NUMERIC_TYPES]
)
def gdo_add_real_(x_op: GibbsDensityOperator, y_op: float):
    """Add a gibbs density operator with a real number."""
    if y_op == 0:
        return x_op
    system = x_op.system
    if y_op > 0:
        return MixtureDensityOperator(
            (ProductDensityOperator({}, y_op, system), x_op), system
        )
    return _wrapper(x_op) + ScalarOperator(0, system=x_op.system)


@Operator.register_add_handler(
    [(GibbsDensityOperator, type_1) for type_1 in COMPLEX_NUMERIC_TYPES]
)
def gdo_add_complex_(x_op: GibbsDensityOperator, y_op: float):
    """Add a gibbs density operator with a complex number."""
    if y_op.imag == 0:
        return x_op + y_op.real
    if y_op == 0:
        return x_op
    return _wrapper(x_op) + ScalarOperator(0, system=x_op.system)


@Operator.register_add_handler(
    [
        (GibbsDensityOperator, DensityOperatorMixin),
        (GibbsDensityOperator, ProductDensityOperator),
        (GibbsDensityOperator, GibbsDensityOperator),
        (GibbsDensityOperator, GibbsProductDensityOperator),
    ]
)
def _(x_op: GibbsDensityOperator, y_op: DensityOperatorMixin):
    """Add a gibbs density operator with another density operator."""
    if x_op is y_op:
        return GibbsDensityOperator(
            x_op.k,
            x_op.system,
            prefactor=2 * x_op.prefactor,
            normalized=True,
            meanfield=x_op._meanfield,
            symmetry_projections=x_op.symmetry_projections,
        )

    return MixtureDensityOperator((x_op, y_op), x_op.system * y_op.system)


@Operator.register_add_handler(
    [
        (GibbsDensityOperator, MixtureDensityOperator),
    ]
)
def _(x_op: GibbsDensityOperator, y_op: MixtureDensityOperator):
    """Add a gibbs density operator with a mixture density operator."""
    return MixtureDensityOperator((x_op,) + y_op.terms, x_op.system * y_op.system)


@Operator.register_add_handler(
    [
        (
            MixtureDensityOperator,
            GibbsDensityOperator,
        ),
    ]
)
def _(y_op: MixtureDensityOperator, x_op: GibbsDensityOperator):
    """Add a gibbs density operator with a mixture density operator."""
    return MixtureDensityOperator((x_op,) + y_op.terms, x_op.system * y_op.system)


@Operator.register_add_handler(
    [(GibbsDensityOperator, type_op) for type_op in BASIC_OPERATOR_TYPES]
)
@Operator.register_add_handler((GibbsDensityOperator, SumOperator))
@Operator.register_add_handler((GibbsDensityOperator, OneBodyOperator))
def add_gdo_sum_(x_op: GibbsDensityOperator, y_op: Operator):
    """Add a gibbs density operator with another general operator."""
    return _wrapper(x_op) + y_op


# ####### Multiplication ###################


# ## with numbers


@Operator.register_mul_handler(
    [(GibbsDensityOperator, type_op) for type_op in REAL_NUMERIC_TYPES]
)
def mul_gibbs_times_float(x_op: GibbsDensityOperator, y_op: float):
    """Multiply a gibbs density operator with a real number."""
    if y_op == 0:
        return GibbsDensityOperator(
            ScalarOperator(0, x_op.system), x_op.system, prefactor=0, normalized=False
        )
    if y_op == 1:
        return x_op
    if 0 < y_op:
        return GibbsDensityOperator(
            x_op.k,
            x_op.system,
            prefactor=x_op.prefactor * y_op,
            normalized=True,
        )
    # Generic
    return _wrapper(x_op, y_op)


@Operator.register_mul_handler(
    [(type_op, GibbsDensityOperator) for type_op in REAL_NUMERIC_TYPES]
)
def _(y_op: float, x_op: GibbsDensityOperator):
    """Multiply a gibbs density operator with a real number."""
    if y_op == 0:
        return GibbsDensityOperator(
            ScalarOperator(0, x_op.system), x_op.system, prefactor=0, normalized=False
        )
    if y_op == 1:
        return x_op
    if 0 < y_op:
        return GibbsDensityOperator(
            x_op.k,
            x_op.system,
            prefactor=x_op.prefactor * y_op,
            normalized=x_op.normalized,
        )
    # Generic
    return _wrapper(x_op, y_op)


@Operator.register_mul_handler(
    [(GibbsDensityOperator, type_op) for type_op in COMPLEX_NUMERIC_TYPES]
)
def _(x_op: GibbsDensityOperator, y_op: complex):
    """Multiply a gibbs density operator with a complex number."""
    if y_op.imag == 0:
        return mul_gibbs_times_float(x_op, y_op.real)
    return _wrapper(x_op, y_op)


@Operator.register_mul_handler(
    [(type_op, GibbsDensityOperator) for type_op in COMPLEX_NUMERIC_TYPES]
)
def _(y_op: complex, x_op: GibbsDensityOperator):
    """Multiply a gibbs density operator with a complex number."""
    if y_op.imag == 0:
        return mul_gibbs_times_float(x_op, y_op.real)
    return _wrapper(x_op, y_op)


# With other DensityOperators:


@Operator.register_mul_handler((GibbsDensityOperator, GibbsDensityOperator))
def _(x_op: GibbsDensityOperator, y_op: GibbsDensityOperator):
    """Multiply two gibbs density operators."""
    return _wrapper(x_op) * _wrapper(y_op)


@Operator.register_mul_handler((GibbsProductDensityOperator, GibbsDensityOperator))
def _(x_op: GibbsProductDensityOperator, y_op: GibbsDensityOperator):
    """Multiply a gibbs density operator with a gibbs product density operator."""
    return _wrapper_gibbs_product(x_op) * _wrapper(y_op)


@Operator.register_mul_handler((GibbsDensityOperator, GibbsProductDensityOperator))
def _(x_op: GibbsDensityOperator, y_op: GibbsProductDensityOperator):
    """Multiply a gibbs density operator with a gibbs product density operator."""
    return _wrapper(x_op) * _wrapper_gibbs_product(y_op)


@Operator.register_mul_handler(
    [
        (GibbsDensityOperator, type_op)
        for type_op in (
            DensityOperatorMixin,
            ProductDensityOperator,
        )
    ]
)
def _(x_op: GibbsDensityOperator, y_op: DensityOperatorMixin):
    """Multiply a gibbs density operator with a general density operator."""
    return _wrapper(x_op) * y_op


@Operator.register_mul_handler(
    [
        (type_op, GibbsDensityOperator)
        for type_op in (
            DensityOperatorMixin,
            ProductDensityOperator,
        )
    ]
)
def _(x_op: DensityOperatorMixin, y_op: GibbsDensityOperator):
    """Multiply a gibbs density operator with a gibbs general density operator."""
    return x_op * _wrapper(y_op)


## With Basic Operators


@Operator.register_mul_handler(
    [
        (GibbsDensityOperator, type_op)
        for type_op in NON_PRODUCT_BASIC_OPERATOR_TYPES
        if type_op is not SumOperator
    ]
)
@Operator.register_mul_handler((GibbsDensityOperator, SumOperator))
def _(x_op: GibbsDensityOperator, y_op: Operator):
    """Multiply a gibbs density operator with a general operator."""
    return _wrapper(x_op) * y_op


@Operator.register_mul_handler(
    [
        (type_op, GibbsDensityOperator)
        for type_op in NON_PRODUCT_BASIC_OPERATOR_TYPES
        if type_op is not SumOperator
    ]
)
@Operator.register_mul_handler((SumOperator, GibbsDensityOperator))
def _(x_op: Operator, y_op: GibbsDensityOperator):
    """Multiply a gibbs density operator with a general operator."""
    return x_op * _wrapper(y_op)


@Operator.register_mul_handler((GibbsDensityOperator, ScalarOperator))
@Operator.register_mul_handler((GibbsDensityOperator, LocalOperator))
@Operator.register_mul_handler((GibbsDensityOperator, ProductOperator))
def _(x_op: GibbsDensityOperator, y_op: Operator):
    """Multiply a gibbs density operator with a local or product operator."""
    return _wrapper(x_op) * y_op


@Operator.register_mul_handler((ScalarOperator, GibbsDensityOperator))
@Operator.register_mul_handler((LocalOperator, GibbsDensityOperator))
@Operator.register_mul_handler((ProductOperator, GibbsDensityOperator))
def _(x_op: Operator, y_op: GibbsDensityOperator):
    """Multiply a gibbs density operator with a local or product operator."""
    return x_op * _wrapper(y_op)
