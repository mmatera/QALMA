"""Utility functions used in Operator classes."""

import logging
from typing import Callable, Optional

from qalma.settings import (
    QALMA_INFER_ARITHMETICS,
)

logger = logging.getLogger(__name__)

__all__ = ["find_arithmetic_implementation", "register_dispatch_fallback"]

# Classes that, when no handler is registered for them, use the handlers of
# another class (e.g. PauliStringOperator -> ProductOperator). This is
# independent of QALMA_INFER_ARITHMETICS: only explicitly declared
# substitutions are used.
DISPATCH_FALLBACK: dict = {}


def register_dispatch_fallback(cls: type, base: type):
    """Use the arithmetic handlers of ``base`` for ``cls`` when ``cls`` has none.

    ``cls`` must be a subclass of ``base`` whose instances provide the
    interface that the handlers of ``base`` rely on.
    """
    assert issubclass(cls, base)
    DISPATCH_FALLBACK[cls] = base


def find_arithmetic_implementation(
    op1, op2, dispatch_table: dict
) -> Optional[Callable]:
    """Lookup for the implementation of an arithmetic operation.

    Find the function that implements the operation op1 [operation] op2 in
    the dispatch table dispatch.

    If the combination of types is not already in the dispatch table,
    store it.
    """
    type_op1, type_op2 = type(op1), type(op2)
    # Explicitly declared substitutions first
    fallback_1 = DISPATCH_FALLBACK.get(type_op1, type_op1)
    fallback_2 = DISPATCH_FALLBACK.get(type_op2, type_op2)
    if (fallback_1, fallback_2) != (type_op1, type_op2):
        for key in ((fallback_1, type_op2), (type_op1, fallback_2), (fallback_1, fallback_2)):
            func = dispatch_table.get(key, None)
            if func is not None:
                dispatch_table[(type_op1, type_op2)] = func
                return func

    op1_parent_classes = type_op1.__mro__
    op2_parent_classes = type_op2.__mro__
    # Go over the combinations of parent classes
    for lhf in op1_parent_classes:
        for rhf in op2_parent_classes:
            key = (lhf, rhf)
            if key in dispatch_table:
                func = dispatch_table[key]
                if QALMA_INFER_ARITHMETICS:
                    dispatch_table[(type_op1, type_op2)] = func
                    return func
                logger.warning("try with %s", func.__code__)
                return None

    # Last resource: try if the operands are instances of one of the keys
    # in the dispatch table.
    # Required for example for keys of the form (Operator, complex).

    for key, func in dispatch_table.items():
        if isinstance(op1, key[0]) and isinstance(op2, key[1]):
            func = dispatch_table[key]
            if QALMA_INFER_ARITHMETICS:
                dispatch_table[(type_op1, type_op2)] = func
                return func
            logger.warning("try with %s", func.__code__)
            return None
    return None
