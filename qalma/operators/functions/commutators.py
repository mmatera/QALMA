"""Functions for operators."""

# from collections.abc import Iterable
# from typing import Callable, List, Optional, Tuple
from typing import Union

from qutip import Qobj

from qalma.operators.arithmetic import SumOperator, iterable_to_operator
from qalma.operators.basic import (
    Operator,
)
from qalma.operators.pauli import (
    anticommutator_pauli,
    commutator_pauli,
    is_pauli_sum,
)
from qalma.operators.simplify import _commutator_term_pairs, collect_nbody_terms
from qalma.parallel import USE_PARALLEL, commutator_qalma_parallel

# from qalma.operators.simplify import simplify_sum_operator


# from datetime import datetime


__all__ = ["anticommutator", "commutator"]


def anticommutator(
    op_1: Union[Qobj, Operator], op_2: Union[Qobj, Operator]
) -> Union[Qobj, Operator]:
    """Compute the anticommutator of two operators, {op1, op2} = op1*op2 + op2*op1.

    Parameters
    ----------
    op_1 : Operator or qutip.Qobj
        First operator.
    op_2 : Operator or qutip.Qobj
        Second operator.

    Returns
    -------
    Operator or qutip.Qobj
        The anticommutator ``op_1 * op_2 + op_2 * op_1``.

    """
    if isinstance(op_1, Qobj):
        if not isinstance(op_2, Qobj):
            op_2 = op_2.to_qutip()
        return op_1 * op_2 + op_2 * op_1
    if isinstance(op_2, Qobj):
        op_1 = op_1.to_qutip()
        return op_1 * op_2 + op_2 * op_1

    return anticommutator_qalma(op_1, op_2)


def anticommutator_qalma_serial(op_1: Operator, op_2: Operator) -> Operator:
    r"""Compute the anticommutator of two QALMA operators.

    The anticommutator is defined as

    .. math::

        \{op_1, op_2\} = op_1 \cdot op_2 + op_2 \cdot op_1.

    Parameters
    ----------
    op_1 : Operator
        First operator.
    op_2 : Operator
        Second operator.

    Returns
    -------
    Operator
        The anticommutator ``op_1 * op_2 + op_2 * op_1``, simplified.

    """
    system = op_1.system or op_2.system
    if isinstance(op_1, SumOperator):
        return SumOperator(
            tuple((anticommutator_qalma(term, op_2) for term in op_1.terms)),
            system,
        ).simplify()
    if isinstance(op_2, SumOperator):
        return SumOperator(
            tuple((anticommutator_qalma(op_1, term) for term in op_2.terms)),
            system,
        ).simplify()

    # TODO: Handle fermions...
    acts_over_1, acts_over_2 = op_1.acts_over(), op_2.acts_over()
    if acts_over_1 is not None:
        if len(acts_over_1) == 0:
            return op_2 * (op_1 * 2)
        if acts_over_2 is not None:
            if len(acts_over_2) == 0:
                return op_1 * (op_2 * 2)
            if len(acts_over_1.intersection(acts_over_2)) == 0:
                return (op_1 * op_2).simplify() * 2
    return (op_1 * op_2 + op_2 * op_1).simplify()


def commutator(
    op_1: Union[Operator, Qobj], op_2: Union[Operator, Qobj]
) -> Union[Qobj, Operator]:
    """Commutator of two operators."""
    if isinstance(op_1, Qobj):
        if not isinstance(op_2, Qobj):
            op_2 = op_2.to_qutip()
        return op_1 * op_2 - op_2 * op_1
    if isinstance(op_2, Qobj):
        op_1 = op_1.to_qutip()
        return op_1 * op_2 - op_2 * op_1

    return commutator_qalma(op_1, op_2)


def commutator_qalma_serial(op_1: Operator, op_2: Operator) -> Operator:
    """Compute the commutator of two Operator objects `op_1` and `op_2` (serial implementation).

    Serial implementation.
    """
    system = op_1.system.union(op_2.system)
    op_1_terms = collect_nbody_terms(op_1.flat())
    op_2_terms = collect_nbody_terms(op_2.flat())

    terms = tuple(
        op_1 * op_2 - op_2 * op_1
        for op_1, op_2 in _commutator_term_pairs(op_1_terms, op_2_terms)
    )
    return iterable_to_operator(terms, system).simplify()


_commutator_qalma_generic = (
    commutator_qalma_parallel if USE_PARALLEL else commutator_qalma_serial
)


def commutator_qalma(op_1: Operator, op_2: Operator) -> Operator:
    """Commutator of two QALMA operators.

    Sums of Pauli strings are handled symbolically by
    :func:`qalma.operators.pauli.commutator_pauli`; any other combination
    goes through the generic (serial or parallel) implementation.
    """
    if is_pauli_sum(op_1) and is_pauli_sum(op_2):
        return commutator_pauli(op_1, op_2)
    return _commutator_qalma_generic(op_1, op_2)


def anticommutator_qalma(op_1: Operator, op_2: Operator) -> Operator:
    """Anticommutator of two QALMA operators (symbolic for Pauli sums)."""
    if is_pauli_sum(op_1) and is_pauli_sum(op_2):
        return anticommutator_pauli(op_1, op_2)
    return anticommutator_qalma_serial(op_1, op_2)
