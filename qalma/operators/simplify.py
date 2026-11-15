# -*- coding: utf-8 -*-
"""Functions to simplify sums of operators."""

import logging
from typing import Callable, Dict, List, Optional, Sequence, cast

from qalma.model import SystemDescriptor
from qalma.operators.arithmetic import OneBodyOperator, SumOperator
from qalma.operators.basic import (
    Operator,
)
from qalma.operators.pauli import (
    PauliStringOperator,
    collect_pauli_terms,
    pauli_dict_to_operator,
)
from qalma.operators.product import (
    ProductOperator,
    ScalarOperator,
)
from qalma.operators.qutip import QutipOperator
from qalma.qutip_tools.tools import (
    is_empty_op,
)

logger = logging.getLogger(__name__)

__all__ = ["collect_nbody_terms", "group_terms_by_blocks", "simplify_qutip_sums"]


def sum_operator_sequence(
    seq: Sequence[Operator], system: SystemDescriptor, **attrs
) -> Operator:
    """Convert a sequence of operators into their sum.

    Parameters
    ----------
    seq : list or tuple of Operator
        The operators to sum.
    system : SystemDescriptor
        The system the operators act on.
    **attrs
        Additional keyword arguments forwarded to :class:`SumOperator`.

    Returns
    -------
    Operator
        The sum of the terms in the sequence.

    """
    if not seq:
        return ScalarOperator(0, system)
    if len(seq) == 1:
        return seq[0]
    return SumOperator(tuple(seq), system=system, **attrs)


def collect_nbody_terms(operator: Operator) -> dict:
    """Collect terms by the block on which they act.

    Builds a dictionary whose keys are subsystems (frozensets of site names)
    and the values are lists of operators acting exactly on that subsystem.

    Parameters
    ----------
    operator : Operator
        The operator to be decomposed.

    Returns
    -------
    dict[frozenset, list[Operator]]
        A dict whose keys are sets of sites, and whose values are the terms
        acting on those sites.

    """
    full_acts_over: frozenset
    terms_by_block: Dict[frozenset, List[Operator]] = {}
    scalar_term: complex = cast(complex, 0)
    system = operator.system

    if not isinstance(operator, SumOperator):
        return {operator.acts_over(): [operator]}

    full_acts_over = frozenset()
    for term in operator.terms:
        acts_over_key = term.acts_over()
        if not acts_over_key:
            scalar_term += cast(complex, term.prefactor)
        else:
            full_acts_over = full_acts_over.union(acts_over_key)
            terms_by_block.setdefault(acts_over_key, []).append(term)

    # Add a scalar term
    if scalar_term:
        terms_by_block[frozenset()] = [ScalarOperator(scalar_term, system)]
    return terms_by_block


def group_terms_by_blocks(operator: Operator, fn: Optional[Callable] = None):
    """
    Rewrite operator as a sum of terms acting on different blocks.

    Rewrite a sum of operators as a sum
    of a ScalarOperator, a OneBodyOperator
    and terms acting on different blocks.

    For many-body terms, apply fn for further
    simplifications.

    For example


    .. code-block:: python

        group_terms_by_blocks(operator, lambda op:op.to_qutip_operator())

    convert these many-body terms into Qutip operators,
    which for small blocks could provide a more efficient
    representation.

    Parameters
    ----------
    operator : Operator
        The operator to be reduced.
    fn : Callable or None, optional
        A function to implement specific simplifications. The default is None.

    Returns
    -------
    Operator
       A monomial operator or a sum operator, with terms acting
       in different blocks.

    """
    if (
        not isinstance(operator, SumOperator)
        or getattr(operator, "_simplified", False)
        or isinstance(operator, OneBodyOperator)
        or hasattr(operator, "expect")
    ):
        return operator

    changed = False
    system = operator.system
    acts_over = operator.acts_over()
    operator_flat = operator.flat()
    isherm = getattr(operator, "_isherm", None)
    if operator is not operator_flat:
        changed = True
    terms_dict = collect_nbody_terms(operator_flat)

    if any(isinstance(t, QutipOperator) for t in terms_dict.get(acts_over, [])):
        return operator.to_qutip_operator()

    new_terms = []
    one_body_terms = []
    scalar_terms = []

    def apply_simplification_fn(op_in: Operator, fn: Optional[Callable]):
        """Apply the simplification function.

        Parameters
        ----------
        op_in : Operator
            The operator to simplify.
        fn : Callable or None
            The simplification function, or ``None`` to skip.

        Returns
        -------
        Operator
            The simplified operator.
        """
        try:
            if isinstance(op_in, SumOperator):
                op_in = simplify_qutip_sums(op_in)
            if fn is None:
                return op_in
            return fn(op_in)
        except Exception as exc:
            logger.warning(exc)
            return op_in

    for block, terms in terms_dict.items():
        if block is None:
            new_terms.extend(terms)
        elif len(block) == 0:
            assert all(
                isinstance(term, ScalarOperator) for term in terms
            ), "Should be a scalar term..."
            scalar_terms.extend(terms)
        elif len(block) == 1:
            one_body_terms.extend(terms)
        else:
            if len(terms) == 1:
                new_term = terms[0]
                if isherm and not new_term.isherm:
                    changed = True
                    new_term = (
                        SumOperator(
                            tuple([new_term, new_term.dag()]),
                            system=system,
                            isherm=isherm,
                        )
                        * 0.5
                    )
            else:
                new_term = SumOperator(tuple(terms), system=system, isherm=isherm)

            new_term_simpl = apply_simplification_fn(new_term, fn)
            if new_term_simpl is not new_term:
                changed = True
                new_term = new_term_simpl

            new_terms.append(new_term)

    new_terms = [term for term in new_terms if term]
    if len(scalar_terms) > 1:
        scalar_terms = [sum(scalar_terms)]

    if not new_terms:
        if one_body_terms:
            result = OneBodyOperator(
                tuple(one_body_terms + scalar_terms), system, isherm=isherm
            ).simplify()
            return result
        return scalar_terms[0] if scalar_terms else ScalarOperator(0.0, system)

    if not changed:
        setattr(operator, "_simplified", True)
        if isherm:
            setattr(operator, "_isherm", True)
        return operator

    new_terms = new_terms + scalar_terms

    if one_body_terms:
        new_term = OneBodyOperator(tuple(one_body_terms), system, isherm=isherm)
        new_terms.append(new_term)
        changed = True

    if not new_terms:
        return ScalarOperator(0, system)
    if len(new_terms) == 1:
        return new_terms[0]

    return sum_operator_sequence(
        new_terms,
        system=system,
        isherm=isherm,
        isdiag=getattr(operator, "_isdiagonal", None),
        simplified=True,
    )


def simplify_qutip_sums(sum_operator: SumOperator) -> Operator:
    """Simplify terms that are qutip operators.

    Collects terms acting on the same block of sites and reduces them to a
    single :class:`~qalma.operators.qutip.QutipOperator`.

    Parameters
    ----------
    sum_operator : SumOperator
        The operator to be reduced.

    Returns
    -------
    Operator
        The simplified operator.

    """
    if not isinstance(sum_operator, SumOperator):
        return sum_operator

    changed: bool = False
    isherm = sum_operator._isherm
    system = sum_operator.system
    terms: List[Operator] = []
    qutip_terms: Dict[frozenset, List[QutipOperator]] = {}
    product_terms: Dict[frozenset, List[ProductOperator]] = {}

    pauli_strings: List[PauliStringOperator] = []
    for term in sum_operator.terms:
        if isinstance(term, PauliStringOperator):
            pauli_strings.append(term)
        elif isinstance(term, ProductOperator):
            product_terms.setdefault(term.acts_over(), []).append(term)
        elif isinstance(term, QutipOperator):
            qutip_terms.setdefault(term.acts_over(), []).append(term)
        else:
            terms.append(term)

    # Pauli strings are merged symbolically, keeping them as Pauli strings.
    if pauli_strings:
        merged = collect_pauli_terms(pauli_strings, system)
        if len(merged) != len(pauli_strings):
            changed = True
        merged_op = pauli_dict_to_operator(merged, system)
        if isinstance(merged_op, SumOperator):
            terms.extend(merged_op.terms)
        elif merged_op:
            terms.append(merged_op)

    # Process first the product operator terms
    for block, p_terms in product_terms.items():
        # If block is in qutip_terms, or there are more than a single
        # product term, and each product term acts on few sites, it is more
        # efficient to store them as a single qutip operator:
        if block in qutip_terms or (len(p_terms) > 1 and len(block) < 6):
            changed = True
            block_tuple = tuple(sorted(block))
            sum_qutip_op = sum(
                term.to_qutip_operator().to_qutip(block_tuple) for term in p_terms
            )
            if not is_empty_op(sum_qutip_op):
                qutip_terms.setdefault(block, []).append(
                    QutipOperator(
                        sum_qutip_op,
                        names={site: idx for idx, site in enumerate(block_tuple)},
                        system=system,
                    )
                )
            continue
        # Otherwise, just add as terms
        terms.extend(p_terms)

    # Now,
    for block, q_terms in qutip_terms.items():
        block_tuple = tuple(sorted(block))
        if len(q_terms) == 1:
            terms.append(q_terms[0])
            continue
        changed = True
        new_qterm = sum(q_term.to_qutip(block_tuple) for q_term in q_terms)
        # Reinforce hermiticity if the operator is hermitian, but the term isn't.
        if isherm and not new_qterm.isherm:
            new_qterm = (new_qterm + new_qterm.dag()) * 0.5

        if not is_empty_op(new_qterm):
            terms.append(
                QutipOperator(
                    new_qterm,
                    names={site: pos for pos, site in enumerate(block_tuple)},
                    system=system,
                )
            )
    strip_terms = tuple((term for term in terms if not term.is_zero))
    if len(strip_terms) != len(terms):
        changed = True

    if not changed:
        return sum_operator
    return sum_operator_sequence(terms, system=system, simplified=True, isherm=isherm)


def _commutator_term_pairs(op_1_terms, op_2_terms):
    """Yield all (term_1, term_2) pairs that can contribute to a commutator.

    Pairs whose support blocks are disjoint are skipped because their
    commutator is identically zero.  Pairs where both terms are the same
    object are also skipped.

    Parameters
    ----------
    op_1_terms : dict
        Output of :func:`collect_nbody_terms` for the first operator.
    op_2_terms : dict
        Output of :func:`collect_nbody_terms` for the second operator.

    Yields
    ------
    tuple[Operator, Operator]
        Pairs ``(t1, t2)`` for which the commutator ``t1 * t2 - t2 * t1``
        may be non-zero.
    """
    for block_1, terms_1 in op_1_terms.items():
        for block_2, terms_2 in op_2_terms.items():
            if (
                block_1 is not None
                and block_2 is not None
                and not block_1.intersection(block_2)
            ):
                continue
            for term_1 in terms_1:
                for term_2 in terms_2:
                    if term_1 is term_2:
                        continue
                    yield (term_1, term_2)
