"""Arithmetic operations with states.

Essentially, arithmetic operations with states involves just mixing of
operators, implemented though the class MixtureDensityOperator.
"""

import logging
import pickle
from typing import Dict, Iterable, Optional, Set, Tuple, Union, cast

import numpy as np

from qalma.model import SystemDescriptor
from qalma.operators.arithmetic import SumOperator
from qalma.operators.basic import (
    Operator,
)
from qalma.operators.product import (
    ScalarOperator,
)
from qalma.operators.states.basic import (
    DensityOperatorMixin,
    DensityOperatorProtocol,
)

__all__ = ["MixtureDensityOperator"]


class MixtureDensityOperator(DensityOperatorMixin, SumOperator):
    """A mixture of density operators."""

    terms: Tuple[Operator]

    def __init__(self, terms: tuple, system: Optional[SystemDescriptor] = None):
        """
        Initialize a MixtureDensityOperator.

        Parameters
        ----------
        terms : tuple[Operator, ...]
            Tuple of density operators to mix. Each term must have a
            ``prefactor`` attribute representing its weight in the mixture.
        system : SystemDescriptor or None, optional
            Descriptor of the full lattice system. Inferred from ``terms``
            if not provided.

        """
        super().__init__(terms, system, True)

    def __neg__(self):
        """Multiply the operator by -1."""
        logging.warning("Negate a DensityOperator leads to a regular operator.")
        new_terms = tuple(((-t) * (t.prefactor) for t in self.terms))
        return SumOperator(new_terms, self.system, isherm=True)

    def acts_over(self) -> frozenset:
        """
        Return the block over which the operator acts over.

        Return a set with the name of the sites where the operator
        nontrivially acts.
        """
        sites: Set[str] = set()
        for term in self.terms:
            acts_over = cast(Operator, term).acts_over()
            sites.update(acts_over)
        return frozenset(sites)

    def expect(
        self,
        obs_objs: Union[Operator, Iterable],
        _local_states: Optional[Dict[frozenset, DensityOperatorProtocol]] = None,
    ) -> Union[np.ndarray, dict, complex]:
        r"""Compute expectation values as a weighted sum over the mixture components.

        For each component :math:`\rho_k` with weight :math:`\lambda_k`,
        computes :math:`\langle O \rangle = \sum_k \lambda_k \langle O \rangle_{\rho_k}`.

        Parameters
        ----------
        obs_objs : Operator or Iterable[Operator] or dict
            Observable or collection of observables.
        _local_states : dict or None, optional
            Pre-computed local states (unused, kept for interface compatibility).

        Returns
        -------
        complex or np.ndarray or dict
            Expectation value(s) of the observable(s).

        """

        def compute_results(curr_obs, sub_averages, prefactors):
            """Combine per-component averages into the mixture expectation value."""
            if isinstance(curr_obs, dict):
                result = {}
                for key in curr_obs:
                    content = curr_obs[key]
                    result[key] = compute_results(
                        content,
                        tuple(contrib[key] for contrib in sub_averages),
                        prefactors,
                    )
                return result
            return sum(
                exp_val * p_refactor
                for exp_val, p_refactor in zip(sub_averages, prefactors)
            )

        averages = tuple(
            cast(DensityOperatorMixin, term).expect(obs_objs) for term in self.terms
        )
        prefactors = tuple(term.prefactor for term in self.terms)
        return compute_results(obs_objs, averages, prefactors)

    def partial_trace(self, sites: Union[frozenset, SystemDescriptor]):
        """Compute the partial trace over the complement of ``sites``.

        Applies partial trace to each component and returns a new
        :class:`MixtureDensityOperator` on the reduced subsystem.

        Parameters
        ----------
        sites : frozenset[str] or SystemDescriptor
            Sites to *keep*. All other sites are traced out.

        Returns
        -------
        MixtureDensityOperator
            The reduced mixture on the subsystem defined by ``sites``.

        """
        new_terms = tuple(cast(Operator, t).partial_trace(sites) for t in self.terms)
        subsystem = new_terms[0].system
        return MixtureDensityOperator(new_terms, subsystem)

    def simplify(self):
        """
        Return a simplified representation of the operator.

        Return ``self`` — mixture density operators are already in
        simplified form.

        Returns
        -------
        MixtureDensityOperator
            ``self``.

        """
        return self

    def __setstate__(self, state):
        """Set the state of the object."""
        state = pickle.loads(state)
        self.__dict__.update(state)
        self._set_system_(self.system)

    def to_qutip(self, block: Optional[Tuple[str, ...]] = None):
        """Produce a qutip compatible object."""
        if len(self.terms) == 0:
            return ScalarOperator(0, self.system).to_qutip()

        acts_over = self.acts_over()
        if block is None or acts_over is None:
            block = tuple(sorted(self.system.sites))
        else:
            block = block + tuple(
                (site for site in sorted(acts_over) if site not in block)
            )

        # TODO: find a more efficient way to avoid element-wise
        # multiplications
        terms = (
            (
                cast(Operator, term).to_qutip(block),
                term.prefactor,
            )
            for term in self.terms
        )
        return sum(term[0] * term[1] for term in terms)
