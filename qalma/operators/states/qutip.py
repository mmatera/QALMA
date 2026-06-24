"""
Qutip representation for density operators.

Be careful: just use this class for states of small systems.
"""

import logging
from numbers import Real
from typing import Optional, Tuple, Union, cast

import numpy as np
from qutip import Qobj as _Qobj, tensor as _tensor_qutip  # type: ignore[import-untyped]

from qalma.model import SystemDescriptor
from qalma.operators.product import ScalarOperator
from qalma.operators.qutip import QutipOperator
from qalma.operators.states.basic import (
    DensityOperatorMixin,
    DensityOperatorProtocol,
)

__all__ = ["QutipDensityOperator"]


class QutipDensityOperator(DensityOperatorMixin, QutipOperator):
    """Qutip representation of a density operator."""

    def __init__(
        self,
        qoperator: _Qobj,
        system: Optional[SystemDescriptor] = None,
        names=None,
        prefactor=1,
        normalized=False,
    ):
        r"""
        Initialize a QutipDensityOperator.

        Parameters
        ----------
        qoperator : Qobj
            The QuTiP density matrix. Normalized to unit trace on construction
            unless ``normalized=True``.
        system : SystemDescriptor or None, optional
            Descriptor of the full lattice system.
        names : dict[str, int] or None, optional
            Mapping from site name to tensor-product index in ``qoperator``.
        prefactor : float, optional
            Scalar weight :math:`\lambda`. Default is ``1``.
        normalized : bool, optional
            If ``True``, skips normalization on construction. Default is
            ``False``.
        """
        self._normalized = normalized
        super().__init__(qoperator, system, names, prefactor)
        self.normalize()

    def __neg__(self):
        """Multiply by -1."""
        logging.warning("Negate a DensityOperator leads to a regular operator.")
        self.normalize()
        return QutipOperator(self.operator, self.system, self.site_names, -1)

    def join_states(self, other: DensityOperatorProtocol | complex):
        """Combine states of two systems.

        Combine the states of two disjoint systems to produce the state of
        the union of both systems.
        """
        if isinstance(other, Real):
            if other < 0:
                raise ValueError
            return QutipDensityOperator(
                self.operator,
                system=self.system,
                names=self.site_names,
                prefactor=self.prefactor * other,
            )
        if isinstance(other, complex):
            raise ValueError("operand is not a positive number")
        rho: DensityOperatorProtocol = other
        if not hasattr(rho, "expect"):
            raise ValueError
        if not rho.prefactor:
            return QutipDensityOperator(
                self.operator, system=self.system, names=self.site_names, prefactor=0
            )
        system_a = self.system
        system_b = rho.system
        if set(system_a.sites).intersection(system_b.sites):
            raise ValueError("Systems have overlap")

        system = system_a.union(system_b)
        acts_over_b = rho.acts_over()
        if len(acts_over_b) == 0:
            return QutipDensityOperator(
                self.operator,
                system=self.system,
                names=self.site_names,
                prefactor=cast(Real, self.prefactor) * cast(Real, rho.prefactor),
            )
        acts_over_a = self.acts_over()
        if len(acts_over_a) == 0:
            return rho * self.prefactor

        block_a = tuple(acts_over_a)
        block_b = tuple(acts_over_b)
        names = {site: pos for pos, site in enumerate(block_a + block_b)}
        qutip_block = _tensor_qutip(self.to_qutip(block_a), rho.to_qutip(block_b))
        prefactor = self.prefactor * rho.prefactor
        return QutipDensityOperator(
            qutip_block, names=names, system=system, prefactor=prefactor
        )

    def logm(self):
        r"""
        Return the matrix logarithm :math:`\ln \rho`.

        Normalizes first, then computes the logarithm via eigendecomposition.
        Eigenvalues below ``1e-30`` are clamped to avoid divergence.

        Returns
        -------
        QutipOperator
            The matrix logarithm as a :class:`~qalma.operators.qutip.QutipOperator`.
        """
        self.normalize()
        operator = self.operator
        evals, evecs = operator.eigenstates()
        evals[abs(evals) < 1.0e-30] = 1.0e-30
        log_op = sum(
            np.log(e_val) * e_vec * e_vec.dag() for e_val, e_vec in zip(evals, evecs)
        )
        return QutipOperator(log_op, self.system, self.site_names)

    def normalize(self):
        """Normalize the operator."""
        if self._normalized:
            return self
        qoperator = self.operator
        tr_op = qoperator.tr()
        if tr_op != 1:
            qoperator = qoperator / tr_op
        self.operator = qoperator
        self._normalized = True
        return self

    def partial_trace(self, sites: Union[frozenset, SystemDescriptor]):
        """Compute the partial trace over the complement of ``sites``.

        Normalizes first, delegates to the parent
        :class:`~qalma.operators.qutip.QutipOperator` partial trace, and
        wraps the result back as a :class:`QutipDensityOperator`.

        Parameters
        ----------
        sites : frozenset[str] or SystemDescriptor
            Sites to *keep*. All other sites are traced out.

        Returns
        -------
        QutipDensityOperator or ScalarOperator
            The reduced density operator on the subsystem defined by
            ``sites``, or a :class:`~qalma.operators.product.ScalarOperator`
            if all sites are traced out.

        """
        self.normalize()
        self_pt = super().partial_trace(sites)
        if isinstance(self_pt, ScalarOperator):
            return self_pt

        return QutipDensityOperator(
            self_pt.operator,
            names=self_pt.site_names,
            system=self_pt.system,
            prefactor=self.prefactor,
        )

    def to_qutip(self, block: Optional[Tuple[str, ...]] = None):
        """Return the normalized QuTiP density matrix over ``block``.

        Normalizes first, then delegates to
        :meth:`~qalma.operators.qutip.QutipOperator.to_qutip` with the
        prefactor temporarily set to ``1`` so that the state sums to unit
        trace (the prefactor is a mixture weight, not part of the matrix).

        Parameters
        ----------
        block : tuple[str, ...] or None, optional
            Ordered list of site names. Defaults to all sites in the order
            stored in ``site_names``.

        Returns
        -------
        qutip.Qobj
            The normalized density matrix restricted to ``block``.

        """
        self.normalize()
        # set the prefactor temporarily to 1, because it should
        # not be taken into account in the conversion of a state.
        prefactor = self.prefactor
        self.prefactor = 1
        qutip_op = super().to_qutip(block)
        # setting back the value
        self.prefactor = prefactor
        return qutip_op
