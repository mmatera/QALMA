"""Density operator classes."""

import logging
import warnings
from typing import Any, Dict, Optional, Tuple, Union, cast

import numpy as np
from numpy.typing import NDArray
from qutip import (  # type: ignore[import-untyped]
    Qobj as _Qobj,
    qeye as _qutip_qeye,
    tensor as _qutip_tensor,
)
from scipy.linalg import logm as _scp_logm
from scipy.linalg._matfuncs_inv_ssq import LogmExactlySingularWarning

from qalma.model import SystemDescriptor
from qalma.operators.arithmetic import OneBodyOperator, SumOperator
from qalma.operators.basic import (
    LocalOperator,
)
from qalma.operators.product import (
    ProductOperator,
    ScalarOperator,
)
from qalma.operators.states.basic import DensityOperatorMixin, DensityOperatorProtocol
from qalma.qutip_tools.tools import (
    _to_array,
)

__all__ = ["ProductDensityOperator"]


class ProductDensityOperator(DensityOperatorMixin, ProductOperator):
    """An uncorrelated density operator."""

    prefactor: complex  # must be float

    def __init__(
        self,
        local_states: Dict[str, Any],
        weight: float = 1.0,
        system: Optional[SystemDescriptor] = None,
        normalized: bool = False,
        _qutip_factors: Optional[Dict[str, _Qobj]] = None,
    ):
        r"""
        Initialize a ProductDensityOperator.

        Parameters
        ----------
        local_states : dict[str, np.ndarray or Qobj]
            Mapping from site name to local density matrix. If
            ``normalized=False``, each matrix is divided by its trace.
            Sites present in ``system`` but absent from ``local_states``
            are filled with the maximally mixed state :math:`\mathbb{I}/d`.
        weight : float, optional
            Non-negative scalar prefactor :math:`\lambda`. Default is
            ``1.0``.
        system : SystemDescriptor or None, optional
            Descriptor of the full lattice system. If ``None``, the system
            is inferred from the dimensions of ``local_states``.
        normalized : bool, optional
            If ``True``, assumes each local matrix already has unit trace.
            Default is ``False``.
        _qutip_factors : dict[str, Qobj] or None, optional
            Pre-computed QuTiP representations of the local factors.
            Used internally to avoid redundant conversions.

        """
        assert weight >= 0

        # Build the local partition functions and normalize
        # if required
        if weight == 0:
            local_states = {}
            local_zs = {}
        else:
            local_states = {key: _to_array(val) for key, val in local_states.items()}
            local_zs = {site: state.trace() for site, state in local_states.items()}
            if normalized:
                if _qutip_factors is not None:
                    self.__dict__["site_factors_qutip"] = _qutip_factors
            else:
                assert all(z > 0 for z in local_zs.values())
                local_states = {
                    site: sigma / local_zs[site] for site, sigma in local_states.items()
                }

        # Complete the scalar factors using the system
        if system is None:
            dimensions = {
                site: operator.data.shape[0] for site, operator in local_states.items()
            }
            # TODO: build a system
        else:
            dimensions = system.dimensions
            local_identities: dict = {}
            for site, dimension in dimensions.items():
                if site not in local_states:
                    local_id = local_identities.get(dimension, None)
                    local_zs[site] = dimension
                    if local_id is None:
                        local_id = _qutip_qeye(dimension) / dimension
                        local_identities[dimension] = local_id
                    local_states[site] = local_id

        super().__init__(local_states, prefactor=weight, system=system)
        self.local_fs = {site: -np.log(z) for site, z in local_zs.items()}

    def __mul__(self, a):
        """Multiply by a number of operator by the right."""
        if isinstance(a, (float, np.float64)):
            if a >= 0:
                return ProductDensityOperator(
                    self.site_factors, self.prefactor * a, self.system, False
                )
            logging.warning(
                (
                    "Multiplication of a non positive number by a "
                    "density operator returns a regular operator."
                )
            )
            return ProductOperator(self.site_factors, 1, self.system) * a
        return ProductOperator(self.site_factors, 1, self.system) * a

    def __neg__(self):
        """Multiply by -1."""
        logging.warning("Negate a DensityOperator leads to a regular operator.")
        return ProductOperator(self.site_factors, -1, self.system)

    def __rmul__(self, a):
        """Multiply by a number of operator by the left."""
        if isinstance(a, (float, np.float64)):
            if a >= 0:
                return ProductDensityOperator(
                    self.site_factors, self.prefactor * a, self.system, False
                )
            logging.warning(
                (
                    "Multiplication of a non positive number by "
                    "a density operator returns a regular operator."
                )
            )
            return ProductOperator(self.site_factors, 1, self.system) * a
        return a * ProductOperator(self.site_factors, 1, self.system)

    def expect(
        self: Any,
        obs_objs: Any,
        _local_states: Optional[Dict[frozenset, "DensityOperatorProtocol"]] = None,
    ) -> Any:
        """
        Compute expectation values.

        Compute the expectation value of an operator or a sequence of
        operators.

        Hot paths use dense numpy arithmetic and bypass Qobj overhead:

        * LocalOperator  -> single _trace2 call (no Qobj allocation)
        * ProductOperator, homogeneous system -> batched einsum over a
          stacked (N, d, d) tensor
        * ProductOperator, heterogeneous system -> per-site _trace2 loop
        * Everything else -> delegate to the parent DensityOperatorMixin
        """
        if isinstance(obs_objs, LocalOperator):
            site = obs_objs.site
            op_dense = obs_objs.operator
            return self._trace2(self.site_factors[site], op_dense)

        if isinstance(obs_objs, ProductOperator):
            obs_prod = cast(ProductOperator, obs_objs)
            result: complex = complex(obs_prod.prefactor)
            if not result:
                return complex(0)

            rhos = self.site_factors  # dict[site -> (d,d)]

            # --- Fast path: homogeneous system, batched einsum -----------
            # try:
            #    raise ValueError
            #    obs_sites, obs_tensor = obs_prod._dense_tensor  # (N, d, d)
            #    rho_tensor = np.stack([rhos[s] for s in obs_sites])  # (N, d, d)
            #    # traces[i] = Tr(rho_i @ obs_i), no intermediate matrices
            #    traces = np.einsum("nij,nji->n", rho_tensor, obs_tensor)
            #    result *= complex(traces.prod())
            # except (ValueError, KeyError):
            # Heterogeneous dims or a site not in rhos: fall back to
            # a per-site loop that is still numpy-only (no Qobj).
            #    for site, obs_op in obs_prod.site_factors.items():
            #        if not result:
            #            break
            #        result *= self._trace2(rhos[site], obs_op)

            for site, obs_op in obs_prod.site_factors.items():
                if not result:
                    break
                result *= self._trace2(rhos[site], obs_op)

            return result

        if isinstance(obs_objs, SumOperator):
            obs_sum = cast(SumOperator, obs_objs)
            return cast(
                NDArray,
                sum(
                    cast(NDArray, self.expect(term, _local_states=_local_states))
                    for term in obs_sum.terms
                    if term.prefactor
                ),
            )

        if isinstance(obs_objs, (tuple, list)):
            return np.array(
                [self.expect(elem, _local_states=_local_states) for elem in obs_objs]
            )

        if isinstance(obs_objs, dict):
            return {
                key: self.expect(val, _local_states=_local_states)
                for key, val in obs_objs.items()
            }

        # Fallback: we know we'll need Qobj representations down the call
        # chain (via to_qutip). Warm the cache now on self so that
        # partial_trace children can inherit it via the existing
        # _qutip_factors mechanism in __init__.
        _ = self.site_factors_qutip
        return super().expect(obs_objs, _local_states=_local_states)

    def logm(self):
        r"""Return the matrix logarithm :math:`\log\rho`.

        Exploits the product structure: since
        :math:`\rho = \lambda \bigotimes_i \rho_i`, we have

        .. math::

            \log\rho = \log\lambda + \sum_i \log\rho_i
                       - \sum_{j \notin \text{support}} \log d_j\, \mathbb{I}

        where the last sum accounts for identity factors from sites not in
        ``site_factors``.

        Returns
        -------
        Operator
            The matrix logarithm as a :class:`~qalma.operators.arithmetic.OneBodyOperator`
            plus a scalar offset.

        """
        system = self.system
        sites_op = self.site_factors
        warnings.filterwarnings("ignore", category=LogmExactlySingularWarning)
        terms = tuple(
            LocalOperator(site, _scp_logm(loc_op), system)
            for site, loc_op in sites_op.items()
        )
        warnings.resetwarnings()
        if system:
            norm = -sum(
                np.log(dim)
                for site, dim in system.dimensions.items()
                if site not in sites_op
            )
            return OneBodyOperator(terms, system, False) + ScalarOperator(norm, system)
        return OneBodyOperator(terms, system, False)

    def partial_trace(self, sites: Union[frozenset, SystemDescriptor]):
        """Compute the partial trace over the complement of ``sites``.

        For a product state the partial trace simply discards the local
        factors of the traced-out sites, returning a new
        :class:`ProductDensityOperator` on the reduced subsystem.

        Parameters
        ----------
        sites : frozenset[str] or SystemDescriptor
            Sites to *keep*. All other sites are traced out.

        Returns
        -------
        ProductDensityOperator
            The reduced product state on the subsystem defined by ``sites``.

        """
        sites_op = self.site_factors
        if isinstance(sites, SystemDescriptor):
            subsystem = sites
            sites = frozenset(sites.sites.keys())
        else:
            subsystem = self.system.subsystem(sites)

        local_states: Dict[str, np.ndarray] = {
            str(site): sites_op[site] for site in sites
        }

        qutip_factors: Optional[Dict[str, _Qobj]] = self.__dict__.get(
            "site_factors_qutip", None
        )
        if qutip_factors is not None:
            qutip_factors = {
                site: qutip_factors[site] for site in sites if site in qutip_factors
            }

        return ProductDensityOperator(
            local_states,
            np.real(self.prefactor),
            subsystem,
            normalized=True,
            _qutip_factors=qutip_factors,
        )

    def to_qutip(self, block: Optional[Tuple[str, ...]] = None):
        r"""Return the QuTiP tensor-product representation of the state.

        Sites in ``block`` that are not in ``site_factors`` contribute an
        identity factor :math:`\mathbb{I}/d`. If ``block`` is ``None``,
        all system sites are used in lexicographical order.

        Parameters
        ----------
        block : tuple[str, ...] or None, optional
            Ordered list of site names defining the tensor-product structure
            of the returned :class:`qutip.Qobj`. Default is all sites in
            lexicographical order.

        Returns
        -------
        qutip.Qobj or float
            The density matrix :math:`\lambda \bigotimes_i \rho_i` over
            ``block``. Returns a scalar if the weight is zero or the system
            has no sites.

        """
        prefactor = self.prefactor
        if prefactor == 0 or len(self.system.dimensions) == 0:
            return np.exp(-sum(np.log(dim) for dim in self.system.dimensions.values()))

        sites_op = self.site_factors_qutip
        dimensions = self.system.dimensions
        if block is None:
            block = tuple(sorted(self.system.sites))
        else:
            block = block + tuple(
                (site for site in sorted(sites_op) if site not in block)
            )

        return _qutip_tensor(
            [
                (
                    sites_op[site]
                    if site in sites_op
                    else _qutip_qeye(dimensions[site]) / dimensions[site]
                )
                for site in block
            ]
        )
