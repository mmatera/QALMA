"""
Dynamic RPA.

"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from qutip import Qobj

from qalma.operators import Operator, QutipOperator
from qalma.operators.basic import LocalOperator

from .environment import _environment_bogoliubov
from .rpa_structure import RPAStructure, _split_sites
from .utils import _DEGENERACY_TOL

LOGGER = logging.getLogger(__name__)


def _dynamic_correction(
    k_mf: Operator,
    structure: RPAStructure,
    sites: frozenset,
    n_tau: int,
) -> Optional[Operator]:
    """Dynamical RPA correction to the mean-field generator."""
    system = structure.system
    sigma_tau, nodes, weights, local_basis, sites_a = rpa_dynamic_kernel(
        structure, sites, n_tau
    )
    if sigma_tau.size == 0 or not np.any(np.abs(sigma_tau) > 1e-14):
        return None
    block = tuple(sorted(sites))
    k0_qobj: Qobj = k_mf.to_qutip(block)
    d_ops = [
        LocalOperator(site, e, system).to_qutip(block).full()
        for site in sites_a
        for e in local_basis[site]
    ]
    d_k = dynamic_generator_correction(k0_qobj.full(), d_ops, sigma_tau, nodes, weights)
    return QutipOperator(
        Qobj(d_k, dims=k0_qobj.dims),
        system,
        {site: pos for pos, site in enumerate(block)},
    )


def rpa_dynamic_kernel(
    structure: RPAStructure,
    sites: frozenset,
    n_tau: int = 32,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, List], List[str]]:
    r"""Compute the imaginary-time kernel :math:`\Sigma_{\mu\nu}(\tau)`.

    The environment is bosonized site by site (see
    :func:`local_boson_modes`) and the resulting quadratic Hamiltonian
    :math:`H_b` is diagonalized with a Bogoliubov transformation. The
    kernel is then a sum of exponentials over the normal modes
    (see the module documentation), which is equivalent to the RPA dressing
    summed over all the Matsubara frequencies.

    Parameters
    ----------
    structure : RPAStructure
        Subsystem-independent data of the RPA. The frequency of the static
        bosons is ``structure.static_gap``.
    sites : frozenset
        Sites of the subsystem ``A``.
    n_tau : int, optional
        Number of Gauss-Legendre nodes in :math:`(0, 1)`. The default is 32.

    Returns
    -------
    sigma_tau : numpy.ndarray
        Complex array ``(n_tau, nA, nA)`` with :math:`\Sigma_{\mu\nu}(u_j)`.
    nodes, weights : numpy.ndarray
        Gauss-Legendre nodes and weights on :math:`(0, 1)`.
    local_basis : dict
        Centred local basis of each site.
    sites_a : list of str
        Sites of ``A`` in the order used to index ``sigma_tau``.

    Raises
    ------
    ValueError
        If the RPA is unstable.
    """
    nodes, weights = np.polynomial.legendre.leggauss(n_tau)
    nodes, weights = 0.5 * (nodes + 1.0), 0.5 * weights
    environment = _environment_bogoliubov(structure, sites)
    if environment is None:
        _, _, sites_a, _ = _split_sites(structure, sites)
        n_a = sum(len(structure.local_basis[site]) for site in sites_a)
        empty = np.zeros((n_tau, n_a, n_a), dtype=complex)
        return empty, nodes, weights, structure.local_basis, sites_a

    # B_mu = sum_m c_{mu m} Psi_m, Sigma(tau) = sum_m c c^* f_m exp(-lam_m tau)
    c_mat, lam = environment.c_mat, environment.lam
    omega = np.abs(lam)
    expo = np.where(
        lam[None, :] > 0,
        -omega[None, :] * nodes[:, None],
        -omega[None, :] * (1.0 - nodes[:, None]),
    )
    occupation = np.exp(expo) / (-np.expm1(-omega))[None, :]
    sigma_tau = np.einsum("im,tm,jm->tij", c_mat, occupation, c_mat.conj())
    return sigma_tau, nodes, weights, environment.local_basis, environment.sites_a


# ### Dynamical correction of the generator ###


def dynamic_generator_correction(
    k0: np.ndarray,
    d_ops: List[np.ndarray],
    sigma_tau: np.ndarray,
    nodes: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    r"""First-order generator correction from an imaginary-time kernel.

    Parameters
    ----------
    k0 : numpy.ndarray
        Dense Hermitian generator :math:`K_{A,MF}` of the subsystem.
    d_ops : list of numpy.ndarray
        Dense centred operators :math:`\delta e_\mu` acting on ``A``.
    sigma_tau : numpy.ndarray
        Kernel ``(n_tau, n_mu, n_mu)`` evaluated at ``nodes``.
    nodes, weights : numpy.ndarray
        Quadrature on :math:`(0, 1)`.

    Returns
    -------
    numpy.ndarray
        Hermitian matrix :math:`\delta K` such that
        :math:`K_A' = K_{A,MF} + \delta K` (in the original basis).
    """
    eps, vecs = np.linalg.eigh(k0)
    eps = eps - eps.min()
    q_ops = np.array([vecs.conj().T @ op @ vecs for op in d_ops])
    delta = eps[:, None] - eps[None, :]  # eps_m - eps_n
    small = np.abs(delta) < _DEGENERACY_TOL
    safe = np.where(small, 1.0, delta)

    def km_weight(a: float) -> np.ndarray:
        """Matrix ``(e^{-a eps_n} - e^{-a eps_m})/(eps_m - eps_n)``."""
        diff = (np.exp(-a * eps)[None, :] - np.exp(-a * eps)[:, None]) / safe
        return np.where(small, a * np.exp(-a * eps)[:, None], diff)

    y_mat = np.zeros_like(k0, dtype=complex)
    for u, w_q, sigma_u in zip(nodes, weights, sigma_tau):
        c_ops = np.einsum("mn,mik->nik", sigma_u, q_ops)
        z_mat = np.einsum("nik,k,nkj->ij", c_ops, np.exp(-u * eps), q_ops)
        y_mat += w_q * z_mat * km_weight(1.0 - u)
    d_k = -y_mat / km_weight(1.0)
    d_k = 0.5 * (d_k + d_k.conj().T)
    return vecs @ d_k @ vecs.conj().T
