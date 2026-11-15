"""
Normal mode approximation of the environment.

"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from qutip import Qobj

from .rpa_structure import RPAStructure, _split_sites
from .utils import _quadrature_map
from .zeromodes import _bogoliubov_reduced, _ZeroModes

LOGGER = logging.getLogger(__name__)


@dataclass
class _EnvironmentBogoliubov:
    """Normal modes of the environment (the subsystem removed).

    ``B_mu = sum_m c_mat[mu, m] Psi_m`` for the normal-mode operators
    ``Psi_m`` of signed frequency ``lam[m]``.
    """

    sites_a: List[str]
    local_basis: Dict[str, List[Qobj]]
    c_mat: np.ndarray
    lam: np.ndarray
    n_zero: int = 0
    zero: Optional["_ZeroModes"] = None


def _environment_bogoliubov(
    structure: RPAStructure, sites: frozenset
) -> Optional[_EnvironmentBogoliubov]:
    """Bosonize the environment of ``sites`` and diagonalize it.

    Returns ``None`` if the subsystem is not coupled to the environment.
    """
    local_basis = structure.local_basis
    m_full = structure.m_full
    idx_a, idx_e, sites_a, sites_e = _split_sites(structure, sites)
    if not idx_a or not idx_e:
        return None
    freqs_list = []
    row = 0
    blocks = []
    for site in sites_e:
        size = len(local_basis[site])
        freqs, coup = structure.boson_modes(site)
        freqs_list.append(freqs)
        blocks.append((row, size, coup))
        row += size
    freqs_all = np.concatenate(freqs_list)
    n_modes = len(freqs_all)
    if n_modes == 0:
        return None
    g_mat = np.zeros((row, n_modes), dtype=complex)
    col = 0
    for r_0, size, coup in blocks:
        g_mat[r_0 : r_0 + size, col : col + coup.shape[1]] = coup
        col += coup.shape[1]

    # Quadratic bosonic Hamiltonian of the environment and its modes.
    m_ae = m_full[np.ix_(idx_a, idx_e)]
    m_ee = m_full[np.ix_(idx_e, idx_e)]
    a_mat = np.diag(freqs_all) + g_mat.conj().T @ m_ee @ g_mat
    b_mat = g_mat.conj().T @ m_ee @ g_mat.conj()
    a_mat = 0.5 * (a_mat + a_mat.conj().T)
    b_mat = 0.5 * (b_mat + b_mat.T)
    h_mat = np.block([[a_mat, b_mat], [b_mat.conj(), a_mat.conj()]])
    static_modes = np.concatenate(
        [freqs <= structure.static_gap * (1.0 + 1e-6) for freqs in freqs_list]
    )
    lam, t_mat, n_zero, zero_data = _bogoliubov_reduced(
        h_mat, structure.zero_rtol, static_modes
    )
    g_b = m_ae @ g_mat
    s_phi = np.hstack([g_b, g_b.conj()])
    c_mat = s_phi @ t_mat
    zero = None
    if zero_data is not None:
        w_map = _quadrature_map(n_modes)
        s_hat = s_phi @ w_map  # real rows of the operators in the quadratures
        zero = _ZeroModes(
            e_coupling=s_hat.real @ zero_data.s0p,
            c_drive=(zero_data.drive_rows @ w_map.conj().T) @ t_mat,
            kappa_eff=zero_data.kappa_eff,
            kappa=zero_data.kappa,
        )
    return _EnvironmentBogoliubov(sites_a, local_basis, c_mat, lam, n_zero, zero)


def rpa_zero_modes(structure: RPAStructure, sites: Optional[frozenset] = None) -> int:
    """Number of zero modes removed from the bosonic RPA problem.

    Parameters
    ----------
    structure : RPAStructure
        Subsystem-independent data of the RPA.
    sites : frozenset, optional
        If given, count the zero modes of the environment of ``sites`` (the
        problem used by the reductions of a subsystem). Otherwise, count
        those of the whole system (the problem of the full bosonization).

    Returns
    -------
    int
        The number of zero modes.
    """
    if sites is None:
        return structure.boson_system().n_zero_modes
    environment = _environment_bogoliubov(structure, frozenset(sites))
    return 0 if environment is None else environment.n_zero


def environment_modes(
    structure: RPAStructure, sites: frozenset
) -> Optional[Tuple[np.ndarray, np.ndarray, List[str], Dict[str, List[Qobj]]]]:
    r"""Normal modes of the bosonized environment, coupled to ``sites``.

    With the subsystem removed, the bosonized environment is a set of
    independent normal modes of frequencies :math:`\omega_m>0` (in the
    units of the generator), and the boundary operators of the subsystem are
    :math:`B_\mu = \sum_m (C_{\mu m} b_m + C^*_{\mu m} b_m^\dagger)`.

    Parameters
    ----------
    structure : RPAStructure
        Subsystem-independent data of the RPA.
    sites : frozenset
        Sites of the subsystem ``A``.

    Returns
    -------
    omega : numpy.ndarray
        Frequencies :math:`\omega_m` of the normal modes.
    coupling : numpy.ndarray
        Complex matrix :math:`C_{\mu m}` of shape ``(nA, n_modes)``.
    sites_a : list of str
        Sites of ``A`` in the order used to index ``mu``.
    local_basis : dict
        Centred local basis of each site.

    ``None`` is returned if ``A`` is not coupled to the environment.

    Raises
    ------
    ValueError
        If the RPA is unstable.
    """
    environment = _environment_bogoliubov(structure, sites)
    if environment is None:
        return None
    positive = environment.lam > 0
    return (
        environment.lam[positive],
        environment.c_mat[:, positive],
        environment.sites_a,
        environment.local_basis,
    )
