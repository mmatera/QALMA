"""
Utils for RPA-based approximations.

Functions used to build Bogoliubov's decompositions, find quadratures
and determine the RPA-efective local modes.

"""

import logging
from typing import List, Optional, Tuple

import numpy as np

_DEGENERACY_TOL = 1e-12
_STATIC_GAP = 1e-6
# The mean-field state is only self-consistent up to the tolerance of its
# solver, so the zero modes of the RPA have eigenvalues of the order 1e-9-1e-7
# (relative to the norm), with random sign. The first non-zero eigenvalues are
# orders of magnitude larger.
_ZERO_MODE_RTOL = 1e-5

LOGGER = logging.getLogger(__name__)


def _bogoliubov(h_mat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    r"""Diagonalize a bosonic quadratic Hamiltonian.

    Given :math:`H = \frac12\Phi^\dagger \mathcal{H}\Phi` with
    :math:`\Phi = (a, a^\dagger)^T` and :math:`\mathcal{H}` positive
    definite, find :math:`T` such that :math:`\Phi = T\tilde\Psi`, with
    :math:`[\tilde\Psi, \tilde\Psi^\dagger] = \tilde\eta` and
    :math:`H = \frac12 \tilde\Psi^\dagger |\Lambda| \tilde\Psi`
    (Colpa's algorithm).

    Parameters
    ----------
    h_mat : numpy.ndarray
        Hermitian positive definite matrix :math:`\mathcal{H}`.

    Returns
    -------
    lam : numpy.ndarray
        Signed mode frequencies :math:`\lambda_m`; the sign is the
        commutator sign :math:`\tilde\eta_m` of the mode.
    t_mat : numpy.ndarray
        Transformation :math:`T`.

    Raises
    ------
    ValueError
        If ``h_mat`` is not positive definite (unstable RPA).
    """
    n_half = h_mat.shape[0] // 2
    try:
        chol = np.linalg.cholesky(h_mat)
    except np.linalg.LinAlgError as err:
        raise ValueError(
            "Unstable RPA: the bosonic Hamiltonian of the environment is not "
            "positive definite (the environment is at or beyond its "
            "critical point)."
        ) from err
    k_mat = chol.conj().T
    eta = np.diag(np.concatenate([np.ones(n_half), -np.ones(n_half)]))
    lam, vecs = np.linalg.eigh(k_mat @ eta @ k_mat.conj().T)
    t_mat = np.linalg.solve(k_mat, vecs) * np.sqrt(np.abs(lam))[None, :]
    return lam, t_mat


def _quadrature_map(n_modes: int) -> np.ndarray:
    """Matrix ``W`` such that ``(b, b^dag) = W (x, p)``, with unit commutator."""
    eye = np.eye(n_modes)
    return np.block([[eye, 1j * eye], [eye, -1j * eye]]) / np.sqrt(2.0)


# ### Local (single-site) quantities ###


def _spectral_data(
    sigma_loc: np.ndarray,
    basis: List[np.ndarray],
    spectrum: Optional[Tuple[np.ndarray, np.ndarray]] = None,
):
    """Return populations, ``-log`` populations and the basis in the eigenbasis.

    If ``spectrum = (populations, eigenvectors)`` is given, it is used
    instead of diagonalizing ``sigma_loc``. This allows to keep the accuracy
    of very small populations, which are lost in a density matrix.
    """
    if spectrum is None:
        p, vecs = np.linalg.eigh(sigma_loc)
    else:
        p, vecs = spectrum
    p = np.clip(p, 1e-300, None)
    ops = np.array([vecs.conj().T @ np.asarray(e) @ vecs for e in basis])
    return p, -np.log(p), ops


def _symplectic_spectrum(sigma: np.ndarray) -> np.ndarray:
    """Symplectic eigenvalues :math:`\\nu_\\alpha` of a covariance matrix."""
    n_sub = sigma.shape[0] // 2
    if n_sub == 0:
        return np.zeros(0)
    eye, zero = np.eye(n_sub), np.zeros((n_sub, n_sub))
    omega = np.block([[zero, eye], [-eye, zero]])
    values = np.sort(np.abs(np.linalg.eigvals(1j * omega @ sigma)))
    return values[::2]


def kubo_mori_susceptibility(
    sigma_loc: np.ndarray,
    basis: List[np.ndarray],
    spectrum: Optional[Tuple[np.ndarray, np.ndarray]] = None,
):
    r"""Connected Kubo-Mori susceptibility of a local state.

    .. math::

        \chi^0_{\mu\nu} = \sum_{mn} \frac{p_m - p_n}{\ln p_m - \ln p_n}
        (e_\mu)_{mn}(e_\nu)_{nm} - \langle e_\mu\rangle\langle e_\nu\rangle

    where the quotient equals :math:`p_m` if :math:`p_m = p_n`.

    Parameters
    ----------
    sigma_loc : numpy.ndarray
        Local density matrix.
    basis : list of numpy.ndarray
        Hermitian local operators :math:`e_\mu`.
    spectrum : tuple of numpy.ndarray, optional
        Populations and eigenvectors of ``sigma_loc``, if known accurately.

    Returns
    -------
    numpy.ndarray
        Real symmetric matrix :math:`\chi^0_{\mu\nu}`.
    """
    p, eps, ops = _spectral_data(sigma_loc, basis, spectrum)
    delta = eps[:, None] - eps[None, :]
    small = np.abs(delta) < _DEGENERACY_TOL
    w = np.where(
        small,
        p[:, None] * np.ones_like(delta),
        (p[:, None] - p[None, :]) / np.where(small, 1.0, -delta),
    )
    avg = np.array([np.sum(p * np.diag(o)).real for o in ops])
    chi = np.einsum("mn,amn,bnm->ab", w, ops, ops).real
    return chi - np.outer(avg, avg)


def local_boson_modes(
    sigma_loc: np.ndarray,
    basis: List[np.ndarray],
    static_gap: float = _STATIC_GAP,
    spectrum: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    r"""Bosonize the fluctuation operators of a single site.

    In the eigenbasis of ``sigma_loc`` (populations :math:`p_i\ge p_j` for
    :math:`i<j`) the off-diagonal operators are mapped to
    :math:`|i\rangle\langle j|\to\sqrt{p_i-p_j}\,a_{ij}`, and
    :math:`\delta e_\mu` is truncated to its part linear in the bosons:

    .. math::

        \delta e_\mu \simeq \sum_k g_{\mu k} a_k + g^*_{\mu k} a_k^\dagger .

    Pairs with :math:`\ln(p_i/p_j) <` ``static_gap`` and the diagonal
    (intraband) covariance of the populations are time independent. They are
    represented by bosons of frequency ``static_gap`` and couplings scaled
    so that the static susceptibility
    :math:`\sum_k (g_{\mu k} g^*_{\nu k} + c.c.)/\Delta_k` is exact.

    Parameters
    ----------
    sigma_loc : numpy.ndarray
        Local density matrix.
    basis : list of numpy.ndarray
        Hermitian local operators :math:`e_\mu` (centred or not).
    static_gap : float, optional
        Frequency assigned to the static modes. The default is 1e-6.
    spectrum : tuple of numpy.ndarray, optional
        Populations and eigenvectors of ``sigma_loc``, if known accurately
        (for instance, from the local generator). Needed when some
        populations are below the numerical precision of a density matrix.

    Returns
    -------
    freqs : numpy.ndarray
        Mode frequencies :math:`\Delta_k \ge` ``static_gap``, shape ``(K,)``.
    couplings : numpy.ndarray
        Complex couplings :math:`g_{\mu k}`, shape ``(len(basis), K)``.
    """
    if spectrum is None:
        p, vecs = np.linalg.eigh(sigma_loc)
    else:
        p, vecs = spectrum
    order = np.argsort(-p)
    p = np.clip(p[order], 1e-300, None)
    vecs = vecs[:, order]
    ops = np.array([vecs.conj().T @ np.asarray(e) @ vecs for e in basis])
    freqs: List[float] = []
    couplings: List[np.ndarray] = []
    for i in range(len(p)):
        for j in range(i + 1, len(p)):
            gap = np.log(p[i] / p[j])
            if gap >= static_gap:
                weight = p[i] - p[j]
            else:
                gap = static_gap
                weight = 0.5 * (p[i] + p[j]) * static_gap
            freqs.append(gap)
            couplings.append(np.sqrt(weight) * ops[:, i, j])
    diag = np.real(np.array([np.diag(o) for o in ops]))
    mean = diag @ p
    cov = (diag * p[None, :]) @ diag.T - np.outer(mean, mean)
    evals, evecs = np.linalg.eigh(cov)
    for val, vec in zip(evals, evecs.T):
        if val > 1e-14:
            freqs.append(static_gap)
            couplings.append(vec * np.sqrt(val * static_gap / 2.0))
    if not freqs:
        return np.zeros(0), np.zeros((len(basis), 0), dtype=complex)
    return np.array(freqs), np.array(couplings).T.astype(complex)
