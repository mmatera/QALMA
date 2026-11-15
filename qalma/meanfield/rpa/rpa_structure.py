r"""RPA structure

RPAStructure contains the information to generate an RPA-like approximation
of any subsystem.

"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from qutip import Qobj

from qalma.meanfield.gibbs_partial_trace import (
    project_boundary_term,
)
from qalma.model import SystemDescriptor
from qalma.operators import Operator
from qalma.operators.arithmetic import SumOperator, iterable_to_operator
from qalma.operators.quadratic.build import (
    build_local_basis,
    build_quadratic_form_matrix,
    classify_terms,
    zero_expectation_value_basis,
)

from .boson import BosonicGaussianSystem
from .utils import (
    _STATIC_GAP,
    _ZERO_MODE_RTOL,
    kubo_mori_susceptibility,
    local_boson_modes,
)
from .zeromodes import _bogoliubov_reduced

LOGGER = logging.getLogger(__name__)


def _check_stability(chi0_static: np.ndarray, m_ee: np.ndarray) -> None:
    """Raise if ``1 + chi0 M_EE`` is not positive definite."""
    stab = np.eye(chi0_static.shape[0]) + chi0_static @ m_ee
    if np.min(np.linalg.eigvals(stab).real) <= 1e-10:
        raise ValueError(
            "Unstable RPA: 1 + chi0 M_EE is not positive definite "
            "(the environment is at or beyond its critical point)."
        )


def _local_spectra(
    generator: Operator, sigma_mf
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Populations and eigenvectors of the local mean-field generators."""
    system = generator.system
    terms = generator.terms if isinstance(generator, SumOperator) else [generator]
    contributions: Dict[str, List[Operator]] = {}
    for term in terms:
        acts_over = term.acts_over()
        for site in acts_over:
            if len(acts_over) == 1:
                projected = term
            else:
                projected = project_boundary_term(term, sigma_mf, frozenset([site]))
            if projected:
                contributions.setdefault(site, []).append(projected)
    spectra = {}
    for site, local_terms in contributions.items():
        k_loc = iterable_to_operator(local_terms, system, isherm=True)
        k_mat = k_loc.to_qutip((site,)).full()
        eps, vecs = np.linalg.eigh(0.5 * (k_mat + k_mat.conj().T))
        weights = np.exp(-(eps - eps.min()))
        spectra[site] = (weights / weights.sum(), vecs)
    return spectra


# ### Coupling matrices ###
@dataclass
class RPAStructure:
    r"""Data of the RPA of a Gibbs state that do not depend on the subsystem.

    The centred coupling matrix, the local bases and the single-site
    quantities (static susceptibilities and bosonic modes) are the same for
    any subsystem ``A``. Storing them allows reusing them when many partial
    traces of the same state are required. The single-site quantities are
    computed on demand and cached.

    Attributes
    ----------
    generator : Operator
        Generator ``K`` of the Gibbs state (flat and simplified).
    sigma_mf : ProductDensityOperator
        Mean-field reference state.
    static_gap : float
        Frequency assigned to the static bosons.
    m_full : numpy.ndarray
        Centred two-body coupling matrix :math:`M` of the whole system.
    local_basis : dict
        Centred Hilbert-Schmidt orthonormal basis of local operators of each
        site.
    positions : dict
        Index of the first element of each site in ``m_full``.
    zero_rtol : float
        Relative tolerance to identify the zero modes of the bosonic RPA
        problems (see :func:`_bogoliubov_reduced`).
    """

    generator: Operator
    sigma_mf: Any
    static_gap: float
    m_full: np.ndarray
    local_basis: Dict[str, List[Qobj]]
    positions: Dict[str, int]
    zero_rtol: float = _ZERO_MODE_RTOL
    _local_data: Dict[str, Tuple[np.ndarray, List[np.ndarray]]] = field(
        default_factory=dict, repr=False
    )
    _chi0: Dict[str, np.ndarray] = field(default_factory=dict, repr=False)
    _modes: Dict[str, Tuple[np.ndarray, np.ndarray]] = field(
        default_factory=dict, repr=False
    )
    _boson_system: Optional[BosonicGaussianSystem] = field(default=None, repr=False)
    _spectra: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]] = field(
        default=None, repr=False
    )

    @property
    def system(self) -> SystemDescriptor:
        """System descriptor of the generator."""
        return self.generator.system

    def local_data(self, site: str) -> Tuple[np.ndarray, List[np.ndarray]]:
        """Return the dense mean-field state and local basis of ``site``."""
        if site not in self._local_data:
            sigma_loc = self.sigma_mf.site_factors_qutip[site].full()
            basis = [e.full() for e in self.local_basis[site]]
            self._local_data[site] = (sigma_loc, basis)
        return self._local_data[site]

    def local_spectrum(self, site: str) -> Tuple[np.ndarray, np.ndarray]:
        """Return the populations and eigenvectors of the local mean-field state.

        They are obtained from the local mean-field generator of the site
        (the terms of ``K`` over the site plus the boundary terms projected
        with the mean-field state), so very small populations keep their
        accuracy even if they cannot be resolved in a density matrix.
        """
        if self._spectra is None:
            self._spectra = _local_spectra(self.generator, self.sigma_mf)
        return self._spectra[site]

    def static_chi0(self, site: str) -> np.ndarray:
        """Return the static susceptibility :math:`\\chi^0` of ``site``."""
        if site not in self._chi0:
            self._chi0[site] = kubo_mori_susceptibility(
                *self.local_data(site), spectrum=self.local_spectrum(site)
            )
        return self._chi0[site]

    def boson_modes(self, site: str) -> Tuple[np.ndarray, np.ndarray]:
        """Return the bosonic modes of ``site`` (see :func:`local_boson_modes`)."""
        if site not in self._modes:
            self._modes[site] = local_boson_modes(
                *self.local_data(site),
                self.static_gap,
                spectrum=self.local_spectrum(site),
            )
        return self._modes[site]

    def boson_system(self) -> BosonicGaussianSystem:
        """Return the bosonic system of the whole state (computed on demand)."""
        if self._boson_system is None:
            self._boson_system = build_boson_system(self)
        return self._boson_system


def build_boson_system(structure: RPAStructure) -> BosonicGaussianSystem:
    """Build the bosonized RPA of the whole system.

    Parameters
    ----------
    structure : RPAStructure
        Subsystem-independent data of the RPA.

    Returns
    -------
    BosonicGaussianSystem
        The bosonic system, with the contractions of its thermal state.

    Raises
    ------
    ValueError
        If the RPA is unstable.
    """
    sites = list(structure.positions)
    freqs_list, mode_site, blocks = [], [], []
    row = 0
    for site in sites:
        size = len(structure.local_basis[site])
        freqs, coup = structure.boson_modes(site)
        dynamic = freqs > structure.static_gap * (1.0 + 1e-6)
        freqs_list.append(freqs[dynamic])
        mode_site.extend([site] * int(np.sum(dynamic)))
        blocks.append((row, size, coup[:, dynamic]))
        row += size
    freqs_all = np.concatenate(freqs_list) if freqs_list else np.zeros(0)
    n_modes = len(freqs_all)
    g_mat = np.zeros((row, n_modes), dtype=complex)
    col = 0
    for r_0, size, coup in blocks:
        g_mat[r_0 : r_0 + size, col : col + coup.shape[1]] = coup
        col += coup.shape[1]
    m_full = structure.m_full
    a_mat = np.diag(freqs_all) + g_mat.conj().T @ m_full @ g_mat
    b_mat = g_mat.conj().T @ m_full @ g_mat.conj()
    a_mat = 0.5 * (a_mat + a_mat.conj().T)
    b_mat = 0.5 * (b_mat + b_mat.T)
    h_mat = np.block([[a_mat, b_mat], [b_mat.conj(), a_mat.conj()]])
    lam, t_mat, n_zero, _ = _bogoliubov_reduced(h_mat, structure.zero_rtol)
    omega = np.abs(lam)
    expo = np.where(lam > 0, 0.0, -omega)  # equal-time occupations
    occupation = np.exp(expo) / (-np.expm1(-omega))
    contractions = (t_mat * occupation[None, :]) @ t_mat.conj().T
    return BosonicGaussianSystem(
        sites=sites,
        mode_site=mode_site,
        freqs=freqs_all,
        couplings=g_mat,
        lam=lam,
        t_mat=t_mat,
        contractions=contractions,
        n_zero_modes=n_zero,
    )


def build_rpa_structure(
    generator: Operator,
    sigma_mf,
    static_gap: float = _STATIC_GAP,
    zero_rtol: float = _ZERO_MODE_RTOL,
) -> RPAStructure:
    """Build the subsystem-independent data of the RPA of a Gibbs state.

    Parameters
    ----------
    generator : Operator
        Generator ``K`` of the Gibbs state.
    sigma_mf : ProductDensityOperator
        Mean-field reference state.
    static_gap : float, optional
        Frequency assigned to the static bosons.
    zero_rtol : float, optional
        Relative tolerance to identify the zero modes of the bosonic
        problems. The default is 1e-5.

    Returns
    -------
    RPAStructure
        The structure.
    """
    generator = generator.flat().simplify()
    blocks, _, _ = classify_terms(generator, sigma_mf)
    local_basis = build_local_basis(blocks)
    m_full, positions = build_quadratic_form_matrix(blocks, local_basis)
    local_basis = zero_expectation_value_basis(local_basis, sigma_mf)
    return RPAStructure(
        generator=generator,
        sigma_mf=sigma_mf,
        static_gap=static_gap,
        m_full=m_full,
        local_basis=local_basis,
        positions=positions,
        zero_rtol=zero_rtol,
    )


def rpa_coupling_kernel(
    structure: RPAStructure, sites: frozenset
) -> Tuple[np.ndarray, Dict[str, List], List[str]]:
    r"""Compute the static kernel :math:`\Sigma` restricted to ``sites``.

    Parameters
    ----------
    structure : RPAStructure
        Subsystem-independent data of the RPA.
    sites : frozenset
        Sites of the subsystem ``A``.

    Returns
    -------
    sigma : numpy.ndarray
        Real symmetric matrix indexed by the local basis elements of the
        sites in ``A``.
    local_basis : dict
        Centred local basis of each site.
    sites_a : list of str
        Sites of ``A`` in the order used to index ``sigma``.
    """
    local_basis = structure.local_basis
    idx_a, idx_e, sites_a, sites_e = _split_sites(structure, sites)
    if not idx_a or not idx_e:
        return np.zeros((len(idx_a), len(idx_a))), local_basis, sites_a
    chi0 = np.zeros((len(idx_e), len(idx_e)))
    pos = 0
    for site in sites_e:
        size = len(local_basis[site])
        chi0[pos : pos + size, pos : pos + size] = structure.static_chi0(site)
        pos += size
    m_ae = structure.m_full[np.ix_(idx_a, idx_e)]
    m_ee = structure.m_full[np.ix_(idx_e, idx_e)]
    _check_stability(chi0, m_ee)
    chi_e = np.linalg.solve(np.eye(len(idx_e)) + chi0 @ m_ee, chi0)
    chi_e = 0.5 * (chi_e + chi_e.T)
    sigma = m_ae @ chi_e @ m_ae.T
    return 0.5 * (sigma + sigma.T), local_basis, sites_a


def _split_sites(
    structure: RPAStructure, sites: frozenset
) -> Tuple[List[int], List[int], List[str], List[str]]:
    """Split the indices of the coupling matrix into subsystem and environment."""
    idx_a: List[int] = []
    idx_e: List[int] = []
    sites_a: List[str] = []
    sites_e: List[str] = []
    for site, start in structure.positions.items():
        indices = range(start, start + len(structure.local_basis[site]))
        if site in sites:
            idx_a.extend(indices)
            sites_a.append(site)
        else:
            idx_e.extend(indices)
            sites_e.append(site)
    return idx_a, idx_e, sites_a, sites_e
