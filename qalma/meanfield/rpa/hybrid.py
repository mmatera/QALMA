"""
RPA Hybrid approach.

Tools to build an approximation of the local state on a subsystem A, by
replacing the environment by some efective bosonic gaussian modes and classical
couplings, when the occupation of the bosonic modes exceed a threshold.

"""

import logging
from typing import List, Tuple

import numpy as np
import scipy.linalg
import scipy.sparse as sparse
from qutip import Qobj

from qalma.operators import Operator
from qalma.operators.basic import LocalOperator

from .environment import _environment_bogoliubov
from .rpa_structure import RPAStructure

LOGGER = logging.getLogger(__name__)


def _bath_operators(n_modes: int, n_fock: int) -> List[sparse.csr_matrix]:
    """Annihilation operators of ``n_modes`` bosons truncated at ``n_fock`` quanta."""
    dim = n_fock + 1
    boson = sparse.diags(np.sqrt(np.arange(1, dim)), 1, format="csr")
    operators = []
    for j in range(n_modes):
        factors = [sparse.identity(dim, format="csr")] * n_modes
        factors[j] = boson
        full = factors[0]
        for factor in factors[1:]:
            full = sparse.kron(full, factor, format="csr")
        operators.append(full)
    return operators


def rpa_hybrid_state(
    structure: RPAStructure,
    sites: frozenset,
    k_mf: Operator,
    n_blocks: int = 1,
    n_fock: int = 6,
    n_cut: float = 3.0,
    n_gauss: int = 6,
    max_dim: int = 6000,
    max_nodes: int = 20000,
    charge_fluctuations: bool = True,
) -> Qobj:
    r"""Reduced state of the subsystem, exact in ``A`` and RPA in the environment.

    The environment is bosonized (RPA) and the subsystem is kept exact. The
    reduced state is the partial trace over the bosons of the Gibbs state of

    .. math::

        K = K_{A,MF} + \sum_j \omega_j b_j^\dagger b_j
            + \sum_\mu \delta e_\mu \otimes (B_\mu^{q} + x_\mu),

    where the quantum modes :math:`B^q_\mu` are the normal modes of the
    environment with thermal occupation not larger than ``n_cut``, compressed
    to a few effective modes with :func:`effective_bath_modes` and truncated
    in Fock space, and the modes with larger occupations are classical static
    Gaussian fields :math:`x_\mu` averaged with a Gauss-Hermite quadrature.
    Unlike the generator corrections of
    :func:`rpa_generator_correction`, it is non perturbative in the coupling
    between the subsystem and the environment. The back-reaction of the
    subsystem on the environment is therefore included.

    Parameters
    ----------
    structure : RPAStructure
        Subsystem-independent data of the RPA.
    sites : frozenset
        Sites of the subsystem ``A``.
    k_mf : Operator
        Mean-field reduced generator :math:`K_{A,MF}` of the subsystem.
    n_blocks : int, optional
        Lanczos blocks of the effective bath. The default is 1.
    n_fock : int, optional
        Maximum number of quanta of each effective boson. The default is 6.
    n_cut : float, optional
        Modes with thermal occupation larger than ``n_cut`` are treated as
        classical. The default is 3.
    n_gauss : int, optional
        Gauss-Hermite nodes per classical direction. The default is 6.
    max_dim : int, optional
        Maximum dimension of the Hilbert space of the hybrid Hamiltonian.
    max_nodes : int, optional
        Maximum number of quadrature nodes of the static fields.
    charge_fluctuations : bool, optional
        If the environment has zero modes (a broken continuous symmetry that
        is not pinned by the subsystem), they are removed from the bosonic
        problem: the orientation is fixed, and the symmetry must be restored
        with the ``symmetry_projections`` of the state. The conserved charge
        conjugate to each zero mode is a static Gaussian variable that
        couples to the subsystem and drives the bath; its thermal
        fluctuations are included if this option is ``True`` (default), and
        neglected otherwise. They are not supported together with classical
        thermal modes.

    Returns
    -------
    qutip.Qobj
        The normalized reduced density matrix, in the tensor-product order of
        ``tuple(sorted(sites))``.

    Raises
    ------
    ValueError
        If the RPA is unstable, or if the hybrid problem is too large.
    """
    block = tuple(sorted(sites))
    system = structure.system
    k0_qobj: Qobj = k_mf.to_qutip(block)
    k0 = k0_qobj.full()
    k0 = 0.5 * (k0 + k0.conj().T)
    dim_a = k0.shape[0]

    def finish(rho: np.ndarray) -> Qobj:
        rho = 0.5 * (rho + rho.conj().T)
        return Qobj(rho / np.real(np.trace(rho)), dims=k0_qobj.dims)

    environment = _environment_bogoliubov(structure, sites)
    if environment is None:
        eps, vecs = np.linalg.eigh(k0)
        return finish((vecs * np.exp(-(eps - eps.min()))) @ vecs.conj().T)
    positive = environment.lam > 0
    omega = environment.lam[positive]
    coupling = environment.c_mat[:, positive]
    sites_a, local_basis = environment.sites_a, environment.local_basis
    d_ops = [
        LocalOperator(site, e, system).to_qutip(block).full()
        for site in sites_a
        for e in local_basis[site]
    ]
    n_a = len(d_ops)

    # Classical (static) and quantum modes
    occupation = 1.0 / np.expm1(omega)
    classical = occupation > n_cut
    zero = environment.zero if charge_fluctuations else None
    if zero is not None and np.any(classical):
        raise ValueError(
            "Zero modes together with classical thermal modes are not "
            "supported: use a lower temperature or charge_fluctuations=False."
        )
    covariance = np.zeros((n_a, n_a))
    if np.any(classical):
        c_cl = coupling[:, classical]
        covariance = np.real(
            (c_cl * (2.0 * occupation[classical] + 1.0)[None, :]) @ c_cl.conj().T
        )
        covariance = 0.5 * (covariance + covariance.T)
    cov_vals, cov_vecs = np.linalg.eigh(covariance)
    keep = cov_vals > 1e-12 * max(1.0, cov_vals.max())
    fields = [
        np.sqrt(cov_vals[a]) * sum(cov_vecs[mu, a] * d_ops[mu] for mu in range(n_a))
        for a in np.nonzero(keep)[0]
    ]

    # Conserved charges conjugate to the zero modes: static Gaussian
    # variables, coupled to the subsystem and driving the bath.
    n_charges = 0
    if zero is not None:
        n_charges = zero.kappa_eff.shape[0]
        k_vals, k_vecs = np.linalg.eigh(zero.kappa_eff)
        if k_vals[0] <= 1e-12 * max(1.0, k_vals[-1]):
            raise ValueError(
                "The charges conjugate to the zero modes are not confined "
                "(their inverse covariance is not positive definite)."
            )
        charge_dirs = k_vecs / np.sqrt(k_vals)[None, :]  # p = charge_dirs @ xi
        start = np.vstack([coupling, zero.c_drive[:, positive]])
    else:
        start = coupling
    omega_eff, coupling_eff = effective_bath_modes(
        omega[~classical], start[:, ~classical], n_blocks
    )
    n_eff = len(omega_eff)
    bath_dim = (n_fock + 1) ** n_eff
    if dim_a * bath_dim > max_dim:
        raise ValueError(
            f"The hybrid problem has dimension {dim_a * bath_dim} > {max_dim}: "
            "reduce n_fock or n_blocks, or increase max_dim."
        )
    n_static = len(fields) + n_charges
    nodes_1d, weights_1d = np.polynomial.hermite_e.hermegauss(n_gauss)
    weights_1d = weights_1d / np.sqrt(2.0 * np.pi)
    n_nodes = n_gauss**n_static
    if n_nodes > max_nodes:
        raise ValueError(
            f"{n_static} static directions need {n_nodes} > {max_nodes} "
            "quadrature nodes: reduce n_gauss or increase max_nodes."
        )

    # Hamiltonian of the bath and coupling to the subsystem
    bosons = _bath_operators(n_eff, n_fock)
    identity_bath = sparse.identity(bath_dim, format="csr")
    identity_a = sparse.identity(dim_a, format="csr")

    def bath_operator(row: np.ndarray) -> sparse.csr_matrix:
        """Linear operator ``sum_j (row_j b_j + h.c.)`` of the effective bath."""
        result = sparse.csr_matrix((bath_dim, bath_dim), dtype=complex)
        for coef, boson in zip(row, bosons):
            result = result + coef * boson + np.conj(coef) * boson.T
        return result

    bath_h = sparse.csr_matrix((bath_dim, bath_dim), dtype=complex)
    for w_j, boson in zip(omega_eff, bosons):
        bath_h = bath_h + w_j * (boson.T @ boson)
    h_fixed = sparse.kron(identity_a, bath_h, format="csr")
    for mu, d_op in enumerate(d_ops):
        h_fixed = h_fixed + sparse.kron(
            sparse.csr_matrix(d_op), bath_operator(coupling_eff[mu]), format="csr"
        )
    # operators of the charges: coupling to the subsystem and drive of the bath
    charge_sub, charge_bath = [], []
    if zero is not None:
        for direction in range(n_charges):
            weights_dir = charge_dirs[:, direction]
            e_op = sum(
                weights_dir[a] * zero.e_coupling[mu, a] * d_ops[mu]
                for a in range(n_charges)
                for mu in range(n_a)
            )
            charge_sub.append(sparse.csr_matrix(e_op))
            drive_row = sum(
                weights_dir[a] * coupling_eff[n_a + a] for a in range(n_charges)
            )
            charge_bath.append(
                sparse.kron(identity_a, bath_operator(drive_row), format="csr")
            )

    log_weights, energies, states = [], [], []
    for index in np.ndindex(*([n_gauss] * n_static)):
        xi = np.array([nodes_1d[i] for i in index])
        log_weight = float(np.sum(np.log([weights_1d[i] for i in index])))
        k_a = k0 + sum((x * f for x, f in zip(xi[: len(fields)], fields)), 0 * k0)
        h_total = h_fixed + sparse.kron(
            sparse.csr_matrix(k_a), identity_bath, format="csr"
        )
        if n_charges and zero is not None:
            xi_p = xi[len(fields) :]
            # importance sampling: Gaussian weight of the charges over the
            # normalized measure of the nodes
            charges = charge_dirs @ xi_p
            log_weight += float(
                -0.5 * charges @ zero.kappa @ charges
                + 0.5 * np.sum(xi_p**2)
                + 0.5 * n_charges * np.log(2.0 * np.pi)
            )
            for x_p, e_op, drive in zip(xi_p, charge_sub, charge_bath):
                h_total = h_total + x_p * (sparse.kron(e_op, identity_bath) + drive)
        h_dense = h_total.toarray()
        eps, vecs = scipy.linalg.eigh(0.5 * (h_dense + h_dense.conj().T))
        log_weights.append(log_weight)
        energies.append(eps[0])
        boltz = np.exp(-(eps - eps[0]))
        psi = vecs.reshape(dim_a, bath_dim, -1)
        states.append(np.einsum("anr,r,bnr->ab", psi, boltz, psi.conj()))
    log_total = np.array(log_weights) - np.array(energies)
    scale = np.exp(log_total - log_total.max())
    total = sum(w * rho for w, rho in zip(scale, states))
    return finish(total)


def effective_bath_modes(
    omega: np.ndarray,
    coupling: np.ndarray,
    n_blocks: int = 1,
    tol: float = 1e-10,
) -> Tuple[np.ndarray, np.ndarray]:
    r"""Compress a bosonic bath to a few effective modes (block Lanczos).

    The bath is :math:`\sum_m \omega_m b_m^\dagger b_m` and it couples to the
    subsystem through :math:`B_\mu = \sum_m (C_{\mu m} b_m + h.c.)`. Only the
    Krylov subspace generated by the coupling vectors is relevant: the
    effective modes are the eigenmodes of the Hamiltonian projected on that
    subspace. This is a Gauss quadrature of the spectral density
    :math:`J_{\mu\nu}(\omega)=\sum_m C_{\mu m}C^*_{\nu m}\delta(\omega-\omega_m)`
    that reproduces its first :math:`2 n_{blocks}` moments, and is exact if the
    subspace is the whole space of modes.

    Parameters
    ----------
    omega : numpy.ndarray
        Frequencies of the bath modes.
    coupling : numpy.ndarray
        Complex matrix :math:`C_{\mu m}` of shape ``(nA, n_modes)``.
    n_blocks : int, optional
        Number of Lanczos blocks. Each block adds as many effective modes as
        the rank of the coupling. The default is 1.
    tol : float, optional
        Relative tolerance to discard linearly dependent directions.

    Returns
    -------
    omega_eff : numpy.ndarray
        Frequencies of the effective modes.
    coupling_eff : numpy.ndarray
        Couplings of the effective modes, shape ``(nA, n_effective)``.
    """
    n_modes = len(omega)
    if n_modes == 0:
        return omega, coupling

    def orthonormal_columns(vectors: np.ndarray) -> np.ndarray:
        u_mat, sing, _ = np.linalg.svd(vectors, full_matrices=False)
        if sing.size == 0 or sing[0] <= 1e-300:
            return u_mat[:, :0]
        return u_mat[:, sing > tol * sing[0]]

    basis = orthonormal_columns(np.conj(coupling).T)
    if basis.shape[1] == 0:
        return omega[:0], coupling[:, :0]
    last = basis
    for _ in range(n_blocks - 1):
        if basis.shape[1] >= n_modes:
            break
        vectors = omega[:, None] * last
        for _ in range(2):  # full reorthogonalization
            vectors = vectors - basis @ (basis.conj().T @ vectors)
        last = orthonormal_columns(vectors)
        if last.shape[1] == 0:
            break
        basis = np.hstack([basis, last])
    projected = basis.conj().T @ (omega[:, None] * basis)
    omega_eff, eigenvectors = np.linalg.eigh(0.5 * (projected + projected.conj().T))
    return omega_eff, coupling @ (basis @ eigenvectors)
