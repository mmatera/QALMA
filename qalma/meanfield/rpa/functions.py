"""
Entanglement & negativity.
"""

from typing import Optional

import numpy as np

from .rpa_structure import RPAStructure
from .utils import _symplectic_spectrum


def rpa_entanglement_entropy(structure: RPAStructure, sites: frozenset) -> float:
    r"""Bosonic RPA entropy of a subsystem, in bits.

    It is the von Neumann entropy of the reduced Gaussian state of the
    bosons of ``sites``, :math:`S=\sum_\alpha[(1+f_\alpha)\log_2(1+f_\alpha)
    - f_\alpha\log_2 f_\alpha]` with :math:`f_\alpha=\nu_\alpha-1/2`.
    For a pure state (zero temperature) it is the entanglement entropy of
    the bipartition (``sites``, complement); otherwise it also includes the
    thermal entropy.

    Parameters
    ----------
    structure : RPAStructure
        Subsystem-independent data of the RPA.
    sites : frozenset
        Sites of the subsystem.

    Returns
    -------
    float
        The entropy in bits.
    """
    bosons = structure.boson_system()
    if bosons.n_zero_modes:
        raise ValueError(
            f"The bosonic RPA of the whole system has {bosons.n_zero_modes} zero "
            "mode(s) (a broken continuous symmetry): the covariance matrix of a "
            "subsystem is not defined without restoring the symmetry. Use the "
            "hybrid reduction instead."
        )
    modes = bosons.modes_of(sites)
    nu = _symplectic_spectrum(bosons.covariance_matrix(modes))
    f = np.clip(nu - 0.5, 0.0, None)
    f = f[f > 1e-15]
    return float(np.sum((1 + f) * np.log2(1 + f) - f * np.log2(f)))


def rpa_negativity(
    structure: RPAStructure, sites_b: frozenset, sites_c: Optional[frozenset] = None
) -> float:
    r"""Bosonic RPA negativity between two subsystems.

    The negativity is computed from the symplectic spectrum of the partial
    transpose of the reduced Gaussian state of ``sites_b`` and ``sites_c``
    (:math:`\mathcal{N} = \frac12[\prod_{\tilde\nu<1/2}(2\tilde\nu)^{-1}-1]`).

    Parameters
    ----------
    structure : RPAStructure
        Subsystem-independent data of the RPA.
    sites_b, sites_c : frozenset
        Sites of the two subsystems. If ``sites_c`` is ``None``, the
        complement of ``sites_b`` is used.

    Returns
    -------
    float
        The negativity.
    """
    bosons = structure.boson_system()
    if bosons.n_zero_modes:
        raise ValueError(
            f"The bosonic RPA of the whole system has {bosons.n_zero_modes} zero "
            "mode(s) (a broken continuous symmetry): the covariance matrix of a "
            "subsystem is not defined without restoring the symmetry. Use the "
            "hybrid reduction instead."
        )
    if sites_c is None:
        sites_c = frozenset(bosons.sites) - sites_b
    modes_b = bosons.modes_of(sites_b)
    modes_c = bosons.modes_of(sites_c)
    modes = modes_b + modes_c
    sigma = bosons.covariance_matrix(modes)
    n_sub = len(modes)
    flip = np.ones(2 * n_sub)
    flip[n_sub + len(modes_b) :] = -1.0  # time reversal of the momenta of C
    sigma_pt = flip[:, None] * sigma * flip[None, :]
    nu = _symplectic_spectrum(sigma_pt)
    nu = nu[nu < 0.5 - 1e-12]
    return float(0.5 * (np.prod(1.0 / (2.0 * nu)) - 1.0))
