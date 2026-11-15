r"""Bosonic Gaussian system.

This module contains the classes and functions that store information
of a Bosonic Gaussian system.

`BosonicGaussianSystem`: dataclass that stores the description of the bosonic system
and its correlations.

`_covariance_matrix`: computes the covariance matrix associated to the gaussian bosonic system.

"""

from dataclasses import dataclass
from typing import List

import numpy as np

# ### Full bosonization: Gaussian state of the whole system ###


@dataclass
class BosonicGaussianSystem:
    r"""Bosonized RPA of the whole system, and its Gaussian (thermal) state.

    The local algebra of every site is bosonized (see
    :func:`local_boson_modes`) and the quadratic bosonic Hamiltonian of the
    whole system is diagonalized with a Bogoliubov transformation. The
    thermal state of the quasi-particles is a Gaussian state, determined by
    the matrix of contractions
    :math:`\mathcal{G} = \langle\Phi\Phi^\dagger\rangle`, with
    :math:`\Phi = (b_1,\dots,b_n,b_1^\dagger,\dots,b_n^\dagger)^T`.

    Only the bosons with a non-vanishing frequency are included: the static
    (zero-frequency) modes are classical fluctuations, and do not contribute
    to the entanglement.

    Attributes
    ----------
    sites : list of str
        Sites of the system.
    mode_site : list of str
        Site to which each boson belongs.
    freqs : numpy.ndarray
        Local frequencies :math:`\Delta_k` of the bosons.
    couplings : numpy.ndarray
        Couplings :math:`g_{\mu k}` of the local operators to the bosons.
    lam : numpy.ndarray
        Signed frequencies of the normal modes.
    t_mat : numpy.ndarray
        Bogoliubov transformation.
    contractions : numpy.ndarray
        Matrix :math:`\langle\Phi\Phi^\dagger\rangle` of the thermal state.
    """

    sites: List[str]
    mode_site: List[str]
    freqs: np.ndarray
    couplings: np.ndarray
    lam: np.ndarray
    t_mat: np.ndarray
    contractions: np.ndarray
    n_zero_modes: int = 0

    @property
    def n_modes(self) -> int:
        """Number of bosons."""
        return len(self.freqs)

    def modes_of(self, sites: frozenset) -> List[int]:
        """Indices of the bosons of the given sites."""
        return [k for k, site in enumerate(self.mode_site) if site in sites]

    def covariance_matrix(self, modes: List[int]) -> np.ndarray:
        r"""Covariance matrix of the quadratures of a subset of modes.

        The quadratures are ordered as :math:`(x_1,\dots,x_n,p_1,\dots,p_n)` with
        :math:`x=(b+b^\dagger)/\sqrt2`, :math:`p=(b-b^\dagger)/(i\sqrt2)`, so that
        the vacuum has covariance :math:`1/2`.
        """
        n_all, n_sub = self.n_modes, len(modes)
        idx = np.array(list(modes) + [n_all + k for k in modes], dtype=int)
        g_sub = self.contractions[np.ix_(idx, idx)]
        eye = np.eye(n_sub)
        u_mat = np.block([[eye, eye], [-1j * eye, 1j * eye]]) / np.sqrt(2.0)
        return np.real(u_mat @ g_sub @ u_mat.conj().T)
