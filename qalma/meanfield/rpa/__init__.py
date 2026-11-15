r"""RPA reduction of a Gibbs state.

Given ``rho = exp(-K)/Z`` and a subsystem ``A``, the generator is split as

.. math::

    K = K_A + K_E + W_{AE}

and the reduced generator is approximated by

.. math::

    K_A' = K_{A,MF} + K_{A,RPA}.

Here :math:`K_{A,MF}` is the mean-field projection of the boundary terms
(see :mod:`qalma.meanfield.gibbs_partial_trace`), and :math:`K_{A,RPA}`
accounts for the Gaussian fluctuations of the environment around its
mean-field state. Let :math:`e_\mu` be a Hilbert-Schmidt orthonormal basis
of local operators and
:math:`\delta e_\mu = e_\mu - \langle e_\mu\rangle_{MF}` the centred basis.
The centred two-body coupling matrix :math:`M` is defined by
:math:`K_2 = \frac12\sum_{\mu\nu} M_{\mu\nu}\,\delta e_\mu\delta e_\nu`.

**Static approximation.**

.. math::

    K_{A,RPA} = -\frac12 \sum_{\mu\nu} \Sigma_{\mu\nu}\,
                \delta e_\mu \delta e_\nu,
    \qquad
    \Sigma = M_{AE}\,\chi_E\,M_{EA},
    \qquad
    \chi_E = (1 + \chi^0 M_{EE})^{-1}\chi^0 ,

where :math:`\chi^0` is the connected Kubo-Mori susceptibility of the
single-site mean-field states.

**Dynamical correction (RPA bosonization).** The static form replaces the
imaginary-time evolved operators :math:`\delta e_\mu(s) =
e^{sK_{A,MF}}\delta e_\mu e^{-sK_{A,MF}}` by their value at :math:`s=0`.
To keep the full imaginary-time kernel
:math:`\Sigma_{\mu\nu}(\tau) = \langle B_\mu(\tau) B_\nu\rangle`, with
:math:`B_\mu = \sum_{\nu\in E} M_{\mu\nu}\delta e_\nu`, the local
algebra of every site of the environment is bosonized in the eigenbasis of
its mean-field state :math:`\sigma = \sum_i p_i |i\rangle\langle i|`
(populations sorted as :math:`p_i \ge p_j` for :math:`i<j`):

.. math::

    |i\rangle\langle j| \to \sqrt{p_i - p_j}\; a_{ij},
    \qquad
    |j\rangle\langle i| \to \sqrt{p_i - p_j}\; a^\dagger_{ij},

so that :math:`[|i\rangle\langle j|, |j\rangle\langle i|]` has expectation
value :math:`p_i-p_j`, as required by :math:`[a,a^\dagger]=1`. Keeping only
the part of :math:`\delta e_\mu` that is linear in the bosons,

.. math::

    \delta e_\mu \simeq \sum_k \left(g_{\mu k} a_k + g^*_{\mu k}
    a_k^\dagger\right),
    \qquad
    g_{\mu,(ij)} = \sqrt{p_i-p_j}\,(e_\mu)_{ij},

the environment becomes a quadratic bosonic Hamiltonian
:math:`H_b = \sum_k \Delta_k a_k^\dagger a_k + \frac12\sum M_{\mu\nu}
\delta e_\mu\delta e_\nu` with :math:`\Delta_k = \ln(p_i/p_j)`, which is
diagonalized by a Bogoliubov transformation. This is equivalent to
integrating the Hubbard-Stratonovich fields analytically and summing over
all Matsubara frequencies: with normal modes :math:`\Psi_m` of signed
frequency :math:`\lambda_m` and :math:`B_\mu = \sum_m c_{\mu m}\Psi_m`,

.. math::

    \Sigma_{\mu\nu}(\tau) = \sum_m c_{\mu m} c^*_{\nu m}\,
    \frac{\eta_m\, e^{-\lambda_m \tau}}{1 - e^{-\lambda_m}},
    \qquad \eta_m = \operatorname{sign}\lambda_m .

The diagonal operators :math:`|i\rangle\langle i|` are bilinear in the
bosons, so they are not part of the linear truncation. Their static
(intraband) response, :math:`\sum_i p_i (e_\mu)_{ii}(e_\nu)_{ii} -
\langle e_\mu\rangle\langle e_\nu\rangle`, and the one of pairs with
:math:`p_i = p_j`, are time independent. They are included as bosons of
vanishing frequency ``static_gap`` whose couplings are scaled so that
:math:`\chi(\omega=0)` is exact.

With :math:`\Sigma(\tau)` at hand, the second-order contribution to the
reduced state is

.. math::

    Y = \int_{0<s'<s<1} ds\,ds'\; \Sigma_{\mu\nu}(s-s')\,
        e^{-K_{A,MF}}\,\delta e_\mu(s)\,\delta e_\nu(s') ,

and the generator correction follows from inverting the derivative of the
matrix exponential (Kubo-Mori) at first order in :math:`Y`:
:math:`\delta K_{mn} = -Y_{mn}/w_{mn}`, with :math:`w_{mn} =
(e^{-\epsilon_n}-e^{-\epsilon_m})/(\epsilon_m-\epsilon_n)` in the
eigenbasis of :math:`K_{A,MF}`. In the limit where :math:`\delta e_\mu`
commutes with :math:`K_{A,MF}` the static expression is recovered.

**Hybrid reduction (subsystem exact, RPA on the environment).** Instead of
correcting the generator at first order, the environment is bosonized with
the subsystem removed, its normal modes are compressed to a few effective
bosons (block Lanczos, a Gauss quadrature of the spectral density), the
modes with large thermal occupation are taken as static classical Gaussian
fields, and the resulting hybrid Hamiltonian (subsystem exact, truncated
bosons) is diagonalized exactly (:func:`rpa_hybrid_state`). It is non
perturbative in the coupling between the subsystem and the environment, so
it includes the back-reaction of the subsystem, and it is valid at low
temperatures, where the generator corrections fail. The symmetry
restoration is the one of the Gibbs state (``symmetry_projections``), which
for the parity is the mixture of the two broken-symmetry branches.

**Zero modes.** If the mean-field state breaks a continuous symmetry, the
bosonic Hamiltonian has zero modes (Goldstone modes): the Hamiltonian does
not depend on a collective coordinate :math:`Q` (the orientation of the
order parameter), and the conjugate charge :math:`P` is conserved. They are
detected and removed (see :func:`_bogoliubov_reduced`): the orientation is
fixed, and the symmetry has to be restored averaging over its orbit, with the
``symmetry_projections`` of the state (for instance,
``project_conserved_quantity`` for a U(1) symmetry). The thermal fluctuations
of the charge :math:`P` are included in the hybrid reduction as a static
Gaussian field that couples to the subsystem and drives the bath. The
entropies and negativities of the full bosonization are not defined with zero
modes, since the covariance of a subsystem is not finite. A mean-field state
that breaks the symmetry in the subsystem pins the orientation of the
environment, so that in many cases the environment has only a soft mode.
:func:`rpa_zero_modes` counts the zero modes.

**Full bosonization.** If the subsystem is also bosonized, the whole system
is a quadratic bosonic Hamiltonian whose thermal state is Gaussian. The
reduced covariance matrix of any subset of sites gives the entropy and the
negativity of the bosonic RPA state (see :func:`rpa_entanglement_entropy`
and :func:`rpa_negativity`), as in the RPA of Matera, Rossignoli and Canosa
(Phys. Rev. A 82, 052332). The populations of the local mean-field states
are obtained from the local mean-field generators, so that the zero
temperature limit can be reached.

Limitations
-----------
The environment is described by the Gaussian (linear) truncation of the
bosonization, completed with the static modes. The quartic terms, which
come from the products of the diagonal operators of different sites
(hopping and pairing contributions in a Wick decoupling), are neglected.
Their inclusion matters for sensitive quantities such as entanglement, but
it is only controlled at low temperatures with not too strong couplings, or
at high temperatures. At intermediate temperatures the thermal occupations
of the bosonic modes become large and the bosons explore regions of the
Fock space that have no counterpart in the local Hilbert space, so the
result should be regarded as qualitative. If the matrix of the bosonic
Hamiltonian is not positive definite, the RPA is unstable and a
``ValueError`` is raised.
"""

from .dynamic import rpa_dynamic_kernel
from .hybrid import effective_bath_modes, rpa_hybrid_state
from .rpa import RPAGibbsDensityOperator, rpa_coupling_kernel, rpa_generator_correction
from .rpa_structure import build_rpa_structure
from .utils import kubo_mori_susceptibility, local_boson_modes

__all__ = [
    "RPAGibbsDensityOperator",
    "build_rpa_structure",
    "effective_bath_modes",
    "kubo_mori_susceptibility",
    "local_boson_modes",
    "rpa_coupling_kernel",
    "rpa_dynamic_kernel",
    "rpa_generator_correction",
    "rpa_hybrid_state",
]
