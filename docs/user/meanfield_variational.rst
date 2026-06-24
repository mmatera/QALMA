Variational Mean-Field Approximation
=====================================

.. contents::
   :depth: 2
   :local:

Overview
--------

The variational mean-field approximation finds the product state
:math:`\sigma = \bigotimes_i \rho_i` that minimises the variational free
energy

.. math::

   F[\sigma] = \mathrm{Tr}\bigl[\sigma\,(\beta H + \log\sigma)\bigr]
             = \beta\langle H\rangle_\sigma - S(\sigma),

where :math:`S(\sigma) = -\mathrm{Tr}[\sigma\log\sigma]` is the von
Neumann entropy.  By the Gibbs inequality,
:math:`F[\sigma] \ge F_{\rm exact} = -\log Z` for all states
:math:`\sigma`, with equality only for the exact Gibbs state
:math:`\rho = e^{-\beta H}/Z`.

QALMA implements the **quadratic variational mean-field** method
(:func:`~qalma.meanfield.variational_quadratic_mfa`), which parametrises
:math:`H` by a quadratic (two-body) generator and optimises over a family
of auxiliary fields to improve the approximation beyond the plain
self-consistent (SC) solution.

Basic usage
-----------

.. code-block:: python

   from qalma.meanfield import variational_quadratic_mfa, compute_free_energy

   beta = 2.0

   # Plain self-consistent solution (numfields=0)
   sigma_sc  = variational_quadratic_mfa(beta * ham, numfields=0,
                                          max_self_consistent_steps=100)

   # Variational solution with 6 auxiliary fields
   sigma_var = variational_quadratic_mfa(beta * ham, numfields=6,
                                          max_self_consistent_steps=30)

   f_sc  = compute_free_energy(sigma_sc,  beta * ham)
   f_var = compute_free_energy(sigma_var, beta * ham)

   print(f"F_sc  = {f_sc:.6f}")   # upper bound on F_exact
   print(f"F_var = {f_var:.6f}")  # tighter upper bound ($\leq$ F_sc)

The variational free energy is non-increasing in ``numfields``: more
auxiliary fields give a tighter upper bound on the exact free energy.

Warm starting
^^^^^^^^^^^^^

When sweeping over ``numfields``, passing the previous solution as
``sigma_ref`` significantly speeds up convergence:

.. code-block:: python

   sigma_ref = sigma_sc
   for nf in [1, 2, 4, 6, 8]:
       sigma_ref = variational_quadratic_mfa(
           beta * ham,
           numfields=nf,
           sigma_ref=sigma_ref,
           max_self_consistent_steps=30,
       )
       f = compute_free_energy(sigma_ref, beta * ham)
       print(f"nf={nf:2d}:  F = {f:.6f}")

Convergence diagnostics
------------------------

Variational free energy alone does not tell the full story: a state can
have a low :math:`F[\sigma]` (close to :math:`F_{\rm exact}`) but still
be a poor approximation to the Gibbs state because its fluctuations are
wrong.  QALMA provides two complementary diagnostics.

T-score
^^^^^^^

The **T-score** measures how much the log-ratio operator
:math:`\hat{F} = \beta H - \kappa` (where :math:`\kappa = -\log\sigma`)
fluctuates relative to its mean under :math:`\sigma`:

.. math::

   T_{\rm score}
       = \frac{N\,\operatorname{Var}_\sigma(\hat{F})}
              {\langle \hat{F} \rangle_\sigma^2},

where :math:`N` is the number of sites.  The factor :math:`N` makes
:math:`T_{\rm score}` an *intensive* quantity: since :math:`\hat{F}` is a
sum of local terms, :math:`\operatorname{Var}_\sigma(\hat{F}) \sim O(N)`
and :math:`\langle\hat{F}\rangle_\sigma^2 \sim O(N^2)`, so without the
prefactor the ratio would decay as :math:`1/N`.

:math:`T_{\rm score} = 0` if and only if :math:`\hat{F}` is constant on
the support of :math:`\sigma`, which happens precisely when
:math:`\sigma = \rho`.  Together with :math:`F[\sigma]`, it provides a
two-dimensional picture of approximation quality:

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - :math:`F[\sigma]`
     - :math:`T_{\rm score}`
     - Interpretation
   * - large
     - small
     - Systematic energy offset; the fluctuation structure is well captured.
   * - small
     - large
     - Good average energy; large residual fluctuations in the distribution.
   * - small
     - small
     - Approximation is close to the exact Gibbs state.

When the exact free energy :math:`F_{\rm exact} = -\log Z` is available
(small systems), pass it to :func:`~qalma.meanfield.compute_t_score`:

.. code-block:: python

   from qalma.meanfield import compute_t_score

   # exact_f_exact = -log Tr[exp(-beta*H)], computed by full diagonalisation
   tscore, mean_F, var_F = compute_t_score(sigma_var, beta * ham, exact_f_exact)
   print(f"T-score = {tscore:.4f}")

The argument ``exact_f_exact`` must be in the same units as ``beta * ham``,
i.e. :math:`-\log\mathrm{Tr}[e^{-\beta H}]` (not the Helmholtz free energy
:math:`F/k_BT`).

Variance ratio (large systems)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For large systems where full diagonalisation is not feasible,
:math:`F_{\rm exact}` is unavailable and the T-score cannot be computed.
Instead, use the **variance ratio**

.. math::

   R_m = \frac{\operatorname{Var}_{\sigma_m}(\hat{F})}
              {\operatorname{Var}_{\sigma_{\rm SC}}(\hat{F})},

where :math:`\sigma_m` is the variational state with :math:`m` fields and
:math:`\sigma_{\rm SC}` is the SC baseline (:math:`m=0`).  This ratio is
:math:`1` at :math:`m=0` and decreases toward :math:`0` as the
approximation improves.  It does not require :math:`F_{\rm exact}` and
is computed using :func:`~qalma.meanfield.compute_variance`:

.. code-block:: python

   from qalma.meanfield import compute_variance

   var_sc  = compute_variance(sigma_sc,  beta * ham)  # baseline
   var_var = compute_variance(sigma_var, beta * ham)  # variational

   R = var_var / var_sc
   print(f"Variance ratio R = {R:.4f}")   # < 1 means improvement over SC

Example: chiral spin Hamiltonian on a triangular strip
------------------------------------------------------

The following example applies the variational MF to the chiral spin model
(see :doc:`loop_operators`) and tracks both :math:`F[\sigma]` and
:math:`R_m` as the number of auxiliary fields grows.

.. code-block:: python

   import numpy as np
   from qalma import graph_from_alps_xml, model_from_alps_xml
   from qalma.model import SystemDescriptor
   from qalma.meanfield import (
       variational_quadratic_mfa,
       compute_free_energy,
       compute_variance,
   )

   # --- Build the system ---------------------------------------------------
   L, J, chi, beta = 8, 1.0, 0.5, 2.0

   graph  = graph_from_alps_xml(name="triangular strip open",
                                parms={"L": L, "a": 1})
   model  = model_from_alps_xml(name="chiral spin")
   system = SystemDescriptor(graph, model, {"J": J, "Wilson2": chi})
   ham    = system.global_operator("Hamiltonian")

   # --- SC baseline --------------------------------------------------------
   sigma_sc = variational_quadratic_mfa(
       beta * ham, numfields=0, max_self_consistent_steps=100
   )
   f_sc    = compute_free_energy(sigma_sc,  beta * ham)
   var_sc  = compute_variance(sigma_sc, beta * ham)

   print(f"SC baseline:  F = {f_sc:.6f}  Var = {var_sc:.4g}")

   # --- Variational sweep --------------------------------------------------
   sigma_ref = sigma_sc
   for nf in [1, 2, 4, 6, 8]:
       sigma_ref = variational_quadratic_mfa(
           beta * ham,
           numfields=nf,
           sigma_ref=sigma_ref,
           max_self_consistent_steps=30,
       )
       f   = compute_free_energy(sigma_ref, beta * ham)
       var = compute_variance(sigma_ref, beta * ham)
       R   = var / var_sc
       print(f"  nf={nf:2d}:  F = {f:.6f}  Var = {var:.4g}  R = {R:.4f}")

A complete worked example including plots is available in the notebook
:doc:`../examples/example_chiral_variational_mf`.

API reference
-------------

.. autofunction:: qalma.meanfield.variational_quadratic_mfa
   :noindex:

.. autofunction:: qalma.meanfield.compute_free_energy
   :noindex:

.. autofunction:: qalma.meanfield.compute_t_score
   :noindex:

.. autofunction:: qalma.meanfield.compute_variance
   :noindex:

.. seealso::

   :doc:`loop_operators` — how to define three-body Hamiltonians.

   :doc:`/api/meanfield` — full API reference for the meanfield module.
