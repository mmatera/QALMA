"""Functions used to run MaxEnt simulations."""

from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union, cast

import numpy as np
from numpy.typing import NDArray

from qalma.meanfield import (
    variational_quadratic_mfa,
)
from qalma.operators import (
    Operator,
)
from qalma.operators.states import (
    GibbsDensityOperator,
    GibbsProductDensityOperator,
    ProductDensityOperator,
)
from qalma.projections import n_body_projection
from qalma.scalarprod import (
    HierarchicalOperatorBasis,
    OperatorBasis,
    covar_scalar_product,
    trim_terms_by_tolerance,
)

from .simulation import Simulation

logger = logging.getLogger(__name__)

# function used to safely and robustly map K-states to states


__all__ = ["adaptive_projected_evolution"]


def compute_mean_field_state(
    k: Operator, sigma: ProductDensityOperator, **kwargs
) -> Tuple[Operator, ProductDensityOperator]:
    """Build the generator associated to the mean field state.

    Parameters
    ----------
    k: Operator
       the generator to be approximated by a OneBodyOperator
    sigma: ProductDensityOperator
       A reference state to start the search of a generator

    Return
    ------

    generator: Operator
       The computed generator

    sigma_result: ProductDensityOperator
       The associated state.

    """
    sigma_result = variational_quadratic_mfa(k, sigma_ref=sigma)
    generator = -sigma_result.logm()
    assert generator.isherm, "generator should be hermitian"
    return generator, sigma_result


def compute_n_body_sector(k: Operator) -> int:
    """Return the n-body sector of the operator ``k``.

    Parameters
    ----------
    k : Operator
        The operator whose n-body sector is queried.

    Returns
    -------
    int
        The maximum number of sites on which any term of ``k`` acts
        non-trivially.

    """
    return k.n_body_sector()


def occupation_factor(phi: NDArray, threshold: float = 0.995) -> int:
    """Compute an estimate of how spread an operator is over a basis.

    Returns the number of terms in the partial sum of squared moduli of the
    components of ``phi`` needed to reach ``threshold``.

    Parameters
    ----------
    phi : numpy.ndarray of float or complex
        Coefficient vector.
    threshold : float, optional
        Target cumulative fraction of the squared norm (default 0.995).

    Returns
    -------
    int
        Number of leading components (sorted by magnitude) required to
        account for ``threshold`` of the total squared norm.

    """
    if len(phi) < 2:
        return 0
    partial_sums = np.array(
        [sum(np.abs(phi[:i]) ** 2) ** 0.5 for i in range(1, len(phi))]
    )
    partial_sums = partial_sums / partial_sums[-1]
    for idx, val in enumerate(partial_sums):
        if val > threshold:
            return idx + 1
    return len(phi)


def update_basis(
    k, sigma, ham, order, n_body, extra_observables, k_ref=None
) -> Tuple[OperatorBasis, Operator, Operator]:
    """Build a hierarchical operator basis adapted to the current state ``k``.

    Computes the mean-field approximation of ``k`` to define a reference
    generator ``k_ref``, then constructs a :class:`HierarchicalOperatorBasis`
    using the covariant scalar product with respect to ``sigma``. The basis
    is projected onto the ``n_body`` sector at each hierarchical level.

    Parameters
    ----------
    k : Operator
        The current generator :math:`K(t)`.
    sigma : ProductDensityOperator
        The current reference state, used to define the scalar product
        and the mean-field approximation.
    ham : Operator
        The Hamiltonian, used to build the hierarchical basis via iterated
        commutators.
    order : int
        The order of the hierarchical basis (number of commutator levels).
    n_body : int
        Maximum n-body sector for the projection. Negative values disable
        the projection.
    extra_observables : Iterable[Operator]
        Additional operators to append to the basis.
    k_ref : Operator or None, optional
        If provided, skip the mean-field computation and use this as the
        reference generator directly.

    Returns
    -------
    new_basis : OperatorBasis
        The updated hierarchical basis.
    sigma : ProductDensityOperator
        The updated mean-field reference state.
    k_ref_new : Operator
        The reference generator used to seed the basis.

    """
    if k_ref is None:
        k_ref_new, sigma = compute_mean_field_state(k, sigma)
        k_ref_new = k_ref_new + sigma.expect(k - k_ref_new)
        k_ref_new = k_ref_new.hermitian_part()
    else:
        k_ref_new = k_ref

    new_basis = HierarchicalOperatorBasis(
        k,
        ham,
        order,
        covar_scalar_product(sigma),
        n_body_projection=trim_and_project_function(sigma, n_body, tol=1e-9),
    )
    rest_elements = tuple(extra_observables)
    if k is not k_ref_new:
        rest_elements = rest_elements + (k_ref_new,)

    if rest_elements:
        new_basis = new_basis + rest_elements

    return (
        new_basis,
        sigma,
        k_ref_new,
    )


def update_basis_heavy(
    k, sigma, ham, order, n_body, extra_observables, k_ref=None
) -> Tuple[OperatorBasis, Operator, Operator]:
    """Build a hierarchical basis with a two-pass n-body projection.

    Like :func:`update_basis` but applies the n-body projection as a
    post-processing step over the full (unprojected) hierarchical basis,
    reusing the pre-computed Gram matrix and generator matrix. This avoids
    recomputing expensive tensor contractions while still enforcing the
    n-body truncation.

    Parameters
    ----------
    k : Operator
        The current generator :math:`K(t)`.
    sigma : ProductDensityOperator
        The current reference state, used to define the scalar product
        and the mean-field approximation.
    ham : Operator
        The Hamiltonian, used to build the hierarchical basis via iterated
        commutators.
    order : int
        The order of the hierarchical basis (number of commutator levels).
    n_body : int
        Maximum n-body sector for the second-pass projection.
    extra_observables : Iterable[Operator]
        Additional operators to append to the basis.
    k_ref : Operator or None, optional
        If provided, skip the mean-field computation and use this as the
        reference generator directly.

    Returns
    -------
    new_basis : OperatorBasis
        The updated basis with n-body projection applied.
    sigma : ProductDensityOperator
        The updated mean-field reference state.
    k_ref_new : Operator
        The reference generator used to seed the basis.

    """
    if k_ref is None:
        k_ref_new, sigma = compute_mean_field_state(k, sigma)
        k_ref_new = k_ref_new + sigma.expect(k - k_ref_new)
        k_ref_new = k_ref_new.hermitian_part()
    else:
        k_ref_new = k_ref

    sp = covar_scalar_product(sigma)
    new_basis: OperatorBasis = HierarchicalOperatorBasis(
        k,
        ham,
        order,
        sp,
    )
    # Now a new basis is built projecting the operators to the n_body sector.
    new_basis = OperatorBasis(
        new_basis.operator_basis,
        ham,
        sp,
        n_body_projection=trim_and_project_function(sigma, n_body, tol=1e-6),
        precomputed_tensors={
            "gram": new_basis.gram,
            "gram_inv": new_basis.gram_inv,
            "errors": new_basis.errors,
            "gen_matrix": new_basis.gen_matrix,
            "hij": new_basis._hij,
        },
    )

    rest_elements = tuple(extra_observables)
    if k is not k_ref_new:
        rest_elements = rest_elements + (k_ref_new,)
    if rest_elements:
        new_basis = new_basis + rest_elements

    return (
        new_basis,
        sigma,
        k_ref_new,
    )


def update_basis_light(
    k, sigma, ham, order, n_body, extra_observables, k_ref=None
) -> Tuple[OperatorBasis, Operator, Operator]:
    """Build a hierarchical basis seeded from the mean-field approximation.

    Lighter alternative to :func:`update_basis`: constructs the
    :class:`HierarchicalOperatorBasis` using ``k_ref`` (the mean-field
    approximation of ``k``) as the seed instead of ``k`` itself, and
    omits the per-level n-body projection. The current generator ``k``
    is prepended to the basis as an extra element.

    This trades accuracy for speed: the basis adapts to the mean-field
    structure rather than the full generator, which is cheaper to build
    but may require more basis updates to maintain accuracy.

    Parameters
    ----------
    k : Operator
        The current generator :math:`K(t)`.
    sigma : ProductDensityOperator
        The current reference state, used to define the scalar product
        and the mean-field approximation.
    ham : Operator
        The Hamiltonian, used to build the hierarchical basis via iterated
        commutators.
    order : int
        The order of the hierarchical basis (number of commutator levels).
    n_body : int
        Maximum n-body sector (passed for interface compatibility; not used
        for per-level projection in this variant).
    extra_observables : Iterable[Operator]
        Additional operators to prepend to the basis alongside ``k``.
    k_ref : Operator or None, optional
        If provided, skip the mean-field computation and use this as the
        reference generator directly.

    Returns
    -------
    new_basis : OperatorBasis
        The updated hierarchical basis seeded from ``k_ref``.
    sigma : ProductDensityOperator
        The updated mean-field reference state.
    k_ref_new : Operator
        The reference generator used to seed the basis.

    """
    if k_ref is None:
        k_ref_new, sigma = compute_mean_field_state(k, sigma)
        k_ref_new = k_ref_new + sigma.expect(k - k_ref_new)
    else:
        k_ref_new = k_ref

    new_basis = HierarchicalOperatorBasis(
        k_ref_new,
        ham,
        order,
        covar_scalar_product(sigma),
    )

    rest_elements = tuple(extra_observables)
    if k is not k_ref_new:
        rest_elements = rest_elements + (k,)
    if rest_elements:
        new_basis = rest_elements + new_basis
    return (
        new_basis,
        sigma,
        k_ref_new,
    )


def adaptive_projected_evolution(
    ham,
    k0,
    t_span,
    order,
    n_body: int = -1,
    tol=1e-3,
    *,
    e_ops: Optional[Dict | List | Callable] = None,
    on_update_basis_callback: Optional[Callable] = None,
    extra_observables: Tuple[Operator, ...] = tuple(),
    include_one_body_projection: bool = False,
    basis_update_callback: Callable[
        ..., Tuple[OperatorBasis, Operator, Operator]
    ] = update_basis,
    update_condition: str = "adaptive",
    store_states=False,
    checkpoint_file: Optional[Union[str, os.PathLike]] = None,
) -> Simulation:
    """Compute the solution of the MaxEnt projected Schrödinger equation.

    dk
    -- = -i [H, k]
    dt

    as a linear combination of a an operator basis

    .. math::

        k = sum phi_a(t) Q_a

    chosen adaptively along the evolution.

    Parameters
    ----------
    ham : Operator
        The Hamiltonian operator

    k0 : Operator
        The initial condition

    t_span: np.array
        the times for with the evolution is computed

    order:
        the order of the solution

    n_body: int
        if non-negative, build a solution projected on
        the subspace of n_body operators.

    tol: float
        the maximum induced distance between the projected solution
        and the exact solution.

    basis_update_callback: Callable[..., Tuple[OperatorBasis, Operator]]
        the function that creates a new basis from the current
        state of the simulation. The default value is the function `update_basis`
        which generates a hierarchical basis from the current generator `K(t)`,
        projecting on each step to the `n_body` sector.
        Alternatively, `update_basis_light` is a light-weight version of `update_basis`
        which uses as the seed the self-consistent mean field approximation of K(t).

    on_update_basis_callback: Callable[dict], optional
        if not None, this function is called each time the basis is rebuilt.

    update_condition: str (case insensitive)
        The condition to update the basis. One of

        * ``"adaptive"`` (default) update the basis when the accuracy goal
          can not be reached.
        * ``"always"``: update the basis on each step
        * ``"never"``: the basis stays fixed.

        Default: adaptive

    store_states: bool
        If True, always store the states.

    checkpoint_file: str or path-like, optional
        If given, the partial result is stored in this HDF5 file after
        each time step, replacing the previous checkpoint. It can be loaded
        with ``Simulation.load_hdf5``. By default, no checkpoint is written.

    Returns
    -------
    Simulation:
        A Simulation object storing the results of the simulation.

    """
    t_0 = t_span[0]
    error = 0
    errors: List[float] = []
    expect_ops: Dict[Any, Operator] = {}
    local_evol_parms: Dict = {
        "t_ref": t_0,
        "last_t": t_0,
        "away": 0,
        "error": 0,
        "cummulated error": 0,
        "k_ref": None,
    }
    oc_factors: List[float] = []
    saturated_tolerance: bool = False
    states: List[Operator] = []
    t_max = t_span[-1]
    tlist: List[float] = []

    update_condition = update_condition.lower()
    parameters: Dict[str, Any] = {
        "n_body": n_body,
        "order": order,
        "tol": tol,
        "include_one_body_projection": include_one_body_projection,
        "basis_update_callback": basis_update_callback.__name__,
        "system": ham.system,
        "update_condition": update_condition,
        "method": "Adaptative Restricted Evolution",
    }
    stats: Dict[str, Any] = {
        "method": "Adaptative Restricted Evolution",
        "errors": errors,
        "basis time costs": [],
        "occupation factor": oc_factors,
        "away_from_ref": [],
        "n_body_sector": [],
        "basis update times": [],
    }
    simulation = Simulation(
        parameters=parameters,
        stats=stats,
        time_span=tlist,
        expect_ops=expect_ops,
        states=states,
    )
    if update_condition == "always":
        always_update, never_update = True, False
    elif update_condition == "never":
        always_update, never_update = False, True
    elif update_condition == "adaptive":
        always_update, never_update = False, False
    else:
        raise ValueError(
            f"update_condition={update_condition} is not one of the valid values 'always', 'never', 'adaptive'."
        )

    ### Handle e_ops #####
    if e_ops is None:

        def call_on_success_evol(t, k):
            """Store the state ``k`` at time ``t``."""
            states.append(k)

    elif hasattr(e_ops, "__call__"):
        call_on_success_evol = cast(Callable, e_ops)
    else:
        if not isinstance(e_ops, dict):
            e_ops = {pos: e_op for pos, e_op in enumerate(cast(Iterable, e_ops))}

        def call_on_success_evol(t, k):
            """Evaluate observables at time ``t`` and optionally store ``k``."""
            curr_e_ops = GibbsDensityOperator(k).expect(e_ops)
            for key, val in curr_e_ops.items():
                expect_ops.setdefault(key, []).append(val)
            if store_states:
                states.append(k)

    ####  Basis update ########
    def call_update_basis(local_evol_parms) -> bool:
        """Update the basis and reference state in ``local_evol_parms``."""
        k_t = local_evol_parms["k_t"].simplify()
        start_basis_time = datetime.now()
        basis, sigma_ref, k_ref = basis_update_callback(
            k_t,
            local_evol_parms["sigma_ref"],
            ham,
            order,
            local_evol_parms["curr_n_body"],
            extra_observables,
            k_ref=local_evol_parms["k_ref"],
        )
        build_basis_time_cost = datetime.now() - start_basis_time
        local_evol_parms["basis time cost"] = (
            build_basis_time_cost.seconds + 1e-6 * build_basis_time_cost.microseconds
        )
        local_evol_parms["sigma_ref"] = sigma_ref
        local_evol_parms["k_ref"] = k_ref
        local_evol_parms["basis"] = basis
        local_evol_parms["t_ref"] = local_evol_parms["last_t"]
        local_evol_parms["phi_0"] = basis.coefficient_expansion(k_t)

        if on_update_basis_callback is not None:
            on_update_basis_callback(local_evol_parms)
        return True

    # Initialize
    local_evol_parms["t"] = 0
    local_evol_parms["k_t"] = k0
    local_evol_parms["curr_n_body"] = n_body  # compute_n_body_sector(k0)
    local_evol_parms["sigma_ref"] = GibbsProductDensityOperator({}, k0.system)
    local_evol_parms["max_error_speed"] = tol / (t_max - t_0)
    tlist.append(local_evol_parms["t_ref"])
    logger.info(f"max_error_speed:{local_evol_parms['max_error_speed']}")

    # Create the first base
    call_update_basis(local_evol_parms)
    stats["basis update times"].append(t_0)
    stats["basis time costs"].append(local_evol_parms["basis time cost"])
    stats["n_body_sector"].append(local_evol_parms["curr_n_body"])
    # Perform tasks that follows to a success evolution:
    call_on_success_evol(t_0, local_evol_parms["k_t"])
    away = abs(
        local_evol_parms["basis"].operator_norm(
            (local_evol_parms["k_t"] - local_evol_parms["k_ref"]).simplify()
        )
    )
    local_evol_parms["away"] = away
    stats["errors"].append(0)
    stats["away_from_ref"].append(away)
    stats["occupation factor"].append(occupation_factor(local_evol_parms["phi_0"]))

    # Main loop
    for t in t_span[1:]:
        local_evol_parms["t"] = t
        last_error = error
        # If K_t is too far from K_ref, update the basis:
        if always_update or (not never_update and away > tol):
            logger.info("updating K_ref")
            local_evol_parms["k_ref"] = None
            call_update_basis(local_evol_parms)
            local_evol_parms["cummulated error"] += last_error
            last_error = 0
            local_evol_parms["max_error_speed"] = (
                tol - local_evol_parms["cummulated error"]
            ) / (t_max - local_evol_parms["t_ref"])
            stats["basis update times"].append(t)
            stats["basis time costs"].append(local_evol_parms["basis time cost"])

        delta_t = t - local_evol_parms["t_ref"]
        phi, error = local_evol_parms["basis"].evolve(
            delta_t, local_evol_parms["phi_0"]
        )

        # If the error is growing faster that the acceptable rate,
        # try to enlarge the n-body sector:
        while error > local_evol_parms["max_error_speed"] * delta_t:
            logger.info("Error max speed saturated. Enlarge the basis.")
            local_evol_parms["error"] = error
            call_update_basis(local_evol_parms)
            delta_t = t - local_evol_parms["t_ref"]
            phi, error = local_evol_parms["basis"].evolve(
                delta_t, local_evol_parms["phi_0"]
            )

            if error <= local_evol_parms["max_error_speed"] * delta_t:
                # Compute the cummulated error until the last successful basis change
                local_evol_parms["cummulated error"] += last_error
                local_evol_parms["max_error_speed"] = (
                    tol - local_evol_parms["cummulated error"]
                ) / (t_max - local_evol_parms["t_ref"])
                stats["basis update times"].append(t)
                stats["basis time costs"].append(local_evol_parms["basis time cost"])
                break
            if n_body > local_evol_parms["curr_n_body"]:
                local_evol_parms["curr_n_body"] += 1
                logger.warning(
                    f"tolerance goal cannot be reached within this subspace. Try in {local_evol_parms['curr_n_body']} sector."
                )
                continue

            if local_evol_parms["cummulated error"] > tol:
                saturated_tolerance = True
                logger.warning(
                    f"tolerance goal cannot be reached within {n_body} sector."
                )
                break
            local_evol_parms["max_error_speed"] = (
                2 * local_evol_parms["max_error_speed"]
            )

        if saturated_tolerance:
            logger.warning("tolerance goal cannot be reached within this subspace.")
            break

        tlist.append(t)
        local_evol_parms["last_t"] = t
        local_evol_parms["k_t"] = local_evol_parms["basis"].operator_from_coefficients(
            phi
        )
        stats["errors"].append(error + local_evol_parms["cummulated error"])
        local_evol_parms["error"] = error
        call_on_success_evol(t, local_evol_parms["k_t"])
        stats["n_body_sector"].append(local_evol_parms["curr_n_body"])
        stats["occupation factor"].append(occupation_factor(phi))
        away = local_evol_parms["basis"].operator_norm(
            (local_evol_parms["k_t"] - local_evol_parms["k_ref"]).simplify()
        )
        local_evol_parms["away"] = away
        stats["away_from_ref"].append(away)

        # Dump the simulation state
        if checkpoint_file is not None:
            simulation.save_hdf5(checkpoint_file, mode="w")

    return simulation


def projected_evolution(ham, k0, t_span, order, n_body: int = -1) -> Simulation:
    """Compute the solution of the MaxEnt projected Schrödinger equation.

    dk
    -- = -i [H, k]
    dt

    as a linear combination of the iterated commutators

    k = sum phi_a(t) Q_a

    Parameters
    ----------
    ham : Operator
        The Hamiltonian operator
    k0 : Operator
        The initial condition
    t_span: np.array
        the times for with the evolution is computed
    order:
        the order of the solution
    n_body: int
        if non-negative, build a solution projected on
        the subspace of n_body operators.

    Returns
    -------
    Simulation:
        A simulation object with the results of the simulation.

    """
    sigma_0 = GibbsProductDensityOperator(k0)
    sp = covar_scalar_product(sigma_0)

    basis = HierarchicalOperatorBasis(
        k0,
        ham,
        order,
        sp,
        n_body_projection=lambda op_b: n_body_projection(
            op_b, n_max=n_body, sigma=sigma_0
        ),
    )
    phi_0 = basis.coefficient_expansion(k0)
    errors = []
    states = []
    for t in t_span:
        phi, error = basis.evolve(t, phi_0)
        errors.append(error)
        states.append(basis.operator_from_coefficients(phi))

    return Simulation(
        parameters={"n_body": n_body, "order": order, "system": ham.system},
        stats={"method": "Static Projected Evolution", "errors": errors},
        time_span=t_span,
        expect_ops={},
        states=states,
    )


def trim_and_project_function(sigma, n_body, tol=1e-6):
    """Build a projection function that trims and projects operators.

    Returns a callable that, given an operator, first projects it onto the
    ``n_body`` sector and then removes terms whose contribution is below
    ``tol`` relative to the scalar product defined by ``sigma``.

    Parameters
    ----------
    sigma : ProductDensityOperator
        Reference state defining the covariant scalar product used for
        trimming.
    n_body : int
        Maximum n-body sector for the projection.
    tol : float, optional
        Tolerance for trimming small terms. Default is ``1e-6``.

    Returns
    -------
    Callable[[Operator], Operator]
        A function that accepts an :class:`Operator` and returns its
        projected and trimmed version.

    """

    def trim_and_project(op_b):
        """Project ``op_b`` onto the n-body sector and trim small terms.

        Parameters
        ----------
        op_b : Operator
            The operator to project and trim.

        Returns
        -------
        Operator
            The projected and trimmed operator, enforcing Hermiticity if
            the input was Hermitian.

        """
        logger.debug(
            "trim and project %s terms in the %s sector up to tol=%s",
            op_b.num_terms(),
            op_b.n_body_sector(),
            tol,
        )
        isherm = op_b.isherm
        op_b = op_b.simplify()
        op_b = n_body_projection(op_b, n_max=n_body, sigma=sigma).simplify()
        op_b = trim_terms_by_tolerance(sigma, op_b, tol)
        if isherm:
            op_b = op_b.hermitian_part()
        else:
            logger.debug("basis element of type %s is not hermitian", type(op_b))
        return op_b

    return trim_and_project
