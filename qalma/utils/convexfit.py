"""
Convex fit tools
----------------

Tools to postprocess results of optimizations.
"""

import logging
from typing import List, Tuple

import numpy as np
from scipy.interpolate import BSpline
from scipy.optimize import minimize


def convex_fit_derivative(
    data: List[Tuple[float, float]], convexity: float = -1.0
) -> List[float]:
    """
    Fits a strictly convex spline to the given data points and returns its
    first derivative evaluated at the original x-coordinates.

    Args:
        data: A list of (x, y) tuples representing the data points.

        convexity: float (optional)
           the sign of convexity:``1.`` assumes a convex function.
           ``-1`` assumes a concave function. ``1.`` by default.

    Returns:
        A list of float values, representing the first derivative of the
        fitted convex spline evaluated at the x-coordinates from the input data.
    """
    # 1. Parse and sort data
    print(data)
    x_data_raw, y_data_raw = [np.array(lst) for lst in zip(*data)]

    sort_idx = np.argsort(x_data_raw)
    x = x_data_raw[sort_idx]
    y = y_data_raw[sort_idx]

    # 2. Define B-spline basis
    k = 3  # Cubic spline degree
    num_internal_knots = max(k + 1, len(x) // 3)
    # Avoid quantile error for very few points by ensuring linspace has at least 2 points
    if num_internal_knots < 1:
        internal_knots = np.array([])
    else:
        internal_knots = np.quantile(x, np.linspace(0, 1, num_internal_knots + 2)[1:-1])

    knots = np.concatenate(([x.min()] * k, internal_knots, [x.max()] * k))
    n_coeffs = len(knots) - k - 1

    def get_spline_from_coeffs(coeffs, knots, k):
        return BSpline(knots, coeffs, k)

    # 3. Define the Objective Function (Mean Squared Error)
    def objective_function(coeffs, x_data, y_data, knots, k):
        spline_func = get_spline_from_coeffs(coeffs, knots, k)
        y_pred = spline_func(x_data)
        return np.sum((y_pred - y_data) ** 2)

    # 4. Define Convexity Constraints
    # f''(x) >= 0 for all x. So, the constraint function should return f''(x).
    x_eval_convexity = np.linspace(x.min(), x.max(), 100)  # Check at 100 points

    def convexity_constraint_func(coeffs, x_eval, knots, k):
        spline_func = get_spline_from_coeffs(coeffs, knots, k)
        spline_deriv2_func = spline_func.derivative(nu=2)
        deriv2_values = spline_deriv2_func(x_eval)
        return (
            deriv2_values * convexity
        )  # We want deriv2_values >= 0 for 'greater_equal' constraint

    constraints = [
        {
            "type": "ineq",
            "fun": convexity_constraint_func,
            "args": (x_eval_convexity, knots, k),
        }
    ]

    # Initial guess for coefficients
    initial_coeffs = np.zeros(n_coeffs)

    # 5. Run the Optimization
    result = minimize(
        objective_function,
        initial_coeffs,
        args=(x, y, knots, k),
        method="SLSQP",
        constraints=constraints,
        options={"disp": False, "maxiter": 1000},
    )

    if not result.success:
        logging.warning(f"Optimization did not converge successfully: {result.message}")

    optimized_coeffs = result.x
    convex_spline_func = get_spline_from_coeffs(optimized_coeffs, knots, k)

    # 6. Evaluate the first derivative
    spline_deriv1_func = convex_spline_func.derivative(nu=1)
    deriv_values_at_original_x = spline_deriv1_func(x)

    return deriv_values_at_original_x.tolist()
