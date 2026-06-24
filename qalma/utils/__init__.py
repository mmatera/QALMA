"""
Utility functions
-----------------

This module contains different utility functions for evaluating expressions
using structures defined in the xml libraries, build different plots
and graphic representation of operators and structures, to convert and process
the output of the different simulation and optimization routines.

"""

from .convexfit import convex_fit_derivative
from .evaluation import (
    eval_expr,
    find_ref,
    next_name,
    replace_variable_type,
)
from .inout import (
    matrix_to_wolfram,
    operator_to_wolfram,
)
from .visualization import (
    draw_ellipse_around_points,
    draw_operator,
)

__all__ = [
    "convex_fit_derivative",
    "draw_ellipse_around_points",
    "draw_operator",
    "operator_to_wolfram",
    "matrix_to_wolfram",
    "eval_expr",
    "find_ref",
    "next_name",
    "replace_variable_type",
]
