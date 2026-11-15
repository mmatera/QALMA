# -*- coding: utf-8 -*-
"""
Operators.

Classes representing different kind of operators.

Subpackages
-----------
functions
    Operator-level functions: commutator, anticommutator, fidelity,
    spectral_norm, relative_entropy, eigenvalues, log_op, compute_dagger.
states
    Density-operator classes: ProductDensityOperator, GibbsDensityOperator,
    GibbsProductDensityOperator, QutipDensityOperator, MixtureDensityOperator.
"""

import importlib

from .arithmetic import (
    OneBodyOperator,
    SumOperator,
    iterable_to_operator,
)
from .basic import (
    LocalOperator,
    Operator,
)
from .pauli import PauliStringOperator, to_pauli_sum
from .product import (
    ProductOperator,
    ScalarOperator,
)
from .quadratic import QuadraticFormOperator
from .qutip import QutipOperator

# register_ops wires arithmetic dispatch tables; must run before states/functions
# so that operator arithmetic works when those subpackages are loaded.
importlib.import_module(".register_ops", __name__)

# states and functions are imported last because they depend on the operator
# classes and dispatch tables being fully initialised above.
from . import functions, states  # noqa: E402

__all__ = [
    # Operator classes
    "LocalOperator",
    "OneBodyOperator",
    "Operator",
    "PauliStringOperator",
    "ProductOperator",
    "QuadraticFormOperator",
    "QutipOperator",
    "ScalarOperator",
    "SumOperator",
    "iterable_to_operator",
    "to_pauli_sum",
    # Subpackages
    "functions",
    "states",
]
