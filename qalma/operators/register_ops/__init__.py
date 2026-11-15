"""Arithmetic operator dispatch registrations.

Submodules are imported in a specific order because the dispatch table uses
last-write-wins for duplicate keys, and the base-class catch-all
(Operator, Operator) must be registered *before* the more specific handlers
so that specific handlers take precedence via the hierarchy walk in
find_arithmetic_implementation.

Import order:
  1. scalar    — base-class catch-all + ScalarOperator
  2. local     — LocalOperator (most primitive concrete type)
  3. product   — ProductOperator (built from LocalOperators)
  4. onebody   — OneBodyOperator (sum of LocalOperators)
  5. sum       — SumOperator (most general sum type)
  6. qutip_op  — QutipOperator (QuTiP-backed fallback)
  7. quadratic — QuadraticFormOperator (highest-level composite)
  8. pauli     — PauliStringOperator (symbolic Pauli algebra)
"""

from . import local  # noqa: F401
from . import onebody  # noqa: F401
from . import pauli  # noqa: F401
from . import product  # noqa: F401
from . import quadratic  # noqa: F401
from . import qutip_op  # noqa: F401
from . import scalar  # noqa: F401
from . import sum  # noqa: F401
