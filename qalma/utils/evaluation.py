"""
Evaluation and access to xml libraries.
---------------------------------------

Utility functions to import and process ALPS specification files.

"""

import ast
import logging
from functools import lru_cache

import numpy as np

logger = logging.getLogger(__name__)

default_parms = {
    "pi": 3.1415926,
    "e": 2.71828183,
    "sqrt": np.sqrt,
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "exp": np.exp,
    "log": np.log,
    "rand": np.random.rand,
}


# Syntax accepted in expressions: numbers, names, arithmetic operators and
# calls to named functions. In particular, there is no attribute access, no
# subscripts, no string literals, no comprehensions and no lambdas.
_ALLOWED_EXPR_NODES = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Call,
    ast.Name,
    ast.Load,
    ast.Constant,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.FloorDiv,
    ast.Mod,
    ast.Pow,
    ast.MatMult,
    ast.UAdd,
    ast.USub,
)


@lru_cache(maxsize=4096)
def _compile_expr(expr: str):
    """Compile an arithmetic expression, rejecting any other Python syntax.

    Expressions come from ALPS libraries and from simulation files, so they
    are data and must not be able to run arbitrary code. Only the syntax
    listed in ``_ALLOWED_EXPR_NODES`` is accepted.

    Raises
    ------
    SyntaxError
        If ``expr`` is not valid Python, or uses syntax that is not allowed.
    """
    tree = ast.parse(expr.strip(), mode="eval")
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_EXPR_NODES):
            raise SyntaxError(
                f"<<{expr}>> is not an arithmetic expression: "
                f"{type(node).__name__} is not allowed."
            )
        if isinstance(node, ast.Constant) and not isinstance(
            node.value, (bool, int, float, complex)
        ):
            raise SyntaxError(
                f"<<{expr}>> is not an arithmetic expression: "
                f"the constant {node.value!r} is not a number."
            )
        if isinstance(node, ast.Call) and (
            not isinstance(node.func, ast.Name) or node.keywords
        ):
            raise SyntaxError(
                f"<<{expr}>> is not an arithmetic expression: only calls of "
                "the form name(arg, ...) are allowed."
            )
    return compile(tree, "<qalma expression>", "eval")


def eval_expr(expr: str, parms: dict):
    """Evaluate the expression `expr` using ``parms``.

    ``expr`` can include numbers, the names defined in ``parms``,
    arithmetic operators and some elementary functions (``sqrt``, ``sin``,
    ``cos``, ``tan``, ``exp``, ``log``, ``rand``).

    Expressions are data: they are read from ALPS libraries and from
    simulation files. To prevent them from running arbitrary code, any
    other Python syntax (attribute access, subscripts, strings, ...) raises
    a ``SyntaxError``, and names are only looked up in ``parms`` (Python
    builtins are not available).
    """
    # TODO: Improve the workflow in a way that numpy functions
    # and constants be loaded just if they are needed.

    if not isinstance(expr, str):
        return expr

    try:
        return float(expr)
    except (ValueError, TypeError):
        try:
            if expr not in ("J", "j"):
                return complex(expr)
        except (ValueError, TypeError):
            pass

    parms = {
        key.replace("'", "_prima"): val for key, val in parms.items() if val is not None
    }
    expr = expr.replace("'", "_prima")

    while expr in parms:
        expr = parms.pop(expr)
        if not isinstance(expr, str):
            return expr

    # Reduce the parameters
    p_vars = list(parms)
    while True:
        changed = False
        for k in p_vars:
            val = parms.pop(k)
            if not isinstance(val, str):
                parms[k] = val
                continue
            try:
                result = eval_expr(val, parms)
                if result is not None:
                    parms[k] = result
                if val != result:
                    changed = True
            except RecursionError:
                logger.warning("A recursion error happens evaluating `%s`.", val)
                raise
        if not changed:
            break
    parms.update(default_parms)
    # Names are resolved in ``parms`` only.
    parms["__builtins__"] = {}
    try:
        # pylint: disable=eval-used
        result = eval(_compile_expr(expr), parms)
        return result
    except NameError:
        pass
    except TypeError as exc:
        logger.warning("Type Error. Undefined variables in [%s] in %s.", exc, expr)
        return None
    except SyntaxError:
        logger.error(
            (
                "expression " f"<<{expr}>>",
                f"\n   with parameters\n{parms}\n" "raised a SyntaxError",
            )
        )
        raise
    return expr


def find_ref(node, root):
    """Find a node in the root.

    Parameters
    ----------
    node : object
        the key of the node.
    root : dict
        a nested tree structure of
        dicts

    Returns
    -------
    dict
        the node corresponding to ``node``.

    """
    node_items = dict(node.items())
    if "ref" in node_items:
        name_ref = node_items["ref"]
        for refnode in root.findall("./" + node.tag):
            if ("name", name_ref) in refnode.items():
                return refnode
    return node


def next_name(dictionary: dict, s: int = 1, prefix: str = "") -> str:
    """Produce a new key for the ``dictionary`` with a ``prefix``."""
    name = f"{prefix}{s}"
    if name in dictionary:
        return next_name(dictionary, s + 1, prefix)
    return name


def replace_variable_type(val, e_type):
    """Replace `#` by type in parametrized variable names.

    If ``val`` is a str representing an unevaluated expression, replace
    occurrences of ``#`` by ``e_type``.
    """
    if isinstance(val, str):
        return val.replace("#", f"{e_type}")
    return val
