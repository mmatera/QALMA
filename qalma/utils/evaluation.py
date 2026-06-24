"""
Evaluation and access to xml libraries.
---------------------------------------

Utility functions to import and process ALPS specification files.

"""

import logging

import numpy as np

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


def eval_expr(expr: str, parms: dict):
    """Evaluate the expression `expr` using ``parms``.

    The function uses the Python ``eval`` method using the
    ``parms`` dict as a context.

    ``expr`` can include python`s arithmetic expressions, and some
    elementary functions.
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
                logging.warning("A recursion error happens evaluating `%s`.", val)
                raise
        if not changed:
            break
    parms.update(default_parms)
    try:
        result = eval(expr, parms)
        return result
    except NameError:
        pass
    except TypeError as exc:
        logging.warning("Type Error. Undefined variables in [%s] in %s.", exc, expr)
        return None
    except SyntaxError:
        logging.error(
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
