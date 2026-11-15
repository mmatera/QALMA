"""Expressions of the models are data: they cannot run arbitrary code."""

import pytest

from qalma.model import build_system
from qalma.utils import eval_expr


@pytest.mark.parametrize(
    ["expr", "parms", "result"],
    [
        ("2*sqrt(x)+J", {"x": "4", "J": 1}, 5.0),
        ("-a**2/4 + b % 3", {"a": 2, "b": 7}, 0.0),
        ("(1+2j)*x", {"x": 2}, 2 + 4j),
        ("J", {"J": "Jz", "Jz": 1.5}, 1.5),
        ("t'", {"t'": 3}, 3),
        ("undefined*2", {}, "undefined*2"),
    ],
)
def test_arithmetic(expr, parms, result):
    assert eval_expr(expr, parms) == result


def test_functions_given_as_parameters_can_be_called():
    assert eval_expr("double(x)+1", {"double": lambda x: 2 * x, "x": 3}) == 7


@pytest.mark.parametrize(
    "expr",
    [
        "__import__('os').getcwd()",
        "().__class__.__bases__",
        "x.real",
        "x[0]",
        "[y for y in (1, 2)]",
        "(lambda: 1)()",
        "sqrt(x=1)",
        "f'{x}'",
        "x if x else 1",
        "x == 1",
        "(sqrt, cos)[0](1)",
    ],
)
def test_other_syntax_is_rejected(expr):
    with pytest.raises(SyntaxError):
        eval_expr(expr, {"x": 1})
    # Also when the expression is the value of a parameter.
    with pytest.raises(SyntaxError):
        eval_expr("a+1", {"a": expr, "x": 1})


@pytest.mark.parametrize("name", ["open", "eval", "exec", "__import__", "print"])
def test_builtins_are_not_available(name):
    assert eval_expr(f"{name}", {}) == name
    assert eval_expr(f"{name}(1)", {}) == f"{name}(1)"


def test_model_parameters_cannot_run_code(tmp_path):
    marker = tmp_path / "marker"
    with pytest.raises(SyntaxError):
        build_system("chain lattice", "spin", L=2, h=f"open({str(marker)!r}, 'w')")
    assert not marker.exists()
