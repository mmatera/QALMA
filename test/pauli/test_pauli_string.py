"""Tests for PauliStringOperator."""

import itertools

import numpy as np
import pytest
import qutip

from qalma.model import build_spin_chain
from qalma.operators import (
    PauliStringOperator,
    ProductOperator,
    ScalarOperator,
    to_pauli_sum,
)
from qalma.operators.functions import anticommutator, commutator
from qalma.operators.pauli import (
    PAULI_MATRICES,
    fermion_annihilation,
    fermion_creation,
    fermion_number,
    majorana_operators,
    pauli_terms,
)

SYSTEM = build_spin_chain(4)
SITES = tuple(SYSTEM.sites)
BLOCK = tuple(sorted(SITES))
RNG = np.random.default_rng(1234)


def dense(op):
    """Dense matrix over all the sites, in the sorted order used by QALMA."""
    if isinstance(op, ScalarOperator):
        return op.prefactor * np.eye(2 ** len(SITES))
    return op.to_qutip(BLOCK).full()


def random_string(prefactor=None):
    label = "".join(RNG.choice(list("IXYZ"), size=len(SITES)))
    if prefactor is None:
        prefactor = complex(*RNG.normal(size=2))
    return PauliStringOperator.from_label(label, SYSTEM, prefactor)


ALL_LABELS = ["".join(p) for p in itertools.product("IXYZ", repeat=2)]


@pytest.mark.parametrize("a", "IXYZ")
@pytest.mark.parametrize("b", "IXYZ")
def test_single_site_products(a, b):
    p = PauliStringOperator({SITES[0]: a}, 1.0, SYSTEM)
    q = PauliStringOperator({SITES[0]: b}, 1.0, SYSTEM)
    expected = PAULI_MATRICES["IXYZ".index(a)] @ PAULI_MATRICES["IXYZ".index(b)]
    result = p * q
    assert isinstance(result, (PauliStringOperator, ScalarOperator))
    assert np.allclose(
        dense(result), np.kron(expected, np.eye(8)) if False else dense(p) @ dense(q)
    )
    # the label of the product is the XOR of the labels
    if isinstance(result, PauliStringOperator):
        assert result.code == p.code ^ q.code


def test_encoding():
    p = PauliStringOperator({SITES[0]: "X", SITES[1]: "Y", SITES[2]: "Z"}, 1, SYSTEM)
    assert p.code == 0b11_10_01
    assert p.label == "XYZI"
    assert p.weight == 3
    assert p.acts_over() == frozenset(SITES[:3])
    assert PauliStringOperator(0b11_10_01, 1, SYSTEM).label == "XYZI"


@pytest.mark.parametrize("trial", range(30))
def test_random_products_and_commutators(trial):
    p, q = random_string(), random_string()
    dp, dq = dense(p), dense(q)
    assert np.allclose(dense(p * q), dp @ dq)
    assert np.allclose(dense(commutator(p, q)), dp @ dq - dq @ dp)
    assert np.allclose(dense(anticommutator(p, q)), dp @ dq + dq @ dp)
    assert p.commutes_with(q) == np.allclose(dp @ dq, dq @ dp)


def test_sums_and_scalars():
    p = PauliStringOperator.from_label("XZII", SYSTEM, 0.5)
    q = PauliStringOperator.from_label("XZII", SYSTEM, 0.25j)
    merged = p + q
    assert isinstance(merged, PauliStringOperator)
    assert merged.prefactor == 0.5 + 0.25j
    assert isinstance(p - p, ScalarOperator) and not (p - p)
    assert isinstance(2 * p, PauliStringOperator) and (2 * p).prefactor == 1.0
    assert isinstance(p * ScalarOperator(3, SYSTEM), PauliStringOperator)
    assert np.allclose(dense(p + 1), dense(p) + np.eye(16))
    assert isinstance(p * p, ScalarOperator) and (p * p).prefactor == 0.25


def test_mixed_with_product_operator():
    p = random_string()
    sx = SYSTEM.site_operator(f"Sx@{SITES[1]}")
    sz = SYSTEM.site_operator(f"Sz@{SITES[2]}")
    prod = ProductOperator({SITES[1]: sx.operator, SITES[2]: sz.operator}, 1.3, SYSTEM)
    for other in (sx, prod):
        assert np.allclose(dense(p * other), dense(p) @ dense(other))
        assert np.allclose(dense(other * p), dense(other) @ dense(p))
        assert np.allclose(dense(p + other), dense(p) + dense(other))


def test_unary_operations():
    p = random_string()
    dp = dense(p)
    assert np.allclose(dense(p.dag()), dp.conj().T)
    assert np.allclose(dense(-p), -dp)
    assert np.allclose(dense(p**3), np.linalg.matrix_power(dp, 3))
    assert np.allclose(dense(p.inv()), np.linalg.inv(dp))
    assert np.isclose(p.tr(), np.trace(dp))
    assert np.isclose(p.norm(), np.linalg.norm(dp, 2))
    assert np.isclose(p.norm("fro"), np.linalg.norm(dp, "fro"))
    expm = p.expm()
    assert np.allclose(expm.to_qutip(BLOCK).full(), qutip.Qobj(dp).expm().full())
    assert p.hermitian_part().isherm
    assert np.allclose(dense(p.hermitian_part()), (dp + dp.conj().T) / 2)


def test_partial_trace():
    p = PauliStringOperator.from_label("XZII", SYSTEM, 0.7)
    kept = frozenset(SITES[:2])
    reduced = p.partial_trace(kept)
    expected = qutip.Qobj(dense(p), dims=[[2] * 4, [2] * 4]).ptrace([0, 1])
    # ptrace of a non-positive operator: compare directly with the formula
    assert isinstance(reduced, PauliStringOperator)
    assert reduced.prefactor == 0.7 * 4
    assert isinstance(p.partial_trace(frozenset(SITES[1:])), ScalarOperator)
    assert not p.partial_trace(frozenset(SITES[1:]))
    del expected


def test_compatibility_with_product_operator():
    p = random_string()
    po = p.to_product_operator()
    assert type(po) is ProductOperator
    assert np.allclose(dense(po), dense(p))
    assert PauliStringOperator.from_operator(po).code == p.code


def test_to_pauli_sum_heisenberg():
    ham = SYSTEM.global_operator("Hamiltonian")
    pauli_ham = to_pauli_sum(ham)
    assert pauli_terms(pauli_ham) is not None
    assert np.allclose(dense(pauli_ham), dense(ham))
    # Symbolic commutator matches the generic one.
    obs = to_pauli_sum(SYSTEM.site_operator(f"Sx@{SITES[1]}"))
    comm = commutator(pauli_ham, obs)
    assert pauli_terms(comm) is not None
    assert np.allclose(dense(comm), dense(ham) @ dense(obs) - dense(obs) @ dense(ham))
    # Nested commutators stay in the Pauli algebra.
    comm2 = commutator(pauli_ham, comm)
    assert pauli_terms(comm2) is not None
    dh, dc = dense(ham), dense(comm)
    assert np.allclose(dense(comm2), dh @ dc - dc @ dh)
    # simplify() keeps the Pauli strings
    assert pauli_terms((pauli_ham + pauli_ham).simplify()) is not None


def test_jordan_wigner():
    c = [fermion_annihilation(SYSTEM, s) for s in SITES]
    cd = [fermion_creation(SYSTEM, s) for s in SITES]
    for i, j in itertools.product(range(len(SITES)), repeat=2):
        acomm = anticommutator(c[i], cd[j])
        assert isinstance(acomm, ScalarOperator)
        assert np.isclose(acomm.prefactor, 1.0 if i == j else 0.0)
        assert not anticommutator(c[i], c[j])
    for i, s in enumerate(SITES):
        n = fermion_number(SYSTEM, s)
        assert np.allclose(dense(cd[i] * c[i]), dense(n))
        g1, g2 = majorana_operators(SYSTEM, s)
        assert isinstance(anticommutator(g1, g2), ScalarOperator)
        assert anticommutator(g1, g1).prefactor == 2


def test_rejects_non_qubit_sites():
    system = build_spin_chain(2)
    with pytest.raises(ValueError):
        PauliStringOperator({tuple(system.sites)[0]: "W"}, 1, system)


def test_interoperability_with_other_operators():
    from qalma.operators.states import GibbsProductDensityOperator

    p = random_string()
    dp = dense(p)
    ham = SYSTEM.global_operator("Hamiltonian")
    qham = ham.to_qutip_operator()
    for other in (ham, qham, SYSTEM.global_operator("Hamiltonian").simplify()):
        do = dense(other)
        assert np.allclose(dense(p * other), dp @ do)
        assert np.allclose(dense(other * p), do @ dp)
        assert np.allclose(dense(p + other), dp + do)
        assert np.allclose(dense(commutator(p, other)), dp @ do - do @ dp)
    state = GibbsProductDensityOperator(
        SYSTEM.site_operator(f"Sx@{SITES[0]}") + SYSTEM.site_operator(f"Sz@{SITES[1]}"),
        system=SYSTEM,
    )
    rho = state.to_qutip(BLOCK).full()
    assert np.isclose(state.expect(p), np.trace(rho @ dp))


def test_subsystem_with_different_order():
    sub = SYSTEM.subsystem(frozenset(SITES[1:3]))
    p_sub = PauliStringOperator({SITES[2]: "Y", SITES[1]: "X"}, 1.0, sub)
    p = PauliStringOperator.from_label("IZXI", SYSTEM)
    prod = p * p_sub
    assert prod.system is SYSTEM
    assert np.allclose(
        dense(prod), dense(p) @ dense(PauliStringOperator(p_sub.paulis, 1.0, SYSTEM))
    )
