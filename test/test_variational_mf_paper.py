"""Benchmarks for the variational mean-field approximation — paper figures.

Two families of tests:

1. Validation against exact diagonalization (L <= 8)
   Models: Ising transverse, XX, XXX, XYZ
   Lattice: "open chain" (simple1d unit cell, NN bonds only)

2. Convergence vs numfields — J1-J2 frustrated chain
   Lattice: "open chain" with "complex1d" unit cell
             bond type 0 = NN (J1), bond type 1 = NNN (J2)
   Requires 'complex1d' unit cell in lattices.xml (already present).

Usage
-----
Full run (saves JSON for plotting):
    python test_variational_mf_paper.py

Quick validation (pytest, no BENCHMARKS flag needed):
    pytest test_variational_mf_paper.py -v

Full benchmark suite:
    BENCHMARKS=1 pytest test_variational_mf_paper.py -v \\
        --benchmark-json=results.json
"""

import json
import os
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pytest

from qalma import graph_from_alps_xml, model_from_alps_xml
from qalma.meanfield import (
    compute_t_score,
    compute_variance,
    variational_quadratic_mfa,
)
from qalma.model import SystemDescriptor
from qalma.operators.states import ProductDensityOperator

# ---------------------------------------------------------------------------
# Incremental JSONL output
# ---------------------------------------------------------------------------


def _append_jsonl(path, records):
    """Append *records* to a JSON Lines file, one record per line.

    The file is created if it does not exist.  Each call flushes and syncs
    to disk so that partial results survive a crash or keyboard interrupt.

    Parameters
    ----------
    path : Path
        Destination ``.jsonl`` file.
    records : dict or list of dict
        One record or a sequence of records to append.
    """
    if isinstance(records, dict):
        records = [records]
    with open(path, "a") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


# ---------------------------------------------------------------------------
# System builders
# ---------------------------------------------------------------------------


def build_nn_chain(L: int, parms: dict) -> Tuple[SystemDescriptor, object]:
    """Spin-1/2 open chain with nearest-neighbor bonds only. Uses 'open chain'
    latticegraph (simple1d unit cell, bond type 0).

    Parameters
    ----------
    L : int
        Number of sites.
    parms : dict
        Model parameters. Keys: Jz, Jxy, Gamma, h, etc.
        Bond type 0 maps to J, Jz, Jxy in the 'spin' Hamiltonian.

    """
    graph = graph_from_alps_xml(name="open chain lattice", parms={"L": L, "a": 1})
    model = model_from_alps_xml(name="spin")
    system = SystemDescriptor(graph, model, parms)
    ham = system.global_operator("Hamiltonian")
    return system, ham


def build_j1j2_chain(L: int, J1: float, J2: float) -> Tuple[SystemDescriptor, object]:
    """Spin-1/2 J1-J2 open chain.
    Uses 'nnn open chain lattice' (complex1d unit cell), which has:
      - bond type 0: nearest neighbors  (J1)
      - bond type 1: next-nearest neighbors (J2).

    Parameters
    ----------
    L : int
        Number of sites.
    J1 : float
        Nearest-neighbor coupling. Negative = ferromagnetic.
    J2 : float
        Next-nearest-neighbor coupling.

    """
    parms = {
        "Jz": J1,
        "Jxy": J1,  # bond type 0 → NN
        "Jz'": J2,
        "Jxy'": J2,  # bond type 1 → NNN
    }
    graph = graph_from_alps_xml(name="nnn open chain lattice", parms={"L": L, "a": 1})
    model = model_from_alps_xml(name="spin")
    system = SystemDescriptor(graph, model, parms)
    ham = system.global_operator("Hamiltonian")
    # print("J1=",J1, "J2=",J2)
    # print(ham)
    return system, ham


# ---------------------------------------------------------------------------
# Helpers: reference quantities
# ---------------------------------------------------------------------------


def exact_free_energy(ham, system, beta: float) -> float:
    """Exact free energy in units of k = beta*H: -log Z = -log Tr[exp(-beta*H)].

    This is the value expected by ``compute_t_score`` as ``_f_exact``,
    and is in the same units as ``mf_free_energy`` = Tr[sigma(log sigma + beta*H)].

    Uses a numerically stable shift by the ground-state energy e0:

        log_Z_shift = log sum_i exp(-beta*(E_i - e0)) = log Z + beta*e0

    so  -log Z = -log_Z_shift + beta*e0.

    Only feasible for L <= 10 (Hilbert-space dim = 2^L).
    """
    sites = tuple(sorted(system.sites.keys()))
    ham_qutip = ham.to_qutip(sites)
    evals = ham_qutip.eigenenergies()
    e0 = evals.min()
    log_Z_shift = np.log(np.exp(-beta * (evals - e0)).sum())
    # log_Z_shift = log Z + beta*e0  =>  -log Z = -log_Z_shift + beta*e0
    return -log_Z_shift / beta + e0


def mf_free_energy(sigma, ham, beta: float) -> float:
    """F_{mf} = Tr[sigma(log sigma + beta H)].

    Equals S_rel(sigma || e^{-beta H}) up to the constant log Z,
    which is sufficient for comparing approximations at fixed (H, beta).
    """
    return sigma.variational_free_energy(beta * ham) / beta


def t_score(sigma, ham, beta, f_exact: Optional[float]):
    """Compute the T-score associated to ham.

    ``f_exact`` is the Helmholtz free energy F = -(1/beta)*log Z returned by
    ``exact_free_energy``.  ``compute_t_score`` expects log Z = -beta * F,
    so we convert here.

    Parameters
    ----------
    sigma : ProductDensityOperator
        Trial state.
    ham : Operator
        Hamiltonian (without beta).
    beta : float
        Inverse temperature.
    f_exact : float or None
        -log Tr[exp(-beta*H)], as returned by ``exact_free_energy``.
        If None, returns None (T-score not computed).
    """
    if f_exact is None:
        return None
    betaf_exact = beta * f_exact  # log Tr[e^{-beta*H}]
    return float(np.real(compute_t_score(sigma, ham * beta, betaf_exact)[0]))


# ---------------------------------------------------------------------------
# Family 1 — Validation against exact diagonalization
# ---------------------------------------------------------------------------

# (label, parms, L_list, beta_list)

MF_LENGTH_TESTS = [8, 12, 16]
LENGTHS_FOR_EXACT_TESTS = [4, 6, 8]
BETAS = [0.5, 1.0, 2.0, 5.0]

EXACT_CASES = [
    (
        "Pure transverse field (J=0, Gamma=1.)",
        {"Jz": 0.0, "Jxy": 0.0, "Gamma": 1.0},
        [4],
        [0.01],
    ),
    (
        "Ising transverse (Gamma=0.5J)",
        {"Jz": 1.0, "Jxy": 0.0, "Gamma": 0.5},
        LENGTHS_FOR_EXACT_TESTS,
        BETAS,
    ),
    (
        "Ising transverse critical (Gamma=J)",
        {"Jz": 1.0, "Jxy": 0.0, "Gamma": 1.0},
        LENGTHS_FOR_EXACT_TESTS,
        [0.5, 1.0, 2.0],
    ),
    (
        "XX chain",
        {"Jz": 0.0, "Jxy": 1.0},
        LENGTHS_FOR_EXACT_TESTS,
        BETAS,
    ),
    (
        "XXX Heisenberg AFM",
        {"Jz": 1.0, "Jxy": 1.0},
        LENGTHS_FOR_EXACT_TESTS,
        BETAS,
    ),
    (
        "XXX Heisenberg FM",
        {"Jz": -1.0, "Jxy": -1.0},
        LENGTHS_FOR_EXACT_TESTS,
        BETAS,
    ),
    (
        "XYZ anisotropic (Jz=1, Jxy=0.5)",
        {"Jz": 1.0, "Jxy": 0.5},
        LENGTHS_FOR_EXACT_TESTS,
        BETAS,
    ),
]


@pytest.mark.parametrize("L", LENGTHS_FOR_EXACT_TESTS)
@pytest.mark.parametrize("beta", BETAS)
@pytest.mark.parametrize("Gamma", [0.5, 1.0, 2.0])
def test_exact_free_energy_noninteracting(L, beta, Gamma):
    """Validate exact_free_energy against the analytic result for a pure
    transverse field H = Gamma * sum_i sigma^x_i (no interactions).

    For non-interacting spins the partition function factorises:

        Z = (2 cosh(beta * Gamma))^L
        => -log Z = -L * log(2 * cosh(beta * Gamma/2))

    This case also serves as a sanity check that F_exact <= F_mixed,
    since the mixed state has F_mixed = -L * log(2) and
    cosh(beta*Gamma) >= 1 implies -log Z <= -L*log(2).

    Furthermore, since H is a sum of single-site terms, the Gibbs state
    is an exact product state, so the mean-field approximation is exact:
    F_mf == F_exact and T_score == 0.
    """
    import numpy as np

    parms = {"Jz": 0.0, "Jxy": 0.0, "Gamma": Gamma}
    system, ham = build_nn_chain(L, parms)

    # --- Analytic reference -----------------------------------------------
    f_analytic = -L * np.log(2 * np.cosh(beta * Gamma / 2.0)) / beta

    # --- exact_free_energy ------------------------------------------------
    f_computed = exact_free_energy(ham, system, beta)
    assert abs(f_computed - f_analytic) < 1e-10, (
        f"exact_free_energy mismatch: got {f_computed:.8f}, "
        f"expected {f_analytic:.8f} (L={L}, beta={beta}, Gamma={Gamma})"
    )

    # --- F_exact <= F_mixed -----------------------------------------------
    f_mixed = -L * np.log(2) / beta  # = mf_free_energy(sigma_mixed, ham, beta)
    assert f_computed <= f_mixed + 1e-10, (
        f"F_exact={f_computed:.6f} > F_mixed={f_mixed:.6f} "
        f"(L={L}, beta={beta}, Gamma={Gamma})"
    )

    # --- MF is exact: F_mf == F_exact and T_score == 0 -------------------
    sigma_var = variational_quadratic_mfa(
        beta * ham,
        numfields=0,
        max_self_consistent_steps=100,
    )
    f_mf = mf_free_energy(sigma_var, ham, beta)
    assert abs(f_mf - f_analytic) < 1e-6, (
        f"F_mf={f_mf:.8f} != F_exact={f_analytic:.8f} "
        f"for non-interacting model (L={L}, beta={beta}, Gamma={Gamma})"
    )

    ts = t_score(sigma_var, ham, beta, f_computed)
    assert ts < 1e-6, (
        f"T_score={ts:.2e} should be ~0 for non-interacting model "
        f"(L={L}, beta={beta}, Gamma={Gamma})"
    )

    print(
        f"  L={L}  beta={beta}  Gamma={Gamma}: "
        f"F_analytic={f_analytic:.6f}  F_computed={f_computed:.6f}  "
        f"F_mf={f_mf:.6f}  T_score={ts:.2e}"
    )


@pytest.mark.parametrize("label,parms,L_list,beta_list", EXACT_CASES)
@pytest.mark.parametrize("L", LENGTHS_FOR_EXACT_TESTS)
@pytest.mark.parametrize("beta", BETAS)
def test_exact_validation(label, parms, L_list, beta_list, L, beta):
    """Variational MF must improve over the fully mixed state.

    The mixed state is the trivial upper bound on F_mf.
    """
    if L not in L_list or beta not in beta_list:
        pytest.skip(f"Not in test matrix for {label}")

    system, ham = build_nn_chain(L, parms)
    sigma_mixed = ProductDensityOperator({}, system=system)

    sigma_var = variational_quadratic_mfa(
        beta * ham,
        numfields=6,
        max_self_consistent_steps=30,
    )
    sigma_sc = variational_quadratic_mfa(
        beta * ham,
        numfields=0,
        max_self_consistent_steps=100,
    )

    f_mixed = mf_free_energy(sigma_mixed, ham, beta)
    f_sc = mf_free_energy(sigma_sc, ham, beta)
    f_var = mf_free_energy(sigma_var, ham, beta)

    print(f"\n{label}  L={L}  beta={beta}")
    print(f"  F_mf mixed: {f_mixed:.6f}")
    print(f"  F_mf SC:    {f_sc:.6f}")
    print(f"  F_mf var:   {f_var:.6f}")
    print(f"  Delta(var vs SC):   {f_sc - f_var:.6f}")

    assert (
        f_var <= f_mixed + 1e-6
    ), f"Variational ({f_var:.4f}) not better than mixed ({f_mixed:.4f})"


# ---------------------------------------------------------------------------
# Family 2 — F_mf vs numfields for the J1-J2 chain
# ---------------------------------------------------------------------------

# (J2/J1, label)
J1J2_CASES = [
    (0.0, "no frustration (J2=0)"),
    (0.2, "weak frustration"),
    (0.4, "moderate frustration"),
    (0.5, "maximum frustration (critical)"),
    (0.6, "spiral phase"),
    (0.8, "strong J2"),
    (1.0, "J1=J2"),
]

NUMFIELDS_LIST = [1, 2, 3, 4, 6, 8, 10]
J1 = -1.0  # AFM nearest-neighbor


def _var_f(sigma, ham, beta: float) -> float:
    """Var_sigma[hat{F}] = <(beta*H - kappa)^2>_sigma - <beta*H - kappa>_sigma^2.

    Delegates to :func:`compute_variance`, which does not require F_exact.
    """
    return compute_variance(sigma, ham * beta)


def worker_numfield_sweep(nf, system, ham, beta, var_f_sc, sigma_ref, J2_ratio):
    """Worker that does single evaluation for numfield sweep"""
    t0 = time.perf_counter()
    Sz_ops = [system.site_operator("Sz", s) for s in system.sites]
    sigma = variational_quadratic_mfa(
        beta * ham,
        numfields=nf,
        sigma_ref=sigma_ref,
        max_self_consistent_steps=30,
    )
    elapsed = time.perf_counter() - t0

    f_mf = mf_free_energy(sigma, ham, beta)
    var_f = _var_f(sigma, ham, beta)
    var_f_ratio = var_f / var_f_sc if var_f_sc > 1e-15 else None
    mag = [float(np.real(sigma.expect(sz))) for sz in Sz_ops]

    ratio_str = f"{var_f_ratio:.4f}" if var_f_ratio is not None else "  n/a"
    print(
        f"  J2/J1={J2_ratio:.2f}  L={L}  beta={beta}  "
        f"nf={nf:2d}:  F_mf={f_mf:.6f}  Var={var_f:.4g}  "
        f"Var_ratio={ratio_str}  t={elapsed:.2f}s  "
        f"<Sz>=[{', '.join(f'{m:.3f}' for m in mag[:5])}...]"
    )

    return {
        "J2_over_J1": J2_ratio,
        "L": L,
        "beta": beta,
        "numfields": nf,
        "f": f_mf,
        "var_f": var_f,
        "var_f_ratio": var_f_ratio,
        "magnetization": mag,
        "time": elapsed,
    }


def run_numfields_sweep(
    J2_ratio: float,
    L: int,
    beta: float,
    numfields_list: List[int] = NUMFIELDS_LIST,
) -> List[dict]:
    """Sweep numfields for a J1-J2 chain, using warm start between runs.

    For large systems where F_exact is not available, the quality of each
    variational state sigma_m is characterised by two quantities:

    * ``var_f``       Var_{sigma_m}[hat{F}], the variance of the log-ratio
                      operator hat{F} = k - kappa_m under sigma_m.  This is
                      the numerator of the T-score and is available without
                      knowing F_exact.

    * ``var_f_ratio`` Var_{sigma_m}[hat{F}] / Var_{sigma_SC}[hat{F}], the
                      ratio of the variance of the variational state to that
                      of the self-consistent (nf=0) baseline.  A value < 1
                      means the variational method reduces the fluctuations of
                      hat{F} relative to the SC solution; a value approaching
                      0 indicates near-exact convergence in distribution.

    Both quantities are intensive (scale as O(L) for local Hamiltonians) and
    do not require F_exact, making them suitable for large systems.

    Returns list of result dicts suitable for JSON serialization.
    """
    import concurrent.futures

    J2 = J2_ratio * abs(J1)
    system, ham = build_j1j2_chain(L, J1, J2)

    # --- Self-consistent baseline (nf=0) -----------------------------------
    t0 = time.perf_counter()
    sigma_sc = variational_quadratic_mfa(
        beta * ham,
        numfields=0,
        max_self_consistent_steps=100,
    )
    t_sc = time.perf_counter() - t0
    f_sc = mf_free_energy(sigma_sc, ham, beta)
    var_f_sc = _var_f(sigma_sc, ham, beta)

    print(
        f"  J2/J1={J2_ratio:.2f}  L={L}  beta={beta}  "
        f"nf= 0 (SC):  F_mf={f_sc:.6f}  Var={var_f_sc:.4g}  t={t_sc:.2f}s"
    )

    results = []
    sigma_ref = sigma_sc  # warm start from SC solution

    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_nf = {
            executor.submit(
                worker_numfield_sweep,
                nf,
                system,
                ham,
                beta,
                var_f_sc,
                sigma_ref,
                J2_ratio,
            ): nf
            for nf in numfields_list
        }

        for future in concurrent.futures.as_completed(future_to_nf):
            nf_val = future_to_nf[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as exc:
                print(f"nf_val={nf_val} generated an exception: {exc}")

    return results


@pytest.mark.skipif(
    not os.environ.get("BENCHMARKS"),
    reason="set BENCHMARKS=1 to run",
)
@pytest.mark.parametrize("J2_ratio,label", J1J2_CASES)
@pytest.mark.parametrize("L", MF_LENGTH_TESTS)
@pytest.mark.parametrize("beta", BETAS)
def test_numfields_convergence(J2_ratio, label, L, beta):
    """F must be non-increasing as numfields grows.

    Tolerance of 1e-4 allows for numerical noise in the optimizer.
    """
    print(f"\n--- {label}  L={L}  beta={beta} ---")
    results = run_numfields_sweep(J2_ratio, L, beta)
    fs = [r["f"] for r in results]
    for i in range(1, len(fs)):
        nf_prev = results[i - 1]["numfields"]
        nf_curr = results[i]["numfields"]
        assert fs[i] <= fs[i - 1] + 1e-4, (
            f"F increased: nf={nf_prev} → {nf_curr}  " f"({fs[i-1]:.5f} → {fs[i]:.5f})"
        )


# ---------------------------------------------------------------------------
# Main: full benchmark run, saves results to JSON for plotting
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    output_dir = Path("benchmark_results")
    output_dir.mkdir(exist_ok=True)

    out_exact = output_dir / "exact_validation_2body.jsonl"
    out_numfields = output_dir / "numfields_convergence_2body.jsonl"

    all_results = {"exact_validation": [], "numfields_convergence": []}

    # ---- Family 1 --------------------------------------------------------
    print("=" * 70)
    print("Family 1: Validation against exact diagonalization")
    print("=" * 70)

    for label, parms, L_list, beta_list in EXACT_CASES:
        for L in L_list:
            for beta in beta_list:
                print(f"\n{label}  L={L}  beta={beta}")
                system, ham = build_nn_chain(L, parms)
                sigma_mixed = ProductDensityOperator({}, system=system)

                t0 = time.perf_counter()
                sigma_var = variational_quadratic_mfa(
                    beta * ham, numfields=6, max_self_consistent_steps=30
                )
                t_var = time.perf_counter() - t0

                t0 = time.perf_counter()
                sigma_sc = variational_quadratic_mfa(
                    beta * ham, numfields=0, max_self_consistent_steps=100
                )
                t_sc = time.perf_counter() - t0

                # F_exact = -log Tr[exp(-beta*H)], same units as F_mixed/sc/var.
                # This is what compute_t_score expects as _f_exact.
                F_exact = exact_free_energy(ham, system, beta) if L <= 8 else None
                print("   @@ build row values")
                assert F_exact < 0
                row = {
                    "label": label,
                    "params": parms,
                    "L": L,
                    "beta": beta,
                    "F_exact": F_exact,
                    "F_mixed": mf_free_energy(sigma_mixed, ham, beta),
                    "F_sc": mf_free_energy(sigma_sc, ham, beta),
                    "F_variational": mf_free_energy(sigma_var, ham, beta),
                    "T_score_mixed": t_score(sigma_mixed, ham, beta, F_exact),
                    "T_score_sc": t_score(sigma_sc, ham, beta, F_exact),
                    "T_score_variational": t_score(sigma_var, ham, beta, F_exact),
                    "time_variational": t_var,
                    "time_sc": t_sc,
                }
                all_results["exact_validation"].append(row)
                _append_jsonl(out_exact, row)

                print(
                    f"  F [beta units]: mixed={row['F_mixed']:.4f}  "
                    f"SC={row['F_sc']:.4f}  "
                    f"var={row['F_variational']:.4f}  "
                    f"(t_var={t_var:.1f}s  t_sc={t_sc:.1f}s)"
                    + (f"  F_exact={F_exact:.4f}" if F_exact is not None else "")
                )

    # ---- Family 2 --------------------------------------------------------
    print("\n" + "=" * 70)
    print("Family 2: F vs numfields  (J1-J2 frustrated chain)")
    print("=" * 70)

    for J2_ratio, label in J1J2_CASES:
        for L in MF_LENGTH_TESTS:
            for beta in BETAS:
                print(f"\n{label}  L={L}  beta={beta}")
                try:
                    rows = run_numfields_sweep(J2_ratio, L, beta)
                    all_results["numfields_convergence"].extend(rows)
                    _append_jsonl(out_numfields, rows)
                except Exception as exc:
                    print(f"  FAILED: {exc}")

    # ---- Results summary -------------------------------------------------
    # Incremental JSONL files were written after each block above.
    print("\nResults saved incrementally:")
    print(f"  {out_exact}")
    print(f"  {out_numfields}")

    # ---- Summary table ---------------------------------------------------
    print("\n--- Convergence summary (L=8, beta=2.0) ---")
    print(
        f"{'Frustration':45s} "
        f"{'F(nf=1)':>10} {'F(nf=4)':>10} {'F(nf=10)':>10}  "
        f"{'R(nf=1)':>10} {'R(nf=4)':>10} {'R(nf=10)':>10}"
    )
    print(f"  {'':43s} {'--- F [beta*E] ---':>32}   {'--- Var_m/Var_SC ---':>32}")
    for J2_ratio, label in J1J2_CASES:
        rows = [
            r
            for r in all_results["numfields_convergence"]
            if r["J2_over_J1"] == J2_ratio and r["L"] == 8 and r["beta"] == 2.0
        ]
        if not rows:
            continue
        by_nf_f = {r["numfields"]: r["f"] for r in rows}
        by_nf_r = {r["numfields"]: r.get("var_f_ratio") for r in rows}

        def _fmt_f(nf):
            v = by_nf_f.get(nf, float("nan"))
            return f"{v:.4f}" if v == v else "  nan"

        def _fmt_r(nf):
            v = by_nf_r.get(nf)
            return f"{v:.4f}" if v is not None else "   --"

        print(
            f"  {label:43s} "
            f"{_fmt_f(1):>10} {_fmt_f(4):>10} {_fmt_f(10):>10}  "
            f"{_fmt_r(1):>10} {_fmt_r(4):>10} {_fmt_r(10):>10}"
        )
