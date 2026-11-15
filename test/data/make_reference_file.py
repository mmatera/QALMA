#!/usr/bin/env python
"""Write the reference file ``simulation_format1.h5``.

The file is part of the repository: ``test/io/test_reference_file.py``
checks that every future version of QALMA still reads it. It only has to be
written again to add a reference file for a *new* version of the format
(with a new name): never replace the file of a released version.

Usage::

    python test/data/make_reference_file.py
"""

import datetime
import sys
from pathlib import Path

import numpy as np
import qutip

from qalma.evolution import Simulation
from qalma.meanfield.rpa import RPAGibbsDensityOperator
from qalma.model import build_system
from qalma.operators import ProductOperator, QutipOperator, ScalarOperator
from qalma.operators.quadratic import build_quadratic_form_from_operator
from qalma.operators.states import (
    GibbsDensityOperator,
    GibbsProductDensityOperator,
    ProductDensityOperator,
)

FILENAME = Path(__file__).parent / "simulation_format1.h5"


def build_simulation() -> Simulation:
    """Build a simulation with all the classes that the format stores."""
    system = build_system("chain lattice", "spin", L=3, h=0.5, J=1)
    sites = tuple(system.sites)
    ham = system.global_operator("Hamiltonian")
    sz_total = system.global_operator("Sz")
    sx = [system.site_operator("Sx", site) for site in sites]
    sz = [system.site_operator("Sz", site) for site in sites]

    product = ProductOperator(
        {sites[0]: sx[0].operator, sites[2]: sz[2].operator}, 2 - 1j, system
    )
    block = QutipOperator(
        qutip.tensor(qutip.sigmay(), qutip.sigmax()),
        system,
        names={sites[2]: 0, sites[0]: 1},
        prefactor=0.25,
    )
    mixed_sum = sx[0] * sx[1] + product + block + ScalarOperator(0.5, system) + sz[1]
    operators = {
        "scalar": ScalarOperator(1.5 - 2j, system),
        "local": sx[0],
        "product": product,
        "sum": mixed_sum,
        "one_body": sz_total,
        "qutip": block,
        "quadratic": build_quadratic_form_from_operator(ham),
    }

    gibbs = GibbsDensityOperator(ham * 0.7, system)
    gibbs.normalize()
    gibbs_product = GibbsProductDensityOperator(sz_total + sx[0] * 0.3, system)
    product_state = ProductDensityOperator(
        {sites[0]: 0.5 * qutip.qeye(2) + 0.2 * qutip.sigmax()}, 1.0, system=system
    )
    states = [
        gibbs,
        gibbs_product,
        product_state,
        0.25 * gibbs + 0.75 * product_state,
        gibbs.to_qutip_operator(),
        RPAGibbsDensityOperator(ham * 0.2, system, n_tau=8),
        GibbsDensityOperator(ham, system).partial_trace(frozenset(sites[:2])),
        # Generators K(t), as stored by the MaxEnt solvers.
        ham + sz_total * 0.1,
    ]
    observables = {"energy": ham, "sz": sz_total, "sx_0": sx[0]}

    def expect(state, operator):
        if not operator.acts_over() <= set(state.system.sites):
            return float("nan")
        if not hasattr(state, "expect"):
            state = GibbsDensityOperator(state)
        return float(np.real(state.expect(operator)))

    expect_ops = {
        name: [expect(state, operator) for state in states]
        for name, operator in observables.items()
    }
    return Simulation(
        parameters={
            "system": system,
            "order": 3,
            "tol": 1.0e-3,
            "method": "reference file",
            "update_condition": None,
            "observables": observables,
            "operators": operators,
            "nested": {"list": [1, 2, 3], "tuple": (1.5, "a", None), 7: 1 + 2j},
        },
        stats={
            "errors": [0.01 * step for step in range(len(states))],
            "run time": datetime.timedelta(seconds=12.5),
            "matrix": np.arange(6.0).reshape(2, 3),
            "names": ["a", "b"],
        },
        time_span=[0.25 * step for step in range(len(states))],
        expect_ops=expect_ops,
        states=states,
    )


if __name__ == "__main__":
    if FILENAME.exists() and "--force" not in sys.argv:
        sys.exit(f"{FILENAME} exists: reference files must not be replaced.")
    build_simulation().save_hdf5(FILENAME, mode="w")
    print("written", FILENAME)
