"""Compatibility with the files written by released versions of the format.

``test/data/simulation_format1.h5`` was written by QALMA 1.0 with
``test/data/make_reference_file.py``. This test must keep passing, without
replacing the file, in every later version: a simulation stored today has to
be readable in the future.
"""

import datetime
import math
from pathlib import Path

import h5py
import numpy as np

from qalma.evolution import Simulation
from qalma.io import FORMAT_VERSION
from qalma.operators.states import GibbsDensityOperator

REFERENCE_FILE = Path(__file__).parents[1] / "data" / "simulation_format1.h5"

STATE_CLASSES = [
    "GibbsDensityOperator",
    "GibbsProductDensityOperator",
    "ProductDensityOperator",
    "MixtureDensityOperator",
    "QutipDensityOperator",
    "RPAGibbsDensityOperator",
    "QutipDensityOperator",
    "SumOperator",
]
OPERATOR_CLASSES = {
    "scalar": "ScalarOperator",
    "local": "LocalOperator",
    "product": "ProductOperator",
    "sum": "SumOperator",
    "one_body": "OneBodyOperator",
    "qutip": "QutipOperator",
    "quadratic": "QuadraticFormOperator",
}
# Some of the values stored in the file, to detect errors that affect in
# the same way the states and the observables.
ENERGY = [-0.43249454706689594, 0.5052727510921726, 0.0, -0.10812363676672399]
SX_0 = [0.0, -0.06885630425691658, 0.2, 0.15]


def test_the_file_is_the_released_one():
    with h5py.File(REFERENCE_FILE, "r") as h5file:
        assert h5file.attrs["format"] == "qalma-simulation"
        assert h5file.attrs["format_version"] == 1
        assert h5file.attrs["qalma_version"].startswith("1.0.0")
    assert FORMAT_VERSION >= 1


def test_load_reference_file():
    simulation = Simulation.load_hdf5(REFERENCE_FILE)
    system = simulation.system
    sites = list(system.sites)
    assert sites == ["1[0]", "1[1]", "1[2]"]
    assert system.dimensions == {site: 2 for site in sites}
    assert system.name == "spin on chain lattice"

    parameters = simulation.parameters
    assert parameters["system"] is system
    assert parameters["order"] == 3
    assert parameters["tol"] == 1.0e-3
    assert parameters["method"] == "reference file"
    assert parameters["update_condition"] is None
    assert parameters["nested"] == {
        "list": [1, 2, 3],
        "tuple": (1.5, "a", None),
        7: 1 + 2j,
    }
    stats = simulation.stats
    assert stats["run time"] == datetime.timedelta(seconds=12.5)
    assert stats["names"] == ["a", "b"]
    assert np.array_equal(stats["matrix"], np.arange(6.0).reshape(2, 3))
    assert np.allclose(stats["errors"], [0.01 * step for step in range(8)])
    assert simulation.time_span == [0.25 * step for step in range(8)]

    operators = parameters["operators"]
    assert {
        name: type(operator).__name__ for name, operator in operators.items()
    } == OPERATOR_CLASSES
    assert operators["scalar"].prefactor == 1.5 - 2j
    assert operators["product"].prefactor == 2 - 1j
    assert operators["qutip"].site_names == {"1[2]": 0, "1[0]": 1}
    assert [type(term).__name__ for term in operators["sum"].terms].count(
        "QutipOperator"
    ) >= 1

    # The operators stored in the file agree with the ones that the system
    # of the file builds from the description of the model.
    observables = parameters["observables"]
    for name, rebuilt in (
        ("energy", system.global_operator("Hamiltonian")),
        ("sz", system.global_operator("Sz")),
        ("sx_0", system.site_operator("Sx", "1[0]")),
        ("energy", operators["quadratic"].as_sum_of_products()),
    ):
        difference = (observables[name] - rebuilt).to_qutip()
        assert abs(difference.full()).max() < 1e-12, name

    states = list(simulation.states)
    assert [type(state).__name__ for state in states] == STATE_CLASSES
    assert states[0].normalized and states[0].free_energy != 0
    assert states[5].n_tau == 8
    assert sorted(states[6].system.sites) == sites[:2]

    # The expectation values of the stored states are the stored ones.
    for name, operator in observables.items():
        expected = simulation.expect_ops[name]
        assert len(expected) == len(states)
        for state, value in zip(states, expected):
            if math.isnan(value):
                continue
            if not hasattr(state, "expect"):
                state = GibbsDensityOperator(state)
            assert abs(state.expect(operator) - value) < 1e-10, name
    assert np.allclose(simulation.expect_ops["energy"][:4], ENERGY)
    assert np.allclose(simulation.expect_ops["sx_0"][:4], SX_0)
