from test.helper import check_equality

import numpy as np

from qalma.evolution import Simulation, qutip_me_solve
from qalma.model import build_system
from qalma.operators.states import GibbsDensityOperator


def test_store_and_recover_dicts(tmp_path):
    """Test stored dict routines."""
    from h5py import File
    from numpy import array as np_array, ndarray

    from qalma.evolution.simulation import load_hdf5_dict, store_hdf5_dict

    test_dict = {
        "str": "string",
        "bool": True,
        "float": 2.1,
        "int": 32,
        "tuple empty": tuple(),
        "tuple of numbers": (
            1,
            2,
            3,
        ),
        "tuple of bools": (
            True,
            False,
        ),
        "tuple of strings": (
            "s1",
            "s2",
            "hola!",
        ),
        "list empty": list(),
        "list of numbers": (1, 2, 3),
        "list of bools": [True, False],
        "list of strings": ["s1", "s2", "hola!"],
        "set of strings": set(("Hola", "no estoy")),
        "1d np array": np_array([1.0, 2.0]),
        "2d np array": np_array([[1.0, 2.0], [3.0, 4.0]]),
        "sub dict": {"a": 1, "b": 2.0},
    }
    with File(tmp_path / "test_dict.h5", "w") as f5:
        group = f5.create_group("test_dict")
        store_hdf5_dict(group, test_dict)
    with File(tmp_path / "test_dict.h5", "r") as f5:
        stored_dict = load_hdf5_dict(f5["test_dict"])

    def check_dict_equality(d1, d2):
        assert isinstance(d1, dict), "d1 is not a dict"
        assert isinstance(d2, dict), "d2 is not a dict"
        assert all(
            key in d2 for key in d1
        ), f"keys {','.join([k for k in d1 if k not in d2])} not in d2."
        assert all(
            key in d1 for key in d2
        ), f"keys {','.join([k for k in d2 if k not in d1])} not in d1."
        for key, val in d1.items():
            if isinstance(val, dict):
                check_dict_equality(val, d2[key])
            elif isinstance(val, ndarray):
                assert (val == d2[key]).all(), f"{val}!={d2[key]}"
            else:
                print(type(val), type(d2[key]))
                assert (
                    val == d2[key]
                ), f"key {key} have different values ({val}!={d2[key]})"

    check_dict_equality(stored_dict, test_dict)


def test_empty(tmp_path):
    sim_empty = Simulation(
        parameters={}, stats={}, time_span=[], expect_ops={}, states=[]
    )
    sim_empty.save_hdf5(str(tmp_path / "sol.h5"), mode="w")
    sim = Simulation.load_hdf5(str(tmp_path / "sol.h5"))
    compare_sim_objects(sim_empty, sim)
    assert len(sim.parameters) == 0
    assert len(sim.stats) == 0
    assert len(sim.time_span) == 0
    assert len(sim.expect_ops) == 0
    assert len(sim.states) == 0


def test_qutip_me_solve_states(tmp_path):
    system = build_system(
        parms={
            "L": 2,
            "a": 1,
            "h": 2,
            "J": 1,
        }
    )
    t_list = np.linspace(0, 3, 10)
    hamiltonian = system.global_operator("Hamiltonian")
    rho0 = GibbsDensityOperator(system.site_operator("Sz", "1[0]"))
    solution = qutip_me_solve(hamiltonian, rho0, t_list)

    solution.save_hdf5(str(tmp_path / "sol.h5"), mode="w")
    stored = Simulation.load_hdf5(str(tmp_path / "sol.h5"))
    compare_sim_objects(solution, stored)


def test_qutip_me_solve_expect(tmp_path):
    system = build_system(
        parms={
            "L": 2,
            "a": 1,
            "h": 2,
            "J": 1,
        }
    )
    t_list = np.linspace(0, 3, 10)
    hamiltonian = system.global_operator("Hamiltonian")
    rho0 = GibbsDensityOperator(system.site_operator("Sz", "1[0]"))
    solution = qutip_me_solve(
        hamiltonian,
        rho0,
        t_list,
        e_ops={
            "Sz_1": system.site_operator("Sz", "1[0]"),
            "Sz_2": system.site_operator("Sz", "1[1]"),
        },
    )
    solution.save_hdf5(str(tmp_path / "sol.h5"), mode="w")
    assert len(solution.states) == 0 or (
        solution.states[0].system is system
    ), "Check that the system was not modified after serialization."
    stored = Simulation.load_hdf5(str(tmp_path / "sol.h5"))
    compare_sim_objects(solution, stored)


def compare_sim_objects(solution, stored):
    assert len(solution.time_span) == len(stored.time_span)
    assert np.allclose(solution.time_span, stored.time_span)
    sol_stats = solution.stats
    sto_stats = stored.stats
    for key in sol_stats:
        assert key in sto_stats
        assert (
            sto_stats[key] == sol_stats[key]
        ), f"{key}  have different values {sto_stats[key]}!={sol_stats[key]}"

    for key in solution.parameters:
        assert key in stored.parameters
        assert stored.parameters[key] == solution.parameters[key], key

    system = solution.states[0].system if solution.states else None
    for state in stored.states:
        assert str(system) == str(state.system), f"{system}!={state.system}"
    assert all(
        key in stored.expect_ops for key in solution.expect_ops
    ), f"{key} is missing in the stored object."
    for key, val in solution.expect_ops.items():
        stored_val = stored.expect_ops[key]
        sim_val = np.array(val)
        try:
            assert check_equality(sim_val, stored_val)
        except AssertionError:
            print(f"{sim_val}!={stored_val} for {key}")
            print(type(sim_val), "=?=", type(stored_val))
            print(
                [
                    (
                        type(x),
                        type(y),
                    )
                    for x, y in zip(sim_val, stored_val)
                ]
            )
            raise

    for rho, sigma in zip(solution.states, stored.states):
        check_equality(rho, sigma)


# ---------------------------------------------------------------------------
#  Format 1
# ---------------------------------------------------------------------------

import importlib.util  # noqa: E402
import os  # noqa: E402
import pickle  # noqa: E402
import warnings  # noqa: E402
from pathlib import Path  # noqa: E402

import h5py  # noqa: E402
import pytest  # noqa: E402

from qalma.evolution import (  # noqa: E402
    SimulationHDF5,
    adaptive_projected_evolution,
    heisenberg_solve,
    projected_evolution,
    series_evolution,
)
from qalma.io import (  # noqa: E402
    HDF5FormatError,
    IncompleteStateWarning,
    PickleNotAllowedError,
)
from qalma.model import build_system_from_dims  # noqa: E402
from qalma.operators import ScalarOperator  # noqa: E402
from qalma.operators.states import GibbsProductDensityOperator  # noqa: E402

from ..helper import (  # noqa: E402
    HAMILTONIAN,
    SITES,
    SX_AB,
    SX_TOTAL,
    SYSTEM,
    SZ_TOTAL,
    TEST_CASES_STATES,
    check_operator_equality,
)

TOOLS = Path(__file__).parents[2] / "tools"


def make_simulation(n_states=4):
    """A simulation with operators and states of different classes."""
    times = [0.5 * step for step in range(4)]
    states = [
        GibbsDensityOperator(HAMILTONIAN * (1 + t) + SZ_TOTAL, SYSTEM) for t in times
    ]
    states[1] = HAMILTONIAN + SX_TOTAL * times[1]
    states[2] = GibbsProductDensityOperator(SZ_TOTAL * times[2], SYSTEM)
    return Simulation(
        parameters={"system": SYSTEM, "order": 3, "tol": 1e-3, "ham": HAMILTONIAN},
        stats={"method": "test", "errors": [0.0, 0.1, 0.2, 0.3][:n_states]},
        time_span=times[:n_states],
        expect_ops={
            0: [1.0, 0.5, 0.25, 0.125][:n_states],
            "sx": [1j, 2j, 3j, 4j][:n_states],
        },
        states=states[:n_states],
    )


def assert_same_simulation(original, loaded):
    assert loaded.time_span == list(original.time_span)
    assert loaded.stats == original.stats
    assert loaded.expect_ops == original.expect_ops
    assert list(loaded.parameters) == list(original.parameters)
    assert len(loaded.states) == len(original.states)
    for state, loaded_state in zip(original.states, loaded.states):
        assert type(loaded_state) is type(state)
        assert check_operator_equality(state.to_qutip(), loaded_state.to_qutip())


def find_pickles(filename):
    """Names of the members of an HDF5 file that can contain pickles."""
    found = []

    def visit(name, node):
        py_type = node.attrs.get("py_type", "")
        if py_type in ("pickle", "pickled_operator") or node.attrs.get("pickled"):
            found.append(name)
        elif isinstance(node, h5py.Dataset) and node.dtype.kind in "OV":
            if h5py.check_string_dtype(node.dtype) is None and name != "system/sites":
                found.append(name)

    with h5py.File(filename, "r") as h5file:
        h5file.visititems(visit)
    return found


def test_format1_layout(tmp_path):
    filename = tmp_path / "sim.h5"
    make_simulation().save_hdf5(filename)
    assert find_pickles(filename) == []
    with h5py.File(filename, "r") as h5file:
        assert h5file.attrs["format"] == "qalma-simulation"
        assert h5file.attrs["format_version"] == 1
        assert "qalma_version" in h5file.attrs
        assert set(h5file) == {
            "system",
            "parameters",
            "stats",
            "expect_ops",
            "time_span",
            "states",
        }
        assert h5file["time_span"][()].tolist() == [0.0, 0.5, 1.0, 1.5]
        assert list(h5file["states"]) == ["000000", "000001", "000002", "000003"]
        assert [g.attrs["type"] for g in h5file["states"].values()] == [
            "gibbs",
            "sum",
            "gibbs_product",
            "gibbs",
        ]
        assert [g.attrs["time"] for g in h5file["states"].values()] == [
            0.0,
            0.5,
            1.0,
            1.5,
        ]
        assert h5file["states/000000/k"].attrs["type"] == "sum"
        assert h5file["expect_ops/sx"].dtype == np.complex128
        assert h5file["stats/errors"][()].tolist() == [0.0, 0.1, 0.2, 0.3]
        assert h5file["parameters/order"][()] == 3


def test_format1_roundtrip(tmp_path):
    simulation = make_simulation()
    filename = tmp_path / "sim.h5"
    simulation.save_hdf5(filename)
    loaded = Simulation.load_hdf5(filename)
    assert isinstance(loaded, SimulationHDF5)
    assert loaded.format_version == 1
    assert_same_simulation(simulation, loaded)
    # Integer keys are kept.
    assert list(loaded.expect_ops) == [0, "sx"]
    # There is a single system, shared by everything in the file.
    assert loaded.system == SYSTEM
    assert loaded.parameters["system"] is loaded.system
    assert loaded.parameters["ham"].system is loaded.system
    assert all(state.system is loaded.system for state in loaded.states)
    # Simulation.load detects the format.
    assert_same_simulation(simulation, Simulation.load(filename))


def test_state_list(tmp_path):
    simulation = make_simulation()
    filename = tmp_path / "sim.h5"
    simulation.save_hdf5(filename)
    states = Simulation.load_hdf5(filename).states
    assert len(states) == 4
    assert type(states[1]) is type(simulation.states[1])
    assert check_operator_equality(states[-1].to_qutip(), states[3].to_qutip())
    assert [type(s) for s in states[1:3]] == [type(s) for s in simulation.states[1:3]]
    assert len(states[::2]) == 2
    for index in (4, -5):
        with pytest.raises(IndexError):
            states[index]


def test_state_list_append_extend_and_replace(tmp_path):
    simulation = make_simulation(2)
    filename = tmp_path / "sim.h5"
    simulation.save_hdf5(filename)
    loaded = Simulation.load_hdf5(filename)

    loaded.states.append(SX_TOTAL)
    loaded.states.extend([SZ_TOTAL, HAMILTONIAN])
    loaded.states[1] = SX_AB
    loaded.states[-1] = TEST_CASES_STATES["gibbs_sz"]
    with pytest.raises(IndexError):
        loaded.states[7] = SX_AB
    with pytest.raises(TypeError, match="Operator was expected"):
        loaded.states.append("not a state")

    with h5py.File(filename, "r") as h5file:
        assert list(h5file["states"]) == [f"{i:06d}" for i in range(5)]
    expected = [
        simulation.states[0],
        SX_AB,
        SX_TOTAL,
        SZ_TOTAL,
        TEST_CASES_STATES["gibbs_sz"],
    ]
    reloaded = Simulation.load_hdf5(filename)
    for state, loaded_state in zip(expected, reloaded.states, strict=True):
        assert type(loaded_state) is type(state)
        assert check_operator_equality(state.to_qutip(), loaded_state.to_qutip())


def test_append_to_empty_file(tmp_path):
    """The system is stored with the first object that needs it."""
    filename = tmp_path / "sim.h5"
    Simulation({}, {}, [], {}, []).save_hdf5(filename)
    loaded = Simulation.load_hdf5(filename)
    assert loaded.system is None
    loaded.states.append(SX_AB)
    assert loaded.system is SYSTEM
    other = ScalarOperator(1.0, build_system_from_dims({"a": 3}))
    with pytest.raises(ValueError, match="not in the system stored in the file"):
        loaded.states.append(other)
    reloaded = Simulation.load_hdf5(filename)
    assert len(reloaded.states) == 1
    assert reloaded.system == SYSTEM


def test_update_file_in_place(tmp_path):
    simulation = make_simulation()
    filename = tmp_path / "sim.h5"
    simulation.save_hdf5(filename)
    loaded = Simulation.load_hdf5(filename)
    loaded.expect_ops["sz"] = [1.0, 2.0]
    del loaded.expect_ops["sx"]
    loaded.stats["errors"].append(0.4)
    loaded.time_span.append(2.0)
    loaded.parameters["obs"] = SZ_TOTAL
    for args in ((), (filename,), (str(filename), "r+")):
        loaded.save_hdf5(*args)
        reloaded = Simulation.load_hdf5(filename)
        assert reloaded.expect_ops == {0: [1.0, 0.5, 0.25, 0.125], "sz": [1.0, 2.0]}
        assert reloaded.stats["errors"] == [0.0, 0.1, 0.2, 0.3, 0.4]
        assert reloaded.time_span == [0.0, 0.5, 1.0, 1.5, 2.0]
        assert len(reloaded.states) == 4
        assert reloaded.parameters["obs"].system is reloaded.system


def test_copy_to_another_file(tmp_path):
    simulation = make_simulation()
    simulation.save_hdf5(tmp_path / "sim.h5")
    loaded = Simulation.load_hdf5(tmp_path / "sim.h5")
    loaded.stats["extra"] = 1
    loaded.save_hdf5(tmp_path / "copy.h5")
    with pytest.raises(FileExistsError):
        loaded.save_hdf5(tmp_path / "copy.h5")
    copy = Simulation.load_hdf5(tmp_path / "copy.h5")
    assert copy.stats["extra"] == 1
    assert "extra" not in Simulation.load_hdf5(tmp_path / "sim.h5").stats
    assert_same_simulation(loaded, copy)


def test_save_modes_and_errors(tmp_path):
    simulation = make_simulation(2)
    filename = tmp_path / "sim.h5"
    simulation.save_hdf5(filename)
    with pytest.raises(FileExistsError, match="mode='w'"):
        simulation.save_hdf5(filename)
    simulation.save_hdf5(filename, mode="w")
    with pytest.raises(ValueError, match="mode"):
        simulation.save_hdf5(filename, mode="r+")
    # Errors are raised, not logged.
    with pytest.raises(OSError):
        simulation.save_hdf5(tmp_path / "missing" / "sim.h5")
    with pytest.raises(OSError):
        Simulation.load_hdf5(tmp_path / "missing.h5")
    with pytest.raises(OSError):
        Simulation.load(tmp_path / "missing.h5")


def test_failed_save_keeps_the_previous_file(tmp_path):
    filename = tmp_path / "sim.h5"
    simulation = make_simulation(2)
    simulation.save_hdf5(filename)
    simulation.parameters["callback"] = len
    with pytest.raises(TypeError, match=r"parameters\['callback'\].*allow_pickle"):
        simulation.save_hdf5(filename, mode="w")
    assert os.listdir(tmp_path) == ["sim.h5"]
    assert "callback" not in Simulation.load_hdf5(filename).parameters


def test_arbitrary_objects_need_allow_pickle(tmp_path):
    filename = tmp_path / "sim.h5"
    simulation = make_simulation(2)
    simulation.parameters["callback"] = len
    simulation.save_hdf5(filename, allow_pickle=True)
    assert find_pickles(filename) == ["parameters/callback"]
    with pytest.raises(PickleNotAllowedError, match="allow_pickle=True"):
        Simulation.load_hdf5(filename)
    loaded = Simulation.load_hdf5(filename, allow_pickle=True)
    assert loaded.parameters["callback"] is len
    assert_same_simulation(simulation, loaded)
    # In-place updates keep the setting used to open the file.
    loaded.save_hdf5()
    assert find_pickles(filename) == ["parameters/callback"]


def test_states_on_subsystems_and_other_systems(tmp_path):
    rho = TEST_CASES_STATES["gibbs_H"]
    reduced = rho.partial_trace(frozenset(SITES[:2]))
    assert reduced.system is not SYSTEM
    simulation = Simulation({}, {}, [0.0, 1.0], {}, [reduced, rho])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", IncompleteStateWarning)
        simulation.save_hdf5(tmp_path / "sim.h5")
        loaded = Simulation.load_hdf5(tmp_path / "sim.h5")
        states = list(loaded.states)
    assert loaded.system == SYSTEM
    assert set(states[0].system.sites) == set(reduced.system.sites)
    assert states[0].system is loaded.system.subsystem(frozenset(SITES[:2]))
    assert states[1].system is loaded.system

    other = ScalarOperator(1.0, build_system_from_dims({"a": 3}))
    simulation = Simulation({}, {}, [0.0, 1.0], {}, [SX_AB, other])
    with pytest.raises(ValueError, match="same system"):
        simulation.save_hdf5(tmp_path / "other.h5")


def test_gibbs_states_with_symmetry_projections(tmp_path):
    rho = TEST_CASES_STATES["gibbs_H"]
    simulation = Simulation({}, {}, [0.0], {}, [rho])
    with pytest.warns(IncompleteStateWarning):
        simulation.save_hdf5(tmp_path / "sim.h5")
    with pytest.warns(IncompleteStateWarning):
        loaded = Simulation.load_hdf5(tmp_path / "sim.h5").states[0]
    assert check_operator_equality(rho.to_qutip(), loaded.to_qutip())


def solver_results():
    t_span = np.linspace(0, 1, 4)
    ham = 0.1 * HAMILTONIAN + SZ_TOTAL
    rho0 = GibbsProductDensityOperator(SX_AB)
    e_ops = {"sz": SZ_TOTAL, "sx": SX_TOTAL}
    return {
        "series": lambda: series_evolution(ham, SX_AB, t_span, 2),
        "series, e_ops": lambda: series_evolution(ham, SX_AB, t_span, 2, e_ops=e_ops),
        "projected": lambda: projected_evolution(ham, SX_AB, t_span, 2, n_body=2),
        "adaptive": lambda: adaptive_projected_evolution(
            ham, SX_AB, t_span, 3, 2, tol=2.0
        ),
        "adaptive, e_ops": lambda: adaptive_projected_evolution(
            ham, SX_AB, t_span, 3, 2, tol=2.0, e_ops=[SZ_TOTAL], store_states=True
        ),
        "heisenberg": lambda: heisenberg_solve(ham, rho0, t_span, e_ops=e_ops),
        "qutip": lambda: qutip_me_solve(ham, rho0, t_span),
        "qutip, e_ops": lambda: qutip_me_solve(ham, rho0, t_span, e_ops=e_ops),
    }


@pytest.mark.parametrize("solver", list(solver_results()))
def test_results_of_the_solvers_are_stored_without_pickle(solver, tmp_path):
    simulation = solver_results()[solver]()
    filename = tmp_path / "sim.h5"
    simulation.save_hdf5(filename)
    assert find_pickles(filename) == []
    loaded = Simulation.load_hdf5(filename)
    assert np.allclose(loaded.time_span, simulation.time_span)
    assert list(loaded.parameters) == list(simulation.parameters)
    assert list(loaded.stats) == list(simulation.stats)
    assert list(loaded.expect_ops) == list(simulation.expect_ops)
    for key, values in simulation.expect_ops.items():
        assert np.allclose(loaded.expect_ops[key], values)
    assert len(loaded.states) == len(simulation.states)
    for state, loaded_state in zip(simulation.states, loaded.states):
        assert type(loaded_state) is type(state)
        assert check_operator_equality(state.to_qutip(), loaded_state.to_qutip())


def test_adaptive_evolution_checkpoints(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    t_span = np.linspace(0, 1, 4)
    ham = 0.1 * HAMILTONIAN + SZ_TOTAL
    # By default, nothing is written.
    adaptive_projected_evolution(ham, SX_AB, t_span, 3, 2, tol=2.0)
    assert os.listdir(tmp_path) == []

    checkpoint = tmp_path / "checkpoints" / "run.h5"
    checkpoint.parent.mkdir()
    simulation = adaptive_projected_evolution(
        ham, SX_AB, t_span, 3, 2, tol=2.0, checkpoint_file=checkpoint
    )
    assert os.listdir(tmp_path) == ["checkpoints"]
    assert os.listdir(checkpoint.parent) == ["run.h5"]
    assert find_pickles(checkpoint) == []
    assert_same_simulation(simulation, Simulation.load_hdf5(checkpoint))


# ---------------------------------------------------------------------------
#  Files written before the format 1, and pickles
# ---------------------------------------------------------------------------


def write_legacy_file(filename, simulation):
    """Write a file as QALMA did before the format 1 (see qalma.io.legacy)."""

    def blob(group, key, value):
        dset = group.create_dataset(key, shape=(1,), dtype=h5py.vlen_dtype("V1"))
        dset[0] = np.frombuffer(pickle.dumps(value), dtype=np.uint8)
        return dset

    def store_dict(group, data):
        for key, value in data.items():
            if isinstance(value, str):
                dset = group.create_dataset(
                    str(key), data=value, dtype=h5py.string_dtype()
                )
                dset.attrs["encoding"] = "utf-8"
            elif isinstance(value, (int, float, bool, np.ndarray)):
                group.create_dataset(str(key), data=value)
            elif isinstance(value, (list, tuple)) and all(
                isinstance(x, (int, float, bool)) for x in value
            ):
                group.create_dataset(str(key), data=np.array(value))
            else:
                blob(group, str(key), value).attrs["pickled"] = True

    with h5py.File(filename, "w") as h5file:
        store_dict(h5file.create_group("parameters"), simulation.parameters)
        store_dict(h5file.create_group("stats"), simulation.stats)
        store_dict(h5file.create_group("expect obs"), simulation.expect_ops)
        h5file.create_dataset("time span", data=np.array(simulation.time_span))
        if simulation.states:
            system = simulation.states[0].system
            blob(h5file, "system", system)
            group = h5file.create_group("states")
            for index, state in enumerate(simulation.states):
                state._set_system_(None)
                try:
                    blob(group, f"rho_{index:06d}", state)
                finally:
                    state._set_system_(system)


def test_legacy_file(tmp_path):
    simulation = make_simulation()
    filename = tmp_path / "legacy.h5"
    write_legacy_file(filename, simulation)

    with pytest.raises(PickleNotAllowedError, match="allow_pickle=True"):
        Simulation.load_hdf5(filename)
    with pytest.raises(PickleNotAllowedError):
        Simulation.load(filename)

    loaded = Simulation.load_hdf5(filename, allow_pickle=True)
    assert loaded.format_version == 0
    assert loaded.time_span == simulation.time_span
    assert loaded.parameters["order"] == 3
    assert np.allclose(loaded.stats["errors"], simulation.stats["errors"])
    # The keys were stored as strings.
    assert np.allclose(loaded.expect_ops["0"], simulation.expect_ops[0])
    assert len(loaded.states) == 4
    for state, loaded_state in zip(simulation.states, loaded.states):
        assert loaded_state.system is loaded.system
        assert check_operator_equality(state.to_qutip(), loaded_state.to_qutip())
    assert type(loaded.states[-1]) is type(simulation.states[-1])

    # Old files are read-only.
    with pytest.raises(HDF5FormatError, match="read-only"):
        loaded.states.append(SX_AB)
    with pytest.raises(HDF5FormatError, match="read-only"):
        loaded.save_hdf5()

    # Saving to another file converts to the current format.
    loaded.save_hdf5(tmp_path / "converted.h5")
    assert find_pickles(tmp_path / "converted.h5") == []
    converted = Simulation.load_hdf5(tmp_path / "converted.h5")
    assert converted.format_version == 1
    assert len(converted.states) == 4
    for state, loaded_state in zip(simulation.states, converted.states):
        assert type(loaded_state) is type(state)
        assert check_operator_equality(state.to_qutip(), loaded_state.to_qutip())


def test_legacy_file_without_pickles(tmp_path):
    """An old file with plain data only does not need allow_pickle."""
    simulation = Simulation({"order": 2}, {"name": "x"}, [0.0, 1.0], {"a": [1, 2]}, [])
    write_legacy_file(tmp_path / "legacy.h5", simulation)
    loaded = Simulation.load_hdf5(tmp_path / "legacy.h5")
    assert loaded.format_version == 0
    assert loaded.parameters == {"order": 2}
    assert loaded.stats == {"name": "x"}
    assert len(loaded.states) == 0


def test_load_pickle_file(tmp_path):
    simulation = make_simulation(2)
    filename = tmp_path / "sim.pkl"
    with open(filename, "wb") as stream:
        pickle.dump(simulation, stream)
    with pytest.raises(PickleNotAllowedError, match="allow_pickle=True"):
        Simulation.load(filename)
    loaded = Simulation.load(filename, allow_pickle=True)
    assert loaded.time_span == simulation.time_span
    with open(filename, "wb") as stream:
        pickle.dump({"not": "a simulation"}, stream)
    with pytest.raises(TypeError, match="not a Simulation"):
        Simulation.load(filename, allow_pickle=True)


def load_converter():
    spec = importlib.util.spec_from_file_location(
        "pickle_to_hdf5", TOOLS / "pickle_to_hdf5.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_converter(tmp_path, capsys):
    converter = load_converter()
    simulation = make_simulation()
    write_legacy_file(tmp_path / "old.h5", simulation)
    write_legacy_file(tmp_path / "replaced.h5", simulation)
    # As in the old versions of QALMA, states that were stored in an HDF5
    # file are pickled without their system (they cache their pickle).
    with open(tmp_path / "run.pkl", "wb") as stream:
        pickle.dump(simulation, stream)
    simulation.save_hdf5(tmp_path / "current.h5")

    files = [str(tmp_path / name) for name in ("old.h5", "run.pkl", "current.h5")]
    assert converter.main(files) == 0
    assert "skipped" in capsys.readouterr().out
    for name in ("old.v1.h5", "run.h5"):
        assert find_pickles(tmp_path / name) == []
        converted = Simulation.load_hdf5(tmp_path / name)
        assert converted.format_version == 1
        assert len(converted.states) == 4
        assert check_operator_equality(
            converted.states[3].to_qutip(), simulation.states[3].to_qutip()
        )
    # The keys of the pickle are kept; the old HDF5 files had them as strings.
    assert list(Simulation.load_hdf5(tmp_path / "run.h5").expect_ops) == [0, "sx"]
    assert not (tmp_path / "current.v1.h5").exists()

    # Existing outputs are not replaced, unless asked.
    assert converter.main(files[:1]) == 1
    assert "FAILED" in capsys.readouterr().err
    assert converter.main(files[:1] + ["--overwrite"]) == 0

    assert converter.main([str(tmp_path / "replaced.h5"), "--in-place"]) == 0
    assert Simulation.load_hdf5(tmp_path / "replaced.h5").format_version == 1
    assert not (tmp_path / "replaced.v1.h5").exists()

    # Objects that the format does not support.
    simulation.parameters["callback"] = len
    with open(tmp_path / "custom.pkl", "wb") as stream:
        pickle.dump(simulation, stream)
    assert converter.main([str(tmp_path / "custom.pkl")]) == 1
    assert "allow_pickle" in capsys.readouterr().err
    assert not (tmp_path / "custom.h5").exists()
    assert converter.main([str(tmp_path / "custom.pkl"), "--keep-pickle"]) == 0
    assert find_pickles(tmp_path / "custom.h5") == ["parameters/callback"]
