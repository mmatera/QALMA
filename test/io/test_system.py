"""Round trip of systems through the HDF5 format."""

import json
import warnings

import h5py
import numpy as np
import pytest

from qalma.io import HDF5FormatError, read_system, write_system
from qalma.io._common import MatrixPool, from_jsonable, json_dumps, to_jsonable
from qalma.io.system import _system_to_spec
from qalma.model import build_system, build_system_from_dims

from ..helper import SYSTEM, check_operator_equality


def build(geometry, model, **parms):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return build_system(geometry, model, **parms)


SYSTEMS = {
    "spin chain (test helper)": lambda: SYSTEM,
    "spin ladder": lambda: build("open ladder", "spin", L=3),
    "spin-1 square lattice": lambda: build("square lattice", "spin", L=2, S=1),
    "triangular lattice": lambda: build("triangular lattice", "spin", L=2, W=2),
    "fermions": lambda: build("open chain lattice", "fermion Hubbard", L=2),
    "from dims": lambda: build_system_from_dims({"a": 2, "b": 3, "c": 2}),
}


def roundtrip(system, tmp_path):
    filename = tmp_path / "system.h5"
    with h5py.File(filename, "w") as h5file:
        write_system(h5file.create_group("system"), system)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with h5py.File(filename, "r") as h5file:
            return read_system(h5file["system"])


def spec_of(system):
    pool = MatrixPool()
    spec, keys = _system_to_spec(system, pool)
    matrices = {dim: [m.tolist() for m in mats] for dim, mats in pool._matrices.items()}
    return json_dumps(spec), keys, json_dumps(to_jsonable(matrices))


@pytest.mark.parametrize("name", list(SYSTEMS))
def test_system_roundtrip(name, tmp_path):
    system = SYSTEMS[name]()
    loaded = roundtrip(system, tmp_path)
    assert loaded is not system
    assert spec_of(loaded) == spec_of(system)
    assert list(loaded.sites) == list(system.sites)
    assert loaded.dimensions == system.dimensions
    assert loaded.name == system.name

    # The operators are rebuilt from the description.
    for site, basis in system.sites.items():
        for op_name, operator in basis["operators"].items():
            other = loaded.sites[site]["operators"][op_name]
            assert operator == other
            assert type(operator.data) is type(other.data)
            assert getattr(operator, "fermionic", None) == getattr(
                other, "fermionic", None
            )
    model = system.spec["model"]
    for op_name in list(model.global_ops):
        original = system.global_operator(op_name)
        if original is None:
            continue
        rebuilt = loaded.global_operator(op_name)
        assert type(rebuilt) is type(original)
        rebuilt._set_system_(system)
        assert check_operator_equality(original, rebuilt)


def test_system_equality(tmp_path):
    """For chains, SystemDescriptor.__eq__ can compare the systems."""
    assert roundtrip(SYSTEM, tmp_path) == SYSTEM


def test_union_system(tmp_path):
    """Systems whose sites are not built from the types of the nodes."""
    full = build_system_from_dims({"a": 2, "b": 3, "c": 4, "d": 5})
    system = full.subsystem(frozenset("ab")).union(full.subsystem(frozenset("c")))
    assert system is not full
    loaded = roundtrip(system, tmp_path)
    assert loaded.dimensions == {"a": 2, "b": 3, "c": 4}
    assert spec_of(loaded) == spec_of(system)


def test_system_is_plain_data(tmp_path):
    """The system is stored as JSON and arrays, readable without QALMA."""
    filename = tmp_path / "system.h5"
    with h5py.File(filename, "w") as h5file:
        write_system(h5file.create_group("system"), SYSTEM)
    with h5py.File(filename, "r") as h5file:
        group = h5file["system"]
        spec = json.loads(group["spec"].asstr()[()])
        assert set(spec) == {"name", "parms", "graph", "model"}
        assert spec["graph"]["edges"]["0"][0] == ["1[0]", "1[1]"]
        coords = SYSTEM.spec["graph"].nodes["1[0]"]["coords"]
        assert spec["graph"]["nodes"]["1[0]"]["coords"] == coords.tolist()
        sites = group["sites"][()]
        assert [name.decode() for name in sites["name"]] == list(SYSTEM.sites)
        assert list(sites["dimension"]) == [2] * len(SYSTEM.sites)
        sz_index = spec["model"]["site_basis"]["0"]["operators"]["Sz"]["matrix"]
        sz_local = SYSTEM.sites["1[0]"]["operators"]["Sz"]
        assert np.array_equal(group["pool_d2"][sz_index], sz_local.full())


def test_custom_parameters_must_be_data(tmp_path):
    system = build_system_from_dims({"a": 2})
    system.spec["parms"]["callback"] = len
    with h5py.File(tmp_path / "system.h5", "w") as h5file:
        with pytest.raises(TypeError, match="cannot be stored as JSON"):
            write_system(h5file.create_group("system"), system)


def edit_spec(filename, edit):
    with h5py.File(filename, "r+") as h5file:
        spec = json.loads(h5file["system/spec"].asstr()[()])
        edit(spec)
        del h5file["system/spec"]
        h5file["system"].create_dataset(
            "spec", data=json.dumps(spec), dtype=h5py.string_dtype()
        )


def test_expressions_cannot_run_code(tmp_path):
    """A crafted description cannot execute code when it is loaded."""
    filename = tmp_path / "system.h5"
    marker = tmp_path / "marker"
    payload = f"__import__('os').mkdir({str(marker)!r})"
    system = build("chain lattice", "spin", L=2)

    def edit(spec):
        spec["model"]["global_ops"]["Hamiltonian"]["site terms"][0]["expr"] = payload
        spec["model"]["parms"]["h"] = payload
        spec["parms"]["h"] = payload

    with h5py.File(filename, "w") as h5file:
        write_system(h5file.create_group("system"), system)
    edit_spec(filename, edit)
    with h5py.File(filename, "r") as h5file:
        with pytest.raises(SyntaxError):
            read_system(h5file["system"])
    assert not marker.exists()


def test_invalid_system(tmp_path):
    filename = tmp_path / "system.h5"
    with h5py.File(filename, "w") as h5file:
        write_system(h5file.create_group("system"), SYSTEM)
    edit_spec(filename, lambda spec: spec.pop("graph"))
    with h5py.File(filename, "r") as h5file:
        with pytest.raises(HDF5FormatError, match="not a valid description"):
            read_system(h5file["system"])


@pytest.mark.parametrize(
    "value",
    [
        None,
        True,
        3,
        2.5,
        float("inf"),
        1 + 2j,
        "text",
        [1, [2, 3]],
        (1, (2, "a")),
        {"a": (1, 2), "b": {"c": None}},
        {1: "int key", (1, 2): "tuple key"},
        {"__tuple__": "a key that looks like a tag"},
        {1, 2, 3},
        frozenset(("a", "b")),
        np.array([1.0, 2.0]),
        np.array([[1, 2], [3, 4]]),
        np.array([1j, 2.0]),
    ],
)
@pytest.mark.parametrize("plain", ["list", "tuple", "ndarray"])
def test_json_roundtrip(value, plain):
    encoded = json.loads(json_dumps(to_jsonable(value, plain)))
    decoded = from_jsonable(encoded, plain)
    assert type(decoded) is type(value)
    if isinstance(value, np.ndarray):
        assert decoded.dtype == value.dtype
        assert np.array_equal(decoded, value)
    else:
        assert decoded == value
