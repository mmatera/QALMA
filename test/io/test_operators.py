"""Round trip of each class of operator and state through the HDF5 format."""

import datetime
import warnings

import h5py
import numpy as np
import pytest
import qutip

from qalma.io import (
    HDF5Context,
    HDF5FormatError,
    IncompleteStateWarning,
    PickleNotAllowedError,
    load_dict,
    load_operators,
    read_operator,
    save_operators,
    store_dict,
    write_operator,
)
from qalma.meanfield.rpa import RPAGibbsDensityOperator
from qalma.model import build_system_from_dims
from qalma.operators import (
    LocalOperator,
    OneBodyOperator,
    ProductOperator,
    QuadraticFormOperator,
    QutipOperator,
    ScalarOperator,
    SumOperator,
)
from qalma.operators.states import (
    GibbsDensityOperator,
    GibbsProductDensityOperator,
    MixtureDensityOperator,
    ProductDensityOperator,
    QutipDensityOperator,
)

from ..helper import (
    FULL_TEST_CASES,
    HAMILTONIAN,
    SITES,
    SX_A,
    SYSTEM,
    SZ_B,
    SZ_TOTAL,
    TEST_CASES_STATES,
    check_operator_equality,
)

CASES = dict(FULL_TEST_CASES)
CASES["hamiltonian"] = HAMILTONIAN
SX, SY = SX_A.operator, (SX_A * 2j).operator
PACKED_SUM = SumOperator(
    tuple(
        ProductOperator({a: SX, b: SY}, 0.5 + pos, SYSTEM)
        for pos, (a, b) in enumerate((a, b) for a in SITES for b in SITES if a != b)
    )
    + (SX_A, ScalarOperator(2j, SYSTEM), ProductOperator({}, 3, SYSTEM)),
    SYSTEM,
)
CASES["sum of products"] = PACKED_SUM
CASES["rpa gibbs"] = RPAGibbsDensityOperator(
    HAMILTONIAN, SYSTEM, n_tau=8, hybrid_options={"n_modes": 2}
)
CASES["gibbs, weight 0"] = GibbsDensityOperator(HAMILTONIAN, SYSTEM, prefactor=0)
CASES["product state, weight 0"] = ProductDensityOperator({}, 0, system=SYSTEM)
CASES["qutip state"] = TEST_CASES_STATES["gibbs_H"].to_qutip_operator() * 0.5
CASES["qutip, scalar"] = QutipOperator(3.0, SYSTEM)
CASES["qutip, permuted sites"] = QutipOperator(
    qutip.tensor(qutip.sigmax(), qutip.sigmaz()),
    SYSTEM,
    names={SITES[2]: 0, SITES[0]: 1},
    prefactor=2j,
)
CASES["reduced state"] = TEST_CASES_STATES["gibbs_H"].partial_trace(
    frozenset(SITES[:2])
)

ALL_CLASSES = {
    GibbsDensityOperator,
    GibbsProductDensityOperator,
    LocalOperator,
    MixtureDensityOperator,
    OneBodyOperator,
    ProductDensityOperator,
    ProductOperator,
    QuadraticFormOperator,
    QutipDensityOperator,
    QutipOperator,
    RPAGibbsDensityOperator,
    ScalarOperator,
    SumOperator,
}


def test_all_classes_are_covered():
    assert {type(operator) for operator in CASES.values()} == ALL_CLASSES


def roundtrip(operator, tmp_path, allow_pickle=False):
    filename = tmp_path / "operator.h5"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", IncompleteStateWarning)
        save_operators(filename, {"op": operator}, "w", allow_pickle=allow_pickle)
        return load_operators(filename, allow_pickle=allow_pickle)["op"]


def as_qutip(operator):
    return operator.to_qutip(tuple(sorted(operator.system.sites)))


def assert_same_structure(original, loaded):
    """Check that the operators are built from the same pieces."""
    assert type(loaded) is type(original)
    assert list(loaded.system.sites) == list(original.system.sites)
    for attr in ("prefactor", "normalized"):
        assert getattr(loaded, attr, None) == getattr(original, attr, None), attr
    if type(original) in (SumOperator, OneBodyOperator):
        # Cached flags, stored for plain sums.
        for attr in ("_isherm", "_isdiagonal", "_simplified"):
            assert getattr(loaded, attr) == getattr(original, attr), attr
    if isinstance(original, SumOperator):
        assert len(loaded.terms) == len(original.terms)
        for term, loaded_term in zip(original.terms, loaded.terms):
            assert_same_structure(term, loaded_term)
    if isinstance(original, QuadraticFormOperator):
        assert loaded.weights == original.weights
        assert len(loaded.basis) == len(original.basis)
    if isinstance(original, ProductOperator):
        assert list(loaded.site_factors) == list(original.site_factors)
        for site, factor in original.site_factors.items():
            assert np.array_equal(loaded.site_factors[site], factor)
    if isinstance(original, QutipOperator):
        assert loaded.site_names == original.site_names
        assert loaded.operator == original.operator
        if original.operator is not None:
            assert type(loaded.operator.data) is type(original.operator.data)
    if isinstance(original, GibbsDensityOperator):
        assert_same_structure(original.k, loaded.k)
        assert loaded._free_energy == original._free_energy
    if isinstance(original, GibbsProductDensityOperator):
        assert loaded.free_energies == original.free_energies
        assert loaded.k_by_site == original.k_by_site
    if isinstance(original, ProductDensityOperator):
        assert loaded.local_fs == original.local_fs


@pytest.mark.parametrize("name", list(CASES))
def test_operator_roundtrip(name, tmp_path):
    original = CASES[name]
    loaded = roundtrip(original, tmp_path)
    assert_same_structure(original, loaded)
    if getattr(original, "operator", "") is None:
        # A QutipOperator built from a number has no matrix to compare.
        return
    if isinstance(as_qutip(original), qutip.Qobj):
        assert check_operator_equality(as_qutip(original), as_qutip(loaded), 1e-12)
    else:
        assert as_qutip(original) == as_qutip(loaded)


@pytest.mark.parametrize("name", list(TEST_CASES_STATES))
def test_state_expectation_values(name, tmp_path):
    original = TEST_CASES_STATES[name]
    loaded = roundtrip(original, tmp_path)
    observables = {"sz": SZ_TOTAL, "h": HAMILTONIAN, "sx": SX_A}
    expected = original.expect(observables)
    system = loaded.system
    loaded_observables = {
        "sz": system.global_operator("Sz"),
        "h": system.global_operator("Hamiltonian"),
        "sx": system.site_operator("Sx", SITES[0]),
    }
    result = loaded.expect(loaded_observables)
    for key, value in expected.items():
        assert abs(result[key] - value) < 1e-10, key


def test_loaded_operators_share_the_system(tmp_path):
    filename = tmp_path / "operators.h5"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", IncompleteStateWarning)
        save_operators(filename, CASES)
        loaded = load_operators(filename)
    assert list(loaded) == list(CASES)
    system = loaded["hamiltonian"].system
    assert system == SYSTEM
    for name, operator in loaded.items():
        if name == "reduced state":
            assert operator.system is system.subsystem(frozenset(SITES[:2]))
        else:
            assert operator.system is system
        for term in getattr(operator, "terms", ()):
            assert term.system is operator.system
    # Loaded operators can be combined, and with the system of the file.
    total = loaded["hamiltonian"] + system.global_operator("Sz")
    assert check_operator_equality(as_qutip(total), as_qutip(HAMILTONIAN + SZ_TOTAL))


def test_file_has_no_pickle_and_fixed_labels(tmp_path):
    filename = tmp_path / "operators.h5"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", IncompleteStateWarning)
        save_operators(filename, CASES)
    labels = set()

    def visit(name, node):
        assert node.attrs.get("py_type", "") not in ("pickle", "pickled_operator")
        if isinstance(node, h5py.Dataset):
            assert node.dtype.kind != "O" or h5py.check_string_dtype(node.dtype)
        if "type" in node.attrs:
            labels.add(node.attrs["type"])

    with h5py.File(filename, "r") as h5file:
        assert h5file.attrs["format"] == "qalma-operators"
        assert h5file.attrs["format_version"] == 1
        h5file.visititems(visit)
    assert labels == {
        "gibbs",
        "gibbs_product",
        "local",
        "mixture",
        "one_body",
        "product",
        "product_state",
        "quadratic",
        "qutip",
        "qutip_state",
        "rpa_gibbs",
        "scalar",
        "sum",
    }


def test_sum_is_packed(tmp_path):
    """Product terms do not get a group each: they are rows of arrays."""
    operator = PACKED_SUM
    filename = tmp_path / "operator.h5"
    save_operators(filename, {"op": operator})
    with h5py.File(filename, "r") as h5file:
        group = h5file["operators/op"]
        n_terms = len(operator.terms)
        assert group["kind"].shape == (n_terms,)
        assert group["prefactors"].shape == (n_terms,)
        assert group["offsets"].shape == (n_terms + 1,)
        nnz = sum(len(getattr(t, "site_factors", "1")) for t in operator.terms)
        assert group["site"].shape == group["matrix"].shape == (nnz,)
        assert "terms" not in group and "blocks" not in group
        # Few distinct local matrices.
        assert len(group["pool_d2"]) == 2
        # The file can be read without QALMA: rebuild the matrix by hand.
        sites = [name.decode() for name in h5file["system/sites"]["name"]]
        total = 0
        for pos in range(n_terms):
            start, stop = group["offsets"][pos : pos + 2]
            factors = {
                sites[group["site"][i]]: qutip.Qobj(
                    group["pool_d2"][group["matrix"][i]]
                )
                for i in range(start, stop)
            }
            total += group["prefactors"][pos] * qutip.tensor(
                [factors.get(site, qutip.qeye(2)) for site in sorted(sites)]
            )
    assert check_operator_equality(total, as_qutip(operator), 1e-10)


def test_symmetry_projections_are_dropped_with_a_warning(tmp_path):
    state = TEST_CASES_STATES["gibbs_H"]
    assert state.symmetry_projections
    filename = tmp_path / "state.h5"
    with pytest.warns(IncompleteStateWarning, match="symmetry_projections"):
        save_operators(filename, {"rho": state})
    with h5py.File(filename, "r") as h5file:
        assert list(h5file["operators/rho"].attrs["dropped"]) == [
            "symmetry_projections"
        ]
    with pytest.warns(IncompleteStateWarning, match="symmetry_projections"):
        loaded = load_operators(filename)["rho"]
    assert loaded.symmetry_projections == ()
    # States without projections do not warn.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        save_operators(filename, {"rho": GibbsDensityOperator(HAMILTONIAN)}, "w")
        load_operators(filename)


def test_meanfield_cache_is_stored(tmp_path):
    state = GibbsDensityOperator(HAMILTONIAN, SYSTEM)
    state.partial_trace(frozenset(SITES[:1]), exact=False)
    assert state._meanfield is not None
    loaded = roundtrip(state, tmp_path)
    assert type(loaded._meanfield) is type(state._meanfield)
    assert check_operator_equality(
        as_qutip(loaded._meanfield), as_qutip(state._meanfield), 1e-12
    )


class CustomOperator(LocalOperator):
    """A subclass that the format does not know."""


def test_unknown_class(tmp_path):
    operator = CustomOperator(SITES[0], SX_A.operator, SYSTEM)
    filename = tmp_path / "operator.h5"
    with pytest.raises(TypeError, match="CustomOperator cannot be stored"):
        save_operators(filename, {"op": operator})
    # A failed save does not leave a file behind.
    assert list(tmp_path.iterdir()) == []

    save_operators(filename, {"op": operator, "h": HAMILTONIAN}, allow_pickle=True)
    with pytest.raises(PickleNotAllowedError, match="allow_pickle=True"):
        load_operators(filename)
    loaded = load_operators(filename, allow_pickle=True)
    assert type(loaded["op"]) is CustomOperator
    assert loaded["op"].system is loaded["h"].system
    assert operator.system is SYSTEM


def test_operators_of_another_system(tmp_path):
    other = build_system_from_dims({"a": 2, "b": 2})
    foreign = ScalarOperator(1.0, other)
    with pytest.raises(ValueError, match="not in the system stored in the file"):
        save_operators(tmp_path / "operators.h5", {"h": HAMILTONIAN, "x": foreign})


def test_save_modes(tmp_path):
    filename = tmp_path / "operators.h5"
    save_operators(filename, {"a": SX_A})
    with pytest.raises(FileExistsError):
        save_operators(filename, {"a": SZ_B})
    assert load_operators(filename)["a"].site == SX_A.site
    save_operators(filename, {"a": SZ_B}, mode="w")
    assert load_operators(filename)["a"].site == SZ_B.site
    with pytest.raises(ValueError, match="mode"):
        save_operators(filename, {"a": SZ_B}, mode="a")


def write_and_edit(tmp_path, operator, edit):
    filename = tmp_path / "operator.h5"
    save_operators(filename, {"op": operator})
    with h5py.File(filename, "r+") as h5file:
        edit(h5file["operators/op"])
    return filename


def replace(group, name, data):
    del group[name]
    group.create_dataset(name, data=data)


@pytest.mark.parametrize(
    ["operator", "edit", "message"],
    [
        (SX_A, lambda g: g.attrs.__setitem__("type", "magic"), "does not know"),
        (SX_A, lambda g: g.attrs.__setitem__("site", 99), "site number 99"),
        (SX_A, lambda g: replace(g, "matrix", np.eye(3)), "shape"),
        (PACKED_SUM, lambda g: replace(g, "site", g["site"][:-1]), "same length"),
        (PACKED_SUM, lambda g: replace(g, "matrix", g["matrix"][()] + 50), "pool"),
        (
            PACKED_SUM,
            lambda g: replace(g, "offsets", g["offsets"][()][::-1]),
            "offsets",
        ),
        (PACKED_SUM, lambda g: replace(g, "kind", g["kind"][()] + 9), "unknown kind"),
        (PACKED_SUM, lambda g: g.__delitem__("prefactors"), "prefactors"),
        (
            CASES["qutip, permuted sites"],
            lambda g: replace(g, "indices", g["indices"][()] + 100),
            "CSR",
        ),
        (
            CASES["qutip, permuted sites"],
            lambda g: replace(g, "sites", [0, 0]),
            r"do not\s+match",
        ),
    ],
)
def test_invalid_files(tmp_path, operator, edit, message):
    filename = write_and_edit(tmp_path, operator, edit)
    with pytest.raises(HDF5FormatError, match=message):
        load_operators(filename)


def test_newer_format_version(tmp_path):
    filename = tmp_path / "operators.h5"
    save_operators(filename, {"a": SX_A})
    with h5py.File(filename, "r+") as h5file:
        h5file.attrs["format_version"] = 2
    with pytest.raises(HDF5FormatError, match="Upgrade QALMA"):
        load_operators(filename)


# ---------------------------------------------------------------------------
#  Dictionaries
# ---------------------------------------------------------------------------

VALUES = {
    "none": None,
    "true": True,
    "int": 32,
    "big int": 2**80,
    "float": 2.5,
    "nan": float("nan"),
    "complex": 1 + 2j,
    "str": "cadena con ñ y / barra",
    "bytes": b"\x00\x01",
    "empty tuple": (),
    "empty list": [],
    "list of ints": [3, 1, 2],
    "tuple of floats": (1.0, 2.5),
    "list of complex": [1j, 2.0],
    "list of bools": [True, False],
    "list of strings": ["s1", "s2", "hola!"],
    "set of strings": {"a", "b"},
    "frozenset of ints": frozenset((1, 2)),
    "table": [[1, 2], [3, 4]],
    "mixed list": [1, "a", None, 2.5, True, [1, 2], {"k": (1, 2)}],
    "array": np.arange(6.0).reshape(2, 3),
    "complex array": np.array([1j, 2]),
    "bool array": np.array([True, False]),
    "0d array": np.array(3.0),
    "string array": np.array(["a", "bc"]),
    "dict": {"z": 1, "a": {"nested": [1, 2]}},
    "timedelta": datetime.timedelta(seconds=90.5),
    "datetime": datetime.datetime(2026, 11, 15, 12, 30),
    "qobj": qutip.tensor(qutip.sigmax(), qutip.qeye(3)),
    "ket": qutip.basis(3, 1),
    0: "integer key",
    (1, "a"): "tuple key",
    "%2F": "a key that looks escaped",
    ".": "dot",
    "operator": HAMILTONIAN,
    "operators": [SX_A, {"sz": SZ_B}],
    "system": SYSTEM,
}


def assert_equal_values(original, loaded, where="value"):
    assert type(loaded) is type(original), where
    if isinstance(original, dict):
        assert list(loaded) == list(original), where
        for key in original:
            assert_equal_values(original[key], loaded[key], f"{where}[{key!r}]")
    elif isinstance(original, (list, tuple)):
        assert len(loaded) == len(original), where
        for pos, item in enumerate(original):
            if isinstance(item, (list, tuple)) and isinstance(loaded[pos], list):
                # The rows of a table of numbers are loaded as lists.
                assert list(item) == loaded[pos], where
            else:
                assert_equal_values(item, loaded[pos], f"{where}[{pos}]")
    elif isinstance(original, np.ndarray):
        assert loaded.shape == original.shape, where
        assert np.array_equal(loaded, original), where
    elif isinstance(original, float) and np.isnan(original):
        assert np.isnan(loaded), where
    elif hasattr(original, "to_qutip"):
        assert check_operator_equality(as_qutip(original), as_qutip(loaded)), where
    else:
        assert loaded == original, where


def test_dict_roundtrip(tmp_path):
    filename = tmp_path / "dict.h5"
    with h5py.File(filename, "w") as h5file:
        store_dict(h5file.create_group("data", track_order=True), VALUES)
    with h5py.File(filename, "r") as h5file:
        loaded = load_dict(h5file["data"])
        assert_equal_values(VALUES, loaded)
        assert loaded["system"] is loaded["operator"].system
        # Values are stored with native types.
        data = h5file["data"]
        assert data["float"][()] == 2.5
        assert data["list of ints"][()].tolist() == [3, 1, 2]
        assert data["list of strings"].asstr()[1] == "s2"
        assert data["dict/a/nested"][()].tolist() == [1, 2]
        assert data["timedelta"][()] == 90.5
        assert data["str"].asstr()[()] == VALUES["str"]


def test_dict_is_replaced(tmp_path):
    with h5py.File(tmp_path / "dict.h5", "w") as h5file:
        group = h5file.create_group("data")
        store_dict(group, {"a": 1, "b": [1, 2]})
        store_dict(group, {"b": "new", "c": 3.0})
        assert load_dict(group) == {"b": "new", "c": 3.0}


@pytest.mark.parametrize(
    "value", [len, object(), {1, "a"}, np.array([object()]), [1, len]]
)
def test_dict_unsupported_values(tmp_path, value):
    with h5py.File(tmp_path / "dict.h5", "w") as h5file:
        with pytest.raises(TypeError, match="cannot be stored"):
            store_dict(h5file.create_group("data"), {"key": value})
        if value is len:
            context = HDF5Context(None, allow_pickle=True)
            store_dict(h5file.create_group("pickled"), {"key": value}, context)
            assert load_dict(h5file["pickled"], context) == {"key": len}
            with pytest.raises(PickleNotAllowedError):
                load_dict(h5file["pickled"])


def test_dict_colliding_keys(tmp_path):
    with h5py.File(tmp_path / "dict.h5", "w") as h5file:
        with pytest.raises(ValueError, match="same name"):
            store_dict(h5file.create_group("data"), {0: 1, "0": 2})


def test_low_level_functions(tmp_path):
    """write_operator and read_operator work on any group of a file."""
    with h5py.File(tmp_path / "custom.h5", "w") as h5file:
        context = HDF5Context.create(h5file, SYSTEM)
        group = h5file.create_group("my/own/layout")
        write_operator(group, "h", HAMILTONIAN, context)
    with h5py.File(tmp_path / "custom.h5", "r") as h5file:
        context = HDF5Context.open(h5file)
        loaded = read_operator(h5file["my/own/layout/h"], context)
    assert check_operator_equality(as_qutip(loaded), as_qutip(HAMILTONIAN))
