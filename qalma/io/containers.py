"""Storage of Python dictionaries in HDF5 groups, using native HDF5 types.

A dictionary is stored as a group, with a member for each item:

=============================  =============================================
Python value                   HDF5 member
=============================  =============================================
bool, int, float, complex      scalar dataset
str                            scalar dataset (variable-length UTF-8 string)
numpy array of numbers         dataset
list, tuple, set of numbers    1D (or rectangular) dataset; the attribute
or of strings                  ``py_type`` stores the kind of container
any other list or tuple        group with the members ``000000``, ...
dict                           group
None                           empty dataset
Operator                       group with the attribute ``type``
SystemDescriptor               empty dataset, which refers to ``/system``
datetime, timedelta            ISO string, seconds
qutip.Qobj                     group with the matrix in CSR format
=============================  =============================================

The attribute ``py_type`` of a member is only needed to restore the Python
type: the data can be read without it.

Any other object raises a ``TypeError``, unless the context allows pickle.
"""

import datetime
import json
from numbers import Number
from typing import Any, Dict, Optional

import h5py
import numpy as np
import qutip  # type: ignore[import-untyped]

from qalma.model import SystemDescriptor
from qalma.operators import Operator

from ._common import (
    STRING_DTYPE,
    HDF5FormatError,
    as_str,
    clear_group,
    create_dataset,
    escape_name,
    from_jsonable,
    index_name,
    json_dumps,
    read_pickle,
    read_qobj,
    to_jsonable,
    unescape_name,
    write_pickle,
    write_qobj,
)
from .operators import HDF5Context, is_operator_node, read_operator, write_operator

__all__ = ["load_dict", "store_dict"]

_INT64_MIN, _INT64_MAX = -(2**63), 2**63 - 1
_SEQUENCE_TYPES = {"list": list, "tuple": tuple, "set": set, "frozenset": frozenset}


def _is_number(value) -> bool:
    return isinstance(value, (Number, np.number)) and not isinstance(
        value, (bool, np.bool_)
    )


def _sequence_as_array(values) -> Optional[np.ndarray]:
    """Convert a sequence to an array of numbers or strings, if possible.

    Returns ``None`` if the items are not of the same kind, in which case
    the sequence has to be stored item by item.
    """
    values = list(values)
    if len(values) == 0:
        return np.zeros(0, dtype=np.float64)
    if all(isinstance(item, str) for item in values):
        return np.array(values, dtype=object)
    if all(isinstance(item, (bool, np.bool_)) for item in values):
        return np.array(values, dtype=bool)
    if all(_is_number(item) for item in values):
        # Mixing integers, floats and complex numbers in an array would
        # change the type of some items.
        kinds = {np.asarray(item).dtype.kind for item in values}
        if len(kinds) > 1:
            return None
        try:
            array = np.asarray(values)
        except (ValueError, OverflowError):
            return None
        return array if array.dtype.kind in "iufc" else None
    if all(isinstance(item, (list, tuple, np.ndarray)) for item in values):
        # A rectangular table of numbers.
        try:
            array = np.asarray(values)
        except (ValueError, OverflowError):
            return None
        if array.dtype.kind in "biufc" and array.ndim >= 2:
            return array
    return None


def _write_array(group: h5py.Group, name: str, array: np.ndarray) -> h5py.Dataset:
    if array.dtype.kind in "OUS":
        if not all(isinstance(item, (str, bytes)) for item in array.flat):
            raise TypeError(
                "arrays of objects cannot be stored in the HDF5 format, only "
                "arrays of numbers or of strings."
            )
        return group.create_dataset(name, data=array.astype(object), dtype=STRING_DTYPE)
    if array.dtype.kind not in "biufc":
        raise TypeError(f"arrays of type {array.dtype} cannot be stored.")
    return create_dataset(group, name, array)


def _write_value(group: h5py.Group, name: str, value, context: HDF5Context, where: str):
    """Store ``value`` as the member ``name`` of ``group``."""
    # pylint: disable=too-many-return-statements,too-many-branches
    if value is None:
        node = group.create_dataset(name, data=h5py.Empty("f"))
        node.attrs["py_type"] = "none"
        return node
    if isinstance(value, str):
        return group.create_dataset(name, data=value, dtype=STRING_DTYPE)
    if isinstance(value, (bool, np.bool_)):
        return group.create_dataset(name, data=bool(value))
    if isinstance(value, (int, np.integer)):
        if _INT64_MIN <= int(value) <= _INT64_MAX:
            return group.create_dataset(name, data=np.int64(value))
        node = group.create_dataset(name, data=str(int(value)), dtype=STRING_DTYPE)
        node.attrs["py_type"] = "int"
        return node
    if isinstance(value, (float, np.floating)):
        return group.create_dataset(name, data=np.float64(value))
    if isinstance(value, (complex, np.complexfloating)):
        return group.create_dataset(name, data=np.complex128(value))
    if isinstance(value, bytes):
        node = group.create_dataset(name, data=np.frombuffer(value, dtype=np.uint8))
        node.attrs["py_type"] = "bytes"
        return node
    if isinstance(value, np.ndarray):
        node = _write_array(group, name, value)
        if value.ndim == 0:
            node.attrs["py_type"] = "ndarray"
        return node
    if isinstance(value, datetime.timedelta):
        node = group.create_dataset(name, data=value.total_seconds())
        node.attrs["py_type"] = "timedelta"
        node.attrs["units"] = "s"
        return node
    if isinstance(value, datetime.datetime):
        node = group.create_dataset(name, data=value.isoformat(), dtype=STRING_DTYPE)
        node.attrs["py_type"] = "datetime"
        return node
    if isinstance(value, SystemDescriptor):
        context.bind(group.file, value)
        if context.subsystem_of(value) is not None:
            raise ValueError(
                f"{where} is a subsystem of the system of the file; "
                "only the system itself can be stored."
            )
        node = group.create_dataset(name, data=h5py.Empty("f"))
        node.attrs["py_type"] = "system"
        return node
    if isinstance(value, Operator):
        return write_operator(group, name, value, context)
    if isinstance(value, qutip.Qobj):
        node = group.create_group(name)
        node.attrs["py_type"] = "qobj"
        write_qobj(node, value)
        return node
    if isinstance(value, dict):
        node = group.create_group(name, track_order=True)
        _write_items(node, value, context, where)
        return node
    for type_name, sequence_type in _SEQUENCE_TYPES.items():
        if type(value) is sequence_type:  # pylint: disable=unidiomatic-typecheck
            array = _sequence_as_array(
                sorted(value, key=repr) if type_name.endswith("set") else value
            )
            if array is not None:
                node = _write_array(group, name, array)
            elif type_name.endswith("set"):
                break
            else:
                node = group.create_group(name)
                for pos, item in enumerate(value):
                    _write_value(
                        node, index_name(pos), item, context, f"{where}[{pos}]"
                    )
            node.attrs["py_type"] = type_name
            return node

    if context.allow_pickle:
        return write_pickle(group, name, value)
    raise TypeError(
        f"{where} is of the type {type(value).__name__}, which cannot be "
        "stored in the HDF5 format. Supported values are numbers, strings, "
        "numpy arrays, operators, and lists, tuples and dicts of them. "
        "Use allow_pickle=True to store it as a Python pickle."
    )


def _write_items(group: h5py.Group, data: dict, context: HDF5Context, where: str):
    group.attrs["py_type"] = "dict"
    for key, value in data.items():
        name = escape_name(key if isinstance(key, str) else str(key))
        item_where = f"{where}[{key!r}]"
        if name in group:
            raise ValueError(f"{item_where}: there is another key with the same name.")
        node = _write_value(group, name, value, context, item_where)
        if not isinstance(key, str):
            # The name of the member is str(key): store the key itself.
            try:
                node.attrs["key_json"] = json_dumps(to_jsonable(key))
            except TypeError as exc:
                raise TypeError(
                    f"{item_where}: keys of the type {type(key).__name__} "
                    "cannot be stored."
                ) from exc


def store_dict(
    group: h5py.Group,
    data: Dict[Any, Any],
    context: Optional[HDF5Context] = None,
    where: str = "",
) -> None:
    """Store a dictionary in an HDF5 group, replacing its content.

    Parameters
    ----------
    group : h5py.Group
        The group where the items are stored. Its content is removed.
    data : dict
        The dictionary. See the documentation of the module for the types
        of values that can be stored.
    context : HDF5Context, optional
        The context of the file, needed to store operators.
    where : str, optional
        Name of the dictionary, used in the error messages.

    Raises
    ------
    TypeError
        If a value cannot be stored and ``context.allow_pickle`` is False.
    """
    if context is None:
        context = HDF5Context.open(group.file)
    clear_group(group)
    _write_items(group, data, context, where or group.name)


def _read_dataset(node: h5py.Dataset, context: HDF5Context):
    # pylint: disable=too-many-return-statements,too-many-branches
    py_type = as_str(node.attrs.get("py_type", ""))
    if py_type == "pickle":
        return read_pickle(node, context.allow_pickle)
    if py_type == "pickled_operator":
        return read_operator(node, context)
    if py_type == "system":
        if context.system is None:
            raise HDF5FormatError(
                f"{node.name} refers to the system, but the file has no system."
            )
        return context.system
    if node.shape is None:
        # Empty dataset
        return None

    if h5py.check_string_dtype(node.dtype) is not None:
        value = node.asstr()[()]
        if isinstance(value, np.ndarray):
            if py_type in _SEQUENCE_TYPES:
                return _SEQUENCE_TYPES[py_type](value.tolist())
            return value
        if py_type == "int":
            return int(value)
        if py_type == "datetime":
            return datetime.datetime.fromisoformat(value)
        return str(value)

    value = node[()]
    if py_type == "bytes":
        return value.tobytes()
    if py_type == "ndarray":
        return np.asarray(value)
    if isinstance(value, np.ndarray):
        if py_type in _SEQUENCE_TYPES:
            items = value.tolist()
            if py_type.endswith("set"):
                items = [tuple(i) if isinstance(i, list) else i for i in items]
            return _SEQUENCE_TYPES[py_type](items)
        return value
    if py_type == "timedelta":
        return datetime.timedelta(seconds=float(value))
    # Scalars are converted to Python types.
    return value.item() if isinstance(value, np.generic) else value


def _read_node(node, context: HDF5Context):
    if isinstance(node, h5py.Dataset):
        return _read_dataset(node, context)
    if is_operator_node(node):
        return read_operator(node, context)
    py_type = as_str(node.attrs.get("py_type", "dict"))
    if py_type == "qobj":
        return read_qobj(node)
    if py_type in ("list", "tuple"):
        items = []
        for pos in range(len(node)):
            name = index_name(pos)
            if name not in node:
                raise HDF5FormatError(f"{node.name}/{name} is missing.")
            items.append(_read_node(node[name], context))
        return _SEQUENCE_TYPES[py_type](items)
    if py_type == "dict":
        return _read_items(node, context)
    raise HDF5FormatError(f"{node.name} has the unknown py_type <<{py_type}>>.")


def _read_items(group: h5py.Group, context: HDF5Context) -> dict:
    result = {}
    # The groups written by QALMA track the order of creation of their
    # members, so the items are visited in the order of the dictionary.
    for name in group:
        node = group[name]
        if "key_json" in node.attrs:
            key = from_jsonable(json.loads(as_str(node.attrs["key_json"])))
        else:
            key = unescape_name(name)
        result[key] = _read_node(node, context)
    return result


def load_dict(group: h5py.Group, context: Optional[HDF5Context] = None) -> dict:
    """Load a dictionary stored by :func:`store_dict`.

    Parameters
    ----------
    group : h5py.Group
        The group that stores the dictionary.
    context : HDF5Context, optional
        The context of the file, needed to load operators and pickles.

    Raises
    ------
    PickleNotAllowedError
        If a value is stored as a pickle and ``context.allow_pickle`` is
        False.
    """
    if context is None:
        context = HDF5Context.open(group.file)
    return _read_items(group, context)
