"""Constants, errors and low-level helpers shared by the ``qalma.io`` modules."""

import json
import math
import pickle
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import unquote

import h5py
import numpy as np
import qutip  # type: ignore[import-untyped]
from scipy.sparse import csr_matrix

__all__ = [
    "FORMAT_VERSION",
    "HDF5FormatError",
    "PickleNotAllowedError",
]

# Value of the ``format`` attribute of the root group, for each kind of file.
FORMAT_SIMULATION = "qalma-simulation"
FORMAT_OPERATORS = "qalma-operators"

# Version written by this version of QALMA. Any version lower or equal to
# this one can be read.
FORMAT_VERSION = 1

# Datasets larger than this number of bytes are stored compressed.
_COMPRESSION_THRESHOLD = 4096

STRING_DTYPE = h5py.string_dtype(encoding="utf-8")


class HDF5FormatError(ValueError):
    """The content of an HDF5 file does not follow the QALMA format."""


class PickleNotAllowedError(HDF5FormatError):
    """The file stores Python pickles, and ``allow_pickle`` is not set.

    Loading a pickle can run arbitrary code, so it has to be requested
    explicitly with ``allow_pickle=True``, and only for trusted files.
    """


# ---------------------------------------------------------------------------
#  Format attributes
# ---------------------------------------------------------------------------


def write_format_attrs(root: h5py.Group, format_name: str) -> None:
    """Write the attributes that identify the format in the root group."""
    # pylint: disable=import-outside-toplevel
    from qalma.version import __version__

    root.attrs["format"] = format_name
    root.attrs["format_version"] = FORMAT_VERSION
    root.attrs["qalma_version"] = __version__


def read_format_version(root: h5py.Group, format_name: str) -> int:
    """Check the format attributes of the root group.

    Returns
    -------
    int
        The version of the format, or 0 if the file does not have the
        ``format`` attribute (files written before the format 1).

    Raises
    ------
    HDF5FormatError
        If the file has a different format, or was written by a newer
        version of QALMA.
    """
    if "format" not in root.attrs:
        return 0
    found = as_str(root.attrs["format"])
    if found != format_name:
        raise HDF5FormatError(
            f"{root.file.filename} has the format <<{found}>>, "
            f"but <<{format_name}>> was expected."
        )
    version = int(root.attrs.get("format_version", 0))
    if version < 1:
        raise HDF5FormatError(
            f"{root.file.filename} has an invalid format_version ({version})."
        )
    if version > FORMAT_VERSION:
        written_by = as_str(root.attrs.get("qalma_version", "unknown"))
        raise HDF5FormatError(
            f"{root.file.filename} uses the version {version} of the format "
            f"(written by QALMA {written_by}), but this version of QALMA only "
            f"reads up to the version {FORMAT_VERSION}. Upgrade QALMA to open it."
        )
    return version


# ---------------------------------------------------------------------------
#  Small conversions
# ---------------------------------------------------------------------------


def as_str(value) -> str:
    """Convert a string read from an HDF5 attribute or dataset to ``str``."""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def as_number(value):
    """Convert a numpy scalar to ``float``, or to ``complex`` if needed."""
    value = complex(value)
    if value.imag == 0:
        return value.real
    return value


def escape_name(key: str) -> str:
    """Build a valid HDF5 link name from a string.

    ``/`` is the path separator in HDF5 and ``.`` is the current group, so
    they are replaced by percent-encoded sequences. :func:`unescape_name`
    is the inverse function.
    """
    if key == "":
        raise ValueError("The empty string cannot be used as a name in HDF5.")
    name = key.replace("%", "%25").replace("/", "%2F")
    if name == ".":
        name = "%2E"
    return name


def unescape_name(name: str) -> str:
    """Inverse of :func:`escape_name`."""
    return unquote(name)


def index_name(index: int) -> str:
    """Name of the ``index``-th element of a sequence stored as a group."""
    return f"{index:06d}"


def create_dataset(group: h5py.Group, name: str, data: np.ndarray) -> h5py.Dataset:
    """Create a dataset, compressing it if it is large enough."""
    kwargs: Dict[str, Any] = {}
    if data.ndim > 0 and data.nbytes > _COMPRESSION_THRESHOLD:
        kwargs = {"compression": "gzip", "shuffle": True}
    return group.create_dataset(name, data=data, **kwargs)


def clear_group(group: h5py.Group) -> None:
    """Remove all the members and attributes of a group."""
    for name in list(group.keys()):
        del group[name]
    for name in list(group.attrs.keys()):
        del group.attrs[name]


# ---------------------------------------------------------------------------
#  Pickle blobs (only with allow_pickle=True)
# ---------------------------------------------------------------------------


def write_pickle(group: h5py.Group, name: str, value) -> h5py.Dataset:
    """Store ``value`` as a pickle byte stream."""
    data = np.frombuffer(pickle.dumps(value), dtype=np.uint8)
    dset = create_dataset(group, name, data)
    dset.attrs["py_type"] = "pickle"
    return dset


def read_pickle(dset: h5py.Dataset, allow_pickle: bool):
    """Load a value stored by :func:`write_pickle`."""
    if not allow_pickle:
        raise PickleNotAllowedError(
            f"{dset.name} in {dset.file.filename} is a Python pickle. Loading "
            "a pickle can run arbitrary code: if you trust the file, load it "
            "with allow_pickle=True."
        )
    return pickle.loads(dset[()].tobytes())


# ---------------------------------------------------------------------------
#  JSON with a few tagged types
# ---------------------------------------------------------------------------

# Objects that JSON cannot represent are stored as an object with a single
# one of these keys (``__ndarray__`` comes with ``dtype``).
_JSON_TAGS = frozenset(
    (
        "__complex__",
        "__dict__",
        "__float__",
        "__frozenset__",
        "__list__",
        "__ndarray__",
        "__set__",
        "__tuple__",
    )
)


def to_jsonable(obj, plain: str = "list"):
    """Convert ``obj`` into something that the ``json`` module can store.

    Numbers, strings, ``None``, lists and dicts with string keys are kept as
    they are. Tuples, sets, complex numbers, non-finite floats, numpy arrays
    and dicts with other keys are stored as tagged objects, like
    ``{"__tuple__": [1, 2]}``, so :func:`from_jsonable` restores the same
    types.

    Parameters
    ----------
    obj :
        The object to convert.
    plain : str
        The type of sequence stored as a bare JSON array: ``"list"``
        (default), ``"tuple"`` or ``"ndarray"`` (1D arrays of floats). It
        only affects ``obj`` itself, not the objects inside it.

    Raises
    ------
    TypeError
        If ``obj`` contains an object of a type that cannot be stored.
    """
    if isinstance(obj, np.generic):
        obj = obj.item()
    if obj is None or isinstance(obj, (bool, int, str)):
        return obj
    if isinstance(obj, float):
        if math.isfinite(obj):
            return obj
        return {"__float__": repr(obj)}
    if isinstance(obj, complex):
        return {"__complex__": [to_jsonable(obj.real), to_jsonable(obj.imag)]}
    if isinstance(obj, np.ndarray):
        if obj.dtype.kind not in "biufc":
            raise TypeError(f"arrays of type {obj.dtype} cannot be stored as JSON.")
        data = to_jsonable(obj.tolist())
        if plain == "ndarray" and obj.ndim == 1 and obj.dtype == np.float64:
            return data
        return {"__ndarray__": data, "dtype": obj.dtype.name}
    if isinstance(obj, list):
        data = [to_jsonable(item) for item in obj]
        return data if plain == "list" else {"__list__": data}
    if isinstance(obj, tuple):
        data = [to_jsonable(item) for item in obj]
        return data if plain == "tuple" else {"__tuple__": data}
    if isinstance(obj, (set, frozenset)):
        tag = "__set__" if isinstance(obj, set) else "__frozenset__"
        return {tag: [to_jsonable(item) for item in sorted(obj, key=repr)]}
    if isinstance(obj, dict):
        if all(isinstance(key, str) for key in obj) and _JSON_TAGS.isdisjoint(obj):
            return {key: to_jsonable(val) for key, val in obj.items()}
        return {
            "__dict__": [
                [to_jsonable(key, "list"), to_jsonable(val)] for key, val in obj.items()
            ]
        }
    raise TypeError(f"objects of type {type(obj).__name__} cannot be stored as JSON.")


def from_jsonable(obj, plain: str = "list"):
    """Inverse of :func:`to_jsonable`."""
    if isinstance(obj, list):
        if plain == "ndarray":
            return np.array([from_jsonable(item) for item in obj], dtype=np.float64)
        items = [from_jsonable(item) for item in obj]
        return tuple(items) if plain == "tuple" else items
    if isinstance(obj, dict):
        if "__ndarray__" in obj:
            data = from_jsonable(obj["__ndarray__"])
            return np.array(data, dtype=np.dtype(obj["dtype"]))
        if len(obj) == 1:
            ((tag, data),) = obj.items()
            if tag == "__float__":
                return float(data)
            if tag == "__complex__":
                return complex(from_jsonable(data[0]), from_jsonable(data[1]))
            if tag == "__list__":
                return [from_jsonable(item) for item in data]
            if tag == "__tuple__":
                return tuple(from_jsonable(item) for item in data)
            if tag == "__set__":
                return set(_hashable(from_jsonable(item)) for item in data)
            if tag == "__frozenset__":
                return frozenset(_hashable(from_jsonable(item)) for item in data)
            if tag == "__dict__":
                return {
                    _hashable(from_jsonable(key)): from_jsonable(val)
                    for key, val in data
                }
        return {key: from_jsonable(val) for key, val in obj.items()}
    return obj


def _hashable(obj):
    """Convert the lists inside a key or a set element into tuples."""
    if isinstance(obj, list):
        return tuple(_hashable(item) for item in obj)
    return obj


def json_dumps(obj) -> str:
    """Serialize an object returned by :func:`to_jsonable`."""
    return json.dumps(obj, ensure_ascii=False, allow_nan=False)


# ---------------------------------------------------------------------------
#  Matrices
# ---------------------------------------------------------------------------


class MatrixPool:
    """Collect square matrices, storing each distinct matrix only once.

    Many-body operators are built from a few distinct local matrices, so
    the factors of an operator are stored as indices in a pool of matrices.
    There is a pool for each dimension, stored in the dataset
    ``pool_d{dimension}`` with shape ``(n_matrices, dimension, dimension)``.
    """

    def __init__(self) -> None:
        self._index: Dict[Tuple[int, bytes], int] = {}
        self._matrices: Dict[int, List[np.ndarray]] = {}

    def add(self, matrix) -> int:
        """Add a matrix and return its index in the pool of its dimension."""
        array = np.ascontiguousarray(matrix, dtype=np.complex128)
        if array.ndim != 2 or array.shape[0] != array.shape[1]:
            raise ValueError(f"a square matrix was expected, got {array.shape}.")
        dimension = array.shape[0]
        key = (dimension, array.tobytes())
        index = self._index.get(key)
        if index is None:
            matrices = self._matrices.setdefault(dimension, [])
            index = len(matrices)
            matrices.append(array)
            self._index[key] = index
        return index

    def write(self, group: h5py.Group) -> None:
        """Write the pools as datasets of ``group``."""
        for dimension, matrices in self._matrices.items():
            create_dataset(group, f"pool_d{dimension}", np.stack(matrices))


def read_pools(group: h5py.Group) -> Dict[int, np.ndarray]:
    """Read the pools of matrices of a group, indexed by dimension."""
    pools = {}
    for name in group:
        if name.startswith("pool_d"):
            pool = np.asarray(group[name][()], dtype=np.complex128)
            if pool.ndim != 3 or pool.shape[1] != pool.shape[2]:
                raise HDF5FormatError(
                    f"{group.name}/{name} has the shape {pool.shape}, "
                    "which is not a stack of square matrices."
                )
            if name != f"pool_d{pool.shape[1]}":
                raise HDF5FormatError(
                    f"{group.name}/{name} stores matrices of dimension {pool.shape[1]}."
                )
            pools[pool.shape[1]] = pool
    return pools


def pool_matrix(
    pools: Dict[int, np.ndarray], dimension: int, index: int, where: str
) -> np.ndarray:
    """Get a matrix from the pools, checking that it exists."""
    pool = pools.get(dimension)
    if pool is None or not 0 <= index < len(pool):
        raise HDF5FormatError(
            f"{where} refers to the matrix {index} of dimension {dimension}, "
            "which is not in the pool."
        )
    return pool[index]


def qobj_storage(qobj: qutip.Qobj) -> str:
    """Name of the data layer type of a Qobj (``CSR``, ``Dense``, ``Dia``)."""
    return type(qobj.data).__name__


def qobj_with_storage(qobj: qutip.Qobj, storage: Optional[str]) -> qutip.Qobj:
    """Convert a Qobj to the given data layer type, if qutip knows it."""
    if storage and qobj_storage(qobj) != storage:
        try:
            return qobj.to(storage)
        except (KeyError, ValueError, TypeError):
            pass
    return qobj


def write_qobj(group: h5py.Group, qobj: qutip.Qobj) -> None:
    """Store a Qobj in CSR format.

    The matrix is stored in the datasets ``data``, ``indices`` and
    ``indptr`` (as in ``scipy.sparse.csr_matrix``), and the dimensions of
    the tensor factors in ``dims_rows`` and ``dims_cols``.
    """
    dims_rows, dims_cols = qobj.dims
    if not all(isinstance(dim, int) for dim in (*dims_rows, *dims_cols)):
        raise TypeError(
            f"Qobj with dims={qobj.dims} cannot be stored: only operators, "
            "kets and bras are supported."
        )
    matrix = csr_matrix(qobj.to("CSR").data.as_scipy())
    matrix.sort_indices()
    group.attrs["storage"] = qobj_storage(qobj)
    group.attrs["shape"] = np.array(matrix.shape, dtype=np.int64)
    group.create_dataset("dims_rows", data=np.array(dims_rows, dtype=np.int64))
    group.create_dataset("dims_cols", data=np.array(dims_cols, dtype=np.int64))
    create_dataset(group, "data", np.asarray(matrix.data, dtype=np.complex128))
    create_dataset(group, "indices", np.asarray(matrix.indices, dtype=np.int64))
    create_dataset(group, "indptr", np.asarray(matrix.indptr, dtype=np.int64))


def read_qobj(group: h5py.Group) -> qutip.Qobj:
    """Load a Qobj stored by :func:`write_qobj`."""
    try:
        shape = tuple(int(n) for n in group.attrs["shape"])
        dims = [
            [int(n) for n in group["dims_rows"][()]],
            [int(n) for n in group["dims_cols"][()]],
        ]
        data = np.asarray(group["data"][()], dtype=np.complex128)
        indices = np.asarray(group["indices"][()], dtype=np.int32)
        indptr = np.asarray(group["indptr"][()], dtype=np.int32)
    except KeyError as exc:
        raise HDF5FormatError(
            f"{group.name} is not a matrix in CSR format: {exc.args[0]}"
        ) from exc
    if (
        len(shape) != 2
        or math.prod(dims[0]) != shape[0]
        or math.prod(dims[1]) != shape[1]
        or len(indptr) != shape[0] + 1
        or len(data) != len(indices)
        or (len(indices) and not 0 <= indices.min() <= indices.max() < shape[1])
        or indptr[0] != 0
        or indptr[-1] != len(data)
        or (np.diff(indptr) < 0).any()
    ):
        raise HDF5FormatError(f"{group.name} is not a valid matrix in CSR format.")
    matrix = csr_matrix((data, indices, indptr), shape=shape)
    qobj = qutip.Qobj(matrix, dims=dims)
    return qobj_with_storage(qobj, as_str(group.attrs.get("storage", "CSR")))
