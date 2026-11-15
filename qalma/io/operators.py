"""Storage of operators and states in HDF5 groups.

Each operator is stored in a group with the attribute ``type``, which is one
of a fixed set of labels. Loading an operator only instantiates the class
registered for its label, with numbers, strings and arrays as arguments, so
opening a file cannot run code stored in the file.

Operators refer to the sites of their system by the position of the site in
the table of sites of the stored system (see :mod:`qalma.io.system`).

The layout of each type of group is described in the user guide
(*HDF5 file format*).
"""

import pickle
import warnings
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import h5py
import numpy as np
import qutip  # type: ignore[import-untyped]

from qalma.model import SystemDescriptor
from qalma.operators import (
    LocalOperator,
    OneBodyOperator,
    Operator,
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

from ._common import (
    HDF5FormatError,
    MatrixPool,
    PickleNotAllowedError,
    as_number,
    as_str,
    create_dataset,
    from_jsonable,
    index_name,
    json_dumps,
    pool_matrix,
    read_pools,
    read_qobj,
    to_jsonable,
    write_qobj,
)
from .system import read_sites, read_system, write_system

__all__ = [
    "HDF5Context",
    "IncompleteStateWarning",
    "read_operator",
    "write_operator",
]

# Kinds of terms in a group of type "sum", "one_body" or "mixture".
KIND_PRODUCT = 0  # ProductOperator, stored in the packed arrays
KIND_LOCAL = 1  # LocalOperator, stored in the packed arrays
KIND_SCALAR = 2  # ScalarOperator, stored in the packed arrays
KIND_QUTIP = 3  # QutipOperator, stored in blocks/
KIND_GROUP = 4  # any other operator, stored in terms/


class IncompleteStateWarning(UserWarning):
    """A state was stored without some of its attributes.

    Functions cannot be stored without pickle, so the symmetry projections
    of a :class:`~qalma.operators.states.GibbsDensityOperator` are not
    stored. The group of the state has the attribute ``dropped`` with the
    names of the missing attributes.
    """


class HDF5Context:
    """The system that the operators stored in a file refer to.

    Parameters
    ----------
    system : SystemDescriptor
        The system stored in the file.
    names : list[str], optional
        The names of the sites, in the order of the table of sites of the
        file. By default, the order of ``system.sites``.
    allow_pickle : bool, optional
        Allow to store (and load) the objects that the format does not
        support as Python pickles. The default is False.
    """

    def __init__(
        self,
        system: Optional[SystemDescriptor],
        names: Optional[Sequence[str]] = None,
        allow_pickle: bool = False,
    ):
        self.system = system
        self.allow_pickle = allow_pickle
        self.names: List[str] = []
        self.index: Dict[str, int] = {}
        if system is not None:
            self._set_names(list(system.sites) if names is None else list(names))

    def _set_names(self, names: List[str]) -> None:
        assert self.system is not None
        if set(names) != set(self.system.sites) or len(names) != len(set(names)):
            raise HDF5FormatError(
                "The table of sites does not match the sites of the system."
            )
        self.names = names
        self.index = {name: pos for pos, name in enumerate(names)}

    # -- construction ------------------------------------------------------

    @classmethod
    def create(
        cls, root: h5py.Group, system: SystemDescriptor, allow_pickle: bool = False
    ) -> "HDF5Context":
        """Store ``system`` in the group ``system`` of ``root``."""
        write_system(root.create_group("system"), system)
        return cls(system, allow_pickle=allow_pickle)

    @classmethod
    def open(
        cls,
        root: h5py.Group,
        system: Optional[SystemDescriptor] = None,
        allow_pickle: bool = False,
    ) -> "HDF5Context":
        """Build the context of a file that already exists.

        Parameters
        ----------
        root : h5py.Group
            The root group of the file.
        system : SystemDescriptor, optional
            The system stored in the file, if it was already loaded.
            Otherwise, it is loaded from the group ``system``, if the file
            has it.
        allow_pickle : bool, optional
            Allow to load Python pickles.
        """
        if "system" not in root:
            return cls(None, allow_pickle=allow_pickle)
        group = root["system"]
        if not isinstance(group, h5py.Group):
            raise HDF5FormatError(f"{group.name} is not a group.")
        if system is None:
            system = read_system(group)
        names, dimensions = read_sites(group)
        context = cls(system, names, allow_pickle)
        if [system.dimensions[name] for name in names] != dimensions:
            raise HDF5FormatError(
                "The system does not match the table of sites of "
                f"{root.file.filename}."
            )
        return context

    def bind(self, root: h5py.Group, system: SystemDescriptor) -> None:
        """Store ``system`` in the file if the file does not have a system."""
        if self.system is None:
            write_system(root.create_group("system"), system)
            self.system = system
            self._set_names(list(system.sites))

    # -- systems and sites ---------------------------------------------------

    def site(self, name: str) -> int:
        """Position of a site in the table of sites."""
        try:
            return self.index[name]
        except KeyError:
            raise ValueError(
                f"The site <<{name}>> is not in the system stored in the file."
            ) from None

    def site_name(self, index: int, where: str) -> str:
        """Name of the site at a position of the table of sites."""
        if not 0 <= index < len(self.names):
            raise HDF5FormatError(
                f"{where} refers to the site number {index}, but the system "
                f"has {len(self.names)} sites."
            )
        return self.names[index]

    def subsystem_of(self, system: Optional[SystemDescriptor]) -> Optional[np.ndarray]:
        """Positions of the sites of ``system``, if it is a proper subsystem.

        Returns ``None`` if ``system`` has all the sites of the stored
        system.

        Raises
        ------
        ValueError
            If ``system`` is not a subsystem of the stored system.
        """
        if self.system is None:
            raise ValueError(
                "The operator does not have a system, and the file does not "
                "have one either."
            )
        if system is None or system is self.system:
            # Old versions of QALMA detached the system of the states to
            # pickle them, so states loaded from old pickles may not have
            # a system: they act on the system of the file.
            return None
        root_dimensions = self.system.dimensions
        for name, dimension in system.dimensions.items():
            if root_dimensions.get(name) != dimension:
                raise ValueError(
                    f"The operator acts on a system with the site <<{name}>> of "
                    f"dimension {dimension}, which is not in the system stored "
                    "in the file. All the operators of a file must act on the "
                    "same system, or on its subsystems."
                )
        if len(system.dimensions) == len(root_dimensions):
            return None
        return np.array(sorted(self.index[name] for name in system.sites), np.int32)

    def system_from(self, group: h5py.Group) -> SystemDescriptor:
        """The system of the operator stored in ``group``."""
        if self.system is None:
            raise HDF5FormatError(
                f"{group.file.filename} stores operators, but not their system."
            )
        if "subsystem" not in group.attrs:
            return self.system
        sites = frozenset(
            self.site_name(int(index), group.name)
            for index in np.atleast_1d(group.attrs["subsystem"])
        )
        return self.system.subsystem(sites)


# ---------------------------------------------------------------------------
#  Registry
# ---------------------------------------------------------------------------

_Writer = Callable[[h5py.Group, Any, HDF5Context], None]
_Reader = Callable[[h5py.Group, SystemDescriptor, HDF5Context], Operator]

# Classes are looked up by their exact type: an instance of a subclass that
# is not registered is not stored as its parent class, because it would be
# loaded as a different kind of object.
_WRITERS: Dict[type, Tuple[str, _Writer]] = {}
_READERS: Dict[str, _Reader] = {}


def _register(label: str, cls: type, writer: _Writer, reader: _Reader) -> None:
    _WRITERS[cls] = (label, writer)
    _READERS[label] = reader


def _registry_ready() -> None:
    """Register the classes defined in modules that depend on this one."""
    if "rpa_gibbs" in _READERS:
        return
    # pylint: disable=import-outside-toplevel
    from qalma.meanfield.rpa import RPAGibbsDensityOperator

    def read_rpa_gibbs(group, system, context):
        attrs = group.attrs
        hybrid_options = None
        if "hybrid_options" in attrs:
            hybrid_options = from_jsonable(_json_attr(group, "hybrid_options"))
        return _read_gibbs(
            group,
            system,
            context,
            cls=RPAGibbsDensityOperator,
            dynamic=bool(attrs["dynamic"]),
            n_tau=int(attrs["n_tau"]),
            static_gap=float(attrs["static_gap"]),
            hybrid=bool(attrs["hybrid"]),
            hybrid_options=hybrid_options,
        )

    def write_rpa_gibbs(group, operator, context):
        _write_gibbs(group, operator, context)
        group.attrs["dynamic"] = bool(operator.dynamic)
        group.attrs["n_tau"] = int(operator.n_tau)
        group.attrs["static_gap"] = float(operator.static_gap)
        group.attrs["hybrid"] = bool(operator.hybrid)
        if operator.hybrid_options is not None:
            group.attrs["hybrid_options"] = json_dumps(
                to_jsonable(operator.hybrid_options)
            )

    _register("rpa_gibbs", RPAGibbsDensityOperator, write_rpa_gibbs, read_rpa_gibbs)


def _json_attr(group: h5py.Group, name: str):
    """Read an attribute that stores a JSON string."""
    import json  # pylint: disable=import-outside-toplevel

    return json.loads(as_str(group.attrs[name]))


# ---------------------------------------------------------------------------
#  Public functions
# ---------------------------------------------------------------------------


def write_operator(
    parent: h5py.Group, name: str, operator: Operator, context: HDF5Context
) -> Union[h5py.Group, h5py.Dataset]:
    """Store an operator as the member ``name`` of the group ``parent``.

    Parameters
    ----------
    parent : h5py.Group
        The group where the operator is stored.
    name : str
        Name of the new group. It must not exist.
    operator : Operator
        The operator or state to store.
    context : HDF5Context
        The context of the file, with the system that the operator must
        act on.

    Raises
    ------
    TypeError
        If the class of the operator is not supported by the format, and
        ``context.allow_pickle`` is False.
    ValueError
        If the operator does not act on the system of the file.
    """
    _registry_ready()
    if context.system is None and operator.system is not None:
        context.bind(parent.file, operator.system)
    subsystem = context.subsystem_of(operator.system)
    entry = _WRITERS.get(type(operator))
    if entry is None:
        return _write_pickled_operator(parent, name, operator, context, subsystem)

    label, writer = entry
    group = parent.create_group(name)
    group.attrs["type"] = label
    if subsystem is not None:
        group.attrs["subsystem"] = subsystem
    writer(group, operator, context)
    return group


def read_operator(
    node: Union[h5py.Group, h5py.Dataset], context: HDF5Context
) -> Operator:
    """Load an operator stored by :func:`write_operator`.

    Raises
    ------
    HDF5FormatError
        If ``node`` is not a valid operator.
    PickleNotAllowedError
        If the operator was stored as a pickle and ``context.allow_pickle``
        is False.
    """
    _registry_ready()
    if isinstance(node, h5py.Dataset):
        return _read_pickled_operator(node, context)
    if "type" not in node.attrs:
        raise HDF5FormatError(
            f"{node.name} in {node.file.filename} is not an operator: "
            "it does not have the attribute <<type>>."
        )
    label = as_str(node.attrs["type"])
    reader = _READERS.get(label)
    if reader is None:
        raise HDF5FormatError(
            f"{node.name} in {node.file.filename} is of the type <<{label}>>, "
            "which this version of QALMA does not know."
        )
    system = context.system_from(node)
    try:
        return reader(node, system, context)
    except (KeyError, IndexError) as exc:
        raise HDF5FormatError(
            f"{node.name} in {node.file.filename} is not a valid operator of "
            f"the type <<{label}>>: {type(exc).__name__}: {exc}"
        ) from exc


def is_operator_node(node) -> bool:
    """Check if a member of a file was stored by :func:`write_operator`."""
    if isinstance(node, h5py.Group):
        return "type" in node.attrs
    return as_str(node.attrs.get("py_type", "")) == "pickled_operator"


# ---------------------------------------------------------------------------
#  Operators that the format does not support (allow_pickle=True)
# ---------------------------------------------------------------------------


def _write_pickled_operator(parent, name, operator, context, subsystem):
    if not context.allow_pickle:
        supported = ", ".join(sorted(cls.__name__ for cls in _WRITERS))
        raise TypeError(
            f"Operators of the class {type(operator).__name__} cannot be "
            f"stored in the HDF5 format. The supported classes are: {supported}. "
            "Use allow_pickle=True to store it as a Python pickle."
        )
    # The system is stored once in the file, not with each operator.
    system = operator.system
    cached = "_serialized" in operator.__dict__
    # pylint: disable=protected-access
    operator._set_system_(None)
    try:
        data = np.frombuffer(pickle.dumps(operator), dtype=np.uint8)
    finally:
        operator._set_system_(system)
        if not cached:
            # States cache their pickle: do not keep the one without system.
            operator.__dict__.pop("_serialized", None)
    dset = create_dataset(parent, name, data)
    dset.attrs["py_type"] = "pickled_operator"
    if subsystem is not None:
        dset.attrs["subsystem"] = subsystem
    return dset


def _read_pickled_operator(dset, context):
    if as_str(dset.attrs.get("py_type", "")) != "pickled_operator":
        raise HDF5FormatError(
            f"{dset.name} in {dset.file.filename} is not an operator."
        )
    if not context.allow_pickle:
        raise PickleNotAllowedError(
            f"{dset.name} in {dset.file.filename} is an operator stored as a "
            "Python pickle. Loading a pickle can run arbitrary code: if you "
            "trust the file, load it with allow_pickle=True."
        )
    operator = pickle.loads(dset[()].tobytes())
    if not isinstance(operator, Operator):
        raise HDF5FormatError(f"{dset.name} is not an operator.")
    # pylint: disable=protected-access
    operator._set_system_(context.system_from(dset))
    return operator


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------


def _write_flags(group: h5py.Group, **flags) -> None:
    """Store the flags that are not None as boolean attributes."""
    for name, value in flags.items():
        if value is not None:
            group.attrs[name] = bool(value)


def _read_flag(group: h5py.Group, name: str) -> Optional[bool]:
    value = group.attrs.get(name)
    return None if value is None else bool(value)


def _real(value) -> float:
    """Convert a real number, maybe stored as complex, to float."""
    return float(np.real(value))


class _FactorWriter:
    """Collect the local factors of a list of terms."""

    def __init__(self, context: HDF5Context):
        self.context = context
        self.pool = MatrixPool()
        self.site: List[int] = []
        self.matrix: List[int] = []

    def add(self, site: str, matrix) -> None:
        """Add the factor of a site."""
        self.site.append(self.context.site(site))
        self.matrix.append(self.pool.add(matrix))

    def write(self, group: h5py.Group) -> None:
        """Write the datasets ``site``, ``matrix`` and ``pool_d*``."""
        create_dataset(group, "site", np.array(self.site, dtype=np.int32))
        create_dataset(group, "matrix", np.array(self.matrix, dtype=np.int32))
        self.pool.write(group)


class _FactorReader:
    """Read the local factors written by :class:`_FactorWriter`."""

    def __init__(self, group: h5py.Group, system: SystemDescriptor, context):
        self.where = group.name
        self.pools = read_pools(group)
        site = group["site"][()]
        matrix = group["matrix"][()]
        if site.ndim != 1 or site.shape != matrix.shape:
            raise HDF5FormatError(
                f"{group.name}: <<site>> and <<matrix>> must be 1D arrays of "
                "the same length."
            )
        self.names = [context.site_name(int(index), group.name) for index in site]
        self.matrix = [int(index) for index in matrix]
        self.dimensions = system.dimensions
        for name in set(self.names):
            if name not in self.dimensions:
                raise HDF5FormatError(
                    f"{group.name} has a factor on the site <<{name}>>, which "
                    "is not in the system of the operator."
                )

    def __len__(self) -> int:
        return len(self.names)

    def factors(self, start: int, stop: int) -> Dict[str, np.ndarray]:
        """The factors in a range of positions, as a dict by site."""
        result = {}
        for pos in range(start, stop):
            name = self.names[pos]
            if name in result:
                raise HDF5FormatError(
                    f"{self.where} has two factors on the site <<{name}>> "
                    "in the same term."
                )
            result[name] = pool_matrix(
                self.pools, self.dimensions[name], self.matrix[pos], self.where
            )
        return result


def _read_aligned(group: h5py.Group, name: str, length: int) -> np.ndarray:
    """Read a 1D dataset that must have a given length."""
    data = group[name][()]
    if data.shape != (length,):
        raise HDF5FormatError(
            f"{group.name}/{name} has the shape {data.shape}, "
            f"but ({length},) was expected."
        )
    return data


# ---------------------------------------------------------------------------
#  Scalar, local and product operators
# ---------------------------------------------------------------------------


def _write_scalar(group, operator: ScalarOperator, _context) -> None:
    group.attrs["prefactor"] = complex(operator.prefactor)


def _read_scalar(group, system, _context) -> ScalarOperator:
    return ScalarOperator(as_number(group.attrs["prefactor"]), system)


def _write_local(group, operator: LocalOperator, context) -> None:
    group.attrs["site"] = context.site(operator.site)
    group.create_dataset(
        "matrix", data=np.asarray(operator.operator, dtype=np.complex128)
    )


def _read_local(group, system, context) -> LocalOperator:
    site = context.site_name(int(group.attrs["site"]), group.name)
    matrix = np.array(group["matrix"][()], dtype=np.complex128)
    dimension = system.dimensions.get(site)
    if matrix.shape != (dimension, dimension):
        raise HDF5FormatError(
            f"{group.name} is a matrix of shape {matrix.shape} on the site "
            f"<<{site}>>, of dimension {dimension}."
        )
    return LocalOperator(site, matrix, system)


def _write_product(group, operator: ProductOperator, context) -> None:
    group.attrs["prefactor"] = complex(operator.prefactor)
    factors = _FactorWriter(context)
    for site, matrix in operator.site_factors.items():
        factors.add(site, matrix)
    factors.write(group)


def _read_product(group, system, context) -> ProductOperator:
    factors = _FactorReader(group, system, context)
    return ProductOperator(
        factors.factors(0, len(factors)),
        as_number(group.attrs["prefactor"]),
        system,
    )


# ---------------------------------------------------------------------------
#  Sums
# ---------------------------------------------------------------------------


def _write_terms(group: h5py.Group, terms: Sequence[Any], context) -> None:
    """Store a sequence of operators.

    Scalar, local and product operators are packed into arrays, with a row
    per term (``kind``, ``prefactors``, ``offsets``) or per local factor
    (``site``, ``matrix``): a group per term would be much slower and
    larger. The other terms are stored as groups, in ``blocks`` (the
    QutipOperator terms) or in ``terms``.
    """
    factors = _FactorWriter(context)
    kinds = np.empty(len(terms), dtype=np.uint8)
    prefactors = np.ones(len(terms), dtype=np.complex128)
    offsets = np.zeros(len(terms) + 1, dtype=np.int64)
    blocks: List[Operator] = []
    others: List[Operator] = []

    for pos, term in enumerate(terms):
        term_type = type(term)
        if term_type is ProductOperator:
            kinds[pos] = KIND_PRODUCT
            prefactors[pos] = term.prefactor
            for site, matrix in term.site_factors.items():
                factors.add(site, matrix)
        elif term_type is LocalOperator:
            kinds[pos] = KIND_LOCAL
            factors.add(term.site, term.operator)
        elif term_type is ScalarOperator:
            kinds[pos] = KIND_SCALAR
            prefactors[pos] = term.prefactor
        elif term_type is QutipOperator:
            kinds[pos] = KIND_QUTIP
            blocks.append(term)
        else:
            kinds[pos] = KIND_GROUP
            others.append(term)
        offsets[pos + 1] = len(factors.site)

    create_dataset(group, "kind", kinds)
    create_dataset(group, "prefactors", prefactors)
    create_dataset(group, "offsets", offsets)
    factors.write(group)
    for name, operators in (("blocks", blocks), ("terms", others)):
        if operators:
            subgroup = group.create_group(name)
            for pos, operator in enumerate(operators):
                write_operator(subgroup, index_name(pos), operator, context)


def _read_terms(group: h5py.Group, system, context) -> Tuple[Operator, ...]:
    """Load the operators stored by :func:`_write_terms`."""
    kinds = group["kind"][()]
    if kinds.ndim != 1:
        raise HDF5FormatError(f"{group.name}/kind must be a 1D array.")
    n_terms = len(kinds)
    prefactors = _read_aligned(group, "prefactors", n_terms)
    offsets = _read_aligned(group, "offsets", n_terms + 1)
    factors = _FactorReader(group, system, context)
    if offsets[0] != 0 or offsets[-1] != len(factors) or (np.diff(offsets) < 0).any():
        raise HDF5FormatError(f"{group.name}/offsets is not consistent.")

    counters = {KIND_QUTIP: 0, KIND_GROUP: 0}
    subgroups = {KIND_QUTIP: "blocks", KIND_GROUP: "terms"}
    terms: List[Operator] = []
    for pos, kind in enumerate(kinds):
        start, stop = int(offsets[pos]), int(offsets[pos + 1])
        if kind == KIND_PRODUCT:
            terms.append(
                ProductOperator(
                    factors.factors(start, stop), as_number(prefactors[pos]), system
                )
            )
        elif kind == KIND_LOCAL:
            if stop != start + 1:
                raise HDF5FormatError(
                    f"{group.name}: the term {pos} is a local operator, but "
                    f"it has {stop - start} factors."
                )
            ((site, matrix),) = factors.factors(start, stop).items()
            terms.append(LocalOperator(site, matrix, system))
        elif kind == KIND_SCALAR:
            terms.append(ScalarOperator(as_number(prefactors[pos]), system))
        elif kind in subgroups:
            name = f"{subgroups[kind]}/{index_name(counters[kind])}"
            counters[kind] += 1
            if name not in group:
                raise HDF5FormatError(f"{group.name}/{name} is missing.")
            terms.append(read_operator(group[name], context))
        else:
            raise HDF5FormatError(
                f"{group.name}: the term {pos} is of the unknown kind {kind}."
            )
    return tuple(terms)


def _write_sum(group, operator: SumOperator, context) -> None:
    # pylint: disable=protected-access
    _write_flags(
        group,
        isherm=operator._isherm,
        isdiagonal=operator._isdiagonal,
        simplified=operator._simplified,
    )
    _write_terms(group, operator.terms, context)


def _read_sum(group, system, context) -> SumOperator:
    return SumOperator(
        _read_terms(group, system, context),
        system,
        isherm=_read_flag(group, "isherm"),
        isdiag=_read_flag(group, "isdiagonal"),
        simplified=_read_flag(group, "simplified"),
    )


def _read_one_body(group, system, context) -> OneBodyOperator:
    return OneBodyOperator(
        _read_terms(group, system, context),
        system,
        check_and_convert=False,
        isherm=_read_flag(group, "isherm"),
        isdiag=_read_flag(group, "isdiagonal"),
        simplified=_read_flag(group, "simplified"),
    )


# ---------------------------------------------------------------------------
#  Qutip operators
# ---------------------------------------------------------------------------


def _write_qutip(group, operator: QutipOperator, context) -> None:
    group.attrs["prefactor"] = complex(operator.prefactor)
    if operator.operator is None:
        # A multiple of the identity.
        return
    names = operator.site_names
    block = sorted(names, key=lambda site: names[site])
    if [names[site] for site in block] != list(range(len(block))):
        raise ValueError(
            f"The positions of the sites of the QutipOperator ({names}) "
            "are not consecutive."
        )
    group.create_dataset(
        "sites", data=np.array([context.site(site) for site in block], dtype=np.int32)
    )
    write_qobj(group, operator.operator)


def _read_qutip_args(group, system, context):
    """Arguments to build a QutipOperator or a QutipDensityOperator."""
    prefactor = as_number(group.attrs["prefactor"])
    if "data" not in group:
        return 1, {}, prefactor
    qobj = read_qobj(group)
    block = [context.site_name(int(index), group.name) for index in group["sites"][()]]
    dimensions = system.dimensions
    if (
        len(set(block)) != len(block)
        or not all(name in dimensions for name in block)
        or qobj.dims[0] != [dimensions[name] for name in block]
        or qobj.dims[1] != qobj.dims[0]
    ):
        raise HDF5FormatError(
            f"{group.name}: the dimensions of the matrix ({qobj.dims}) do not "
            f"match the sites {block}."
        )
    return qobj, {name: pos for pos, name in enumerate(block)}, prefactor


def _read_qutip(group, system, context) -> QutipOperator:
    qobj, names, prefactor = _read_qutip_args(group, system, context)
    return QutipOperator(qobj, system, names, prefactor)


# ---------------------------------------------------------------------------
#  Quadratic forms
# ---------------------------------------------------------------------------


def _write_quadratic(group, operator: QuadraticFormOperator, context) -> None:
    group.create_dataset(
        "weights", data=np.array(operator.weights, dtype=np.complex128)
    )
    if operator.basis:
        basis = group.create_group("basis")
        for pos, generator in enumerate(operator.basis):
            write_operator(basis, index_name(pos), generator, context)
    if operator.linear_term is not None:
        write_operator(group, "linear_term", operator.linear_term, context)
    if operator.offset is not None:
        write_operator(group, "offset", operator.offset, context)


def _read_quadratic(group, system, context) -> QuadraticFormOperator:
    weights = tuple(as_number(weight) for weight in group["weights"][()])
    basis = tuple(
        read_operator(group["basis"][index_name(pos)], context)
        for pos in range(len(weights))
    )
    linear_term = offset = None
    if "linear_term" in group:
        linear_term = read_operator(group["linear_term"], context)
    if "offset" in group:
        offset = read_operator(group["offset"], context)
    return QuadraticFormOperator(
        basis, weights, system, linear_term=linear_term, offset=offset
    )


# ---------------------------------------------------------------------------
#  States
# ---------------------------------------------------------------------------


def _warn_dropped(group, dropped, stacklevel) -> None:
    warnings.warn(
        f"{group.name}: the state is stored without its "
        f"{' and '.join(dropped)}, because functions cannot be stored in the "
        "HDF5 format.",
        IncompleteStateWarning,
        stacklevel=stacklevel,
    )


def _write_gibbs(group, operator: GibbsDensityOperator, context) -> None:
    # pylint: disable=protected-access
    group.attrs["prefactor"] = _real(operator.prefactor)
    group.attrs["normalized"] = bool(operator.normalized)
    group.attrs["free_energy"] = _real(operator._free_energy)
    write_operator(group, "k", operator.k, context)
    # The mean field state is a cache, but it can be expensive to compute.
    # With prefactor=0, the constructor sets it.
    if operator._meanfield is not None and operator.prefactor != 0:
        write_operator(group, "meanfield", operator._meanfield, context)
    if operator.symmetry_projections:
        dropped = ["symmetry_projections"]
        group.attrs["dropped"] = dropped
        _warn_dropped(group, dropped, stacklevel=4)


def _read_gibbs(group, system, context, cls=GibbsDensityOperator, **kwargs):
    # pylint: disable=protected-access
    meanfield = None
    if "meanfield" in group:
        meanfield = read_operator(group["meanfield"], context)
    state = cls(
        read_operator(group["k"], context),
        system=system,
        prefactor=float(group.attrs["prefactor"]),
        normalized=bool(group.attrs["normalized"]),
        meanfield=meanfield,
        **kwargs,
    )
    state._free_energy = float(group.attrs.get("free_energy", 0.0))
    if "dropped" in group.attrs:
        dropped = [as_str(name) for name in np.atleast_1d(group.attrs["dropped"])]
        _warn_dropped(group, dropped, stacklevel=4)
    return state


def _write_gibbs_product(group, operator: GibbsProductDensityOperator, context):
    group.attrs["prefactor"] = _real(operator.prefactor)
    factors = _FactorWriter(context)
    free_energies = []
    for site, local_k in operator.k_by_site.items():
        factors.add(site, local_k.full())  # type: ignore[attr-defined]
        free_energies.append(_real(operator.free_energies[site]))
    factors.write(group)
    group.create_dataset("free_energies", data=np.array(free_energies, np.float64))


def _read_gibbs_product(group, system, context) -> GibbsProductDensityOperator:
    factors = _FactorReader(group, system, context)
    free_energies = _read_aligned(group, "free_energies", len(factors))
    k_by_site = {
        site: qutip.Qobj(np.array(matrix))
        for site, matrix in factors.factors(0, len(factors)).items()
    }
    state = GibbsProductDensityOperator(
        k_by_site, system, float(group.attrs["prefactor"]), normalized=True
    )
    # With normalized=True, the constructor sets the free energies to 0.
    state.free_energies.update(
        {site: float(f_site) for site, f_site in zip(factors.names, free_energies)}
    )
    return state


def _write_product_state(group, operator: ProductDensityOperator, context) -> None:
    group.attrs["prefactor"] = _real(operator.prefactor)
    factors = _FactorWriter(context)
    for site, matrix in operator.site_factors.items():
        factors.add(site, matrix)
    factors.write(group)
    local_fs = operator.local_fs
    group.create_dataset(
        "local_fs_site",
        data=np.array([context.site(site) for site in local_fs], dtype=np.int32),
    )
    group.create_dataset(
        "local_fs",
        data=np.array([_real(f_site) for f_site in local_fs.values()], np.float64),
    )


def _read_product_state(group, system, context) -> ProductDensityOperator:
    factors = _FactorReader(group, system, context)
    state = ProductDensityOperator(
        factors.factors(0, len(factors)),
        float(group.attrs["prefactor"]),
        system=system,
        normalized=True,
    )
    # With normalized=True, the constructor sets the local free energies
    # to 0.
    sites = group["local_fs_site"][()]
    local_fs = _read_aligned(group, "local_fs", len(sites))
    state.local_fs = {
        context.site_name(int(index), group.name): float(f_site)
        for index, f_site in zip(sites, local_fs)
    }
    return state


def _write_mixture(group, operator: MixtureDensityOperator, context) -> None:
    _write_terms(group, operator.terms, context)


def _read_mixture(group, system, context) -> MixtureDensityOperator:
    return MixtureDensityOperator(_read_terms(group, system, context), system)


def _write_qutip_state(group, operator: QutipDensityOperator, context) -> None:
    # pylint: disable=protected-access
    _write_qutip(group, operator, context)
    group.attrs["prefactor"] = _real(operator.prefactor)
    group.attrs["normalized"] = bool(operator._normalized)


def _read_qutip_state(group, system, context) -> QutipDensityOperator:
    qobj, names, prefactor = _read_qutip_args(group, system, context)
    normalized = bool(group.attrs["normalized"]) or not isinstance(qobj, qutip.Qobj)
    return QutipDensityOperator(
        qobj, system, names, _real(prefactor), normalized=normalized
    )


_register("scalar", ScalarOperator, _write_scalar, _read_scalar)
_register("local", LocalOperator, _write_local, _read_local)
_register("product", ProductOperator, _write_product, _read_product)
_register("sum", SumOperator, _write_sum, _read_sum)
_register("one_body", OneBodyOperator, _write_sum, _read_one_body)
_register("qutip", QutipOperator, _write_qutip, _read_qutip)
_register("quadratic", QuadraticFormOperator, _write_quadratic, _read_quadratic)
_register("gibbs", GibbsDensityOperator, _write_gibbs, _read_gibbs)
_register(
    "gibbs_product",
    GibbsProductDensityOperator,
    _write_gibbs_product,
    _read_gibbs_product,
)
_register(
    "product_state", ProductDensityOperator, _write_product_state, _read_product_state
)
_register("mixture", MixtureDensityOperator, _write_mixture, _read_mixture)
_register("qutip_state", QutipDensityOperator, _write_qutip_state, _read_qutip_state)
