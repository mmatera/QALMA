"""Storage of a :class:`~qalma.model.SystemDescriptor` in an HDF5 group.

The system is stored as a self-contained description, which does not depend
on the ALPS libraries from which the system was built:

``spec``
    A JSON string with the graph, the model and the parameters.
``sites``
    A table with the name, the dimension and the kind of local basis of
    each site. Operators refer to the sites by their position in this table.
``pool_d{dim}``
    The matrices of the local operators of each kind of site.

No Python object is serialized: loading a system only builds graphs, models
and matrices from numbers and strings.
"""

import json
from typing import Any, Dict, List, Tuple

import h5py
import numpy as np
import qutip  # type: ignore[import-untyped]

from qalma.alpsmodels import ModelDescriptor
from qalma.geometry import GraphDescriptor
from qalma.model import SystemDescriptor

from ._common import (
    STRING_DTYPE,
    HDF5FormatError,
    MatrixPool,
    as_str,
    from_jsonable,
    json_dumps,
    pool_matrix,
    qobj_storage,
    qobj_with_storage,
    read_pools,
    to_jsonable,
)

__all__ = ["read_system", "write_system"]

SITES_DTYPE = np.dtype(
    [("name", STRING_DTYPE), ("dimension", np.int64), ("basis", STRING_DTYPE)]
)


def _string_keys(mapping: dict, what: str) -> dict:
    """Check that all the keys of a dict are strings."""
    for key in mapping:
        if not isinstance(key, str):
            raise TypeError(
                f"{what} must have strings as keys, but {key!r} is of type "
                f"{type(key).__name__}."
            )
    return mapping


# ---------------------------------------------------------------------------
#  Graph
# ---------------------------------------------------------------------------


def _graph_to_spec(graph: GraphDescriptor) -> dict:
    """Describe a graph with JSON types."""
    return {
        "name": graph.name,
        "nodes": {
            name: {
                key: to_jsonable(val, "ndarray" if key == "coords" else "list")
                for key, val in _string_keys(attrs, "The attributes of a node").items()
            }
            for name, attrs in _string_keys(graph.nodes, "graph.nodes").items()
        },
        "edges": {
            edge_type: [to_jsonable(edge, "tuple") for edge in edges]
            for edge_type, edges in _string_keys(graph.edges, "graph.edges").items()
        },
        "loops": to_jsonable(graph.loops),
        "lattice": to_jsonable(graph.lattice),
        "parms": to_jsonable(graph.parms),
    }


def _graph_from_spec(spec: dict) -> GraphDescriptor:
    """Build a graph from its description."""
    nodes = {
        name: {
            key: from_jsonable(val, "ndarray" if key == "coords" else "list")
            for key, val in attrs.items()
        }
        for name, attrs in spec["nodes"].items()
    }
    edges = {
        edge_type: [from_jsonable(edge, "tuple") for edge in edge_list]
        for edge_type, edge_list in spec["edges"].items()
    }
    return GraphDescriptor(
        name=spec["name"],
        nodes=nodes,
        edges=edges,
        loops=from_jsonable(spec["loops"]),
        lattice=from_jsonable(spec["lattice"]),
        parms=from_jsonable(spec["parms"]),
    )


# ---------------------------------------------------------------------------
#  Site basis
# ---------------------------------------------------------------------------


def _site_basis_to_spec(site_basis: dict, pool: MatrixPool) -> dict:
    """Describe the local basis of a kind of site.

    The matrices of the local operators are added to ``pool``, and the
    description stores their positions.
    """
    spec: Dict[str, Any] = {}
    for key, val in _string_keys(site_basis, "A site basis").items():
        if key in ("identity", "operators"):
            continue
        if key == "localstates":
            spec[key] = [to_jsonable(state, "tuple") for state in val]
        else:
            spec[key] = to_jsonable(val)

    operators = {}
    for name, operator in _string_keys(
        site_basis.get("operators", {}), "The operators of a site basis"
    ).items():
        if not isinstance(operator, qutip.Qobj):
            raise TypeError(
                f"The local operator <<{name}>> is of type "
                f"{type(operator).__name__}; a qutip.Qobj was expected."
            )
        op_spec = {
            "matrix": pool.add(operator.full()),
            "storage": qobj_storage(operator),
        }
        if hasattr(operator, "fermionic"):
            op_spec["fermionic"] = bool(operator.fermionic)
        operators[name] = op_spec
    spec["operators"] = operators
    return spec


def _site_basis_from_spec(spec: dict, pools: Dict[int, np.ndarray]) -> dict:
    """Build the local basis of a kind of site from its description."""
    site_basis: Dict[str, Any] = {}
    for key, val in spec.items():
        if key == "operators":
            continue
        if key == "localstates":
            site_basis[key] = [from_jsonable(state, "tuple") for state in val]
        else:
            site_basis[key] = from_jsonable(val)

    dimension = int(site_basis["dimension"])
    operators = {}
    for name, op_spec in spec["operators"].items():
        matrix = pool_matrix(
            pools, dimension, int(op_spec["matrix"]), f"The local operator <<{name}>>"
        )
        operator = qobj_with_storage(
            qutip.Qobj(np.array(matrix)), op_spec.get("storage")
        )
        if "fermionic" in op_spec:
            operator.fermionic = bool(op_spec["fermionic"])
        operators[name] = operator

    # In the systems built by QALMA, the identity of the site and the
    # operator called "identity" are the same object.
    identity = operators.get("identity")
    if identity is None:
        identity = qutip.qeye(dimension)
    site_basis["identity"] = identity
    site_basis["operators"] = operators
    return site_basis


# ---------------------------------------------------------------------------
#  System
# ---------------------------------------------------------------------------


def _system_to_spec(
    system: SystemDescriptor, pool: MatrixPool
) -> Tuple[dict, List[str]]:
    """Describe a system with JSON types.

    Returns
    -------
    spec : dict
        The description of the system.
    site_basis_keys : list[str]
        For each site of the system, the name of its local basis in
        ``spec["model"]["site_basis"]`` or ``spec["extra_site_basis"]``.
    """
    graph: GraphDescriptor = system.spec["graph"]
    model: ModelDescriptor = system.spec["model"]

    model_basis = _string_keys(model.site_basis, "model.site_basis")
    basis_specs = {
        key: _site_basis_to_spec(site_basis, pool)
        for key, site_basis in model_basis.items()
    }
    # Each site uses one of the local basis of the model. This is not
    # enforced by SystemDescriptor, so any other basis is stored apart.
    extra_basis: Dict[str, dict] = {}
    extra_basis_objs: List[Tuple[dict, str]] = []
    site_basis_keys = []
    for name, site in _string_keys(system.sites, "system.sites").items():
        key = next((k for k, basis in model_basis.items() if basis is site), None)
        if key is None:
            key = next((k for basis, k in extra_basis_objs if basis is site), None)
        if key is None:
            key = f"site:{name}"
            extra_basis[key] = _site_basis_to_spec(site, pool)
            extra_basis_objs.append((site, key))
        site_basis_keys.append(key)

    spec = {
        "name": system.name,
        "parms": to_jsonable(system.spec["parms"]),
        "graph": _graph_to_spec(graph),
        "model": {
            "name": model.name,
            "parms": to_jsonable(model.parms),
            "constraints": to_jsonable(model.constraints),
            "bond_ops": to_jsonable(model.bond_ops),
            "global_ops": to_jsonable(model.global_ops),
            "site_basis": basis_specs,
        },
    }
    if extra_basis:
        spec["extra_site_basis"] = extra_basis
    return spec, site_basis_keys


def write_system(group: h5py.Group, system: SystemDescriptor) -> None:
    """Store a system in an HDF5 group.

    Parameters
    ----------
    group : h5py.Group
        An empty group.
    system : SystemDescriptor
        The system to store.

    Raises
    ------
    TypeError
        If the description of the system contains objects that cannot be
        stored (for instance, parameters that are not numbers or strings).
    """
    pool = MatrixPool()
    spec, site_basis_keys = _system_to_spec(system, pool)
    group.create_dataset("spec", data=json_dumps(spec), dtype=STRING_DTYPE)
    sites = np.empty(len(system.sites), dtype=SITES_DTYPE)
    for pos, (name, basis) in enumerate(zip(system.sites, site_basis_keys)):
        sites[pos] = (name, system.dimensions[name], basis)
    group.create_dataset("sites", data=sites)
    pool.write(group)


def read_sites(group: h5py.Group) -> Tuple[List[str], List[int]]:
    """Read the names and the dimensions of the sites of a stored system."""
    try:
        sites = group["sites"]
        names = [as_str(name) for name in sites.fields("name")[()]]
        dimensions = [int(dim) for dim in sites.fields("dimension")[()]]
    except (KeyError, ValueError) as exc:
        raise HDF5FormatError(
            f"{group.name} does not have a valid table of sites: {exc}"
        ) from exc
    return names, dimensions


def read_system(group: h5py.Group) -> SystemDescriptor:
    """Load a system stored by :func:`write_system`.

    The system is built from its description: nothing is unpickled, and
    the expressions that define the operators of the model are restricted
    to arithmetic (see :func:`qalma.utils.eval_expr`).

    Raises
    ------
    HDF5FormatError
        If the group does not contain a valid description of a system.
    """
    where = f"{group.name} in {group.file.filename}"
    try:
        spec = json.loads(as_str(group["spec"][()]))
        names, dimensions = read_sites(group)
        basis_keys = [as_str(key) for key in group["sites"].fields("basis")[()]]
        pools = read_pools(group)

        model_spec = spec["model"]
        model_basis = {
            key: _site_basis_from_spec(basis_spec, pools)
            for key, basis_spec in model_spec["site_basis"].items()
        }
        all_basis = dict(model_basis)
        for key, basis_spec in spec.get("extra_site_basis", {}).items():
            all_basis[key] = _site_basis_from_spec(basis_spec, pools)

        model = ModelDescriptor(
            site_basis=model_basis,
            constraints=from_jsonable(model_spec["constraints"]),
            bond_op_descr=from_jsonable(model_spec["bond_ops"]),
            global_op_descr=from_jsonable(model_spec["global_ops"]),
            parms=from_jsonable(model_spec["parms"]),
            name=model_spec["name"],
        )
        graph = _graph_from_spec(spec["graph"])
        sites = {name: all_basis[key] for name, key in zip(names, basis_keys)}
        parms = from_jsonable(spec["parms"])
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        if isinstance(exc, HDF5FormatError):
            raise
        raise HDF5FormatError(
            f"{where} is not a valid description of a system: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    system = SystemDescriptor(graph, model, parms, sites=sites)
    if dict(system.dimensions) != dict(zip(names, dimensions)):
        raise HDF5FormatError(
            f"The dimensions in the table of sites of {where} do not match "
            "the description of the system."
        )
    return system
