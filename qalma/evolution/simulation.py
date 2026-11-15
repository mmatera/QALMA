"""
Simulation class.

This module defines the `Simulation` dataclass containing the
state and the result of a simulation.
`Simulation` objects can be serialized both as Python pickle and
as  HDF5 files.

To serialize a Simulation object, use the method `save_hdf5`:

.. code-block:: python

    sim.save_hdf5(filename)


To load back the simulation, use the classmethod ``load_hdf5``:

.. code-block:: python

    sim = Simulation.load_hdf5(filename)

"""

import logging
import pickle
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import h5py
import numpy as np

from qalma.model import SystemDescriptor
from qalma.operators.basic import Operator
from qalma.operators.pauli import PauliStringOperator
from qalma.operators.pauli_sum import PauliSumOperator
from qalma.operators.hdf5 import (
    is_operator_group,
    load_operator,
    save_operator,
)

logger = logging.getLogger(__name__)

__all__ = ["Simulation", "SimulationHDF5"]


def store_hdf5_dict(group: h5py.Group, data_dict: Dict[str, Any]):
    """Store data in a Python dict in a group of a hdf5 file.

    Basic types and numpy.ndarray values in data_dict are stored as
    native HDF5 data types. Operators with a native HDF5 format (Pauli
    strings and sums of them, see :mod:`qalma.operators.hdf5`) are stored as
    subgroups. Other objects are stored as pickle byte streams.
    """
    for key, value in data_dict.items():
        key_str = str(key)
        if isinstance(value, (PauliStringOperator, PauliSumOperator)):
            if key_str in group:
                del group[key_str]
            save_operator(group, key_str, value)
            continue
        if isinstance(value, str):
            if key_str in group:
                del group[key_str]
            dset = group.create_dataset(
                key_str, data=value, dtype=h5py.string_dtype(encoding="utf-8")
            )
            dset.attrs["encoding"] = "utf-8"
            continue
        if any(isinstance(value, t) for t in [int, float, bool, np.ndarray]):
            if key_str in group:
                del group[key_str]

            dset = group.create_dataset(key_str, data=value)
        elif isinstance(value, (list, tuple)) and all(
            isinstance(
                x,
                (
                    int,
                    float,
                    bool,
                ),
            )
            for x in value
        ):
            if key_str in group:
                del group[key_str]
            group.create_dataset(key_str, data=np.array(value))
        else:
            data = np.frombuffer(pickle.dumps(value), dtype=np.uint8)
            if key_str in group:
                del group[key_str]
            dset = group.create_dataset(
                key_str, shape=(1,), dtype=h5py.vlen_dtype(np.dtype("V1"))
            )
            dset.attrs["pickled"] = True
            dset[0] = data


def store_system(group, system):
    """Store a SystemDescription on the group."""
    data = np.frombuffer(pickle.dumps(system), dtype=np.uint8)
    dset = group.create_dataset(
        "system", shape=(1,), dtype=h5py.vlen_dtype(np.dtype("V1"))
    )
    dset[0] = data


def store_state(key, state, group, system=None):
    """Serialize a single state operator into an HDF5 group.

    The state is pickled and stored as a variable-length byte dataset
    under ``key`` in ``group``, with gzip compression. If ``system`` is
    provided, the system reference is temporarily detached before pickling
    to avoid duplicating the system object across every stored state.

    Parameters
    ----------
    key : str
        Dataset name to use within ``group``.
    state : Operator
        The density operator or state to store.
    group : h5py.Group
        HDF5 group in which the dataset will be created.
    system : SystemDescriptor or None, optional
        If provided, the state's system reference is cleared before
        pickling and restored afterwards, reducing file size.

    """
    if system is not None:

        def serialize_state(rho):
            """
            Serialize the state attribute.

            Pickle ``rho`` after temporarily clearing its system
            reference.
            """
            state_sys = rho.system
            rho._set_system_(None)
            data = pickle.dumps(rho)
            rho._set_system_(state_sys)
            return data

    else:

        def serialize_state(rho):
            """
            Serialize the state attribute.

            Pickle ``rho`` with its system reference intact.
            """
            return pickle.dumps(rho)

    group.create_dataset(
        key,
        shape=(1,),
        compression="gzip",
        dtype=h5py.vlen_dtype(np.dtype("V1")),
    )[0] = np.frombuffer(serialize_state(state), dtype=np.uint8)


def load_hdf5_dict(
    group: h5py.Group, system: Optional[SystemDescriptor] = None
) -> Dict[str, Any]:
    """Load data from a Python dict stored as a group in a hdf5 file.

    Operators stored natively are loaded into ``system`` (or, if it is
    ``None``, into a system built from the stored site names).
    """
    loaded_dict = {}
    for key_str in group.keys():
        dataset = group[key_str]
        if is_operator_group(dataset):
            loaded_dict[key_str] = load_operator(dataset, system)
            continue
        if isinstance(dataset, h5py.Group):
            logger.warning("group %s is not an operator: skipped.", dataset.name)
            continue
        raw_data = dataset[()]
        if dataset.dtype.kind == "O":
            if isinstance(raw_data, (bytes, np.bytes_)):
                if "encoding" in dataset.attrs:
                    loaded_dict[key_str] = raw_data.decode("utf-8")
                    continue
        if dataset.attrs.get("pickled", False):
            try:
                loaded_dict[key_str] = pickle.loads(dataset[0].tobytes())
                continue
            except (pickle.UnpicklingError, EOFError, ValueError):
                logger.warning(
                    " key %s was marked as containing a pickled object, but cannot be loaded.",
                    key_str,
                )
                pass
        loaded_dict[key_str] = raw_data
        continue

    return loaded_dict


@dataclass
class Simulation:
    """Hold the state and result of a simulation."""

    parameters: Dict[Any, Any]
    stats: Dict[Any, Any]
    time_span: List[float]
    expect_ops: Dict[Any, Any]
    states: List[Operator]

    def save_hdf5(self, filename: str, mode="w-"):
        """Serialize the object as an hdf5 file."""
        try:
            with h5py.File(filename, mode) as f:
                store_hdf5_dict(f.create_group("parameters"), self.parameters)
                store_hdf5_dict(f.create_group("stats"), self.stats)
                store_hdf5_dict(f.create_group("expect obs"), self.expect_ops)
                time_span = self.time_span
                f.create_dataset("time span", data=np.array(time_span))
                states = self.states
                if states:
                    states_group = f.create_group("states")
                    system = states[0].system
                    # Store the system
                    store_system(f, system)
                    for i, rho in enumerate(states):
                        key = "rho_" + f"{i}".rjust(6, "0")
                        store_state(key, rho, states_group, system)
        except (PermissionError,) as exc:
            logger.warning(
                "The object could not be stored in %s. (%s)", filename, str(exc)
            )
            return

    @classmethod
    def load(cls, filename: str):
        """Load a simulation from a file, trying HDF5 first then pickle.

        Attempts to deserialize from an HDF5 file via :meth:`load_hdf5`.
        If the file is not a valid HDF5 file, falls back to unpickling.

        Parameters
        ----------
        filename : str
            Path to the file to load.

        Returns
        -------
        Simulation or None
            The loaded simulation, or ``None`` if neither format succeeds.

        """
        try:
            sim = cls.load_hdf5(filename)
        except OSError:
            sim = None
        if sim is not None:
            return sim
        try:
            with open(filename, "rb") as f:
                return pickle.load(f)
        except pickle.UnpicklingError:
            return None

    @classmethod
    def load_hdf5(cls, filename: str):
        """Load an object serialized as an hdf5 file."""
        return SimulationHDF5(filename)
        try:
            with h5py.File(filename, "r") as f:
                parameters = load_hdf5_dict(f["parameters"])
                stats = load_hdf5_dict(f["stats"])
                expect_ops = load_hdf5_dict(f["expect obs"])
                stored_time_span = f["time span"]
                # Ensure that time span follows the right order
                for key, data in expect_ops.items():
                    expect_ops[key] = np.array(
                        [
                            q
                            for t, q in sorted(
                                zip(stored_time_span, data), key=lambda x: x[0]
                            )
                        ]
                    )
                time_span = sorted(stored_time_span)
                states = []
                for state in hdf5_state_iterator(f):
                    states.append(state)
                    assert (
                        states[0].system == states[-1].system
                    ), f"{states[0].system}!={states[-1].system}"

            return cls(parameters, stats, time_span, expect_ops, states)

        except (
            FileNotFoundError,
            OSError,
            PermissionError,
        ) as exc:
            logger.warning(
                "File %s not found. Simulation object could not be loaded (%s).",
                filename,
                str(exc),
            )
            return None


class SimulationHDF5(Simulation):
    """Interface to read and write simulations stored as HDF5 files.

    The interface avois to load all the states of a file at once, to
    avoid saturate the memory.
    """

    class StateList:
        """Lazy list of states backed by an HDF5 file.

        Provides list-like access (index, iteration, append, extend) to
        states stored in the ``states`` group of the HDF5 file, loading
        each state on demand rather than all at once. This avoids
        saturating memory for long simulations with many stored states.
        """

        def __init__(self, filename, system):
            """
            Initialize the Simulation object.

            Parameters
            ----------
            filename : str
                Path to the HDF5 file containing the simulation.
            system : SystemDescriptor
                System descriptor used to restore state references after
                unpickling.

            """
            self.filename = filename
            with h5py.File(filename, "r") as f:
                self.system = system_from_hdf5(f)

        def __iter__(self):
            """Retrieve items."""
            with h5py.File(self.filename, "r") as f:
                group = f.get("states", None)
                if group is None:
                    return
                yield from hdf5_state_iterator(f, self.system)

        def __getitem__(self, idx: int):
            """Get item from serialization."""
            key = "rho_" + f"{idx}".rjust(6, "0")
            with h5py.File(self.filename, "r") as f:
                group = f.get("states", None)
                if group is None:
                    raise ValueError("there are no stored elements")
                return state_from_hdf5(f, key, self.system)
            raise ValueError("element out of range")

        def __setitem__(self, idx: int, value: Operator):
            """Set item into serialization."""
            key = "rho_" + f"{idx}".rjust(6, "0")
            with h5py.File(self.filename, "r+") as f:
                group = f.get("states", None)
                if group is None:
                    group = f.create_group("states")

                store_state(key, value, group, self.system)

        def __len__(self):
            """Compute the length."""
            with h5py.File(self.filename, "r") as f:
                group = f.get("states", None)
                if group is None:
                    return 0
                return len(group)

        @staticmethod
        def _next_key(group) -> str:
            """Return the next available state key in ``group``."""
            idx = len(group)
            key = "rho_" + f"{idx}".rjust(6, "0")
            while key in group:
                idx += 1
                key = "rho_" + f"{idx}".rjust(6, "0")
            return key

        def append(self, elem):
            """Append a state to the end of the HDF5 states group.

            Parameters
            ----------
            elem : Operator
                The state to append.

            """
            with h5py.File(self.filename, "r+") as f:
                group = f.require_group("states")
                self[self._next_key(group)] = elem

        def extend(self, elems):
            """Append multiple states to the HDF5 states group.

            Parameters
            ----------
            elems : Iterable[Operator]
                The states to append, in order.

            """
            with h5py.File(self.filename, "r+") as f:
                group = f.require_group("states")
                for elem in elems:
                    self[self._next_key(group)] = elem

    def __init__(self, filename):
        """Create an interface with an HDF5 file that stores a simulation."""
        self.filename = filename
        with h5py.File(filename, "r") as f:
            self.system = system_from_hdf5(f)
            self.parameters = load_hdf5_dict(f["parameters"], self.system)
            self.stats = load_hdf5_dict(f["stats"], self.system)
            self.expect_ops = load_hdf5_dict(f["expect obs"], self.system)
            stored_time_span = list(f["time span"])
            self.time_span = stored_time_span
            self.key_map = {
                t: f"{pos}".rjust(6, "0") for pos, t in enumerate(stored_time_span)
            }
        self.states = self.StateList(self.filename, self.system)

    def save_hdf5(self, filename: str, mode="r+"):
        """Serialize the object as an hdf5 file."""
        # Here we assume that the states are serialized on the fly.
        try:
            with h5py.File(self.filename, mode) as f:
                store_hdf5_dict(f["parameters"], self.parameters)
                store_hdf5_dict(f["stats"], self.stats)
                store_hdf5_dict(f["expect obs"], self.expect_ops)
                time_span = self.time_span
                f["time span"][:] = np.array(time_span)
        except (PermissionError,) as exc:
            logger.warning(
                "The object could not be stored in %s. (%s)", filename, str(exc)
            )
            return


def hdf5_state_iterator(group: h5py.Group, system: Optional[SystemDescriptor] = None):
    """Yield states from a hdf5 group."""
    if system is None:
        system = system_from_hdf5(group)

    key_time_map = {
        t: "rho_" + f"{i}".rjust(6, "0")
        for i, t in enumerate(group.get("time span", []))
    }
    if "states" not in group:
        logger.info("no states defined in %s", group)
        return
    assert system is not None

    for t in sorted(key_time_map):
        key = key_time_map[t]
        state = state_from_hdf5(group, key, system)
        assert state.system is system
        yield state


def state_from_hdf5(
    group: h5py.Group, key: str, system: Optional[SystemDescriptor] = None
):
    """Read a state stored in a hdf5 file by its key name."""
    if system is None:
        system = system_from_hdf5(group)
    try:
        state_bytes = group["states"][key][0]
    except KeyError:
        logger.info("key %s not in %s", key, group)
        return None

    assert system is not None
    entry = pickle.loads(
        state_bytes.tobytes() if isinstance(state_bytes, np.void) else state_bytes
    )
    state = entry if isinstance(entry, Operator) else None
    # If the element is a tuple or an iterable object,
    # return the first element.
    if state is None:
        for elem in entry:
            if isinstance(elem, Operator):
                state = elem
                break

    if state is not None:
        if system:
            state._set_system_(system)

    return state


def system_from_hdf5(group: h5py.Group):
    """Read the SystemDescriptor stored in a hdf5 group."""
    try:
        system_bytes = group["system"][0]
        return pickle.loads(
            system_bytes.tobytes()
            if isinstance(system_bytes, np.void)
            else system_bytes
        )
    except KeyError:
        return None
