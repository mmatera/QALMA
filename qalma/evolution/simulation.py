"""
Simulation class.

This module defines the `Simulation` dataclass containing the
state and the result of a simulation.

`Simulation` objects are stored as HDF5 files with the method `save_hdf5`:

.. code-block:: python

    sim.save_hdf5(filename)


To load back the simulation, use the classmethod ``load_hdf5``:

.. code-block:: python

    sim = Simulation.load_hdf5(filename)


The files do not contain Python pickles: the system, the operators and the
states are stored as numbers, strings and arrays (see :mod:`qalma.io` and
the section *HDF5 file format* of the user guide). Files written by older
versions of QALMA, which contain pickles, are loaded with

.. code-block:: python

    sim = Simulation.load_hdf5(filename, allow_pickle=True)

"""

import logging
import os
import pickle
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

import h5py
import numpy as np

from qalma.io import (
    FORMAT_SIMULATION,
    HDF5Context,
    HDF5FormatError,
    PickleNotAllowedError,
    atomic_file,
    legacy,
    load_dict,
    read_format_version,
    read_operator,
    store_dict,
    write_format_attrs,
    write_operator,
)
from qalma.io._common import index_name
from qalma.model import SystemDescriptor
from qalma.operators.basic import Operator

logger = logging.getLogger(__name__)

__all__ = ["Simulation", "SimulationHDF5"]

# Attributes of a Simulation stored as dictionaries, and their groups.
_DICT_GROUPS = ("parameters", "stats", "expect_ops")
_TIME_SPAN = "time_span"


def store_hdf5_dict(group: h5py.Group, data_dict: Dict[str, Any]):
    """Store data in a Python dict in a group of a hdf5 file.

    Alias of :func:`qalma.io.store_dict`.
    """
    store_dict(group, data_dict)


def load_hdf5_dict(group: h5py.Group) -> Dict[str, Any]:
    """Load data from a Python dict stored as a group in a hdf5 file.

    Alias of :func:`qalma.io.load_dict`.
    """
    return load_dict(group)


def _write_metadata(h5file: h5py.File, simulation, context: HDF5Context) -> None:
    """Write the dictionaries and the times of a simulation."""
    for name in _DICT_GROUPS:
        if name in h5file:
            group = h5file[name]
        else:
            group = h5file.create_group(name, track_order=True)
        store_dict(group, getattr(simulation, name), context, name)
    if _TIME_SPAN in h5file:
        del h5file[_TIME_SPAN]
    h5file.create_dataset(
        _TIME_SPAN, data=np.array(simulation.time_span, dtype=np.float64)
    )


def _write_state(group: h5py.Group, name: str, state, context, time=None) -> None:
    """Write a state as the member ``name`` of the group of states."""
    if not isinstance(state, Operator):
        raise TypeError(
            f"The state {name} is of the type {type(state).__name__}; "
            "an Operator was expected."
        )
    node = write_operator(group, name, state, context)
    if time is not None:
        node.attrs["time"] = float(time)


class _OpenFile:
    """Open the file of a simulation, with its context."""

    def __init__(self, simulation: "SimulationHDF5", mode: str):
        """Check that the file can be opened in the given mode."""
        if mode != "r" and simulation.format_version == 0:
            raise HDF5FormatError(
                f"{simulation.filename} was written by an old version of "
                "QALMA, and it is read-only. To convert it to the current "
                "format, save the simulation to a new file "
                "(simulation.save_hdf5(new_filename)) or use "
                "tools/pickle_to_hdf5.py."
            )
        self.simulation = simulation
        self.mode = mode
        self.h5file = None

    def __enter__(self):
        simulation = self.simulation
        self.h5file = h5py.File(simulation.filename, self.mode)
        try:
            if simulation.format_version == 0:
                context = HDF5Context(
                    simulation.system, allow_pickle=simulation.allow_pickle
                )
            else:
                context = HDF5Context.open(
                    self.h5file, simulation.system, simulation.allow_pickle
                )
        except BaseException:
            self.h5file.close()
            raise
        return self.h5file, context

    def __exit__(self, *exc_info):
        self.h5file.close()
        return False


@dataclass
class Simulation:
    """Hold the state and result of a simulation."""

    parameters: Dict[Any, Any]
    stats: Dict[Any, Any]
    time_span: List[float]
    expect_ops: Dict[Any, Any]
    states: List[Operator]

    def _root_system(self) -> Optional[SystemDescriptor]:
        """The system of the simulation: the largest one of its objects."""
        systems = [
            value
            for value in self.parameters.values()
            if isinstance(value, SystemDescriptor)
        ]
        systems.extend(state.system for state in self.states)
        systems = [system for system in systems if system is not None]
        if not systems:
            return None
        return max(systems, key=lambda system: len(system.sites))

    def save_hdf5(self, filename, mode: str = "w-", *, allow_pickle: bool = False):
        """Store the simulation in an HDF5 file.

        The file is written with a temporary name and then renamed, so an
        error does not leave an incomplete file.

        Parameters
        ----------
        filename : str or path-like
            The file to write.
        mode : str, optional
            ``"w-"`` (default) fails if the file exists; ``"w"`` overwrites
            it.
        allow_pickle : bool, optional
            Store the objects that the format does not support (for
            instance, arbitrary objects in ``parameters``) as Python
            pickles. By default, those objects raise a ``TypeError``, so
            the file does not contain pickles.

        Raises
        ------
        FileExistsError
            If the file exists and ``mode`` is not ``"w"``.
        TypeError
            If the simulation has objects that cannot be stored.
        ValueError
            If the states do not act on the same system.
        """
        system = self._root_system()
        with atomic_file(filename, mode) as h5file:
            write_format_attrs(h5file, FORMAT_SIMULATION)
            context = HDF5Context(None, allow_pickle=allow_pickle)
            if system is not None:
                context.bind(h5file, system)
            _write_metadata(h5file, self, context)
            states_group = h5file.create_group("states")
            times = list(self.time_span)
            aligned = len(times) == len(self.states)
            for index, state in enumerate(self.states):
                _write_state(
                    states_group,
                    index_name(index),
                    state,
                    context,
                    times[index] if aligned else None,
                )

    @classmethod
    def load(cls, filename, *, allow_pickle: bool = False):
        """Load a simulation from an HDF5 file or from a pickle file.

        Parameters
        ----------
        filename : str or path-like
            Path to the file to load.
        allow_pickle : bool, optional
            Allow to load pickle files, and HDF5 files that contain
            pickles. Loading a pickle can run arbitrary code: use it only
            with files that you trust.

        Returns
        -------
        Simulation
            The loaded simulation.

        Raises
        ------
        PickleNotAllowedError
            If the file is, or contains, a pickle and ``allow_pickle`` is
            False.
        """
        if h5py.is_hdf5(filename):
            return cls.load_hdf5(filename, allow_pickle=allow_pickle)
        # Raise the usual errors if the file cannot be read.
        with open(filename, "rb") as stream:
            if not allow_pickle:
                raise PickleNotAllowedError(
                    f"{filename} is not an HDF5 file. If it is a pickle of a "
                    "Simulation and you trust it, load it with "
                    "allow_pickle=True: loading a pickle can run arbitrary code."
                )
            simulation = pickle.load(stream)
        if not isinstance(simulation, Simulation):
            raise TypeError(
                f"{filename} contains an object of the type "
                f"{type(simulation).__name__}, not a Simulation."
            )
        return simulation

    @classmethod
    def load_hdf5(cls, filename, *, allow_pickle: bool = False):
        """Load a simulation stored as an HDF5 file.

        Parameters
        ----------
        filename : str or path-like
            Path to the file to load.
        allow_pickle : bool, optional
            Allow to load the Python pickles in the file. Files written by
            QALMA before the format 1 store the system and the states as
            pickles, so they need ``allow_pickle=True``. Loading a pickle
            can run arbitrary code: use it only with files that you trust.

        Returns
        -------
        SimulationHDF5
            An interface to the file, which loads the states on demand.
        """
        return SimulationHDF5(filename, allow_pickle=allow_pickle)


class SimulationHDF5(Simulation):
    """Interface to read and write simulations stored as HDF5 files.

    The states are not loaded when the file is opened, but when they are
    accessed, to avoid saturating the memory.

    Parameters
    ----------
    filename : str or path-like
        The file that stores the simulation.
    allow_pickle : bool, optional
        Allow to load the Python pickles in the file (see
        :meth:`Simulation.load_hdf5`).
    """

    class StateList:
        """Lazy list of states backed by an HDF5 file.

        Provides list-like access (index, iteration, append, extend) to
        states stored in the ``states`` group of the HDF5 file, loading
        each state on demand rather than all at once. This avoids
        saturating memory for long simulations with many stored states.
        """

        def __init__(self, simulation: "SimulationHDF5"):
            """
            Initialize the list.

            Parameters
            ----------
            simulation : SimulationHDF5
                The simulation that owns the states.
            """
            self._simulation = simulation

        def __len__(self):
            """Compute the length."""
            # pylint: disable=protected-access
            with h5py.File(self._simulation.filename, "r") as h5file:
                if self._simulation.format_version == 0:
                    return len(legacy.state_keys(h5file))
                return len(h5file.get("states", ()))

        def _load(self, h5file, context, index: int):
            if self._simulation.format_version == 0:
                keys = legacy.state_keys(h5file)
                return legacy.load_state(h5file, keys[index], context.system)
            name = f"states/{index_name(index)}"
            if name not in h5file:
                raise HDF5FormatError(
                    f"{name} is missing in {self._simulation.filename}."
                )
            return read_operator(h5file[name], context)

        def __iter__(self):
            """Retrieve items."""
            # pylint: disable=protected-access
            for index in range(len(self)):
                with self._simulation._open("r") as (h5file, context):
                    state = self._load(h5file, context, index)
                yield state

        def __getitem__(self, idx):
            """Get a state, or a list of states if ``idx`` is a slice."""
            # pylint: disable=protected-access
            length = len(self)
            if isinstance(idx, slice):
                with self._simulation._open("r") as (h5file, context):
                    return [
                        self._load(h5file, context, index)
                        for index in range(*idx.indices(length))
                    ]
            index = idx + length if idx < 0 else idx
            if not 0 <= index < length:
                raise IndexError(
                    f"state index {idx} out of range: the file has {length} states."
                )
            with self._simulation._open("r") as (h5file, context):
                return self._load(h5file, context, index)

        def __setitem__(self, idx: int, value: Operator):
            """Replace a stored state, or append it if ``idx == len(self)``."""
            # pylint: disable=protected-access
            simulation = self._simulation
            with simulation._open("r+") as (h5file, context):
                group = h5file.require_group("states")
                length = len(group)
                index = idx + length if idx < 0 else idx
                if not 0 <= index <= length:
                    raise IndexError(
                        f"state index {idx} out of range: the file has "
                        f"{length} states."
                    )
                name = index_name(index)
                # Write the new state before removing the one it replaces.
                _write_state(group, f"{name}.new", value, context)
                if name in group:
                    del group[name]
                group.move(f"{name}.new", name)
                if index < len(simulation.time_span):
                    group[name].attrs["time"] = float(simulation.time_span[index])
                simulation.system = context.system

        def append(self, elem):
            """Append a state to the end of the HDF5 states group.

            Parameters
            ----------
            elem : Operator
                The state to append.

            """
            self[len(self)] = elem

        def extend(self, elems: Iterable[Operator]):
            """Append multiple states to the HDF5 states group.

            Parameters
            ----------
            elems : Iterable[Operator]
                The states to append, in order.

            """
            for elem in elems:
                self.append(elem)

    def _open(self, mode: str) -> "_OpenFile":
        """Open the file, returning the file and its context."""
        return _OpenFile(self, mode)

    def __init__(self, filename, *, allow_pickle: bool = False):
        """Create an interface with an HDF5 file that stores a simulation."""
        # pylint: disable=super-init-not-called
        self.filename = os.fspath(filename)
        self.allow_pickle = allow_pickle
        #: The system of the simulation, shared by all the states.
        self.system: Optional[SystemDescriptor] = None
        with h5py.File(self.filename, "r") as h5file:
            #: Version of the format of the file; 0 for the files written
            #: before the format 1, which are read-only.
            self.format_version: int = read_format_version(h5file, FORMAT_SIMULATION)
            if self.format_version == 0:
                legacy.check_pickle_allowed(h5file, allow_pickle)
                self.system = legacy.load_system(h5file)
                self.parameters = legacy.load_dict(h5file, "parameters")
                self.stats = legacy.load_dict(h5file, "stats")
                self.expect_ops = legacy.load_dict(h5file, "expect_ops")
                self.time_span = legacy.load_time_span(h5file)
            else:
                context = HDF5Context.open(h5file, allow_pickle=allow_pickle)
                self.system = context.system
                for name in _DICT_GROUPS:
                    setattr(self, name, load_dict(h5file[name], context))
                self.time_span = h5file[_TIME_SPAN][()].tolist()
        self.states = self.StateList(self)  # type: ignore[assignment]

    def __repr__(self):
        """Build the repr str."""
        return (
            f"SimulationHDF5({self.filename!r}, format_version="
            f"{self.format_version}, states={len(self.states)})"
        )

    def _root_system(self) -> Optional[SystemDescriptor]:
        # Do not load all the states to find the system.
        return self.system or super()._root_system()

    def save_hdf5(
        self,
        filename=None,
        mode: Optional[str] = None,
        *,
        allow_pickle: Optional[bool] = None,
    ):
        """Store the simulation in an HDF5 file.

        Parameters
        ----------
        filename : str or path-like, optional
            The file to write. By default, or if it is the file of this
            simulation, ``parameters``, ``stats``, ``expect_ops`` and
            ``time_span`` are updated in the file (the states are written
            to the file when they are set or appended). Otherwise, the
            simulation is copied to a new file, in the current format.
        mode : str, optional
            Only used to write a new file: ``"w-"`` (default) fails if
            the file exists; ``"w"`` overwrites it.
        allow_pickle : bool, optional
            Store the objects that the format does not support as Python
            pickles. By default, the value given when the file was opened.
        """
        if allow_pickle is None:
            allow_pickle = self.allow_pickle
        if filename is not None and not (
            os.path.exists(filename) and os.path.samefile(filename, self.filename)
        ):
            if mode in (None, "r+", "a"):
                mode = "w-"
            super().save_hdf5(filename, mode, allow_pickle=allow_pickle)
            return

        with self._open("r+") as (h5file, context):
            context.allow_pickle = allow_pickle
            _write_metadata(h5file, self, context)
            self.system = context.system
