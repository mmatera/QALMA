"""Storage of systems, operators and states in HDF5 files, without pickle.

Files written by this package only contain numbers, strings and arrays:
they can be read from other languages and inspected with ``h5dump``, and
opening a file does not run code stored in the file. The format is
described in the user guide (*HDF5 file format*).

Simulations are stored with
:meth:`Simulation.save_hdf5 <qalma.evolution.simulation.Simulation.save_hdf5>`.
This package also stores collections of operators:

.. code-block:: python

    from qalma.io import load_operators, save_operators

    save_operators("operators.h5", {"H": hamiltonian, "rho": state})
    operators = load_operators("operators.h5")

The functions that work on groups of an open file (``write_system``,
``write_operator``, ``store_dict`` and their readers) are the building
blocks of both kinds of files.
"""

import os
from typing import Dict, Mapping

import h5py

from . import legacy  # noqa: E402,F401
from ._common import (
    FORMAT_OPERATORS,
    FORMAT_SIMULATION,
    FORMAT_VERSION,
    HDF5FormatError,
    PickleNotAllowedError,
    read_format_version,
    write_format_attrs,
)
from .containers import load_dict, store_dict
from .operators import (
    HDF5Context,
    IncompleteStateWarning,
    read_operator,
    write_operator,
)
from .system import read_system, write_system

__all__ = [
    "atomic_file",
    "legacy",
    "read_format_version",
    "write_format_attrs",
    "FORMAT_OPERATORS",
    "FORMAT_SIMULATION",
    "FORMAT_VERSION",
    "HDF5Context",
    "HDF5FormatError",
    "IncompleteStateWarning",
    "PickleNotAllowedError",
    "load_dict",
    "load_operators",
    "read_operator",
    "read_system",
    "save_operators",
    "store_dict",
    "write_operator",
    "write_system",
]


def save_operators(
    filename,
    operators: Mapping,
    mode: str = "w-",
    *,
    allow_pickle: bool = False,
) -> None:
    """Store a collection of operators and states in an HDF5 file.

    Parameters
    ----------
    filename : str or path-like
        The file to write.
    operators : dict
        The operators to store, by name. All of them must act on the same
        system, or on subsystems of it. The values can also be numbers,
        arrays, or lists and dicts of operators (see
        :mod:`qalma.io.containers`).
    mode : str, optional
        ``"w-"`` (default) fails if the file exists; ``"w"`` overwrites it.
    allow_pickle : bool, optional
        Store the objects that the format does not support as Python
        pickles. The default is False, which raises a ``TypeError``.
    """
    with atomic_file(filename, mode) as h5file:
        write_format_attrs(h5file, FORMAT_OPERATORS)
        context = HDF5Context(None, allow_pickle=allow_pickle)
        systems = [
            op.system
            for op in operators.values()
            if getattr(op, "system", None) is not None
        ]
        if systems:
            context.bind(h5file, max(systems, key=lambda system: len(system.sites)))
        store_dict(
            h5file.create_group("operators", track_order=True),
            dict(operators),
            context,
            "operators",
        )


def load_operators(filename, *, allow_pickle: bool = False) -> Dict:
    """Load the operators stored by :func:`save_operators`.

    Parameters
    ----------
    filename : str or path-like
        The file to read.
    allow_pickle : bool, optional
        Allow to load the objects stored as Python pickles. Loading a
        pickle can run arbitrary code: use it only with trusted files.

    Returns
    -------
    dict
        The operators, by name. All of them share the same system.
    """
    with h5py.File(filename, "r") as h5file:
        if read_format_version(h5file, FORMAT_OPERATORS) == 0:
            raise HDF5FormatError(f"{filename} is not a file of QALMA operators.")
        context = HDF5Context.open(h5file, allow_pickle=allow_pickle)
        return load_dict(h5file["operators"], context)


class atomic_file:  # pylint: disable=invalid-name
    """Context manager that writes an HDF5 file through a temporary file.

    The file is written with a temporary name in the same directory, and
    renamed when the block finishes. If the block raises an exception,
    the temporary file is removed, so a failure never leaves a truncated
    file, and never destroys the file that was going to be replaced.

    Parameters
    ----------
    filename : str or path-like
        The file to write.
    mode : str
        ``"w"`` to overwrite the file if it exists; ``"w-"`` or ``"x"`` to
        fail with ``FileExistsError``.
    """

    def __init__(self, filename, mode: str = "w-"):
        if mode not in ("w", "w-", "x"):
            raise ValueError(
                f"mode must be 'w', 'w-' or 'x' to write a new file, not {mode!r}."
            )
        self.filename = os.fspath(filename)
        self.mode = mode
        self.tmp_filename = f"{self.filename}.tmp{os.getpid()}"
        self.h5file = None

    def __enter__(self) -> h5py.File:
        if self.mode != "w" and os.path.exists(self.filename):
            raise FileExistsError(
                f"{self.filename} already exists. Use mode='w' to overwrite it."
            )
        self.h5file = h5py.File(self.tmp_filename, "w")
        return self.h5file

    def __exit__(self, exc_type, exc, traceback):
        self.h5file.close()
        if exc_type is None:
            os.replace(self.tmp_filename, self.filename)
        else:
            os.remove(self.tmp_filename)
        return False
