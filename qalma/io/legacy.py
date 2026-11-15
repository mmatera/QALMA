"""Reader of the HDF5 files written before the format 1.

Those files do not have the attribute ``format`` in the root group. They
store the system, the states and some values of the dictionaries as Python
pickles, so they can only be loaded with ``allow_pickle=True``, and only if
the classes of QALMA did not change since the file was written. They are
never written: use ``tools/pickle_to_hdf5.py`` to convert them.

Layout::

    /parameters, /stats, /expect obs    groups, with a dataset for each item
    /time span                          dataset
    /system                             pickle of the SystemDescriptor
    /states/rho_000000, ...             pickle of each state, without system
"""

import logging
import pickle
import re
from typing import Any, Dict, List, Optional

import h5py
import numpy as np

from qalma.model import SystemDescriptor
from qalma.operators import Operator

from ._common import HDF5FormatError, PickleNotAllowedError

logger = logging.getLogger(__name__)

__all__ = [
    "check_pickle_allowed",
    "load_dict",
    "load_state",
    "load_system",
    "state_keys",
]

DICT_GROUPS = {
    "parameters": "parameters",
    "stats": "stats",
    "expect_ops": "expect obs",
}
TIME_SPAN = "time span"


def _has_pickle(root: h5py.Group) -> bool:
    if "system" in root or len(root.get("states", ())) > 0:
        return True
    return any(
        dset.attrs.get("pickled", False)
        for name in DICT_GROUPS.values()
        for dset in root.get(name, {}).values()
    )


def check_pickle_allowed(root: h5py.Group, allow_pickle: bool) -> None:
    """Raise an error if the file has pickles and they are not allowed."""
    if not allow_pickle and _has_pickle(root):
        raise PickleNotAllowedError(
            f"{root.file.filename} was written by an old version of QALMA, "
            "which stored the system and the states as Python pickles. "
            "Loading a pickle can run arbitrary code: if you trust the file, "
            "load it with allow_pickle=True. To convert it to the current "
            "format, which does not use pickle, use tools/pickle_to_hdf5.py "
            "or save the loaded simulation to a new file."
        )


def _unpickle(data, where: str):
    try:
        return pickle.loads(data.tobytes() if hasattr(data, "tobytes") else data)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        # Unpickling fails in many ways if the classes changed.
        raise HDF5FormatError(
            f"{where} cannot be unpickled ({type(exc).__name__}: {exc}). The "
            "file was probably written by an incompatible version of QALMA."
        ) from exc


def load_dict(root: h5py.Group, attribute: str) -> Dict[str, Any]:
    """Load one of the dictionaries of a simulation.

    Parameters
    ----------
    root : h5py.Group
        The root group of the file.
    attribute : str
        ``"parameters"``, ``"stats"`` or ``"expect_ops"``.
    """
    result: Dict[str, Any] = {}
    group = root.get(DICT_GROUPS[attribute])
    if group is None:
        return result
    for key, dataset in group.items():
        raw_data = dataset[()]
        if "encoding" in dataset.attrs and isinstance(raw_data, (bytes, np.bytes_)):
            result[key] = raw_data.decode("utf-8")
        elif dataset.attrs.get("pickled", False):
            result[key] = _unpickle(dataset[0], dataset.name)
        else:
            result[key] = raw_data
    return result


def load_time_span(root: h5py.Group) -> list:
    """Load the times of a simulation."""
    if TIME_SPAN not in root:
        return []
    return list(root[TIME_SPAN][()])


def load_system(root: h5py.Group) -> Optional[SystemDescriptor]:
    """Load the system of the file, or ``None`` if it does not have one."""
    if "system" not in root:
        return None
    system = _unpickle(root["system"][0], "/system")
    if not isinstance(system, SystemDescriptor):
        raise HDF5FormatError("/system is not a SystemDescriptor.")
    return system


def state_keys(root: h5py.Group) -> List[str]:
    """Names of the datasets of the states, in the order of the states."""
    if "states" not in root:
        return []

    def number(key):
        match = re.search(r"(\d+)$", key)
        return int(match.group(1)) if match else -1

    return sorted(root["states"], key=number)


def load_state(root: h5py.Group, key: str, system: Optional[SystemDescriptor]):
    """Load the state stored in the dataset ``key`` of ``/states``."""
    entry = _unpickle(root["states"][key][0], f"/states/{key}")
    state = entry if isinstance(entry, Operator) else None
    # If the entry is a tuple, the state is its first operator.
    if state is None and isinstance(entry, (tuple, list)):
        state = next((elem for elem in entry if isinstance(elem, Operator)), None)
    if state is None:
        raise HDF5FormatError(f"/states/{key} is not an operator.")
    if system is not None:
        state._set_system_(system)  # pylint: disable=protected-access
    return state
