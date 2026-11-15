#!/usr/bin/env python
"""Convert simulations stored with pickle to the HDF5 format of QALMA.

The input files can be pickles of ``Simulation`` objects (``*.pkl``) or HDF5
files written by QALMA before the format 1, which store the system and the
states as pickles. The output files do not contain pickles.

Loading a pickle can run arbitrary code: convert only files that you trust.

Examples
--------
::

    tools/pickle_to_hdf5.py run1.pkl run2.pkl     # writes run1.h5, run2.h5
    tools/pickle_to_hdf5.py old.h5                # writes old.v1.h5
    tools/pickle_to_hdf5.py old.h5 -o new.h5
    tools/pickle_to_hdf5.py --in-place *.h5       # replaces each file
"""

import argparse
import os
import sys
import warnings

import h5py

from qalma.evolution.simulation import Simulation
from qalma.io import IncompleteStateWarning


def default_output(filename: str) -> str:
    """Name of the converted file."""
    base, ext = os.path.splitext(filename)
    if ext in (".h5", ".hdf5"):
        return f"{base}.v1{ext}"
    if ext == ".pkl":
        return f"{base}.h5"
    return f"{filename}.h5"


def convert(filename, output, overwrite=False, keep_pickle=False) -> str:
    """Convert a file, and return a message about the result.

    Parameters
    ----------
    filename : str
        The file to convert.
    output : str
        The file to write. It can be ``filename`` to replace the file.
    overwrite : bool
        Replace ``output`` if it exists.
    keep_pickle : bool
        Store the objects that the format does not support (arbitrary
        objects in ``parameters`` or ``stats``) as pickles in the new
        file. By default, they make the conversion fail.
    """
    if h5py.is_hdf5(filename):
        with h5py.File(filename, "r") as h5file:
            if "format" in h5file.attrs:
                return "already in the current format, skipped"
    in_place = os.path.exists(output) and os.path.samefile(filename, output)
    if os.path.exists(output) and not (overwrite or in_place):
        raise FileExistsError(f"{output} exists; use --overwrite to replace it")

    simulation = Simulation.load(filename, allow_pickle=True)
    # Simulation.save_hdf5 writes a temporary file and renames it at the
    # end, so the input can be replaced safely. Calling the method of the
    # base class copies the simulation instead of updating its own file.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", IncompleteStateWarning)
        Simulation.save_hdf5(simulation, output, mode="w", allow_pickle=keep_pickle)
    message = f"-> {output}"
    if any(issubclass(w.category, IncompleteStateWarning) for w in caught):
        message += (
            "\n    warning: the symmetry projections of the Gibbs states are "
            "functions, and were not stored"
        )
    return message


def main(argv=None) -> int:
    """Run the command line tool."""
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        epilog="Loading a pickle can run arbitrary code: "
        "convert only files that you trust.",
    )
    parser.add_argument("files", nargs="+", help="pickle or old HDF5 files")
    parser.add_argument("-o", "--output", help="output file (for a single input)")
    parser.add_argument(
        "--in-place", action="store_true", help="replace the old HDF5 files"
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="replace existing output files"
    )
    parser.add_argument(
        "--keep-pickle",
        action="store_true",
        help="store as pickles the objects that the format does not support",
    )
    args = parser.parse_args(argv)
    if args.output and len(args.files) > 1:
        parser.error("--output can only be used with a single input file")

    failed = 0
    for filename in args.files:
        if args.output:
            output = args.output
        elif args.in_place and h5py.is_hdf5(filename):
            output = filename
        else:
            output = default_output(filename)
        try:
            print(
                f"{filename}:",
                convert(filename, output, args.overwrite, args.keep_pickle),
            )
        except Exception as exc:  # pylint: disable=broad-exception-caught
            failed += 1
            print(f"{filename}: FAILED ({type(exc).__name__}: {exc})", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
