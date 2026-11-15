Changelog
=========

All notable changes to QALMA are documented in this file. Version numbers
follow `Semantic Versioning <https://semver.org>`_.


1.0.0 (unreleased)
------------------

First public release.

Packaging
+++++++++

* QALMA is now distributed under the LGPL-3.0 (previously GPL-3.0). The ALPS
  model and lattice libraries in ``qalma/lib`` keep the Boost Software
  License 1.0.
* Python 3.11 or newer is required.
* Runtime dependencies are declared with tested lower bounds:
  ``numpy>=1.26``, ``scipy>=1.11``, ``qutip>=5.1``, ``h5py>=3.10`` and
  ``packaging``.
* ``matplotlib`` is now an optional dependency, needed only by the plotting
  helpers. Install it with ``pip install "qalma[plot]"``.
* ``pyproject.toml`` is the only source of package metadata; ``setup.py``
  was removed.
* The wheel installs only the ``qalma`` package. Previously it also
  installed a top-level ``test`` package.
* ``qalma.__version__`` is available and is the single source of the
  version number.

HDF5 files
++++++++++

* ``Simulation.save_hdf5`` writes a new format (format 1) without Python
  pickles: the system, the operators and the states are stored as numbers,
  strings and arrays. The files can be read from other languages, and
  opening a file does not run code stored in it. The format is documented
  in the user guide and has a version number; QALMA 1.x reads every version
  up to its own.
* New package ``qalma.io``, with ``save_operators`` and ``load_operators``
  to store collections of operators and states, and the functions that
  store systems, operators and dictionaries in groups of an HDF5 file.
* HDF5 files written by earlier versions contain pickles. They are now
  read-only and need ``Simulation.load_hdf5(path, allow_pickle=True)``;
  without it, the error explains how to open and convert them.
  ``tools/pickle_to_hdf5.py`` converts old HDF5 files and pickles of
  ``Simulation`` objects to the new format.
* ``Simulation.load`` raises an error instead of returning ``None`` when
  the file cannot be loaded, and needs ``allow_pickle=True`` to load a
  pickle file.
* ``Simulation.save_hdf5`` raises the errors (for instance,
  ``PermissionError``) instead of logging a warning, and writes through a
  temporary file, so a failure does not leave an incomplete file.
* The groups ``expect obs`` and ``time span`` are now ``expect_ops`` and
  ``time_span``. Keys of ``expect_ops`` that are not strings (like the
  positions of a list of observables) are kept: they used to be loaded as
  strings.
* Gibbs states are stored without their ``symmetry_projections``, which
  are functions, with a ``qalma.io.IncompleteStateWarning``. Objects in
  ``parameters``, ``stats`` or ``expect_ops`` that the format does not
  support raise a ``TypeError``, unless ``allow_pickle=True`` is given.
* Fixed: ``append`` and ``extend`` on the states of a ``SimulationHDF5``
  stored the states under a wrong name.
* ``adaptive_projected_evolution`` no longer writes a pickle checkpoint
  (``__adaptative_*.pkl``) in the working directory after each step. Pass
  ``checkpoint_file`` to get an HDF5 checkpoint in the file of your choice.
* The expressions of the models (``qalma.utils.eval_expr``) are restricted
  to arithmetic: numbers, parameters, operators and function calls. Other
  Python syntax raises a ``SyntaxError`` and Python builtins are not
  available, so neither an ALPS library nor a simulation file can run
  arbitrary code.

Behaviour changes
+++++++++++++++++

* Importing ``qalma`` no longer calls ``logging.basicConfig``. All messages
  are emitted through loggers under the ``"qalma"`` name, so INFO and DEBUG
  messages are hidden unless the application configures logging, e.g.::

      import logging
      logging.basicConfig(level=logging.INFO)

* Diagnostic ``print`` calls in the library were removed or turned into
  ``DEBUG`` log messages.
* Building a system from a model with fermionic operators now emits a
  ``qalma.FermionicSignWarning``: fermionic signs between different sites
  are not implemented (see *Known limitations* in the documentation).
* The ``USE_PARALLEL`` environment variable is now parsed as a boolean:
  ``USE_PARALLEL=0`` used to *enable* the parallel routines. The preferred
  name is ``QALMA_USE_PARALLEL``.
* ``qalma.settings.FIGURES_DIR`` was removed; it pointed outside the
  installed package.
* The code paths for QuTiP 4 were removed (QuTiP >= 5.1 is required).
