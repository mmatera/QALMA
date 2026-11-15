HDF5 file format
================

This page describes the files written by
:meth:`Simulation.save_hdf5 <qalma.evolution.simulation.Simulation.save_hdf5>`
and :func:`qalma.io.save_operators`, with enough detail to read them without
QALMA.

The files contain numbers, strings and arrays only. There are no Python
pickles, unless they were requested with ``allow_pickle=True``.

Versions
--------

The root group has three attributes:

``format``
    ``"qalma-simulation"`` or ``"qalma-operators"``.
``format_version``
    An integer; this page describes the version 1.
``qalma_version``
    The version of QALMA that wrote the file.

QALMA reads every version lower or equal to its own, and writes the latest
one. A file with a higher version raises an error that asks to upgrade
QALMA. A file without the attribute ``format`` was written before the format
1: see :ref:`hdf5-legacy`.

The test suite keeps a reference file of each released version
(``test/data/simulation_format1.h5``), so later versions of QALMA are
checked against it.

Layout of a simulation
----------------------

::

    /                      attrs: format, format_version, qalma_version
    ├── system/
    │   ├── spec           JSON string: graph, model and parameters
    │   ├── sites          table: name, dimension, basis
    │   └── pool_d2, ...   matrices of the local operators
    ├── parameters/        one member per item
    ├── stats/             idem
    ├── expect_ops/        idem: usually, a dataset per observable
    ├── time_span          float64 (n_times,)
    └── states/
        ├── 000000/        attrs: type, time, ...
        ├── 000001/
        └── ...

``states/NNNNNN`` is the state number ``N``, stored as an operator (see
below). If there is a state for each time, the attribute ``time`` repeats
``time_span[N]``. The group ``system`` is missing if the simulation has no
operators.

A file of operators has ``system`` and a group ``operators``, which is a
dictionary like ``parameters``.

Conventions
-----------

* Complex numbers are stored as h5py does: a compound type with the fields
  ``r`` and ``i``.
* Strings are variable-length UTF-8 strings.
* Sites are referred to by their position (from 0) in ``/system/sites``.
* Sequences stored as groups have the members ``000000``, ``000001``, ...
* In names of members that come from keys of dictionaries, ``%``, ``/`` and
  the name ``.`` are written as ``%25``, ``%2F`` and ``%2E``.

The system
----------

``/system/sites`` is a 1D table (compound type) with a row per site:
``name``, ``dimension`` of its Hilbert space, and ``basis``, the name of its
kind of local basis in ``spec``.

``/system/spec`` is a JSON string with the complete description of the
system. It does not refer to the ALPS libraries from which the system was
built:

.. code-block:: text

    {
      "name": "spin on chain lattice",
      "parms": {"L": 4, "J": 1, "Jz": "J", ...},
      "graph": {
        "name": "chain lattice",
        "nodes": {"1[0]": {"type": "0", "coords": [0.0]}, ...},
        "edges": {"0": [["1[0]", "1[1]"], ...]},
        "loops": {},
        "lattice": {"basis": [[1]], "dimension": 1, ...},
        "parms": {...}
      },
      "model": {
        "name": "spin",
        "parms": {...},
        "constraints": {...},
        "bond_ops": {"exchange": "Sz@src*Sz@dst+...", ...},
        "global_ops": {"Hamiltonian": {"site terms": [...], "bond terms": [...],
                                       "loop terms": [...]}, ...},
        "site_basis": {
          "0": {
            "name": "spin", "dimension": 2, "qn": {...}, "parms": {...},
            "localstates": [[0.5, -0.5], [0.5, 0.5]],
            "operators": {"Sz": {"matrix": 3, "storage": "CSR",
                                 "fermionic": false}, ...}
          }
        }
      }
    }

The matrix of a local operator is ``pool_d{dimension}[matrix]``: each
``/system/pool_d{d}`` dataset has the shape ``(n, d, d)``. ``storage`` is
the data layer of QuTiP used for the matrix in memory.

In a system whose sites are split into qubits (fermionic models through the
Jordan-Wigner transformation, see :doc:`systems`), the table of sites lists
the qubits, like ``1[0]:Nup``, and their local bases are the entries of
``extra_site_basis``. The nodes of the graph are the physical sites. If the
split was forced or disabled when the system was built, ``spec`` has the key
``qubit_split`` (``true`` or ``false``); otherwise QALMA decides it again
from the model when the system is loaded.

Values that JSON cannot represent are written as objects with a single key:
``{"__tuple__": [...]}``, ``{"__complex__": [re, im]}``,
``{"__float__": "nan"}``, ``{"__set__": [...]}``,
``{"__ndarray__": [...], "dtype": "int64"}`` and ``{"__dict__": [[key,
value], ...]}`` for dictionaries whose keys are not strings. The edges of
the graph and the local states are tuples, and the coordinates of the nodes
are arrays of floats, so they are written as plain JSON arrays.

The global operators of the model are stored as the expressions of the ALPS
library. When QALMA loads the system it evaluates them again, with an
evaluator that only accepts arithmetic (numbers, names, ``+ - * / ** %``
and function calls like ``sqrt(x)``), so the expressions of a file cannot
run code.

Operators
---------

Each operator is a group with the attribute ``type``, one of the labels of
the table. QALMA only builds objects of the class registered for each label.
An operator that acts on a subsystem has the attribute ``subsystem``, with
the positions of the sites of the subsystem.

.. list-table::
   :header-rows: 1
   :widths: 18 30 52

   * - ``type``
     - Class
     - Content
   * - ``scalar``
     - ``ScalarOperator``
     - attr ``prefactor``
   * - ``local``
     - ``LocalOperator``
     - attr ``site``; dataset ``matrix``
   * - ``product``
     - ``ProductOperator``
     - attr ``prefactor``; factors
   * - ``sum``
     - ``SumOperator``
     - terms; attrs ``isherm``, ``isdiagonal``, ``simplified`` (missing if unknown)
   * - ``one_body``
     - ``OneBodyOperator``
     - as ``sum``
   * - ``qutip``
     - ``QutipOperator``
     - attr ``prefactor``; dataset ``sites``; matrix in CSR
   * - ``pauli_string``
     - ``PauliStringOperator``
     - attr ``prefactor``; dataset ``code (W,)``
   * - ``pauli_sum``
     - ``PauliSumOperator``
     - datasets ``codes (m, W)`` and ``coeffs (m,)``
   * - ``quadratic``
     - ``QuadraticFormOperator``
     - dataset ``weights``; groups ``basis/NNNNNN``, ``linear_term`` and ``offset`` (optional)
   * - ``gibbs``
     - ``GibbsDensityOperator``
     - attrs ``prefactor``, ``normalized``, ``free_energy``, ``dropped``; groups ``k`` and ``meanfield`` (optional)
   * - ``rpa_gibbs``
     - ``RPAGibbsDensityOperator``
     - as ``gibbs``, plus the attrs ``dynamic``, ``n_tau``, ``static_gap``, ``hybrid``, ``hybrid_options`` (JSON)
   * - ``gibbs_product``
     - ``GibbsProductDensityOperator``
     - attr ``prefactor``; factors (the local generators); dataset ``free_energies``
   * - ``product_state``
     - ``ProductDensityOperator``
     - attr ``prefactor``; factors (the local states); datasets ``local_fs_site``, ``local_fs``
   * - ``mixture``
     - ``MixtureDensityOperator``
     - terms
   * - ``qutip_state``
     - ``QutipDensityOperator``
     - as ``qutip``, plus the attr ``normalized``

**Factors.** The local factors of an operator are stored in two aligned 1D
datasets and a pool of matrices:

``site (nnz,)``
    Position of the site of each factor.
``matrix (nnz,)``
    Position of the matrix of each factor in the pool of its dimension.
``pool_d{d} (n, d, d)``
    The distinct ``d × d`` matrices used by the factors.

**Terms.** A group per term is slow and large for sums of thousands of
terms, so the terms of a sum are packed:

``kind (n_terms,)``
    0: product; 1: local operator; 2: scalar; 3: ``QutipOperator``; 4: any
    other operator.
``prefactors (n_terms,)``
    Coefficient of the terms of kind 0 and 2 (1 for the others).
``offsets (n_terms + 1,)``
    The factors of the term ``t`` are the rows ``offsets[t]`` to
    ``offsets[t + 1] - 1`` of ``site`` and ``matrix``.
``site``, ``matrix``, ``pool_d{d}``
    The factors, as above.
``blocks/NNNNNN``
    The terms of kind 3, in order: the first term of kind 3 is
    ``blocks/000000``.
``terms/NNNNNN``
    The terms of kind 4, in order.

So, for a sum of products,

.. math::

    O = \sum_t \mathtt{prefactors}[t]
        \bigotimes_{i=\mathtt{offsets}[t]}^{\mathtt{offsets}[t+1]-1}
        \mathtt{pool}[\mathtt{matrix}[i]]_{\mathtt{site}[i]}

**Matrices in CSR format.** ``data``, ``indices`` and ``indptr`` as in
``scipy.sparse.csr_matrix``; the attribute ``shape``; and ``dims_rows`` and
``dims_cols``, the dimensions of the tensor factors. In a ``qutip`` group,
``sites[p]`` is the site of the tensor factor ``p``. A ``qutip`` group
without matrix is ``prefactor`` times the identity.

**Pauli strings.** A product of Pauli matrices is a code of two bits per
site of the table of sites: the bits ``(2k+1, 2k)`` hold the matrix on the
site ``k`` (``00`` identity, ``01`` :math:`\sigma^x`, ``10``
:math:`\sigma^y`, ``11`` :math:`\sigma^z`). The code is split into
``W = ceil(2 n_sites / 64)`` words of type ``uint64``, the least
significant first. A ``pauli_sum`` is
:math:`\sum_t \mathtt{coeffs}[t]\, P(\mathtt{codes}[t])`, with the rows
of ``codes`` sorted and without repetitions. The codes always refer to the
table of sites of the file, also for operators on a subsystem.

**States.** ``prefactor`` is the weight of the state in a mixture. A
``gibbs`` state is :math:`e^{-K}/\mathrm{Tr}\,e^{-K}`, with :math:`K` in
the group ``k``; if ``normalized`` is true, :math:`\mathrm{Tr}\,e^{-K}=1`.
Functions cannot be stored: a Gibbs state with symmetry projections is
stored without them, and the attribute ``dropped`` lists what is missing.

Dictionaries
------------

``parameters``, ``stats`` and ``expect_ops`` are stored with a member for
each item:

================================ ===========================================
Python value                     HDF5 member
================================ ===========================================
bool, int, float, complex        scalar dataset
str                              scalar string dataset
numpy array                      dataset
list, tuple, set of numbers      dataset
or of strings
other lists and tuples           group with the members ``000000``, ...
dict                             group
``None``                         empty dataset
operator or state                group with the attribute ``type``
the system                       empty dataset (it refers to ``/system``)
``timedelta``, ``datetime``      seconds (float), ISO 8601 string
``qutip.Qobj``                   group with a matrix in CSR format
================================ ===========================================

The attribute ``py_type`` (``list``, ``tuple``, ``set``, ``none``,
``system``, ``timedelta``, ...) lets QALMA restore the Python type; the data
can be read without it. A key that is not a string is written with
``str(key)`` as the name of the member, and kept as JSON in the attribute
``key_json``. Any other value is an error, unless ``allow_pickle=True``: then
it is stored as a dataset of bytes with ``py_type = "pickle"``.

Reading a file without QALMA
----------------------------

.. code-block:: python

    import json
    import h5py
    import numpy as np

    with h5py.File("my_simulation.h5", "r") as f:
        times = f["time_span"][()]
        energy = f["expect_ops/energy"][()]
        spec = json.loads(f["system/spec"].asstr()[()])
        sites = [name.decode() for name in f["system/sites"]["name"]]

        # Terms of the generator of the first state, a sum of products
        # (for a "pauli_sum", read "codes" and "coeffs" instead)
        k = f["states/000000/k"]
        offsets = k["offsets"][()]
        for t, prefactor in enumerate(k["prefactors"][()]):
            factors = {
                sites[k["site"][i]]: k["pool_d2"][k["matrix"][i]]
                for i in range(offsets[t], offsets[t + 1])
            }

.. _hdf5-legacy:

Files written before the format 1
---------------------------------

Those files do not have the attribute ``format``. The system (``/system``)
and each state (``/states/rho_NNNNNN``) are pickles, and the other groups
are called ``expect obs`` and ``time span``. They are read-only for QALMA,
and are loaded with ``Simulation.load_hdf5(path, allow_pickle=True)``; see
:doc:`evolution` to convert them.
