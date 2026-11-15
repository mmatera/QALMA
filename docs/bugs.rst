.. index:: bugs
.. _bugs:

Known limitations and reporting bugs
====================================

Known limitations
-----------------

* **Fermionic models.** Models whose sites are products of two-level
  quantum numbers (``spinless fermions``, ``fermion Hubbard``,
  ``Kondo lattice``) are represented by one qubit per quantum number, and
  their fermionic operators are mapped to Pauli strings through the
  Jordan-Wigner transformation, so the signs are exact (see
  :doc:`user/systems`). For other local spaces, like the constrained basis
  of ``t-J``, the signs coming from the anticommutation of fermionic
  operators on different sites are not implemented: those operators
  commute, the results are those of the corresponding hard-core particles,
  and building the system emits a :class:`qalma.FermionicSignWarning`.

* **Exact methods scale exponentially.** Operations that need the full
  matrix representation of an operator (``to_qutip``, exact
  exponentials and spectra) are limited to small systems.

* **Simulation files.** HDF5 files written by QALMA 1.0 or newer do not
  contain Python pickles (see :doc:`user/hdf5_format`). Files written by
  older versions do: they are only loaded with ``allow_pickle=True``, may
  not be readable by other versions of QALMA and, as with any pickle,
  should only be opened if you trust them. Functions cannot be stored, so
  Gibbs states are stored without their symmetry projections.

Reporting bugs
--------------

Bugs and feature requests are tracked in the
`issue tracker <https://github.com/QILPCM-IFLP-CONICET/QALMA/issues>`_.
A useful report includes:

* a short, self-contained script that reproduces the problem;
* what you got (the full traceback or output) and what you expected;
* the versions involved, which you can get with

  .. code-block:: python

     import qalma, qutip, numpy, scipy
     print(qalma.__version__, qutip.__version__, numpy.__version__, scipy.__version__)

If you already know how to fix the problem, pull requests are welcome: see
``CONTRIBUTING.rst`` in the repository.
