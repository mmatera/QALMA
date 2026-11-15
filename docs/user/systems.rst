Systems and Geometries
======================

.. contents::
   :depth: 2
   :local:

Overview
--------

A QALMA simulation starts by defining the lattice geometry and the local
Hilbert space at each site. These two ingredients together produce a
:class:`~qalma.model.SystemDescriptor`, which is the object passed to every
operator and evolution routine.

Defining a system
-----------------

The primary entry point is :func:`~qalma.model.build_system`::

    from qalma import build_system, model_from_alps_xml, graph_from_alps_xml

    model = model_from_alps_xml(name="spin", parms={"S": "1/2"})
    graph = graph_from_alps_xml(name="open chain", parms={"L": 4})
    system = build_system(model, graph)

:func:`~qalma.model.build_system` combines the local basis descriptors from
the model with the connectivity information from the graph and returns a
:class:`~qalma.model.SystemDescriptor`.

The SystemDescriptor
--------------------

:class:`~qalma.model.SystemDescriptor` stores all site-level information:

* ``system.sites`` — ordered dict of site names to their local data
  (dimension, identity operator, available local operators).
* ``system.dimensions`` — mapping from site name to local Hilbert space
  dimension :math:`d_i`.
* ``system.bonds`` — list of ``(site_a, site_b, bond_type)`` tuples
  describing the lattice connectivity.

You can inspect the system directly::

    print(system.sites)      # {'site0': {...}, 'site1': {...}, ...}
    print(system.dimensions) # {'site0': 2, 'site1': 2, ...}

Subsystems
----------

QALMA supports partial traces and reductions over subsystems. A subsystem
is created with::

    sub = system.subsystem(frozenset({"site0", "site1"}))

which returns a new :class:`~qalma.model.SystemDescriptor` containing only
the specified sites, preserving all local data.

Loading geometries and models from ALPS
---------------------------------------

QALMA ships with the standard ALPS model and lattice library. You can
list available models and geometries::

    from qalma import list_models_in_alps_xml, list_geometries_in_alps_xml

    print(list_models_in_alps_xml())
    print(list_geometries_in_alps_xml())

Custom parameters are passed via the ``parms`` dictionary::

    model = model_from_alps_xml(name="Heisenberg", parms={"S": "1", "Jz": 1.0})
    graph = graph_from_alps_xml(name="square lattice", parms={"L": 4, "W": 4})

Fermionic models and qubits
---------------------------

When a model has fermionic quantum numbers and the local space of every
site is a product of quantum numbers that take two values (``N`` in
``spinless fermions``; ``Nup`` and ``Ndown`` in ``fermion Hubbard``; also
the spin-1/2 sites of ``Kondo lattice``), each of those quantum numbers
becomes a qubit::

    system = build_system("chain lattice", "fermion Hubbard", L=4, U=2)
    list(system.sites)        # ['1[0]:Nup', '1[0]:Ndown', '1[1]:Nup', ...]
    system.site_qubits["1[0]"]  # ('1[0]:Nup', '1[0]:Ndown')

The fermionic modes are mapped to Pauli strings through the Jordan-Wigner
transformation, in the order of ``system.sites``, so they anticommute also
within a site. Site operators are still requested by the name of the node
(``system.site_operator("c_up", "1[0]")``), and they are, like the global
operators, sums of Pauli strings
(:class:`~qalma.operators.PauliSumOperator`). A site with a single qubit
keeps the name of the node.

``build_system(..., qubit_split=True)`` forces the representation for any
model whose sites are products of two-level quantum numbers (e.g. spin 1/2
chains), and ``qubit_split=False`` disables it.

.. seealso::

   :mod:`qalma.model`, :mod:`qalma.geometry`, :mod:`qalma.alpsmodels`
