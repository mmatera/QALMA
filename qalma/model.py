"""System Descriptors.

A ``SystemDescriptor`` contains all the information relative to the
system to be simulated.

``SystemDescriptor``s can be generated from information in ALPS xml file
libraries by means of ``build_system``.

An alternative is just to build the system through
``build_system_from_dims`` which takes as argument a  dictionary,
specifiying the names of the system sites as keys, and their
corresponding dimensions as values.

``build_spin_chain`` is a short-cut to build a spin chain lattice.
"""

import itertools
import logging
import pickle
import re
import warnings
from typing import Dict, Optional, Tuple

import numpy as np
import qutip  # type: ignore[import-untyped]

from qalma.alpsmodels import ModelDescriptor, qutip_model_from_dims
from qalma.geometry import GraphDescriptor
from qalma.settings import LATTICE_LIB_FILE, MODEL_LIB_FILE
from qalma.utils import eval_expr, replace_variable_type

logger = logging.getLogger(__name__)


class FermionicSignWarning(UserWarning):
    """Warn that a model with fermionic operators is treated as bosonic.

    When every site of the system is a single fermionic mode (a two-level
    site with a fermionic quantum number), the operators are built through
    the Jordan-Wigner transformation and the signs are exact. Otherwise
    (e.g. sites with two spin species, or models that mix fermionic and
    non-fermionic sites), operators acting on different sites are assumed
    to commute: the signs coming from the anticommutation of fermionic
    operators are not implemented. Results are then only correct when
    those signs cancel out (e.g. nearest-neighbour hopping of a single
    species on an open chain).
    """


def _binary_quantum_numbers(site_basis: dict) -> Optional[list]:
    """Split a local basis into two-level factors, one per quantum number.

    Returns a list of ``(qn_name, position, lower_value, fermionic)``, in
    the order of the quantum numbers, if the local basis is the full tensor
    product of quantum numbers that take two consecutive values. Quantum
    numbers that are functions of the chosen ones (e.g. ``N = Nup + Ndown``,
    or a fixed ``S``) are ignored. Returns ``None`` otherwise, e.g. for
    spin-1 sites or for the constrained t-J basis.
    """
    qns = site_basis.get("qn", None)
    qn_index = site_basis.get("qn_index", None)
    states = site_basis.get("localstates", None)
    if not qns or qn_index is None or not states:
        return None
    binary: list = []
    distinct = {()}
    for qn_name, position in sorted(qn_index.items(), key=lambda item: item[1]):
        values = sorted({state[position] for state in states})
        if len(values) != 2 or abs(values[1] - values[0] - 1) > 1e-10:
            continue
        lower = values[0]
        candidate = binary + [(qn_name, position, lower)]
        new_distinct = {
            tuple(int(round(state[pos] - low)) for _, pos, low in candidate)
            for state in states
        }
        if len(new_distinct) == len(distinct):
            # a function of the quantum numbers already chosen
            continue
        binary, distinct = candidate, new_distinct
    if not binary or len(states) != 2 ** len(binary) or len(distinct) != len(states):
        return None
    return [
        (qn_name, position, lower, bool(qns.get(qn_name, {}).get("fermionic", False)))
        for qn_name, position, lower in binary
    ]


def _qubit_layout(physical_sites: dict, qubit_split: Optional[bool]) -> Optional[dict]:
    """Decide whether the system is represented by qubits, and how.

    Returns ``None`` if the sites keep their own local spaces. Otherwise
    returns a mapping ``node -> [(qubit_name, qn_name, position, lower,
    fermionic), ...]``. With ``qubit_split=None`` the qubit representation
    is used when the model has fermionic quantum numbers and every site
    factorizes into two-level quantum numbers.
    """
    if qubit_split is False or not physical_sites:
        return None
    layout = {}
    has_fermions = False
    for node, site_basis in physical_sites.items():
        binary = _binary_quantum_numbers(site_basis)
        if binary is None:
            if qubit_split:
                raise ValueError(
                    f"the local space of site {node} is not a product of two-level "
                    "quantum numbers, and cannot be split into qubits."
                )
            return None
        single = len(binary) == 1
        layout[node] = [
            (node if single else f"{node}:{qn_name}", qn_name, position, lower, fermionic)
            for qn_name, position, lower, fermionic in binary
        ]
        has_fermions = has_fermions or any(entry[3] for entry in binary)
    if qubit_split is None and not has_fermions:
        return None
    return layout


def _qubit_site_basis(node, qn_name, lower, fermionic, parent_basis) -> dict:
    """Local basis descriptor of one qubit of a split site."""
    identity = qutip.qeye(2)
    return {
        "name": f"{parent_basis.get('name', '')}:{qn_name}",
        "qn": {qn_name: {"min": lower, "max": lower + 1, "fermionic": fermionic}},
        "qn_index": {qn_name: 0},
        "dimension": 2,
        "identity": identity,
        "operators": {"identity": identity},
        "parms": parent_basis.get("parms", {}),
        "localstates": [(lower,), (lower + 1,)],
        "parent": node,
        "fermionic": fermionic,
        # parity of the first basis state, (-1)**lower, for fermionic qubits
        "parity0": (-1.0) ** (int(round(lower)) % 2) if fermionic else 1.0,
    }


class SystemDescriptor:
    """System Descriptor class.

    A SystemDescriptor represents a quantum system as a collection of
    Hilbert spaces associated to the nodes of a graph or lattice, and a
    set of parameters defining a Hamiltonian operator.
    """

    _subsystems_cache: dict
    dimensions: dict
    name: str
    sites: dict
    spec: dict
    operators: dict

    def __init__(
        self,
        graph: GraphDescriptor,
        model: ModelDescriptor,
        parms: Optional[dict] = None,
        sites=None,
        _load_operators=True,
        qubit_split: Optional[bool] = None,
    ):
        """Build a system from a graph and a model.

        Parameters
        ----------
        graph, model, parms :
            Geometry, model and parameters.
        sites : dict, optional
            Local spaces, overriding the ones derived from ``graph`` (used
            internally by ``union`` and ``subsystem``).
        qubit_split : bool or None
            Whether to represent each site by one qubit per two-level
            quantum number, with fermionic modes mapped to Pauli strings
            through the Jordan-Wigner transformation. ``None`` (default)
            does it when the model has fermionic quantum numbers and every
            site can be split; ``True`` forces it (raising ``ValueError`` if
            some site cannot be split); ``False`` disables it.
        """
        if parms is None:
            parms = {}
        if model:
            model_parms = model.parms.copy()
            model_parms.update(parms)
            parms = model_parms

        self.spec = {"graph": graph, "model": model, "parms": parms}
        self.name = f"{self.spec['model'].name} on {self.spec['graph'].name}"
        site_basis = model.site_basis
        try:
            physical_sites = {
                node: site_basis[attr["type"]] for node, attr in graph.nodes.items()
            }
        except KeyError as ex:
            if not sites:
                raise ValueError(
                    (
                        f"Model <<{model.name}>> does not provide the specification "
                        f"for site of type <<{ex.args[0]}>> used in nodes of <<{graph.name}>>."
                    )
                ) from ex
            physical_sites = {}
        given_qubits = bool(sites) and all("parent" in s for s in sites.values())
        if sites and not given_qubits:
            physical_sites = sites
        self.qubit_split = qubit_split
        self.physical_sites = physical_sites
        layout = _qubit_layout(physical_sites, qubit_split)
        # With the qubit representation, every two-level quantum number of
        # every site becomes a qubit, ordered as the nodes and, inside each
        # node, as the quantum numbers. Fermionic modes are mapped to Pauli
        # strings through the Jordan-Wigner transformation in that order.
        self.qubit_representation = layout is not None
        if layout is None:
            self.sites = sites if sites else physical_sites
            self.site_qubits = {node: (node,) for node in self.sites}
        else:
            qubit_sites = {
                name: _qubit_site_basis(node, qn_name, lower, fermionic, physical_sites[node])
                for node, entries in layout.items()
                for name, qn_name, _, lower, fermionic in entries
            }
            self.sites = sites if given_qubits else qubit_sites
            site_qubits = {
                node: tuple(entry[0] for entry in entries if entry[0] in self.sites)
                for node, entries in layout.items()
            }
            self.site_qubits = {node: qs for node, qs in site_qubits.items() if qs}
        self.qubit_parent = {
            qubit: node for node, qubits in self.site_qubits.items() for qubit in qubits
        }

        self.dimensions = {name: site["dimension"] for name, site in self.sites.items()}
        self.operators = {
            "site_operators": {},
            "bond_operators": {},
            "loop_operators": {},
            "global_operators": {},
        }
        self._subsystems_cache = {}
        self._jw_strings = None
        if _load_operators:
            if not self.qubit_representation and any(
                getattr(op, "fermionic", False)
                for site in self.physical_sites.values()
                for op in site.get("operators", {}).values()
            ):
                warnings.warn(
                    f"The model <<{model.name}>> has fermionic operators, but "
                    "fermionic signs are not implemented: operators on different "
                    "sites are assumed to commute, so in general the results "
                    "correspond to hard-core particles, not to fermions.",
                    FermionicSignWarning,
                    stacklevel=2,
                )
            self._load_site_operators()
            self._load_global_ops()

    def __eq__(self, other):
        """Check equality."""
        # To be the equal, two SystemDescriptors
        # should have the same values in the
        # `spec` attribute:
        if self is other:
            return True
        if other is None:
            return False
        assert isinstance(other, SystemDescriptor)
        for key, my_spec in self.spec.items():
            if other.spec[key] != my_spec:
                return False
        for site in other.sites:
            if site not in self.sites:
                return False

        for site, descr in self.sites.items():
            if descr != other.sites[site]:
                return False

        return True

    def __repr__(self):
        """Build repr str."""
        result = (
            "graph:"
            + repr(self.spec["graph"])
            + "\n"
            + "sites:"
            + repr(self.sites.keys())
            + "\n"
            + "dimensions:"
            + repr(self.dimensions)
        )
        return result

    def __getstate__(self):
        """Get state."""
        # Copy the object's state and exclude _subsystems_cache and operators
        if hasattr(self, "_serialized"):
            return self._serialized

        state = {
            key: {} if key in ("_subsystems_cache", "operators") else val
            for key, val in self.__dict__.items()
        }
        self._serialized = pickle.dumps(state)
        return self._serialized

    def __iter__(self):
        """Iterate over sites."""
        yield from self.sites

    def __setstate__(self, state):
        """Set state."""
        state_dict = pickle.loads(state)
        # Files written before the qubit representation was introduced.
        if "physical_sites" not in state_dict:
            sites = state_dict.get("sites", {})
            state_dict["physical_sites"] = sites
            state_dict["site_qubits"] = {node: (node,) for node in sites}
            state_dict["qubit_parent"] = {node: node for node in sites}
            state_dict["qubit_representation"] = False
            state_dict["qubit_split"] = False
        state_dict["_jw_strings"] = None
        state_dict["operators"] = {
            "site_operators": {},
            "bond_operators": {},
            "loop_operators": {},
            "global_operators": {},
        }
        self.__dict__.update(state_dict)

    def subsystem(self, sites: frozenset):
        """Build a subsystem including the sites listed in sites."""
        # Try to find the subsystem in the cache
        assert isinstance(sites, frozenset)
        result = self._subsystems_cache.get(sites, None)
        if result is not None:
            return result

        # To avoid circular references, the cache
        # only stores proper subsystems
        self_sites = self.sites
        if len(self_sites) == len(sites):
            if all(name in self_sites for name in sites):
                return self

        # Build a new subsystem
        parms = self.spec["parms"].copy()
        model = self.spec["model"]
        if self.qubit_representation:
            # ``sites`` are qubits: keep the nodes they belong to, and the
            # qubits in the same order as in ``self``.
            qubit_parent = self.qubit_parent
            graph = self.spec["graph"].subgraph(
                frozenset(qubit_parent[qubit] for qubit in sites)
            )
            qubits = {q: basis for q, basis in self_sites.items() if q in sites}
            result = SystemDescriptor(
                graph,
                model,
                parms,
                sites=qubits,
                _load_operators=False,
                qubit_split=self.qubit_split,
            )
            return self._subsystems_cache.setdefault(sites, result)
        graph = self.spec["graph"].subgraph(sites)
        result = SystemDescriptor(
            graph, model, parms, _load_operators=False, qubit_split=self.qubit_split
        )
        return self._subsystems_cache.setdefault(sites, result)

    def _load_site_operators(self):
        for site_name, site in self.physical_sites.items():
            for op_name in site["operators"]:
                op_site = f"{op_name}@{site_name}"
                self.site_operator(op_site)

    def _load_global_ops(self):
        # Import here to avoid circular dependency
        # pylint: disable=import-outside-toplevel
        from qalma.operators import LocalOperator, OneBodyOperator

        # First, load conserved quantum numbers:
        for constraint_qn in self.spec["model"].constraints:
            if self.qubit_representation:
                # pylint: disable=import-outside-toplevel
                from qalma.operators import SumOperator
                from qalma.operators.pauli import to_pauli_sum

                local_terms = tuple(
                    self.site_operator(site_basis["qn"][constraint_qn]["operator"], node)
                    for node, site_basis in self.physical_sites.items()
                    if constraint_qn in site_basis["qn"]
                )
                if local_terms:
                    global_qn = to_pauli_sum(SumOperator(local_terms, self))
                    if bool(global_qn):
                        self.operators["global_operators"][constraint_qn] = global_qn
                continue
            global_qn_terms = []
            for site, site_basis in self.sites.items():
                local_qn = site_basis["qn"].get(constraint_qn, None)
                if local_qn is None:
                    continue
                op_name = local_qn["operator"]
                operator = site_basis["operators"][op_name]

                global_qn_terms.append(LocalOperator(site, operator, self))

            global_qn = OneBodyOperator(tuple(global_qn_terms), self, True).simplify()

            if bool(global_qn):
                self.operators["global_operators"][constraint_qn] = global_qn

        names = list(self.spec["model"].global_ops)
        for gop in names:
            self.global_operator(gop)

    def __mul__(self, system):
        """Return the union of two models."""
        if system is None or system is self:
            return self
        return self.union(system)

    def __len__(self):
        """Return the number of sites in the model."""
        return len(self.sites)

    def _repr_latex_(self):
        return "System \textbf{" + self.name + "}"

    def contains(self, system):
        """Check if the object contains `system` as a subsystem."""
        if self is system:
            return True
        block = frozenset(system.sites)
        if system is self._subsystems_cache.get(block, None):
            return True

        for subsystem in self._subsystems_cache.values():
            if subsystem.contains(system):
                self._subsystems_cache[block] = system
                return True

        logger.info("check model and graph")
        if (
            self.spec["model"] == system.spec["model"]
            and self.spec["graph"].contains(system.spec["graph"])
            and all(site in self.sites for site in system.sites)
        ):
            self._subsystems_cache[block] = system
            return True
        if not self.spec["model"] == system.spec["model"]:
            logger.debug(
                "models differ: %s vs %s", self.spec["model"], system.spec["model"]
            )
        if not self.spec["graph"].contains(system.spec["graph"]):
            logger.debug(
                "graph %s does not contain %s", self.spec["graph"], system.spec["graph"]
            )
        return False

    def union(self, system):
        """Return a SystemDescritor containing system and self."""
        if system is None or system is self:
            return self
        if all(site in self.sites for site in system.sites):
            return self
        if all(site in system.sites for site in self.sites):
            return system

        model = self.spec["model"]
        assert (
            model is system.spec["model"]
        ), "Join systems with different base models is not supported."
        parms = self.spec["parms"].copy()
        union_graph = self.spec["graph"] + system.spec["graph"]
        sites = self.sites.copy()
        sites.update(system.sites)
        # raise NotImplementedError("Union of disjoint systems are not implemented.")
        result = SystemDescriptor(
            union_graph, model, parms, sites, qubit_split=self.qubit_split
        )
        result._subsystems_cache[frozenset(self.sites)] = self
        result._subsystems_cache[frozenset(system.sites)] = system

        return result

    def site_identity(self, site: str):  # -> Qobj
        """
        Return the identity operator for the site.

        Returns the internal representation of the identity associated to
        `site`.
        """
        return self.sites[site]["identity"]

    def site_operator(self, name: str, site: str = ""):  # -> "Operator"
        """Return a site operator.

        Return a global operator representing an operator `name` acting over
        the site `site`.

        By default, the name is assumed to specify both the name and
        site in the form `"name@site"`.
        """
        # Import here to avoid circular dependency
        # pylint: disable=import-outside-toplevel
        from qalma.operators import LocalOperator

        if site != "":
            op_name = name
            name = f"{name}@{site}"
        else:
            op_name, site = name.split("@")

        site_op = self.operators["site_operators"].get(site, {}).get(op_name, None)
        if site_op is not None:
            return site_op

        local_op = self.physical_sites[site]["operators"].get(op_name, None)
        if local_op is None:
            return None
        if self.qubit_representation:
            result_op = self.jordan_wigner_operator(site, local_op)
        else:
            result_op = LocalOperator(site, local_op, system=self)
        self.operators["site_operators"].setdefault(site, {})
        self.operators["site_operators"][site][op_name] = result_op
        return result_op

    @property
    def jordan_wigner(self) -> bool:
        """``True`` if the fermionic modes are mapped to Pauli strings.

        That is, if the system uses the qubit representation and some qubit
        is a fermionic mode.
        """
        return self.qubit_representation and any(
            basis.get("fermionic", False) for basis in self.sites.values()
        )

    def jordan_wigner_string(self, qubit: str):
        r"""Return :math:`\prod_{q' < q} (-1)^{N_{q'}}` as a Pauli string.

        The product runs over the fermionic qubits that come before
        ``qubit`` in ``self.sites``. Only available when
        ``self.qubit_representation`` is ``True``.
        """
        # pylint: disable=import-outside-toplevel
        from qalma.operators.pauli import PauliStringOperator

        code, sign = self._jordan_wigner_strings()[qubit]
        return PauliStringOperator(code, sign, self)

    def _jordan_wigner_strings(self) -> dict:
        """Map each qubit to ``(code, sign)`` of its Jordan-Wigner string."""
        if self._jw_strings is not None:
            return self._jw_strings
        assert self.qubit_representation, "the system is not represented by qubits."
        result = {}
        code, sign = 0, 1.0
        for position, (qubit, basis) in enumerate(self.sites.items()):
            result[qubit] = (code, sign)
            if basis.get("fermionic", False):
                # (-1)^N = parity0 * sigma_z
                code |= 3 << (2 * position)
                sign *= basis["parity0"]
        self._jw_strings = result
        return result

    def jordan_wigner_operator(self, site: str, local_op):
        r"""Map an operator on the node ``site`` to a sum of Pauli strings.

        The local space of ``site`` is the product of its qubits, one per
        two-level quantum number. The operator is expanded in products of
        Pauli matrices on those qubits, and each product is read as the
        product of single-mode operators in the order of the quantum
        numbers, with each :math:`\sigma^{x,y}` on a fermionic qubit
        replaced by its Jordan-Wigner image
        :math:`\prod_{q'<q}(-1)^{N_{q'}}\,\sigma^{x,y}_q`.

        This is the convention in which the local matrices of the ALPS
        library are written: ``c_up`` and ``c_down`` act on their own
        quantum number with matrix element 1, ``Splus`` is
        :math:`c^\dagger_\uparrow c_\downarrow`, and so on. The resulting
        mode operators satisfy the canonical anticommutation relations,
        also between modes of the same site.
        """
        # pylint: disable=import-outside-toplevel
        from qalma.operators.pauli import (
            PAULI_MATRICES,
            _I_POWERS,
            _product_phase,
            _site_index,
            pauli_dict_to_operator,
        )
        from qalma.qutip_tools.tools import _to_array

        qubits = self.site_qubits[site]
        n_qubits = len(qubits)
        site_basis = self.physical_sites[site]
        qn_index = site_basis["qn_index"]
        qubit_bases = [self.sites[q] for q in qubits]
        positions = [qn_index[next(iter(b["qn"]))] for b in qubit_bases]
        lowers = [b["localstates"][0][0] for b in qubit_bases]

        # Rewrite the local matrix in the product basis of the qubits.
        matrix = _to_array(local_op)
        permutation = np.array(
            [
                sum(
                    int(round(state[pos] - low)) << (n_qubits - 1 - r)
                    for r, (pos, low) in enumerate(zip(positions, lowers))
                )
                for state in site_basis["localstates"]
            ]
        )
        product_matrix = np.zeros_like(matrix, dtype=complex)
        product_matrix[np.ix_(permutation, permutation)] = matrix

        index = _site_index(self)
        mask = index.mask
        qubit_shift = [2 * index.position[q] for q in qubits]
        strings = self._jordan_wigner_strings()
        coefficients: dict = {}
        scale = 2.0**n_qubits
        for labels in itertools.product(range(4), repeat=n_qubits):
            pauli = PAULI_MATRICES[labels[0]]
            for label in labels[1:]:
                pauli = np.kron(pauli, PAULI_MATRICES[label])
            coeff = np.trace(pauli @ product_matrix) / scale
            if abs(coeff) < 1e-14:
                continue
            # Ordered product of the single-mode factors.
            code, value = 0, complex(coeff)
            for r, label in enumerate(labels):
                if label == 0:
                    continue
                factor_code = label << qubit_shift[r]
                if label in (1, 2) and qubit_bases[r]["fermionic"]:
                    string_code, sign = strings[qubits[r]]
                    factor_code |= string_code
                    value *= sign
                k, code = _product_phase(code, factor_code, mask)
                value *= _I_POWERS[k]
            coefficients[code] = coefficients.get(code, 0) + value
        coefficients = {c: v for c, v in coefficients.items() if abs(v) > 1e-14}
        return pauli_dict_to_operator(coefficients, self)

    def bond_operator(self, name: str, src: str, dst: str, skip=None):  # -> "Operator":
        """Return a bond operator by name and sites."""
        result_op = self.operators["global_operators"].get(
            (
                name,
                src,
                dst,
            ),
            None,
        )
        if result_op is not None:
            return result_op
        # Try to build the bond operator from the descriptors.
        bond_op_descriptors = self.spec["model"].bond_ops
        bond_op_descriptor = bond_op_descriptors.get(name, None)
        if bond_op_descriptor is None:
            return None

        bond_dependencies = [
            bop
            for bop in bond_op_descriptors
            if bond_op_descriptor.find(bop + "@") >= 0
        ]
        bond_op_descriptor = bond_op_descriptor.replace("@", "__")
        src_operators = self.physical_sites[src]["operators"]
        dst_operators = self.physical_sites[dst]["operators"]

        # Load site operators on src and dst
        parms_and_ops = {
            f"{name_src}__src": self.site_operator(name_src, src)
            for name_src in src_operators
        }
        parms_and_ops.update(
            {
                f"{name_dst}__dst": self.site_operator(name_dst, dst)
                for name_dst in dst_operators
            }
        )
        self_parms = self.spec["parms"]
        if self_parms:
            parms_and_ops.update(self_parms)

        # Try to evaluate
        result = eval_expr(bond_op_descriptor, parms_and_ops)
        if not (result is None or isinstance(result, str)):
            self.operators["bond_operators"][
                (
                    name,
                    src,
                    dst,
                )
            ] = result
            return result
        # Now, try to include existent bond operators
        parms_and_ops.update(
            {
                f"{tup_op[0]}__src_dst": op
                for tup_op, op in self.operators["global_operators"].items()
                if tup_op[0] == src and tup_op[1] == dst
            }
        )
        parms_and_ops.update(
            {
                f"{tup_op[0]}__dst_src": op
                for tup_op, op in self.operators["global_operators"].items()
                if tup_op[0] == dst and tup_op[1] == src
            }
        )

        result = eval_expr(bond_op_descriptor, parms_and_ops)
        if not (result is None or isinstance(result, str)):
            self.operators["bond_operators"][
                tuple(
                    (
                        name,
                        src,
                        dst,
                    )
                )
            ] = result
            return result

        # Finally, try to load other operators
        if skip is None:
            skip = [name]
        else:
            skip.append(name)
        for bop in bond_dependencies:
            # Skip this name
            if bop in skip:
                continue

            # src->dst
            new_bond_op = self.bond_operator(bop, src, dst, skip)
            if new_bond_op is None:
                continue

            parms_and_ops[f"{bop}__src_dst"] = new_bond_op

            new_bond_op = self.bond_operator(bop, dst, src, skip)
            if new_bond_op is None:
                continue

            parms_and_ops[f"{bop}__dst_src"] = new_bond_op

            result = eval_expr(bond_op_descriptor, parms_and_ops)
            if result is not None and not isinstance(result, str):
                result = result.simplify()
                self.operators["bond_operators"][
                    (
                        name,
                        src,
                        dst,
                    )
                ] = result
                return result

        # If this fails after exploring all the operators, then it means that
        # the operator is not in the basis.
        # if skip[-1]==name:
        #    self.bond_operators[(name, src, dst,)] = None
        return None

    def loop_operator(self, name: str, loop: Tuple[str], _skip=None):  # -> "Operator":
        """Return a loop operator by name and sites."""
        result_op = self.operators["global_operators"].get(
            (
                name,
                loop,
            ),
            None,
        )
        if result_op is not None:
            return result_op
        loop_op_descriptors = self.spec["model"].loop_ops
        loop_op_descriptor = loop_op_descriptors.get(name, None)
        if loop_op_descriptor is None:
            return None

        # loop_dependencies = [
        #    bop
        #    for bop in loop_op_descriptors
        #    if loop_op_descriptor.find(bop + "@") >= 0
        # ]
        loop_op_descriptor = loop_op_descriptor.replace("@", "__")
        # vertex_operators  = tuple((self.sites[src]["operators"] for src in loop))
        raise ValueError(f"{loop_op_descriptor} not implemented for  {loop}.")

    def site_term_from_descriptor(self, term_spec, graph, parms):
        """Build a site term from a site term specification."""
        # Import here to avoid circular dependency
        # pylint: disable=import-outside-toplevel
        from qalma.operators import OneBodyOperator, ScalarOperator

        expr = term_spec["expr"]
        site_type = term_spec.get("type", None)
        term_ops = []
        t_parm = {}
        t_parm.update(term_spec.get("parms", {}))
        if parms:
            t_parm.update(parms)
        for node_name, node in graph.nodes.items():
            node_type = node.get("type", None)
            if site_type is not None and site_type != node_type:
                continue
            s_expr = expr.replace("#", node_type)
            operator_names = set(re.findall(r"\b([a-zA-Z_]+)\b@", s_expr))

            s_expr = s_expr.replace("@", "__")
            s_parm = {
                replace_variable_type(key, node_type): replace_variable_type(
                    val, node_type
                )
                for key, val in t_parm.items()
            }
            s_parm.update(
                {
                    f"{name_op}_local": local_op
                    for name_op, local_op in self.operators["site_operators"][
                        node_name
                    ].items()
                }
            )
            term_op = eval_expr(s_expr, s_parm)
            if term_op is None or isinstance(term_op, str):
                raise ValueError(
                    f"<<{s_expr}>> could not be evaluated.", operator_names
                )
            term_ops.append(term_op)

        if len(term_ops) == 0:
            return ScalarOperator(0, self)
        if len(term_ops) == 1:
            return term_ops[0]
        if self.qubit_representation:
            # Odd site terms carry Jordan-Wigner strings: they are not
            # one-body operators in the qubit representation.
            from qalma.operators import SumOperator
            from qalma.operators.pauli import to_pauli_sum

            term_ops = [
                op if hasattr(op, "system") else ScalarOperator(op, self)
                for op in term_ops
            ]
            return to_pauli_sum(SumOperator(tuple(term_ops), self))
        return OneBodyOperator(tuple(term_ops), self)

    def bond_term_from_descriptor(self, term_spec, graph, model, parms):
        """Build a bond term from a bond term specification."""
        # Import here to avoid circular dependency
        # pylint: disable=import-outside-toplevel
        from qalma.operators import ScalarOperator, SumOperator

        def process_edge(e_expr, bond, model, t_parm):
            edge_type, src, dst = bond
            e_parms = {
                replace_variable_type(key, edge_type): replace_variable_type(
                    val, edge_type
                )
                for key, val in t_parm.items()
            }
            for op_idx in ([src, "src"], [dst, "dst"]):
                e_parms.update(
                    {
                        f"{key}__{op_idx[1]}": val
                        for key, val in self.operators["site_operators"][
                            op_idx[0]
                        ].items()
                    }
                )

            # Try to compute using only site terms

            term_op = eval_expr(e_expr, e_parms)
            if not isinstance(term_op, str):
                return term_op

            # Try now adding the bond operators
            for name_bop in model.bond_ops:
                self.bond_operator(name_bop, src, dst)
                # TODO: for fermions, take into account the sign.
                self.bond_operator(name_bop, dst, src)

            for src_idx, dst_idx in ((1, 2), (2, 1)):
                e_parms.update(
                    {
                        f"{key[0]}__src_dst": val
                        for key, val in self.operators["bond_operators"].items()
                        if key[src_idx] == src and key[dst_idx] == dst
                    }
                )

            return eval_expr(e_expr, e_parms)

        expr = term_spec["expr"]
        term_type = term_spec.get("type", None)
        t_parm = {}
        t_parm.update(term_spec.get("parms", {}))
        if parms:
            t_parm.update(parms)
        result_terms = []
        for edge_type, edges in graph.edges.items():
            if term_type is not None and term_type != edge_type:
                continue

            e_expr = expr.replace("#", edge_type)
            operator_names = set(re.findall(r"\b([a-zA-Z_]+)\b@", e_expr))
            e_expr = e_expr.replace("@", "__")
            for src, dst in edges:
                term_op = process_edge(e_expr, (edge_type, src, dst), model, t_parm)
                if isinstance(term_op, str):
                    raise ValueError(
                        f"   Bond term <<{term_op}>> could not be evaluated.",
                        operator_names,
                    )

                result_terms.append(term_op)

        if len(result_terms) == 0:
            return ScalarOperator(0.0, self)
        if len(result_terms) == 1:
            return result_terms[0]
        return SumOperator(tuple(result_terms), self, True)

    def loop_term_from_descriptor(self, term_spec, graph, model, parms):
        """Build loop term from aloop term specification."""
        # Import here to avoid circular dependency
        # pylint: disable=import-outside-toplevel
        from qalma.operators import ScalarOperator, SumOperator

        def process_loop(loop_expr, loop_type, vertices_map, _model, t_parm):
            loop_parms = {
                replace_variable_type(key, loop_type): replace_variable_type(
                    val, loop_type
                )
                for key, val in t_parm.items()
            }

            for key, site in vertices_map.items():
                loop_expr = loop_expr.replace(f"@[{key}]", f"__{key}_")
                loop_parms.update(
                    {
                        f"{op_key}__{key}_": val
                        for op_key, val in self.operators["site_operators"][
                            site
                        ].items()
                    }
                )

            term_op = eval_expr(loop_expr, loop_parms)
            assert not isinstance(term_op, str), f"got {term_op}"
            return term_op

        expr = term_spec["expr"]
        term_type = term_spec.get("type", None)
        index_names = term_spec.get("indices", None)
        t_parm = {}
        t_parm.update(term_spec.get("parms", {}))
        if parms:
            t_parm.update(parms)
        result_terms = []
        for loop_type, loops in graph.loops.items():
            if term_type is not None and term_type != loop_type:
                continue
            loop_expr = expr.replace("#", loop_type)
            operator_names = set(re.findall(r"\b([a-zA-Z_]+)\b@", loop_expr))
            for vertices in loops:
                vertices_site_map = dict(zip(index_names, vertices))
                term_op = process_loop(
                    loop_expr, loop_type, vertices_site_map, model, t_parm
                )
                if isinstance(term_op, str):
                    raise ValueError(
                        f"   Loop term <<{term_op}>> could not be evaluated.",
                        operator_names,
                    )

                result_terms.append(term_op)

        if len(result_terms) == 0:
            return ScalarOperator(0.0, self)
        if len(result_terms) == 1:
            return result_terms[0]
        return SumOperator(tuple(result_terms), self, True)

    def global_operator(self, name):
        """Return a global operator by its name."""
        # pylint: disable=import-outside-toplevel
        from qalma.operators import OneBodyOperator, ScalarOperator, SumOperator

        if len(self.operators["site_operators"]) == 0:
            self._load_site_operators()
            self._load_global_ops()

        result = self.operators["global_operators"].get(name, None)
        if result is not None:
            logger.info("%s already defined. Return from cache.", name)
            return result
        # Build the global_operator from the descriptor
        op_descr = self.spec["model"].global_ops.get(name, None)
        if op_descr is None:
            logger.warning(
                "%s not defined in %s.", name, str(self.spec["model"].global_ops.keys())
            )
            return None

        graph = self.spec["graph"]
        parms = self.spec["parms"]
        model = self.spec["model"]

        # Process site terms
        try:
            site_terms = (
                self.site_term_from_descriptor(term_spec, graph, parms)
                for term_spec in op_descr["site terms"]
            )
            site_terms = tuple(term for term in site_terms if term)
        except ValueError as exc:
            logger.debug("%s aborting evaluation of %s.", exc.args, name)
            model.global_ops.pop(name)
            return None

        # Process bond terms
        try:
            bond_terms = tuple(
                self.bond_term_from_descriptor(term_spec, graph, model, parms)
                for term_spec in op_descr["bond terms"]
            )
            bond_terms = tuple(term for term in bond_terms if term)

        except ValueError as exc:
            logger.debug("%s aborting evaluation of %s.", exc.args, name)
            model.global_ops.pop(name)
            return None

        # Process loop terms
        try:
            loop_terms = tuple(
                self.loop_term_from_descriptor(term_spec, graph, model, parms)
                for term_spec in op_descr["loop terms"]
            )
            loop_terms = tuple(term for term in loop_terms if term)

        except ValueError as exc:
            logger.debug("%s Aborting evaluation of %s.", exc.args, name)
            model.global_ops.pop(name)
            return None

        interaction_terms = bond_terms + loop_terms
        if self.qubit_representation:
            # pylint: disable=import-outside-toplevel
            from qalma.operators.pauli import to_pauli_sum

            all_terms = site_terms + interaction_terms
            if not all_terms:
                result = ScalarOperator(0, self)
            elif len(all_terms) == 1:
                result = to_pauli_sum(all_terms[0])
            else:
                result = to_pauli_sum(SumOperator(all_terms, self))
            self.operators["global_operators"][name] = result
            return result
        if interaction_terms:
            result = SumOperator(site_terms + interaction_terms, self, True)
        else:
            result = OneBodyOperator(site_terms, self, True)
        result = result.simplify()
        self.operators["global_operators"][name] = result
        logger.debug("  return operator of type %s.", str(type(result)))
        return result


def build_spin_chain(length: int = 2, field=0.0):
    """Build a nn spin chain of length ``l`` in an external magnetic field."""
    parameters = {"L": length, "a": 1, "h": field, "J": 1, "Jz0": 1, "Jxy0": 1}
    return build_system("chain lattice", "spin", **parameters)


def build_system(
    geometry_name: str = "chain lattice",
    model_name: str = "spin",
    models_lib_file=MODEL_LIB_FILE,
    lattice_lib_file=LATTICE_LIB_FILE,
    qubit_split: Optional[bool] = None,
    **kwargs,
) -> SystemDescriptor:
    """Build a SystemDescriptor from the names of the geometry and the model.

    lattice_lib_file: str

    qubit_split: bool or None
        Represent the sites by qubits (one per two-level quantum number),
        with fermions mapped through the Jordan-Wigner transformation.
        See :class:`SystemDescriptor`. By default, this is done for
        fermionic models whose sites can be split into qubits.

    kwargs: Optional keyword parameters are passed to the model.
    """
    # pylint: disable=import-outside-toplevel
    from qalma.alpsmodels import model_from_alps_xml

    # pylint: disable=import-outside-toplevel
    from qalma.geometry import graph_from_alps_xml

    logger.info("loading model %s over graph %s.", model_name, geometry_name)

    parms = {"L": 4, "J": 1, "a": 1}
    parms.update(kwargs)
    model = model_from_alps_xml(models_lib_file, model_name, parms)
    graph = graph_from_alps_xml(lattice_lib_file, geometry_name, parms)

    assert model is not None
    assert graph is not None
    return SystemDescriptor(graph, model, parms, qubit_split=qubit_split)


def build_system_from_dims(dims_by_name: Dict[str, int]) -> SystemDescriptor:
    """Build a System from the dimension of each site.

    Parameters
    ----------
    dims_by_name: dict[str, int]
       a dictionary specifying the dimension of each site in the system.

    Return
    =======

    system: SystemDescriptor
        A system with the specified local sites.

    """
    model = qutip_model_from_dims(dims_by_name.values())
    sitebasis = model.site_basis
    sites = {name: sitebasis[f"dim={dim}"] for name, dim in dims_by_name.items()}
    graph = GraphDescriptor(
        "disconnected graph",
        {name: {"type": f"dim={dim}"} for name, dim in dims_by_name.items()},
        {},
        {},
    )
    return SystemDescriptor(graph, model, sites=sites)
