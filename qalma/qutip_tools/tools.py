"""Functions for basic interface with qutip objects."""

import logging
from functools import reduce
from numbers import Real
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np

# type: ignore[import-untyped]
from numpy import zeros as _np_zeros
from numpy.linalg import eigh as _eigh
from packaging.version import parse as parse_version
from qutip import (  # type: ignore[import-untyped]
    Qobj as _Qobj,
    __version__ as qutip_version_string,
    tensor as _qutip_tensor,
)
from qutip.core.data.csr import CSR as _Qutip_CSR, fast_from_scipy as _fast_from_scipy
from qutip.core.data.dense import (
    Dense as _Qutip_Dense,
    fast_from_numpy as _fast_from_numpy,
)
from scipy.linalg import norm as _scipy_norm, svd as _svd
from scipy.sparse import csr_matrix as _sp_csr_matrix
from scipy.sparse.linalg import (
    ArpackError as _ArpackError,
    ArpackNoConvergence as _ArpackNoConvergence,
)

qutip_version = parse_version(qutip_version_string)


__all__ = [
    "get_proper_spaces",
    "data_element_iterator",
    "is_empty_op",
    "decompose_qutip_operator_hermitian",
]


def ishermitian(array: np.ndarray):
    """Determine if the array is hermitian."""
    return np.allclose(array, array.T.conj())


def _to_array(op) -> np.ndarray:
    """Convert a local operator to np.ndarray complex128.

    Accepts:
      - np.ndarray  → return a C-contiguous copy of type complex128
      - qutip.Qobj  → get a dense matrix representation via `.full()`
      - int / float / complex → error (should be handled as prefactors)
    """
    if isinstance(op, np.ndarray):
        return np.asarray(op, dtype=complex, order="C")
    if isinstance(op, _Qobj):
        return np.asarray(op.full(), dtype=complex, order="C")
    raise TypeError(
        f"Local operators must be np.ndarray o qutip.Qobj, " f"got {type(op)}"
    )


if qutip_version < parse_version("5.0.0"):

    def data_element_iterator(data) -> Iterator:
        """Retrieve a generator for the nontrivial elements."""
        i_idx, j_idx = data.nonzero()
        yield from zip(i_idx, j_idx, data.data)

    def data_get_coeff(data, i_idx, j_idx):
        """Access to a matrix entry."""
        return data[i_idx, j_idx]

    def data_get_type(data) -> type:
        """Get the type of the elements in data."""
        return data.dtype

    def data_is_diagonal(data) -> bool:
        """Check if data is diagonal."""
        if data.nnz == 0 or all(a == b for a, b in zip(*data.nonzero())):
            return True
        return all(val == 0 for val, a, b in zip(data.data, *data.nonzero()) if a != b)

    def data_is_zero(data) -> bool:
        """Check if the matrix is empty."""
        if data.nnz == 0:
            return True
        return not any(data.data)

    def scalar_value(data):
        """If data is a scalar matrix, return any of its diagonal elements.

        Otherwise, return `None`
        """
        if data.nnz == 0:
            return 0.0
        if all(a == b for a, b in zip(*data.nonzero())):
            dim1, _ = data.shape
            elems = data.data
            if len(elems) < dim1:
                return None
            val = elems[0]
            return val if all(val == elem for elem in elems) else None

        if any(val for val, a, b in zip(data.data, *data.nonzero()) if a != b):
            return None
        vals = [val for val, a, b in zip(data.data, *data.nonzero()) if a == b]
        elem = vals[0]
        return elem if all(elem == val for val in vals) else None

    def fast_tensor(*factors):
        """
        Compute a fast tensor product using a CSR representation.

        If some of the factors are not in Dense representation, convert
        everthing to CSR to speedup the computation.
        """
        return _qutip_tensor(*factors)

else:

    def data_element_iterator(data) -> Iterator:
        """Walk over data elements."""

        def do_dense():
            arr = data.as_ndarray() if hasattr(data, "as_ndarray") else data.to_array()
            dim_i, dim_j = arr.shape
            for i in range(dim_i):
                for j in range(dim_j):
                    v = arr[i, j]
                    if v != 0:
                        yield (i, j, v)

        # Backward compatibility v5.0 and v5.1
        def do_dia_5_0(data_dia):
            data = data_dia.as_scipy()
            dim_i, dim_j = data.shape
            for offset, diag_data in zip(data.offsets, data.data):
                if offset < 0:
                    for j_pos, value in enumerate(diag_data):
                        i_pos = j_pos - offset
                        if i_pos >= dim_i:
                            break
                        yield (
                            i_pos,
                            j_pos,
                            value,
                        )
                else:
                    for i_pos, value in enumerate(diag_data):
                        j_pos = i_pos + offset
                        if j_pos >= dim_j:
                            break
                        yield (
                            i_pos,
                            j_pos,
                            value,
                        )

        def do_dia_5_2(data_dia):
            data = data_dia.as_scipy()
            dim_i, _ = data.shape
            for offset, diag_data in zip(data.offsets, data.data):
                if offset < 0:
                    for j_pos, value in enumerate(diag_data):
                        i_pos = j_pos - offset
                        if i_pos >= dim_i:
                            break
                        yield (
                            i_pos,
                            j_pos,
                            value,
                        )
                else:
                    for indx, value in enumerate(diag_data[offset:]):
                        i_pos = indx
                        j_pos = i_pos + offset
                        yield (
                            i_pos,
                            j_pos,
                            value,
                        )

        # Diagonal format
        if hasattr(data, "num_diag"):
            if qutip_version < parse_version("5.2.0"):
                yield from do_dia_5_0(data)
            # For 5.2
            else:
                yield from do_dia_5_2(data)
        elif hasattr(data, "as_scipy"):
            data = data.as_scipy()
            if hasattr(data, "tocoo"):
                coo = data.tocoo()
                for i, j, v in zip(coo.row, coo.col, coo.data):
                    yield (i, j, v)
            else:
                # Fallback: try nonzero and data.data
                try:
                    i_ind, j_ind = data.nonzero()
                    for idx, i_val in enumerate(i_ind):
                        yield (i_val, j_ind[idx], data.data[idx])
                except Exception:
                    # Last resort: try dense
                    yield from do_dense()
        else:
            yield from do_dense()

    def data_get_coeff(data, i_idx, j_idx):
        """Access to a matrix entry."""
        if hasattr(data, "num_diag"):
            data_sp = data.as_scipy()
            offset = j_idx - i_idx
            offsets = data_sp.offsets
            if offset not in offsets:
                return 0
            return data_sp.diagonal(offset)[j_idx]
        if hasattr(data, "as_scipy"):
            return data.as_scipy()[i_idx, j_idx]
        return data.as_ndarray()[i_idx, j_idx]

    def data_get_type(data) -> type:
        """Get the type of the elements in data."""
        if hasattr(data, "as_scipy"):
            return data.as_scipy().dtype
        return data.as_ndarray().dtype

    def data_is_diagonal(data) -> bool:
        """Check if data is diagonal."""
        if hasattr(data, "num_diag"):
            if data.num_diag == 0:
                return True
            if data.num_diag > 1:
                return False
            offsets = data.as_scipy().offsets
            return bool(offsets[0] == 0)
        if hasattr(data, "as_scipy"):
            data = data.as_scipy()
            if data.nnz == 0:
                return True
            return all(a == b for a, b in zip(*data.nonzero()))
        if hasattr(data, "as_ndarray"):
            data = data.as_ndarray()
        dim_i, dim_j = data.shape
        return not any(
            data[i_idx, j_idx]
            for i_idx in range(dim_i)
            for j_idx in range(dim_j)
            if i_idx != j_idx
        )

    def data_is_zero(data) -> bool:
        """Check if the matrix is empty."""
        if hasattr(data, "num_diag"):
            return data.num_diag == 0
        if hasattr(data, "as_scipy"):
            return data.as_scipy().nnz == 0
        return np.count_nonzero(data.as_ndarray()) == 0

    def scalar_value(data):
        """If data is a scalar matrix, return any of its diagonal elements.

        Otherwise, return `None`
        """
        dim1, _ = data.shape
        if hasattr(data, "num_diag"):
            if data.num_diag == 0:
                return 0.0
            if data.num_diag > 1:
                return None
            data = data.as_scipy()
            offsets = data.offsets
            if bool(offsets[0] != 0):
                return None
            diagonal = data.diagonal(0)
            scalar = diagonal[0]
            return (
                scalar
                if len(diagonal) == dim1 and all(elem == scalar for elem in diagonal)
                else None
            )

        if hasattr(data, "as_scipy"):
            data = data.as_scipy()
            if data.nnz == 0:
                return 0.0
            try:
                if not all(a == b for a, b in zip(*data.nonzero())):
                    return None
                dim = data.shape[0]
                data = data.data
                scalar = data[0]
                return (
                    scalar
                    if len(data) == dim and all(value == scalar for value in data)
                    else None
                )
            except ValueError:
                a00 = data[0, 0]
                for i in range(dim1):
                    for j in range(dim1):
                        if i == j:
                            if data[i, i] != a00:
                                return None
                        else:
                            if data[i, i]:
                                return None
                return a00

        # Must be dense...
        if hasattr(data, "as_ndarray"):
            data = data.as_ndarray()
        dim_i, dim_j = data.shape
        if any(
            data[i_idx, j_idx]
            for i_idx in range(dim_i)
            for j_idx in range(dim_j)
            if i_idx != j_idx
        ):
            return None
        scalar = data[0, 0]
        return (
            scalar if all(scalar == data[i, i] for i in range(data.shape[0])) else None
        )

    if qutip_version < parse_version("5.2.0"):

        def fast_tensor(*factors):
            """
            Compute a fast tensor product using a CSR representation.

            If some of the factors are not in Dense representation, convert
            everthing to CSR to speedup the computation.
            """
            return _qutip_tensor(*factors)

    else:

        def fast_tensor(*factors):
            """Compute a fast tensor product using a CSR representation."""
            if all(isinstance(factor.data, _Qutip_Dense) for factor in factors):
                return _qutip_tensor(*factors)
            return _qutip_tensor((factor.to(_Qutip_CSR) for factor in factors))


def data_has_nan(data) -> bool:
    """Check if data has `nan` entries."""
    for _, _, val in data_element_iterator(data):
        if not val == val:
            return True
    return False


def data_is_scalar(data) -> bool:
    """Check if data is a multiple of the identity matrix."""
    return scalar_value(data) is not None


def is_empty_op(op) -> bool:
    """Check if op is an sparse operator without non-zero elements."""
    if isinstance(op, complex):
        return op == 0

    if getattr(op, "prefactor", 1) == 0:
        return True

    if hasattr(op, "nonzero"):
        return len(op.nonzero()) == 0

    if hasattr(op, "data"):
        return data_is_zero(op.data)

    if hasattr(op, "operator_qutip"):
        return is_empty_op(op.operator_qutip)
    if any(is_empty_op(factor) for factor in getattr(op, "site_factors", {}).values()):
        return True
    return False


def hermitian_part(op: _Qobj, tol=None) -> _Qobj:
    """Return the hermitian part of the operator `op`."""
    if op.isherm:
        return op
    return (op + op.dag()).tidyup(tol) * 0.5


def is_diagonal_op(op: _Qobj | np.ndarray) -> bool:
    """Check if a ``Qobj`` operator is diagonal."""
    if isinstance(op, np.ndarray):
        return data_is_diagonal(op)
    return data_is_diagonal(op.data)


def is_scalar_op(op: _Qobj) -> bool:
    """Check if op is a multiple of the identity operator."""
    if isinstance(op, _Qobj):
        return data_is_scalar(op.data)
    return data_is_scalar(op)


def isnan_qutip(op: _Qobj) -> bool:
    """Check if a ``Qobj`` operator has ``nan`` entries."""
    return data_has_nan(op.data)


def norm(
    op: _Qobj,
    ord: Optional[int | str | float] = None,
    axis: Optional[int | Tuple[int, int]] = None,
    keepdims: bool = False,
    check_finite: bool = True,
):
    """Compute the norm of `op` by converting it to a numpy.array.

    Parameters
    ----------
    a : array_like
        Input array. If `axis` is None, `a` must be 1-D or 2-D, unless `ord`
        is None. If both `axis` and `ord` are None, the 2-norm of
        ``a.ravel`` will be returned.
    ord : {int, inf, -inf, 'fro', 'nuc', None}, optional
        Order of the norm (see table under ``Notes``). inf means NumPy's
        `inf` object.
    axis : {int, 2-tuple of ints, None}, optional
        If `axis` is an integer, it specifies the axis of `a` along which to
        compute the vector norms. If `axis` is a 2-tuple, it specifies the
        axes that hold 2-D matrices, and the matrix norms of these matrices
        are computed. If `axis` is None then either a vector norm (when `a`
        is 1-D) or a matrix norm (when `a` is 2-D) is returned.
    keepdims : bool, optional
        If this is set to True, the axes which are normed over are left in the
        result as dimensions with size one. With this option the result will
        broadcast correctly against the original `a`.

    check_finite : bool, optional
        Whether to check that the input matrix contains only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination) if the inputs do contain infinities or NaNs.

    Returns
    -------
    n : float or ndarray
        Norm of the matrix or vector(s).

    Notes
    -----
    ``ord`` is interpreted as:

    =====  ============================  ==========================
    ord    norm for matrices             norm for vectors
    =====  ============================  ==========================
    None   Frobenius norm                2-norm
    'fro'  Frobenius norm                --
    'nuc'  nuclear norm                  --
    inf    max(sum(abs(a), axis=1))      max(abs(a))
    -inf   min(sum(abs(a), axis=1))      min(abs(a))
    0      --                            sum(a != 0)
    1      max(sum(abs(a), axis=0))      as below
    -1     min(sum(abs(a), axis=0))      as below
    2      2-norm (largest sing. value)  as below
    -2     smallest singular value       as below
    other  --                            sum(abs(a)**ord)**(1./ord)
    =====  ============================  ==========================

    See also ``scipy.linalg.norm``.

    """
    if isinstance(op, _Qobj):
        if is_empty_op(op):
            return 0.0

        data = op.data
        if op.isbra or op.isket:
            return _scipy_norm(data.to_array(), ord, axis, keepdims, check_finite)
        assert op.isoper, "op is not valid."
        return _scipy_norm(data.to_array(), ord, axis, keepdims, check_finite)
    else:
        data = op
        return _scipy_norm(data, ord, axis, keepdims, check_finite)


def reshape_qutip_data(data, dims, bs=1) -> np.ndarray:
    """
    Reshape tensor indices.

    Reshape the data representing an operator with dimensions
    dims = [[dim1, dim2,...],[dim1, dim2,...]]
    as an array with shape
    dims' = [[dim1,dim1],[dim2,dim3,... dim2,dim3,...]].
    """
    data_type = data_get_type(data)

    dim_1 = reduce(lambda x, y: x * y, dims[:bs])
    dim_2 = int(data.shape[0] / dim_1)
    new_data: np.ndarray = _np_zeros(
        (
            dim_1**2,
            dim_2**2,
        ),
        dtype=data_type,
    )
    # reshape the operator
    # TODO: see to exploit the sparse structure of data to build the matrix
    for alpha, beta, value in data_element_iterator(data):
        i_idx, k_idx = divmod(alpha, dim_2)
        j_idx, l_idx = divmod(beta, dim_2)
        gamma = dim_1 * i_idx + j_idx
        delta = dim_2 * k_idx + l_idx
        new_data[gamma, delta] = value

    return new_data


def schmidt_dec_first_rest_qutip_operator(
    operator: _Qobj, tol: float = 1e-10
) -> Tuple[List[_Qobj], ...]:
    """
    Decompose an operator as a sum of tensor products.

    Decompose a qutip operator acting over H_1 (x) H_2 (x) H_3 (x) as a sum
    of terms of the form Q_{k} (x) Rest_{k}.
    """
    dims = operator.dims[0]
    if len(dims) < 2:
        return ([operator],)
    data = operator.data
    dim_1 = dims[0]
    dim_2 = int(data.shape[0] / dim_1)
    dims_1 = [[dim_1], [dim_1]]
    dims_2 = [dims[1:], dims[1:]]
    u_mat, s_mat, vh_mat = _svd(
        reshape_qutip_data(data, dims, 1), full_matrices=False, overwrite_a=True
    )
    ops_1 = [
        _Qobj(
            (s * u_mat[:, i].reshape(dim_1, dim_1)),
            dims=dims_1,
            copy=False,
        )
        for i, s in enumerate(s_mat)
        if s > tol
    ]
    ops_2 = [
        _Qobj((vh_mat_row.reshape(dim_2, dim_2)), dims=dims_2, copy=False)
        for vh_mat_row, s in zip(vh_mat, s_mat)
        if s > tol
    ]
    return ops_1, ops_2


def schmidt_dec_first_rest_qutip_operator_hermitian(
    operator: _Qobj, tol: float = 1e-10
) -> Tuple[List[_Qobj], ...]:
    """
    Decompose an hermitian operator as a sum of tensor products.

    Decompose a hermitian qutip operator acting over H_1 (x) H_2 (x) H_3
    (x) as a sum of terms of the form Q_{k} (x) Rest_{k}.
    """
    opsh_1 = []
    opsh_2 = []
    ops_1, *rest = schmidt_dec_first_rest_qutip_operator(operator, tol)
    if not rest:
        return (ops_1,)
    ops_2 = rest[0]

    # First, remove pairs of non-hermitian terms whose
    # sum is hermitian:
    candidates = [i for i, op_1 in enumerate(ops_1) if not op_1.isherm]
    remove_me = []
    while candidates:
        src = candidates.pop()
        op_1 = ops_1[src]
        for tgt in candidates:
            if data_is_zero(((ops_1[tgt].dag() - op_1).tidyup(tol)).data):
                remove_me.append(tgt)
                candidates.remove(tgt)
                ops_1[src] = op_1 * 2
                break

    ops_1 = [op_1 for pos, op_1 in enumerate(ops_1) if pos not in remove_me]
    ops_2 = [op_2 for pos, op_2 in enumerate(ops_2) if pos not in remove_me]

    # Process products of hermitian terms
    for op_1, op_2 in zip(ops_1, ops_2):
        if op_1.isherm:
            opsh_1.append(op_1)
            opsh_2.append(hermitian_part(op_2, tol))
            continue

        op_1h = hermitian_part(op_1, tol)
        if data_is_zero(op_1h.data):
            continue
        op_2h = hermitian_part(op_2, tol)
        opsh_1.append(op_1h)
        opsh_2.append(op_2h)

    # process products of anti-hermitian terms
    for op_1, op_2 in zip(ops_1, ops_2):
        if op_1.isherm:
            continue
        op_1h = hermitian_part(op_1 * 1j, tol)
        if data_is_zero(op_1h.data):
            continue
        op_2h = hermitian_part(op_2 * (-1j), tol)
        opsh_1.append(op_1h)
        opsh_2.append(op_2h)

    return opsh_1, opsh_2


def schmidt_dec_rest_last_qutip_operator(
    operator: _Qobj, tol: float = 1e-10
) -> Tuple[List[_Qobj], ...]:
    """Decompose a qutip operator acting over H_1 (x) H_2 (x) ...

    (x) H_n (x) as a sum of terms of the form Rest_{k} (x)  Q_{k}
    """
    dims = operator.dims[0]
    if len(dims) < 2:
        return ([operator],)
    data = operator.data
    dim_2 = dims[-1]
    dim_1 = int(data.shape[0] / dim_2)
    dims_1 = [dims[:-1], dims[:-1]]
    dims_2 = [[dim_2], [dim_2]]
    u_mat, s_mat, vh_mat = _svd(
        reshape_qutip_data(data, dims, -1), full_matrices=False, overwrite_a=True
    )
    ops_1 = [
        _Qobj(
            (s * u_mat[:, i].reshape(dim_1, dim_1)),
            dims=dims_1,
            copy=False,
        )
        for i, s in enumerate(s_mat)
        if s > tol
    ]
    ops_2 = [
        _Qobj((vh_mat_row.reshape(dim_2, dim_2)), dims=dims_2, copy=False)
        for vh_mat_row, s in zip(vh_mat, s_mat)
        if s > tol
    ]
    return ops_1, ops_2


def schmidt_dec_rest_last_qutip_operator_hermitian(
    operator: _Qobj, tol: float = 1e-10
) -> Tuple[List[_Qobj], ...]:
    """
    Decompose an hermitian operator as a sum of tensor products.

    Decompose a qutip operator acting over H_1 (x) H_2 (x) H_3 (x) as a sum
    of terms of the form Q_{k} (x) Rest_{k}.
    """
    opsh_1 = []
    opsh_2 = []
    ops_1, *rest = schmidt_dec_rest_last_qutip_operator(operator, tol)
    if not rest:
        return (ops_1,)
    ops_2 = rest[0]

    # First, remove pairs of non-hermitian terms whose
    # sum is hermitian:
    candidates = [i for i, op_2 in enumerate(ops_2) if not op_2.isherm]
    remove_me = []
    while candidates:
        src = candidates.pop()
        op_2 = ops_2[src]
        for tgt in candidates:
            if data_is_zero(((ops_2[tgt].dag() - op_2).tidyup(tol)).data):
                remove_me.append(tgt)
                candidates.remove(tgt)
                ops_2[src] = op_2 * 2
                break

    ops_1 = [op_1 for pos, op_1 in enumerate(ops_1) if pos not in remove_me]
    ops_2 = [op_2 for pos, op_2 in enumerate(ops_2) if pos not in remove_me]

    for op_1, op_2 in zip(ops_1, ops_2):
        if op_2.isherm:
            opsh_1.append(hermitian_part(op_1, tol))
            opsh_2.append(op_2)
            continue

        op_2h = hermitian_part(op_2, tol)
        if data_is_zero(op_2h.data):
            continue
        op_1h = hermitian_part(op_1)
        opsh_1.append(op_1h)
        opsh_2.append(op_2h)

    for op_1, op_2 in zip(ops_1, ops_2):
        if op_2.isherm:
            continue
        op_2h = hermitian_part(op_2 * 1j, tol)
        if data_is_zero(op_2h.data):
            continue
        op_1h = hermitian_part(op_1 * (-1j), tol)
        opsh_1.append(op_1h)
        opsh_2.append(op_2h)

    return opsh_1, opsh_2


def to_qobj(array: np.ndarray, atol: float = 1e-12) -> _Qobj:
    """Build a _Qobj with CSR storage directly from a dense numpy array."""
    shape = array.shape
    dims = [[d] for d in shape]
    zero_pos = np.abs(array) < atol
    if shape[0] < 64 or np.count_nonzero(zero_pos):
        array[zero_pos] = 0
        return _Qobj(_fast_from_scipy(_sp_csr_matrix(array)), dims=dims, copy=False)
    return _Qobj(_fast_from_numpy(array), dims=dims, copy=False)


def decompose_qutip_operator(
    operator: _Qobj, tol: float = 1e-10, hermitian: bool = False
) -> List[Tuple]:
    r"""Decompose a qutip operator into a sum of tensor products.

    Decomposes ``operator`` acting on :math:`H_1 \otimes H_2 \otimes \cdots`
    into a list of tuples ``(q1, q2, ...)`` such that
    ``operator \approx \Sigma_k q_1^k \otimes q_2^k \otimes \ldots``.

    Parameters
    ----------
    operator : qutip.Qobj
        The operator to decompose.
    tol : float, optional
        Schmidt coefficients below this threshold are discarded.
    hermitian : bool, optional
        If ``True``, use the Hermitian-aware Schmidt decomposition
        (:func:`schmidt_dec_first_rest_qutip_operator_hermitian`) so that
        all factors in the output are Hermitian operators.  Default is
        ``False``.

    Returns
    -------
    list of tuple of qutip.Qobj
        Each tuple is one tensor-product term in the decomposition.
    """
    _dec = (
        schmidt_dec_first_rest_qutip_operator_hermitian
        if hermitian
        else schmidt_dec_first_rest_qutip_operator
    )
    dims = operator.dims[0]
    ops_1, *rest = _dec(operator, tol)
    if len(rest) == 0:
        return [(op_l,) for op_l in ops_1]
    ops_2 = rest[0]
    if not ops_1:
        return []
    if len(dims) < 3:
        return list(zip(ops_1, ops_2))
    ops_2_factors = [
        decompose_qutip_operator(op2, tol, hermitian=hermitian) for op2 in ops_2
    ]
    result = [
        (op1,) + tuple(op_2 for op_2 in factors)
        for op1, op21_factors in zip(ops_1, ops_2_factors)
        for factors in op21_factors
    ]
    return result if result else []


def decompose_qutip_operator_hermitian(
    operator: _Qobj, tol: float = 1e-10
) -> List[Tuple]:
    """Decompose a Hermitian qutip operator into Hermitian tensor-product terms.

    Convenience alias for :func:`decompose_qutip_operator` with
    ``hermitian=True``.
    """
    return decompose_qutip_operator(operator, tol, hermitian=True)


def get_proper_spaces(spectrum: Iterable) -> List[List[int]]:
    """
    Find the proper space of each eigenvalue.

    Given a diagonal operator, this function finds the proper spaces
    associated to each eigenvalue.
    """
    sectors_dict: Dict[Real, List[int]] = {}
    for idx, sector in enumerate(spectrum):
        sectors_dict.setdefault(np.real(sector), []).append(idx)
    return list(sectors_dict.values())


def reduce_to_proper_spaces(operator: _Qobj, observable: _Qobj) -> _Qobj:
    """
    Reduce operator to a block diagonal operator on each sector.

    If ``observable`` is of the form

    .. code-block:: python

        Q =  sum_i  lambda_i Pi_i

    with ``Pi_i`` a set of orthogonal projectors,
    for an ``operator`` ``T``, this function returns

    .. code-block:: python

        Delta(T) = sum_i  Pi_i T P_i


    """
    # TODO: consider extend for the case of non-diagonal observables
    assert observable.isherm

    def build_proj_data(old_data, spectrum):
        proj_data = np.zeros(operator.shape, dtype=full_operator.dtype)
        for sector in get_proper_spaces(spectrum):
            for i in sector:
                for j in sector:
                    proj_data[i, j] = old_data[i, j]
        return proj_data

    is_diag = data_is_diagonal(observable.data)
    full_operator = operator.full()
    if is_diag:
        spectrum = observable.diag()
        new_data = build_proj_data(full_operator, spectrum)
    else:
        spectrum, unitary = _eigh(observable.full())
        unitary_dag = unitary.T.conj()
        full_operator = unitary_dag @ full_operator @ unitary
        new_data = build_proj_data(full_operator, spectrum)
        new_data = unitary @ new_data @ unitary_dag

    return _Qobj(
        _fast_from_numpy(new_data),
        dims=operator._dims,
        isherm=operator.isherm,
        isunitary=False,
        copy=False,
        # dtype=operator.dtype, # Not supported in Qutip <5.2
    )


def safe_exp_and_normalize(operator: _Qobj) -> Tuple[_Qobj, float]:
    """
    Compute the exponential of a ``Qobj`` operator avoiding overflows.

    Compute the decomposition of exp(operator) as rho*exp(f)
    with f = Tr[exp(operator)], for operator a Qutip operator.

    operator: Qobj

    result: Tuple[Qobj, float]
         (exp(operator)/f , f)

    """
    assert isinstance(operator, _Qobj)
    num_eigvals = min(3, operator.shape[0])
    try:
        k_0 = max(
            np.real(
                operator.eigenenergies(sparse=True, sort="high", eigvals=num_eigvals)
            )
        )
    except np.linalg.LinAlgError as err_la:
        logging.warning(err_la)
        k_0 = 0
    except _ArpackNoConvergence as err_arpack:
        logging.warning(
            "Convergence failed. try with "
            f"{type(err_arpack.eigenvalues)}-> {err_arpack.eigenvalues}"
        )
        k_0 = (
            max(np.real(x) for x in err_arpack.eigenvalues)
            if len(err_arpack.eigenvalues)
            else 0.0
        )
    except _ArpackError:
        return operator * 0 + 1, 0

    op_exp = (operator - k_0).expm()
    op_exp_tr = op_exp.tr()
    op_exp = op_exp * (1.0 / op_exp_tr)
    k_0 = np.log(op_exp_tr) + k_0
    return op_exp, k_0
