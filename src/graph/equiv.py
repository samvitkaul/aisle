"""Structural ``WorkloadGraph`` equivalence

Two tolerance presets:

* :data:`STRICT`     — every field must match (used by the JSON round-trip
  property test).
* :data:`ONNX_LOSSY` — fields that ONNX cannot round-trip are skipped
  (used by the ONNX round-trip property test in the follow-up PR).
"""
from dataclasses import dataclass

import numpy as np

from ..bten.op import RemovalReason
from ..utils.sym import SymDim, SymExpr
from .graph import WorkloadGraph


@dataclass(frozen=True)
class EquivTolerance:
    check_is_param: bool = True
    check_is_view: bool = True
    check_location: bool = True
    check_const_data: bool = True
    check_op_id: bool = True
    check_op_resource: bool = True
    check_op_repeat_count: bool = True
    check_op_precision: bool = True
    # If True, SymExpr nodes must match structurally; if False they collapse
    # to a name-equality check (the ONNX-side lossy behaviour where
    # ``SymExpr`` is rehydrated as a single-name ``SymDim``).
    check_symexpr_structural: bool = True


STRICT = EquivTolerance()
ONNX_LOSSY = EquivTolerance(
    check_is_param=False,
    check_is_view=False,
    check_location=False,
    check_const_data=False,
    check_op_id=False,
    check_op_resource=False,
    check_op_repeat_count=False,
    check_op_precision=False,
    check_symexpr_structural=False,
)


def _axis_equal(a, b, tolerance: EquivTolerance) -> bool:
    if isinstance(a, int) and isinstance(b, int):
        return a == b
    if isinstance(a, SymDim) and isinstance(b, SymDim):
        return a.name == b.name
    if isinstance(a, SymExpr) and isinstance(b, SymExpr):
        if not tolerance.check_symexpr_structural:
            return repr(a) == repr(b)
        if a.op != b.op:
            return False
        return (_axis_equal(a.left, b.left, tolerance)
                and _axis_equal(a.right, b.right, tolerance))
    if not tolerance.check_symexpr_structural:
        # Allow SymExpr on one side, SymDim(repr) on the other (ONNX collapse).
        return repr(a) == repr(b)
    return False


def _shape_equal(s1, s2, tolerance: EquivTolerance) -> bool:
    if s1 is None and s2 is None:
        return True
    if s1 is None or s2 is None:
        return False
    if len(s1) != len(s2):
        return False
    return all(_axis_equal(a, b, tolerance) for a, b in zip(s1, s2))


def _data_equal(d1, d2) -> bool:
    if d1 is None and d2 is None:
        return True
    if d1 is None or d2 is None:
        return False
    try:
        a1 = np.asarray(d1)
        a2 = np.asarray(d2)
    except Exception:
        return d1 == d2
    if a1.shape != a2.shape:
        # Const data may have been flattened by the original writer; compare
        # flattened element-wise.
        return np.array_equal(a1.flatten(), a2.flatten())
    return np.array_equal(a1, a2)


def _attrs_equal(a1, a2) -> bool:
    if a1 == a2:
        return True
    if set(a1.keys()) != set(a2.keys()):
        return False
    for k in a1:
        v1, v2 = a1[k], a2[k]
        if isinstance(v1, float) or isinstance(v2, float):
            if float(v1) != float(v2):
                return False
        elif v1 != v2:
            return False
    return True


def graph_equiv(g1: WorkloadGraph,
                g2: WorkloadGraph,
                /,
                *,
                tolerance: EquivTolerance = STRICT) -> bool:
    """Structural equivalence"""
    if g1._name != g2._name:
        return False
    if set(g1._tensors.keys()) != set(g2._tensors.keys()):
        return False
    if set(g1._ops.keys()) != set(g2._ops.keys()):
        return False
    if g1.get_ordered_nodes() != g2.get_ordered_nodes():
        return False

    for tname in g1._tensors:
        t1, t2 = g1._tensors[tname], g2._tensors[tname]
        if t1.name != t2.name:
            return False
        if t1.dtype != t2.dtype:
            return False
        if not _shape_equal(t1.shape, t2.shape, tolerance):
            return False
        if tolerance.check_is_param and t1.is_param != t2.is_param:
            return False
        if t1.is_const != t2.is_const:
            return False
        if tolerance.check_is_view and t1.is_view != t2.is_view:
            return False
        if tolerance.check_location and t1.location != t2.location:
            return False
        if tolerance.check_const_data and t1.is_const:
            if not _data_equal(t1.data, t2.data):
                return False
        if sorted(t1.op_in) != sorted(t2.op_in):
            return False
        if sorted(t1.op_out) != sorted(t2.op_out):
            return False

    for oname in g1._ops:
        o1, o2 = g1._ops[oname], g2._ops[oname]
        if (o1.optype or '') != (o2.optype or ''):
            return False
        if list(o1.inList) != list(o2.inList):
            return False
        if list(o1.outList) != list(o2.outList):
            return False
        if not _attrs_equal(dict(o1.attrs), dict(o2.attrs)):
            return False
        if tolerance.check_op_id and o1.id != o2.id:
            return False
        if tolerance.check_op_resource and o1.resource != o2.resource:
            return False
        if tolerance.check_op_repeat_count and o1.repeat_count != o2.repeat_count:
            return False
        if tolerance.check_op_precision and o1.precision != o2.precision:
            return False
        if o1.kernel_desc is not None or o2.kernel_desc is not None:
            return False
        if (o1.removal_reason is not RemovalReason.NONE
                or o2.removal_reason is not RemovalReason.NONE):
            return False
        if o1.fused_in_optimization or o2.fused_in_optimization:
            return False
        if o1.fused_with_op is not None or o2.fused_with_op is not None:
            return False

    return True
