"""JSON serializer for ``WorkloadGraph``

"""
import base64
import json
from typing import Any, Dict, List, Optional

import numpy as np

from ..bten.registry import get_op_registry
from ..utils.data_types import DataType, dt2np
from ..utils.sym import SymDim, SymExpr
from .graph import WorkloadGraph

SCHEMA_VERSION = 1
_CONST_INLINE_THRESHOLD = 1024
_ALLOWED_EXPR_OPS = {'+', '-', '*', '//', '%'}


def _collect_sym_names(shape, out: set):
    if shape is None:
        return
    for axis in shape:
        if isinstance(axis, SymDim):
            out.add(axis.name)
        elif isinstance(axis, SymExpr):
            _collect_sym_names_expr(axis, out)


def _collect_sym_names_expr(expr: SymExpr, out: set):
    stack = [expr]
    while stack:
        n = stack.pop()
        if isinstance(n, SymDim):
            out.add(n.name)
        elif isinstance(n, SymExpr):
            stack.append(n.left)
            stack.append(n.right)


def _dim_to_json(d, tensor_name: str, axis_idx: int):
    if isinstance(d, bool):
        raise ValueError(
            f"tensors[{tensor_name!r}].shape[{axis_idx}]: bool is not a valid axis"
        )
    if isinstance(d, int):
        return {"kind": "int", "value": d}
    if isinstance(d, SymDim):
        return {"kind": "sym", "name": d.name}
    if isinstance(d, SymExpr):
        return _expr_to_json(d)
    raise ValueError(
        f"tensors[{tensor_name!r}].shape[{axis_idx}]: unsupported axis type "
        f"{type(d).__name__}"
    )


def _expr_to_json(e: SymExpr) -> Dict[str, Any]:
    return {
        "kind": "expr",
        "op": e.op,
        "left": _operand_to_json(e.left),
        "right": _operand_to_json(e.right),
    }


def _operand_to_json(v):
    if isinstance(v, bool):
        raise ValueError(f"SymExpr operand: bool not supported, got {v!r}")
    if isinstance(v, int):
        return {"kind": "int", "value": v}
    if isinstance(v, SymDim):
        return {"kind": "sym", "name": v.name}
    if isinstance(v, SymExpr):
        return _expr_to_json(v)
    raise ValueError(f"SymExpr operand: unsupported type {type(v).__name__}")


def _encode_const_data(tval, tname: str, include: bool):
    if not include:
        return None
    if tval.data is None:
        return None
    np_dt = dt2np(tval.dtype)
    arr = np.asarray(tval.data, dtype=np_dt)
    # The on-graph shape may be the symbolic Tensor.shape; const tensors are
    # required to be concrete by graph2onnx so use arr.shape for storage.
    data_shape = list(arr.shape)
    flat = arr.flatten()
    if flat.size <= _CONST_INLINE_THRESHOLD:
        return {
            "dtype": tval.dtype.name,
            "shape": data_shape,
            "values": flat.tolist(),
        }
    return {
        "dtype": tval.dtype.name,
        "shape": data_shape,
        "b64": base64.b64encode(arr.tobytes()).decode("ascii"),
    }


def _encode_location(loc, tname: str):
    if loc is None:
        return None
    try:
        json.dumps(loc)
    except TypeError as exc:
        raise TypeError(
            f"tensors[{tname!r}].location is not JSON-serialisable: {exc}"
        ) from exc
    return loc


def _encode_attrs(attrs: Dict[str, Any], op_name: str):
    out: Dict[str, Any] = {}
    for k, v in attrs.items():
        try:
            json.dumps(v)
        except TypeError as exc:
            raise TypeError(
                f"ops[{op_name!r}].attrs[{k!r}] is not JSON-serialisable: {exc}"
            ) from exc
        out[k] = v
    return out


def graph2json(G: WorkloadGraph,
               json_filename: str,
               /,
               *,
               include_const_data: bool = True,
               indent: Optional[int] = 2) -> None:
    """Serialise ``G`` to JSON at ``json_filename`` per the v1 schema (§5).

    See ``docs/tasks/035_graph_roundtrip_deserialization.md`` for the
    full schema. Enforces TENSOR-SHAPE-COMPLETE (§3 item 7): every
    tensor must have a non-None ``shape`` whose every axis is a
    fully-resolved dim record.
    """
    sym_names: set = set()
    for tval in G._tensors.values():
        _collect_sym_names(tval.shape, sym_names)

    tensors_json: List[Dict[str, Any]] = []
    for tname, tval in G._tensors.items():
        if tval.shape is None:
            raise ValueError(
                f"tensors[{tname!r}].shape is None — TENSOR-SHAPE-COMPLETE "
                f"requires a non-None shape on every tensor"
            )
        shape_json = [
            _dim_to_json(d, tname, i) for i, d in enumerate(tval.shape)
        ]
        if tval.dtype not in DataType:
            raise ValueError(
                f"tensors[{tname!r}].dtype: unknown DataType {tval.dtype!r}"
            )
        tensors_json.append({
            "name": tname,
            "dtype": tval.dtype.name,
            "shape": shape_json,
            "is_param": bool(tval.is_param),
            "is_const": bool(tval.is_const),
            "is_view": bool(tval.is_view),
            "location": _encode_location(tval.location, tname),
            "data": (_encode_const_data(tval, tname, include_const_data)
                     if tval.is_const else None),
        })

    registry = get_op_registry()
    ops_json: List[Dict[str, Any]] = []
    for oname in G.get_ordered_nodes():
        op = G.get_op(oname)
        optype = op.optype or ''
        ops_json.append({
            "name": op.name,
            "optype": optype,
            "domain": registry.get_op_domain(optype),
            "inList": list(op.inList),
            "outList": list(op.outList),
            "attrs": _encode_attrs(op.attrs, op.name),
            "id": int(op.id),
            "resource": op.resource,
            "repeat_count": int(op.repeat_count),
            "precision": op.precision,
        })

    payload = {
        "schema_version": SCHEMA_VERSION,
        "name": G._name,
        "sym_dims": sorted(sym_names),
        "tensors": tensors_json,
        "ops": ops_json,
    }

    with open(json_filename, 'w') as f:
        json.dump(payload, f, indent=indent)
