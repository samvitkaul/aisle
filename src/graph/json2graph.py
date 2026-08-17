"""JSON deserializer for ``WorkloadGraph``

Schema v1

Enforces TENSOR-SHAPE-COMPLETE. Does NOT invoke any op's
shape inference, ``__call__``, or ``onnx.shape_inference.infer_shapes``.
"""
import base64
import json
from typing import Dict, List

import numpy as np

from ..bten.op import TensorOp
from ..bten.tensor import Tensor
from ..bten.registry import get_op_registry
from ..utils.data_types import DataType, dt2np, str2dt
from ..utils.sym import SymDim, SymExpr
from .graph import WorkloadGraph

SCHEMA_VERSION = 1
_ALLOWED_EXPR_OPS = {'+', '-', '*', '//', '%'}
_TENSOR_KEYS = {"name", "dtype", "shape", "is_param", "is_const",
                "is_view", "location", "data"}
_OP_KEYS = {"name", "optype", "domain", "inList", "outList",
            "attrs", "id", "resource", "repeat_count", "precision"}
_TOP_KEYS = {"schema_version", "name", "sym_dims", "tensors", "ops"}


def _err(msg: str) -> ValueError:
    return ValueError(msg)


def _check_keys(obj: dict, allowed: set, path: str):
    if not isinstance(obj, dict):
        raise _err(f"{path}: expected object, got {type(obj).__name__}")
    extras = set(obj.keys()) - allowed
    if extras:
        raise _err(f"{path}: unknown fields {sorted(extras)!r}")


def _parse_dim(rec, sym_cache: Dict[str, SymDim], path: str):
    if not isinstance(rec, dict):
        raise _err(f"{path}: expected dim record (object), got {type(rec).__name__}")
    if "kind" not in rec:
        raise _err(f"{path}.kind: missing")
    kind = rec["kind"]
    if kind == "int":
        if "value" not in rec:
            raise _err(f"{path}.value: missing for kind=int")
        v = rec["value"]
        if not isinstance(v, int) or isinstance(v, bool):
            raise _err(f"{path}.value: expected int, got {type(v).__name__}={v!r}")
        return v
    if kind == "sym":
        if "name" not in rec:
            raise _err(f"{path}.name: missing for kind=sym")
        nm = rec["name"]
        if not isinstance(nm, str):
            raise _err(f"{path}.name: expected str, got {type(nm).__name__}")
        if nm not in sym_cache:
            raise _err(f"{path}.name: {nm!r} not declared in top-level sym_dims")
        return sym_cache[nm]
    if kind == "expr":
        for k in ("op", "left", "right"):
            if k not in rec:
                raise _err(f"{path}.{k}: missing for kind=expr")
        op = rec["op"]
        if op not in _ALLOWED_EXPR_OPS:
            raise _err(
                f"{path}.op: {op!r} not in allowed expr ops {sorted(_ALLOWED_EXPR_OPS)!r}"
            )
        left = _parse_dim(rec["left"], sym_cache, f"{path}.left")
        right = _parse_dim(rec["right"], sym_cache, f"{path}.right")
        return SymExpr(op, left, right)
    raise _err(f"{path}.kind: {kind!r} not in {{'int','sym','expr'}}")


def _parse_shape(shape, sym_cache: Dict[str, SymDim], path: str):
    if shape is None:
        raise _err(
            f"{path}: shape is null — TENSOR-SHAPE-COMPLETE requires "
            f"a fully-resolved shape on every tensor"
        )
    if not isinstance(shape, list):
        raise _err(f"{path}: expected list, got {type(shape).__name__}")
    out = []
    for i, axis in enumerate(shape):
        out.append(_parse_dim(axis, sym_cache, f"{path}[{i}]"))
    return out


def _decode_const_data(rec, path: str):
    if rec is None:
        return None
    if not isinstance(rec, dict):
        raise _err(f"{path}: expected object or null, got {type(rec).__name__}")
    if "dtype" not in rec or "shape" not in rec:
        raise _err(f"{path}: const data record requires 'dtype' and 'shape'")
    dt = str2dt(rec["dtype"])
    if dt == DataType.UNDEF:
        raise _err(f"{path}.dtype: unknown DataType name {rec['dtype']!r}")
    np_dt = dt2np(dt)
    shape = tuple(rec["shape"])
    if "values" in rec:
        vals = rec["values"]
        if not isinstance(vals, list):
            raise _err(f"{path}.values: expected list, got {type(vals).__name__}")
        if shape == ():
            if len(vals) != 1:
                raise _err(
                    f"{path}.values: rank-0 const must have exactly one value, "
                    f"got {len(vals)}"
                )
            return np.asarray(vals, dtype=np_dt).reshape(())
        return np.asarray(vals, dtype=np_dt).reshape(shape)
    if "b64" in rec:
        raw = base64.b64decode(rec["b64"])
        return np.frombuffer(raw, dtype=np_dt).reshape(shape)
    raise _err(f"{path}: const data record requires either 'values' or 'b64'")


def json2graph(json_filename: str, /) -> WorkloadGraph:
    """Inverse of :func:`graph2json`. Lossless round-trip per §6 STRICT.

    Raises ``ValueError`` on schema violations (with the offending JSON
    path in the message) and ``KeyError`` from the op registry when an
    unknown ``optype`` is referenced.
    """
    with open(json_filename, 'r') as f:
        payload = json.load(f)

    if not isinstance(payload, dict):
        raise _err(f"<root>: expected object, got {type(payload).__name__}")
    _check_keys(payload, _TOP_KEYS, "<root>")

    if "schema_version" not in payload:
        raise _err("<root>.schema_version: missing")
    ver = payload["schema_version"]
    if ver != SCHEMA_VERSION:
        raise _err(
            f"<root>.schema_version: expected {SCHEMA_VERSION}, got {ver!r}"
        )

    if "name" not in payload or not isinstance(payload["name"], str):
        raise _err("<root>.name: missing or not a string")

    if "sym_dims" not in payload:
        raise _err("<root>.sym_dims: missing")
    sym_names = payload["sym_dims"]
    if not isinstance(sym_names, list):
        raise _err("<root>.sym_dims: expected list")
    sym_cache: Dict[str, SymDim] = {}
    for i, nm in enumerate(sym_names):
        if not isinstance(nm, str):
            raise _err(f"<root>.sym_dims[{i}]: expected str, got {type(nm).__name__}")
        if nm in sym_cache:
            raise _err(f"<root>.sym_dims[{i}]: duplicate name {nm!r}")
        sym_cache[nm] = SymDim(nm)

    if "tensors" not in payload or not isinstance(payload["tensors"], list):
        raise _err("<root>.tensors: missing or not a list")
    if "ops" not in payload or not isinstance(payload["ops"], list):
        raise _err("<root>.ops: missing or not a list")

    G = WorkloadGraph(payload["name"])

    # ---- tensors ---------------------------------------------------------
    tensor_records: List[dict] = []
    for i, t_rec in enumerate(payload["tensors"]):
        path = f"tensors[{i}]"
        _check_keys(t_rec, _TENSOR_KEYS, path)
        for required in ("name", "dtype", "shape"):
            if required not in t_rec:
                raise _err(f"{path}.{required}: missing")
        name = t_rec["name"]
        if not isinstance(name, str):
            raise _err(f"{path}.name: expected str")
        dt = str2dt(t_rec["dtype"])
        if dt == DataType.UNDEF:
            raise _err(
                f"{path}.dtype: unknown DataType name {t_rec['dtype']!r}"
            )
        shape = _parse_shape(t_rec["shape"], sym_cache, f"{path}.shape")
        data = _decode_const_data(t_rec.get("data"), f"{path}.data")
        tensor = Tensor(
            name=name,
            dtype=t_rec["dtype"],
            shape=shape,
            op_in=[],
            op_out=[],
            is_param=bool(t_rec.get("is_param", False)),
            is_const=bool(t_rec.get("is_const", False)),
            is_view=bool(t_rec.get("is_view", False)),
            data=data,
            location=t_rec.get("location"),
        )
        tensor_records.append(t_rec)
        G.add_tensor(tensor)

    # ---- ops -------------------------------------------------------------
    registry = get_op_registry()
    max_id = 0
    op_records: List[dict] = []
    for i, o_rec in enumerate(payload["ops"]):
        path = f"ops[{i}]"
        _check_keys(o_rec, _OP_KEYS, path)
        for required in ("name", "optype", "inList", "outList"):
            if required not in o_rec:
                raise _err(f"{path}.{required}: missing")
        name = o_rec["name"]
        optype = o_rec["optype"]
        # Cross-check op_type via registry (raises KeyError for unknown).
        registry.get_op(optype)
        # Domain drift check.
        if "domain" in o_rec:
            stored_domain = o_rec["domain"]
            actual_domain = registry.get_op_domain(optype)
            if stored_domain != actual_domain:
                raise _err(
                    f"{path}.domain: stored {stored_domain!r} but registry "
                    f"reports {actual_domain!r} for optype {optype!r}"
                )
        in_list = list(o_rec["inList"])
        out_list = list(o_rec["outList"])
        for tn in in_list + out_list:
            if tn not in G._tensors:
                raise _err(
                    f"{path}: references unknown tensor {tn!r}"
                )
        attrs = dict(o_rec.get("attrs", {}))
        op = TensorOp(
            name=name,
            optype=optype,
            inList=in_list,
            outList=out_list,
            attrs=attrs,
        )
        op_id = o_rec.get("id")
        if op_id is not None:
            if not isinstance(op_id, int) or isinstance(op_id, bool):
                raise _err(f"{path}.id: expected int, got {type(op_id).__name__}")
            op.id = op_id
            if op_id > max_id:
                max_id = op_id
        op.resource = o_rec.get("resource")
        rc = o_rec.get("repeat_count", 1)
        if not isinstance(rc, int) or isinstance(rc, bool):
            raise _err(f"{path}.repeat_count: expected int, got {type(rc).__name__}")
        op.repeat_count = rc
        op.precision = o_rec.get("precision")
        op_records.append(o_rec)
        G.add_op(op)

        # Wire tensor.op_in / op_out from the op's in/out lists.
        for tn in in_list:
            t = G._tensors[tn]
            if name not in t.op_in:
                t.op_in.append(name)
        for tn in out_list:
            t = G._tensors[tn]
            if name not in t.op_out:
                t.op_out.append(name)

    G.construct_graph()

    # Advance the op counter past the max loaded id so subsequent in-process
    # ops keep monotonic uniqueness (§5 last bullet, §10 Q11).
    TensorOp.reset_counter()
    if max_id > 0:
        TensorOp.advance_counter_past(max_id)

    return G
