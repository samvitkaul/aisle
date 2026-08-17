"""ONNX deserializer for ``WorkloadGraph``

Inverse of :func:`src.graph.graph2onnx`. Lossy on some fields
(notably ``SymExpr`` shape elements collapse to ``SymDim(repr_str)``
and ``is_param`` is conflated with activation)

Enforces TENSOR-SHAPE-COMPLETE. Does NOT invoke any op's
``__call__`` / shape-inference, does NOT call
``onnx.shape_inference.infer_shapes`` on the round-trip path.
"""
from typing import Any

from ..bten.op import TensorOp
from ..bten.registry import get_op_registry
from ..bten.tensor import Tensor
from ..utils.data_types import DataType
from ..utils.sym import SymDim
from .graph import WorkloadGraph


def _build_onnx_to_dtype():
    from onnx import TensorProto
    return {
        TensorProto.FLOAT16 : DataType.FLOAT16,
        TensorProto.BFLOAT16: DataType.BFLOAT16,
        TensorProto.FLOAT   : DataType.FLOAT32,
        TensorProto.DOUBLE  : DataType.FLOAT64,
        TensorProto.INT8    : DataType.INT8,
        TensorProto.INT16   : DataType.INT16,
        TensorProto.INT32   : DataType.INT32,
        TensorProto.INT64   : DataType.INT64,
        TensorProto.UINT8   : DataType.UINT8,
        TensorProto.BOOL    : DataType.BOOL,
    }


def _onnx_to_dtype(elem_type: int) -> DataType:
    tbl = _build_onnx_to_dtype()
    if elem_type not in tbl:
        raise ValueError(
            f"unsupported ONNX TensorProto.data_type={elem_type}"
        )
    return tbl[elem_type]


def _parse_shape(type_proto, sym_cache: dict[str, SymDim],
                 tensor_name: str) -> list[Any]:
    """Decode a TypeProto's tensor shape into Racksim axes.

    ``dim_value`` → ``int``; ``dim_param`` → ``SymDim`` (shared by name
    via ``sym_cache`` so repeated references stay object-identical, per
    §6 "SymDim identity"). An axis with neither field set violates
    TENSOR-SHAPE-COMPLETE and raises ``ValueError``.
    """
    shape: list[Any] = []
    for axis, dim in enumerate(type_proto.tensor_type.shape.dim):
        if dim.HasField("dim_param"):
            nm = dim.dim_param
            if nm not in sym_cache:
                sym_cache[nm] = SymDim(nm)
            shape.append(sym_cache[nm])
        elif dim.HasField("dim_value"):
            shape.append(int(dim.dim_value))
        else:
            raise ValueError(
                f"tensor {tensor_name!r} axis {axis}: neither dim_value "
                f"nor dim_param set (TENSOR-SHAPE-COMPLETE violation)"
            )
    return shape


def _decode_attr(attr, op_name: str):
    """Decode a single ``NodeProto`` attribute to its Python value.

    Switches on ``attr.type`` (not on "which list is non-empty") so
    scalar-vs-list fidelity survives (e.g. ``attr.ints=[5]`` → ``[5]``,
    ``attr.i=5`` → ``5``). Subgraph-bearing attributes (``GRAPH`` /
    ``GRAPHS``) raise ``NotImplementedError`` per §3 out-of-scope.
    """
    from onnx import AttributeProto, numpy_helper
    t = attr.type
    if t == AttributeProto.INT:
        return int(attr.i)
    if t == AttributeProto.FLOAT:
        return float(attr.f)
    if t == AttributeProto.STRING:
        return attr.s.decode("utf-8")
    if t == AttributeProto.TENSOR:
        return numpy_helper.to_array(attr.t)
    if t == AttributeProto.INTS:
        return [int(v) for v in attr.ints]
    if t == AttributeProto.FLOATS:
        return [float(v) for v in attr.floats]
    if t == AttributeProto.STRINGS:
        return [v.decode("utf-8") for v in attr.strings]
    if t == AttributeProto.TENSORS:
        return [numpy_helper.to_array(x) for x in attr.tensors]
    if t in (AttributeProto.GRAPH, AttributeProto.GRAPHS):
        raise NotImplementedError(
            f"subgraph attr {attr.name!r} on op {op_name!r} "
            f"(Task 035 §3 out-of-scope)"
        )
    raise ValueError(
        f"op {op_name!r} attr {attr.name!r}: unsupported "
        f"AttributeProto.type={t}"
    )


def onnx2graph(onnx_filename: str,
               /,
               *,
               do_model_check: bool = True,
               strict_optypes: bool = True,
               load_external_data: bool = False,
               ) -> WorkloadGraph:
    """Inverse of :func:`graph2onnx`. Reads ``onnx_filename`` and
    reconstructs a :class:`WorkloadGraph` in the same state as
    ``construct_graph()`` leaves it.

    See module docstring and Task 035 §4 / §6 for the lossy-field list.
    """
    import onnx
    from onnx import checker

    modelpb = onnx.load_model(onnx_filename,
                              load_external_data=load_external_data)
    if do_model_check:
        checker.check_model(modelpb)

    if len(modelpb.functions) > 0:
        raise NotImplementedError(
            f"model.functions populated ({len(modelpb.functions)} entries); "
            f"FunctionProto bodies not supported (Task 035 §3 out-of-scope)"
        )

    graph = modelpb.graph
    registry = get_op_registry()
    sym_cache: dict[str, SymDim] = {}

    G = WorkloadGraph(graph.name)
    seen: set = set()

    # 1) Initializers → const tensors. Data is *not* re-read on the ONNX
    #    round-trip path (§3 out-of-scope); we keep only dtype + shape.
    for init in graph.initializer:
        if init.name in seen:
            continue
        try:
            dt = _onnx_to_dtype(init.data_type)
        except ValueError as e:
            raise ValueError(f"initializer {init.name!r}: {e}") from None
        shape = [int(d) for d in init.dims]
        G.add_tensor(Tensor(
            name=init.name,
            dtype=dt.name.lower(),
            shape=shape,
            is_const=True,
            data=None,
        ))
        seen.add(init.name)

    # 2) Inputs / outputs / value_info → activation tensors. Initializer-set
    #    membership takes precedence (the ``seen`` guard skips echoes).
    def _add_value_info_tensor(vi):
        if vi.name in seen:
            return
        try:
            dt = _onnx_to_dtype(vi.type.tensor_type.elem_type)
        except ValueError as e:
            raise ValueError(f"tensor {vi.name!r}: {e}") from None
        shape = _parse_shape(vi.type, sym_cache, vi.name)
        G.add_tensor(Tensor(
            name=vi.name,
            dtype=dt.name.lower(),
            shape=shape,
        ))
        seen.add(vi.name)

    for vi in graph.input:
        _add_value_info_tensor(vi)
    for vi in graph.output:
        _add_value_info_tensor(vi)
    for vi in graph.value_info:
        _add_value_info_tensor(vi)

    # 3) Nodes → TensorOps. Track loaded ops so we can advance the global
    #    op counter past the max issued id at the end.
    load_warnings: list[str] = []
    loaded_ops: list[TensorOp] = []

    for node in graph.node:
        optype = node.op_type
        try:
            registry.get_op(optype)
        except KeyError:
            if strict_optypes:
                raise
            load_warnings.append(
                f"node {node.name!r}: unknown optype {optype!r}; skipped"
            )
            continue

        if strict_optypes:
            expected_domain = registry.get_op_domain(optype)
            if node.domain != expected_domain:
                raise ValueError(
                    f"node {node.name!r}: domain {node.domain!r} does not "
                    f"match registry-expected domain {expected_domain!r} "
                    f"for optype {optype!r}"
                )

        # TENSOR-SHAPE-COMPLETE: every referenced tensor must already be
        # in G (came from initializer / input / output / value_info).
        for tname in list(node.input) + list(node.output):
            if tname == "":
                continue  # ONNX allows optional empty-string IO slots.
            if tname not in G._tensors:
                raise ValueError(
                    f"node {node.name!r}: references tensor {tname!r} that "
                    f"is missing from initializer/input/output/value_info "
                    f"(TENSOR-SHAPE-COMPLETE violation)"
                )

        attrs: dict[str, Any] = {}
        for attr in node.attribute:
            attrs[attr.name] = _decode_attr(attr, node.name)

        op = TensorOp(
            name=node.name,
            optype=optype,
            inList=list(node.input),
            outList=list(node.output),
            attrs=attrs,
        )
        G.add_op(op)
        loaded_ops.append(op)

        for tn in op.inList:
            if tn == "":
                continue
            t = G._tensors[tn]
            if op.name not in t.op_in:
                t.op_in.append(op.name)
        for tn in op.outList:
            if tn == "":
                continue
            t = G._tensors[tn]
            if op.name not in t.op_out:
                t.op_out.append(op.name)

    G.construct_graph()

    # Advance the op counter past the max loaded id (§9 acceptance).
    if loaded_ops:
        max_id = max(op.id for op in loaded_ops)
        TensorOp.advance_counter_past(max_id)

    if load_warnings:
        # Lazy-init the attribute only when needed (additive on
        # WorkloadGraph; strict-mode loads never set it).
        G._load_warnings = load_warnings  # type: ignore[attr-defined]

    del modelpb
    return G
