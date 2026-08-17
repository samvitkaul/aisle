from collections.abc import Callable

from ..utils.data_types import DataType, dt2np
from ..utils.sym import SymDim, SymExpr
from .graph import WorkloadGraph


def _dim_to_onnx(d):
    """Convert a shape dimension to ONNX-compatible format.
    int → int, SymDim → str(name), SymExpr → str(repr)
    """
    if isinstance(d, int):
        return d
    if isinstance(d, SymDim):
        return d.name
    if isinstance(d, SymExpr):
        return repr(d)
    raise TypeError(f"Unsupported dimension type: {type(d)}")


def graph2onnx(G: WorkloadGraph,
               onnx_filename: str,
               /,
               producer_name: str="",
               do_model_check: bool=True,
               filter_op_attrs: Callable | None=None):

    import numpy as np
    import onnx
    from onnx import TensorProto
    from onnx.checker import check_model
    from onnx.helper import (
        make_graph,
        make_model,
        make_node,
        make_opsetid,
        make_tensor,
        make_tensor_value_info,
    )

    from ..bten.registry import custom_domains_for, get_op_registry

    _type_map = {
            DataType.FLOAT16 : TensorProto.FLOAT16,
            DataType.BFLOAT16: TensorProto.BFLOAT16,
            DataType.FLOAT32 : TensorProto.FLOAT,
            DataType.FLOAT64 : TensorProto.DOUBLE,
            DataType.INT8    : TensorProto.INT8,
            DataType.INT16   : TensorProto.INT16,
            DataType.INT32   : TensorProto.INT32,
            DataType.INT64   : TensorProto.INT64,
            DataType.UINT8   : TensorProto.UINT8,
            DataType.BOOL    : TensorProto.BOOL,
            }

    onnx_tensors    = {}   # activation ValueInfoProtos (shape only)
    onnx_param_infos = {}  # param ValueInfoProtos (shape only, added to inputs)
    onnx_constants  = {}   # const TensorProtos (with data, added to initializers)

    for tname, tval in G._tensors.items():
        if tval.shape is None:
            raise ValueError(f"Illegal tensor shape for {tval}")
        if tval.dtype not in _type_map:
            raise ValueError(f"dtype for tensor {tval} not yet supported")

        onnx_shape = [_dim_to_onnx(d) for d in tval.shape] if tval.shape else []
        onnx_dtype = _type_map[tval.dtype]

        if tval.is_const:
            # Constants require concrete dims and actual data
            if any(isinstance(d, (SymDim, SymExpr)) for d in tval.shape):
                raise ValueError(f"Constant tensor {tname!r} has symbolic dims — constants must be concrete")
            concrete_shape = tuple(tval.shape)
            if concrete_shape == ():
                if tval.data is None:
                    _data = np.random.randn(1).astype(dt2np(tval.dtype))
                    tval.data = _data[0]
                val_list = [tval.data]
            else:
                if tval.data is None:
                    tval.data = np.random.randn(*concrete_shape).astype(dt2np(tval.dtype))
                    tval.data = tval.data.flatten().tolist()
                val_list = tval.data
            onnx_constants[tname] = make_tensor(
                name=tname, data_type=onnx_dtype, dims=concrete_shape, vals=val_list)
        elif tval.is_param:
            # Params: shape-only declaration (no data generation)
            onnx_param_infos[tname] = make_tensor_value_info(tname, onnx_dtype, onnx_shape)
        else:
            # Activations: shape-only declaration (no data generation)
            onnx_tensors[tname] = make_tensor_value_info(tname, onnx_dtype, onnx_shape)

    # Build ONNX nodes
    _registry = get_op_registry()
    onnx_nodes = {}
    for oname, op in G._ops.items():
        if filter_op_attrs is not None:
            onnx_attrs = filter_op_attrs(op.attrs)  # type: ignore[arg-type]
        else:
            onnx_attrs = op.attrs
        # Custom-domain ops (CCL today; FusedAttention/etc. in the future)
        # are tagged via the registry's per-group domain map. Standard ops
        # resolve to '' which make_node treats as the default ONNX domain.
        optype = op.optype or ''
        op_domain = _registry.get_op_domain(optype)
        onnx_nodes[oname] = make_node(
            optype, op.inList, op.outList, name=oname,
            domain=op_domain, **onnx_attrs)  # type: ignore[arg-type]

    # Assemble graph
    input_list  = [onnx_tensors[x] for x in G.get_input_tensors() if x in onnx_tensors]
    output_list = [onnx_tensors[x] for x in G.get_output_tensors() if x in onnx_tensors]

    # Params go into inputs (shape-only, no initializer data needed)
    param_list = list(onnx_param_infos.values())

    # Constants go into initializers (with data)
    initializer_list = list(onnx_constants.values())

    # Intermediates (activations that are neither inputs nor outputs of the
    # graph) get a ValueInfoProto so the loader can rehydrate every
    # tensor's shape without invoking shape inference
    _io_set = set(G.get_input_tensors()) | set(G.get_output_tensors())
    intermediate_list = [vi for tname, vi in onnx_tensors.items()
                         if tname not in _io_set]

    node_list = [onnx_nodes[node] for node in G.get_ordered_nodes()]

    onnx_graph = make_graph(
            nodes       = node_list,
            name        = G._name,
            inputs      = input_list + param_list,
            outputs     = output_list,
            initializer = initializer_list,
            value_info  = intermediate_list,
            )
    model_def = make_model(onnx_graph, producer_name=producer_name)
    # For each custom domain referenced by any op in this graph, append an
    # opset_import entry so ONNX's checker accepts the model without
    # requiring full schemas for the custom ops (mirrors how ORT ships ops
    # under 'com.microsoft').
    used_domains = custom_domains_for(
        (op.optype or '') for op in G._ops.values())
    for d_name, d_ver in used_domains.items():
        model_def.opset_import.append(make_opsetid(d_name, d_ver))
    if do_model_check:
        check_model(model_def)
    onnx.save(model_def, onnx_filename)
