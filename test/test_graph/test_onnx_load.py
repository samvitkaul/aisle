"""Unit tests for :func:`src.graph.onnx2graph` (Task 035 §7.1)."""
import onnx
import pytest
from onnx import TensorProto, helper

from src.bten.op import TensorOp
from src.bten.tensor import make_tensor
from src.utils.sym import SymDim, SymExpr
from src.graph import (
    ONNX_LOSSY,
    WorkloadGraph,
    graph2onnx,
    graph_equiv,
    onnx2graph,
)

from ._fixtures import (
    ALL_GRAPH_FIXTURES,
    make_check_test,
    make_const_param,
    make_filter_test,
    make_large_param_test,
    make_large_test,
    make_linear_chain,
    make_rank0_const,
    make_sym_large_test,
    make_sym_test,
)


@pytest.fixture(autouse=True)
def _restore_op_counter():
    saved = TensorOp.op_counter
    yield
    TensorOp.op_counter = saved


CONCRETE_FIXTURES = [
    make_linear_chain,
    make_const_param,
    make_rank0_const,
    make_check_test,
    make_filter_test,
    make_large_test,
    make_large_param_test,
]


# --------------------------------------------------------------------------
# Round-trip property tests
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    'fixture',
    CONCRETE_FIXTURES,
    ids=[f.__name__ for f in CONCRETE_FIXTURES],
)
def test_round_trip_concrete(fixture, tmp_path):
    G = fixture()
    fpath = str(tmp_path / 'g.onnx')
    graph2onnx(G, fpath, do_model_check=False)
    G2 = onnx2graph(fpath, do_model_check=False)
    assert graph_equiv(G, G2, tolerance=ONNX_LOSSY), (
        f"ONNX_LOSSY round-trip failed for {fixture.__name__}"
    )


@pytest.mark.parametrize(
    'fixture',
    [make_sym_test, make_sym_large_test],
    ids=lambda f: f.__name__,
)
def test_round_trip_symdim(fixture, tmp_path):
    G = fixture()
    fpath = str(tmp_path / 'g.onnx')
    graph2onnx(G, fpath, do_model_check=False)
    G2 = onnx2graph(fpath, do_model_check=False)
    assert graph_equiv(G, G2, tolerance=ONNX_LOSSY), (
        f"ONNX_LOSSY round-trip failed for {fixture.__name__}"
    )
    # SymDim identity: every tensor that mentioned 'B' on the source
    # references the same SymDim object on the loaded graph.
    bdims = [d for t in G2._tensors.values() if t.shape is not None
             for d in t.shape
             if isinstance(d, SymDim) and d.name == 'B']
    assert len(bdims) >= 2
    first = bdims[0]
    assert all(d is first for d in bdims), (
        "SymDim 'B' lost reference identity across tensors"
    )


@pytest.mark.parametrize(
    'fixture',
    ALL_GRAPH_FIXTURES,
    ids=[f.__name__ for f in ALL_GRAPH_FIXTURES],
)
def test_onnx_roundtrip_lossy(fixture, tmp_path):
    """Property test over every shipped fixture (Task 035 §7.2)."""
    G = fixture()
    fpath = str(tmp_path / 'g.onnx')
    graph2onnx(G, fpath, do_model_check=False)
    G2 = onnx2graph(fpath, do_model_check=False)
    assert graph_equiv(G, G2, tolerance=ONNX_LOSSY)


# --------------------------------------------------------------------------
# SymExpr collapse — pinned lossy behaviour
# --------------------------------------------------------------------------
def _build_symexpr_fixture() -> WorkloadGraph:
    B = SymDim('B')
    expr = SymExpr('*', B, 2)
    tensors = [
        dict(name='x', dtype='float32', shape=[expr, 8],
             op_in=['Relu_0'], op_out=[]),
        dict(name='y', dtype='float32', shape=[expr, 8],
             op_in=[], op_out=['Relu_0']),
    ]
    ops = [dict(name='Relu_0', optype='Relu', inList=['x'], outList=['y'])]
    G = WorkloadGraph('symexpr_test')
    for t in tensors:
        G.add_tensor(make_tensor(**t))
    for o in ops:
        from src.bten.op import make_op
        G.add_op(make_op(**o))
    G.construct_graph()
    return G


def test_round_trip_symexpr_collapses_to_symdim(tmp_path):
    G = _build_symexpr_fixture()
    expr = G._tensors['x'].shape[0]
    expected_name = repr(expr)
    fpath = str(tmp_path / 'g.onnx')
    graph2onnx(G, fpath, do_model_check=False)
    G2 = onnx2graph(fpath, do_model_check=False)

    loaded_axis = G2._tensors['x'].shape[0]
    assert isinstance(loaded_axis, SymDim), (
        f"expected SymDim after ONNX collapse, got {type(loaded_axis).__name__}"
    )
    assert loaded_axis.name == expected_name, (
        f"SymDim name {loaded_axis.name!r} != repr(SymExpr) {expected_name!r}"
    )
    # Identity preserved between x and y (both referenced the same SymExpr
    # on the source graph; the loader collapses to a shared SymDim).
    assert loaded_axis is G2._tensors['y'].shape[0]


# --------------------------------------------------------------------------
# Out-of-scope: subgraphs and FunctionProtos must raise NotImplementedError.
# --------------------------------------------------------------------------
def _save_minimal_model(tmp_path, *, with_subgraph_attr: bool = False,
                        with_function: bool = False) -> str:
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [4])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [4])
    if with_subgraph_attr:
        sub = helper.make_graph(nodes=[], name='sub', inputs=[], outputs=[])
        subgraph_attr = helper.make_attribute('then_branch', sub)
        node = helper.make_node('Relu', ['x'], ['y'], name='node_with_sub')
        node.attribute.append(subgraph_attr)
    else:
        node = helper.make_node('Relu', ['x'], ['y'], name='good_node')
    graph = helper.make_graph([node], 'g', [x], [y])
    model = helper.make_model(graph, producer_name='test')
    if with_function:
        fn = helper.make_function(
            domain='com.racksim',
            fname='MyFunc',
            inputs=[],
            outputs=[],
            nodes=[],
            opset_imports=[helper.make_opsetid('', 17)],
        )
        model.functions.append(fn)
    path = str(tmp_path / 'mdl.onnx')
    onnx.save(model, path)
    return path


def test_subgraph_attr_raises_notimplemented(tmp_path):
    path = _save_minimal_model(tmp_path, with_subgraph_attr=True)
    with pytest.raises(NotImplementedError) as exc:
        onnx2graph(path, do_model_check=False)
    msg = str(exc.value)
    assert 'then_branch' in msg
    assert 'node_with_sub' in msg


def test_functions_present_raises_notimplemented(tmp_path):
    path = _save_minimal_model(tmp_path, with_function=True)
    with pytest.raises(NotImplementedError, match='FunctionProto'):
        onnx2graph(path, do_model_check=False)


# --------------------------------------------------------------------------
# strict_optypes: domain drift / unknown optype.
# --------------------------------------------------------------------------
def _save_unknown_optype_model(tmp_path, *, op_type: str = 'NoSuchOpRacksimDoesNotHave',
                               domain: str = '') -> str:
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [4])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [4])
    node = helper.make_node(op_type, ['x'], ['y'], name='odd_node', domain=domain)
    graph = helper.make_graph([node], 'g', [x], [y])
    model = helper.make_model(graph, producer_name='test')
    path = str(tmp_path / 'unknown.onnx')
    onnx.save(model, path)
    return path


def test_strict_optypes_rejects_unknown(tmp_path):
    path = _save_unknown_optype_model(tmp_path)
    with pytest.raises(KeyError):
        onnx2graph(path, do_model_check=False)


def test_strict_optypes_rejects_domain_drift(tmp_path):
    path = _save_unknown_optype_model(
        tmp_path, op_type='MatMul', domain='com.someoneelse'
    )
    with pytest.raises(ValueError) as exc:
        onnx2graph(path, do_model_check=False)
    msg = str(exc.value)
    assert 'com.someoneelse' in msg
    assert 'MatMul' in msg
    # Expected (registry) domain '' should also be named.
    assert "''" in msg or '""' in msg


def test_strict_optypes_false_skips_unknown(tmp_path):
    path = _save_unknown_optype_model(tmp_path)
    G = onnx2graph(path, do_model_check=False, strict_optypes=False)
    warnings = getattr(G, '_load_warnings', None)
    assert warnings is not None and len(warnings) == 1
    assert 'odd_node' in warnings[0]
    assert 'NoSuchOpRacksimDoesNotHave' in warnings[0]


# --------------------------------------------------------------------------
# Loader must not call shape inference or any op's forward / __call__.
# --------------------------------------------------------------------------
def test_loader_does_not_call_shape_inference(monkeypatch, tmp_path):
    G = make_linear_chain()
    fpath = str(tmp_path / 'g.onnx')
    graph2onnx(G, fpath, do_model_check=False)

    def _explode(*a, **kw):
        raise AssertionError(
            'onnx.shape_inference.infer_shapes was unexpectedly invoked'
        )

    import onnx.shape_inference as _osi
    monkeypatch.setattr(_osi, 'infer_shapes', _explode)

    loaded = onnx2graph(fpath, do_model_check=False)
    assert graph_equiv(G, loaded, tolerance=ONNX_LOSSY)


def test_loader_does_not_call_forward(monkeypatch, tmp_path):
    G = make_linear_chain()
    fpath = str(tmp_path / 'g.onnx')
    graph2onnx(G, fpath, do_model_check=False)

    def _explode(self, *a, **kw):
        raise AssertionError(
            'TensorOp.__call__ (shape-inference entry point) was unexpectedly invoked'
        )

    monkeypatch.setattr(TensorOp, '__call__', _explode)

    loaded = onnx2graph(fpath, do_model_check=False)
    assert graph_equiv(G, loaded, tolerance=ONNX_LOSSY)
