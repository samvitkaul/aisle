"""Unit tests for :func:`src.graph.json2graph` (Task 035 §7.1)."""
import json

import pytest

from src.bten.op import TensorOp, make_op
from src.bten.tensor import make_tensor
from src.utils.sym import SymDim
from src.graph import (
    STRICT,
    WorkloadGraph,
    graph2json,
    graph_equiv,
    json2graph,
)

from ._fixtures import (
    ALL_GRAPH_FIXTURES,
    make_linear_chain,
    make_sym_test,
)


@pytest.fixture(autouse=True)
def _restore_op_counter():
    """Snapshot/restore the global op counter across each test."""
    saved = TensorOp.op_counter
    yield
    TensorOp.op_counter = saved


def _roundtrip_paths(tmp_path, name='g.json'):
    src = tmp_path / name
    return src


def _dump_payload(G: WorkloadGraph, path) -> dict:
    graph2json(G, str(path))
    with open(path) as f:
        return json.load(f)


def _rewrite(payload: dict, path) -> str:
    with open(path, 'w') as f:
        json.dump(payload, f)
    return str(path)


# --------------------------------------------------------------------------
# Property test — full round-trip equivalence over every fixture.
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    'fixture',
    ALL_GRAPH_FIXTURES,
    ids=[f.__name__ for f in ALL_GRAPH_FIXTURES],
)
def test_round_trip_property(fixture, tmp_path):
    G = fixture()
    path = tmp_path / 'g.json'
    graph2json(G, str(path))
    loaded = json2graph(str(path))
    assert graph_equiv(G, loaded, tolerance=STRICT), (
        f"STRICT round-trip failed for fixture {fixture.__name__}"
    )


# --------------------------------------------------------------------------
# Schema / structural validation
# --------------------------------------------------------------------------
def test_schema_version_mismatch_raises(tmp_path):
    G = make_linear_chain()
    path = tmp_path / 'g.json'
    payload = _dump_payload(G, path)
    payload['schema_version'] = 2
    tampered = _rewrite(payload, tmp_path / 'bad.json')
    with pytest.raises(ValueError, match='schema_version'):
        json2graph(tampered)


def test_unknown_field_raises(tmp_path):
    G = make_linear_chain()
    path = tmp_path / 'g.json'
    payload = _dump_payload(G, path)
    payload['foo'] = 42
    tampered = _rewrite(payload, tmp_path / 'bad.json')
    with pytest.raises(ValueError, match='unknown fields'):
        json2graph(tampered)


def test_unknown_optype_raises_keyerror(tmp_path):
    G = make_linear_chain()
    path = tmp_path / 'g.json'
    payload = _dump_payload(G, path)
    payload['ops'][0]['optype'] = 'NoSuchOpType_XYZ'
    tampered = _rewrite(payload, tmp_path / 'bad.json')
    with pytest.raises(KeyError):
        json2graph(tampered)


def test_tensor_shape_complete_rejects_null_shape(tmp_path):
    G = make_linear_chain()
    path = tmp_path / 'g.json'
    payload = _dump_payload(G, path)
    payload['tensors'][0]['shape'] = None
    tampered = _rewrite(payload, tmp_path / 'bad.json')
    with pytest.raises(ValueError, match='TENSOR-SHAPE-COMPLETE'):
        json2graph(tampered)


def test_tensor_shape_complete_rejects_null_axis(tmp_path):
    G = make_linear_chain()
    path = tmp_path / 'g.json'
    payload = _dump_payload(G, path)
    payload['tensors'][0]['shape'][0] = None
    tampered = _rewrite(payload, tmp_path / 'bad.json')
    with pytest.raises(ValueError, match='expected dim record'):
        json2graph(tampered)


# --------------------------------------------------------------------------
# Semantic invariants of the loaded graph
# --------------------------------------------------------------------------
def test_sym_dim_identity_preserved(tmp_path):
    G = make_sym_test()  # tensors x and y share SymDim('B')
    path = tmp_path / 'g.json'
    graph2json(G, str(path))
    loaded = json2graph(str(path))
    x_axis0 = loaded.get_tensor('x').shape[0]
    y_axis0 = loaded.get_tensor('y').shape[0]
    assert isinstance(x_axis0, SymDim)
    assert isinstance(y_axis0, SymDim)
    assert x_axis0 is y_axis0, (
        "After load, every reference to SymDim('B') must be the same Python "
        "object (canonicalised through sym_cache)."
    )


def test_round_trip_intermediate_shapes_preserved(tmp_path):
    G = make_linear_chain()
    path = tmp_path / 'g.json'
    graph2json(G, str(path))
    loaded = json2graph(str(path))
    for name, src_t in G._tensors.items():
        dst_t = loaded.get_tensor(name)
        assert dst_t.shape is not None, f"tensor {name!r}: loaded shape is None"
        assert src_t.shape is not None, f"tensor {name!r}: source shape is None"
        assert list(src_t.shape) == list(dst_t.shape), (
            f"tensor {name!r}: shape drift {src_t.shape} -> {dst_t.shape}"
        )


def test_loader_does_not_call_shape_inference(monkeypatch, tmp_path):
    """The JSON loader must rehydrate tensor shapes verbatim — it must not
    invoke any op's shape-inference machinery (``TensorOp.__call__`` or
    ``onnx.shape_inference.infer_shapes``).
    """
    G = make_linear_chain()
    path = tmp_path / 'g.json'
    graph2json(G, str(path))

    def _explode(*a, **kw):
        raise AssertionError(
            "shape-inference path was unexpectedly invoked during JSON load"
        )

    # NOTE: There is no ``forward`` method on TensorOp in this codebase.
    # The actual shape-inference entry point is ``TensorOp.__call__``,
    # which dispatches to ``opinfo.shape_inf_func``. Patching ``__call__``
    # at the class level covers every registered op class.
    monkeypatch.setattr(TensorOp, '__call__', _explode)

    # onnx.shape_inference.infer_shapes is the secondary surface the spec
    # calls out. Patch it too; the loader does not import onnx, but if it
    # ever did, this would fail loudly.
    import onnx.shape_inference as _osi
    monkeypatch.setattr(_osi, 'infer_shapes', _explode)

    loaded = json2graph(str(path))
    assert graph_equiv(G, loaded, tolerance=STRICT)


# --------------------------------------------------------------------------
# Canonical SymDims, construct_graph(), counter advance
# --------------------------------------------------------------------------
def _build_canonical_symdim_graph() -> WorkloadGraph:
    """Build a tiny graph exercising all 6 canonical SymDims."""
    B, S, tp, dp, ep, pp = (SymDim(n) for n in ('B', 'S', 'tp', 'dp', 'ep', 'pp'))
    shape = [B, S, tp, dp, ep, pp]
    tensors = [
        dict(name='x', dtype='float32', shape=shape,
             op_in=['Relu_0'], op_out=[]),
        dict(name='y', dtype='float32', shape=shape,
             op_in=[], op_out=['Relu_0']),
    ]
    ops = [dict(name='Relu_0', optype='Relu',
                inList=['x'], outList=['y'])]
    G = WorkloadGraph('canonical_sym')
    for t in tensors:
        G.add_tensor(make_tensor(**t))
    for o in ops:
        G.add_op(make_op(**o))
    G.construct_graph()
    return G


def test_canonical_workload_symdims_round_trip(tmp_path):
    G = _build_canonical_symdim_graph()
    path = tmp_path / 'g.json'
    graph2json(G, str(path))
    loaded = json2graph(str(path))
    assert graph_equiv(G, loaded, tolerance=STRICT)
    # Every named axis preserved.
    for name in ('x', 'y'):
        shp = loaded.get_tensor(name).shape
        assert [a.name for a in shp] == ['B', 'S', 'tp', 'dp', 'ep', 'pp']
        for axis in shp:
            assert isinstance(axis, SymDim)


def test_construct_graph_called(tmp_path):
    G = make_linear_chain()
    path = tmp_path / 'g.json'
    graph2json(G, str(path))
    loaded = json2graph(str(path))
    assert loaded._graph.number_of_nodes() > 0
    assert len(loaded._inodes) > 0
    assert len(loaded._onodes) > 0
    assert len(loaded._itensors) > 0
    assert len(loaded._otensors) > 0


def test_op_counter_advanced(tmp_path):
    G = make_linear_chain()
    path = tmp_path / 'g.json'
    graph2json(G, str(path))
    max_id = max(op.id for op in G._ops.values())
    assert max_id > 0

    TensorOp.reset_counter()
    _ = json2graph(str(path))
    # The counter is an itertools.count(); pop one value and confirm it
    # exceeds the max id observed in the source graph (the loader resets
    # to start=1 then advances past max_id, so the next emitted value
    # must be >= max_id + 1).
    next_val = next(TensorOp.op_counter)
    assert next_val > max_id, (
        f"op_counter not advanced past max loaded id {max_id}: next={next_val}"
    )
