"""Unit tests for :func:`src.graph.rebind.rebind_symbolic_dims`"""
import pytest

from src.bten.op import TensorOp, make_op
from src.bten.tensor import make_tensor
from src.utils.sym import SymDim, SymExpr
from src.graph import WorkloadGraph, rebind_symbolic_dims

#from tests.test_graph_serde._fixtures import make_sym_test
from ._fixtures import make_sym_test


@pytest.fixture(autouse=True)
def _restore_op_counter():
    saved = TensorOp.op_counter
    yield
    TensorOp.op_counter = saved


def _build_multi_sym_graph(*sym_names: str) -> WorkloadGraph:
    """Build a graph whose single tensor's shape is the listed SymDims."""
    dims = [SymDim(n) for n in sym_names]
    tensors = [
        dict(name='x', dtype='float32', shape=list(dims),
             op_in=['Relu_0'], op_out=[]),
        dict(name='y', dtype='float32', shape=list(dims),
             op_in=[], op_out=['Relu_0']),
    ]
    ops = [dict(name='Relu_0', optype='Relu',
                inList=['x'], outList=['y'])]
    G = WorkloadGraph(f"sym_{'_'.join(sym_names)}")
    for t in tensors:
        G.add_tensor(make_tensor(**t))
    for o in ops:
        G.add_op(make_op(**o))
    G.construct_graph()
    return G


# --------------------------------------------------------------------------
# Behaviour
# --------------------------------------------------------------------------
def test_rebind_batch_no_forward_call(monkeypatch):
    G = make_sym_test()

    def _explode(*a, **kw):
        raise AssertionError(
            "op shape-inference path was unexpectedly invoked during rebind"
        )

    # There is no ``forward`` method on TensorOp; the shape-inference
    # entry point is ``__call__``. Patch it at the class level.
    monkeypatch.setattr(TensorOp, '__call__', _explode)

    rebound = rebind_symbolic_dims(G, {'B': 64})
    # Sanity: B is now concrete on every tensor.
    for name in ('x', 'y'):
        assert rebound.get_tensor(name).shape[0] == 64


def test_rebind_partial():
    G = _build_multi_sym_graph('B', 'S')
    rebound = rebind_symbolic_dims(G, {'B': 64})
    shape = rebound.get_tensor('x').shape
    assert shape[0] == 64
    assert isinstance(shape[1], SymDim)
    assert shape[1].name == 'S'


def test_rebind_all_canonical():
    G = _build_multi_sym_graph('B', 'S', 'tp', 'dp', 'ep', 'pp')
    env = {'B': 64, 'S': 2048, 'tp': 4, 'dp': 2, 'ep': 1, 'pp': 1}
    rebound = rebind_symbolic_dims(G, env)
    for name in ('x', 'y'):
        shp = rebound.get_tensor(name).shape
        for axis in shp:
            assert isinstance(axis, int), (
                f"tensor {name!r}: axis {axis!r} is not concrete int"
            )
            assert not isinstance(axis, bool)
        assert list(shp) == [64, 2048, 4, 2, 1, 1]


def test_rebind_symexpr_evaluates():
    B = SymDim('B')
    tp = SymDim('tp')
    expr = SymExpr('*', B, tp)
    tensors = [
        dict(name='x', dtype='float32', shape=[expr, 8],
             op_in=['Relu_0'], op_out=[]),
        dict(name='y', dtype='float32', shape=[expr, 8],
             op_in=[], op_out=['Relu_0']),
    ]
    ops = [dict(name='Relu_0', optype='Relu',
                inList=['x'], outList=['y'])]
    G = WorkloadGraph('sym_expr')
    for t in tensors:
        G.add_tensor(make_tensor(**t))
    for o in ops:
        G.add_op(make_op(**o))
    G.construct_graph()

    rebound = rebind_symbolic_dims(G, {'B': 16, 'tp': 4})
    axis0 = rebound.get_tensor('x').shape[0]
    assert isinstance(axis0, int)
    assert axis0 == 64


# --------------------------------------------------------------------------
# Validation of ``env``
# --------------------------------------------------------------------------
def test_rebind_non_positive_raises():
    G = _build_multi_sym_graph('B')
    with pytest.raises(ValueError):
        rebind_symbolic_dims(G, {'B': 0})
    with pytest.raises(ValueError):
        rebind_symbolic_dims(G, {'B': -1})


def test_rebind_non_int_raises():
    G = _build_multi_sym_graph('B')
    with pytest.raises(TypeError):
        rebind_symbolic_dims(G, {'B': 1.5})
    with pytest.raises(TypeError):
        rebind_symbolic_dims(G, {'B': 'eight'})
    # bool is rejected explicitly even though bool is a subclass of int.
    with pytest.raises(TypeError):
        rebind_symbolic_dims(G, {'B': True})


def test_rebind_unknown_env_key_ignored():
    G = _build_multi_sym_graph('B')
    rebound = rebind_symbolic_dims(G, {'B': 8, 'Z': 99})
    assert rebound.get_tensor('x').shape[0] == 8


# --------------------------------------------------------------------------
# Isolation between input and output graphs
# --------------------------------------------------------------------------
def test_rebind_returns_fresh_graph():
    G = make_sym_test()
    rebound = rebind_symbolic_dims(G, {'B': 64})
    # Distinct object.
    assert rebound is not G
    assert id(rebound) != id(G)
    # Mutating the rebound graph's tensor shape must not bleed back into G.
    rebound.get_tensor('x').shape = [999, 999]
    assert G.get_tensor('x').shape[0] != 999
