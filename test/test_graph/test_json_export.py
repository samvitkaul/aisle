"""Unit tests for :func:`src.graph.graph2json` (Task 035 §7.1)."""
import base64
import json

import numpy as np
import pytest

from src.bten.op import TensorOp, make_op
from src.bten.tensor import make_tensor
from src.graph import WorkloadGraph, graph2json
from src.utils.data_types import DataType
from src.utils.sym import SymDim, SymExpr

from ._fixtures import (
    make_const_param,
    make_linear_chain,
    make_rank0_const,
    make_sym_test,
)


@pytest.fixture(autouse=True)
def _restore_op_counter():
    """Snapshot/restore the global op counter across each test."""
    saved = TensorOp.op_counter
    yield
    TensorOp.op_counter = saved


def _dump(G: WorkloadGraph, path, **kw) -> dict:
    graph2json(G, str(path), **kw)
    with open(path) as f:
        return json.load(f)


def test_schema_version_present(tmp_path):
    G = make_linear_chain()
    payload = _dump(G, tmp_path / 'g.json')
    assert payload['schema_version'] == 1


def test_dtype_canonical_name(tmp_path):
    G = make_linear_chain()
    payload = _dump(G, tmp_path / 'g.json')
    for t in payload['tensors']:
        # Every dtype string must be a valid DataType enum name (the ".name"
        # form, e.g. "FLOAT32") — not the lowercase alias "float32".
        assert t['dtype'] in DataType.__members__, (
            f"tensor {t['name']!r}: dtype {t['dtype']!r} is not a DataType enum name"
        )


def test_sym_dims_deduplicated(tmp_path):
    G = make_sym_test()  # uses SymDim('B') on three tensors
    payload = _dump(G, tmp_path / 'g.json')
    sym_dims = payload['sym_dims']
    assert sym_dims.count('B') == 1
    # Every name referenced by any shape must appear exactly once in sym_dims.
    referenced: set = set()

    def _walk(rec):
        if not isinstance(rec, dict):
            return
        if rec.get('kind') == 'sym':
            referenced.add(rec['name'])
        elif rec.get('kind') == 'expr':
            _walk(rec['left'])
            _walk(rec['right'])
    for t in payload['tensors']:
        for axis in t['shape']:
            _walk(axis)
    assert sorted(referenced) == sorted(sym_dims)
    assert len(sym_dims) == len(set(sym_dims))


def test_expr_shape_ast_serialised(tmp_path):
    B = SymDim('B')
    expr_axis = SymExpr('*', B, 2)
    tensors = [
        dict(name='x', dtype='float32', shape=[expr_axis, 8],
             op_in=['MatMul_0'], op_out=[]),
        dict(name='w', dtype='float32', shape=[8, 16],
             op_in=['MatMul_0'], op_out=[], is_param=True),
        dict(name='y', dtype='float32', shape=[expr_axis, 16],
             op_in=[], op_out=['MatMul_0']),
    ]
    ops = [dict(name='MatMul_0', optype='MatMul',
                inList=['x', 'w'], outList=['y'])]
    G = WorkloadGraph('expr_test')
    for t in tensors:
        G.add_tensor(make_tensor(**t))
    for o in ops:
        G.add_op(make_op(**o))
    G.construct_graph()

    payload = _dump(G, tmp_path / 'g.json')
    x_axis0 = payload['tensors'][[t['name'] for t in payload['tensors']].index('x')]['shape'][0]
    assert isinstance(x_axis0, dict)
    assert x_axis0['kind'] == 'expr'
    assert x_axis0['op'] == '*'
    # Operands themselves are nested dim records (not bare repr strings).
    assert isinstance(x_axis0['left'], dict)
    assert isinstance(x_axis0['right'], dict)
    # Exactly one of {sym, int} on each side
    assert x_axis0['left']['kind'] == 'sym'
    assert x_axis0['left']['name'] == 'B'
    assert x_axis0['right']['kind'] == 'int'
    assert x_axis0['right']['value'] == 2


def test_const_small_inline(tmp_path):
    G = make_rank0_const()  # const scalar tensor 's' with data=2.5
    payload = _dump(G, tmp_path / 'g.json')
    const_recs = [t for t in payload['tensors'] if t['is_const']]
    assert len(const_recs) == 1
    data = const_recs[0]['data']
    assert data is not None
    assert 'values' in data and isinstance(data['values'], list)
    assert 'b64' not in data


def test_const_large_b64(tmp_path):
    arr = np.arange(2048, dtype=np.float32)  # > 1024 elements
    tensors = [
        dict(name='x', dtype='float32', shape=[2048],
             op_in=['Identity_0'], op_out=[]),
        dict(name='big', dtype='float32', shape=[2048],
             op_in=['Identity_0'], op_out=[], is_const=True, data=arr),
        dict(name='y', dtype='float32', shape=[2048],
             op_in=[], op_out=['Identity_0']),
    ]
    ops = [dict(name='Identity_0', optype='Identity',
                inList=['x', 'big'], outList=['y'])]
    G = WorkloadGraph('big_const')
    for t in tensors:
        G.add_tensor(make_tensor(**t))
    for o in ops:
        G.add_op(make_op(**o))
    G.construct_graph()

    payload = _dump(G, tmp_path / 'g.json')
    big_rec = next(t for t in payload['tensors'] if t['name'] == 'big')
    data = big_rec['data']
    assert data is not None
    assert 'b64' in data and isinstance(data['b64'], str)
    assert 'values' not in data
    # Round-trip the base64 payload and confirm contents.
    decoded = np.frombuffer(base64.b64decode(data['b64']), dtype=np.float32)
    np.testing.assert_array_equal(decoded, arr)


def test_include_const_data_false_skips_data(tmp_path):
    G = make_rank0_const()
    payload = _dump(G, tmp_path / 'g.json', include_const_data=False)
    const_recs = [t for t in payload['tensors'] if t['is_const']]
    assert len(const_recs) >= 1
    for rec in const_recs:
        assert rec['data'] is None


def test_attr_non_jsonable_raises_typeerror(tmp_path):
    G = make_const_param()
    op = G.get_op('MatMul_0')
    op.attrs = {'bad_attr': np.array([1.0, 2.0, 3.0])}
    with pytest.raises(TypeError) as excinfo:
        graph2json(G, str(tmp_path / 'g.json'))
    msg = str(excinfo.value)
    assert 'MatMul_0' in msg
    assert 'bad_attr' in msg
