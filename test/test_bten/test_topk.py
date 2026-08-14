
import numpy as np
import pytest

from src.bten.op import make_op
from src.bten.tensor import make_tensor
from src.utils.data_types import DataType


def _make_topk_op(opname, axis=None, largest=None, sorted_=None):
    attrs = {}
    if axis    is not None: attrs['axis']    = axis
    if largest is not None: attrs['largest'] = largest
    if sorted_ is not None: attrs['sorted']  = sorted_

    return make_op(
            name = opname,
            optype='TopK',
            inList=['X', 'K'],
            outList=['V', 'I'],
            attrs=attrs
            )

def _run(op, x_shape, k, dtype='float32', kdtype='int64'):
    k_data = np.array([k], dtype=np.int64)
    iT = [
            make_tensor(name='X', shape=list(x_shape), dtype=dtype),
            make_tensor(name='K', shape=list(k_data.shape), data=k_data, dtype=kdtype, is_const=True),
            ]
    oT = [
            make_tensor(name='V'),
            make_tensor(name='I'),
          ]
    for x in iT: x.op_in  = [op.name]
    for x in oT: x.op_out = [op.name]
    op(iT, oT)
    return iT, oT


@pytest.mark.unit
@pytest.mark.parametrize("tname, x_shape, k, axis, expected", [
    ('1d_default_axis', [10],             3,   None, [3]          ),
    ('1d_axis0',        [10],             5,   0,    [5]          ),
    ('2d_neg_axis',     [4, 8],           3,  -1,    [4, 3]       ),
    ('2d_axis0',        [4, 8],           2,   0,    [2, 8]       ),
    ('2d_axis1',        [4, 8],           5,   1,    [4, 5]       ),
    ('3d_neg_axis',     [2, 3, 16],       4,  -1,    [2, 3, 4]    ),
    ('3d_axis1',        [2, 3, 16],       1,   1,    [2, 1, 16]   ),
    ('4d_axis2',        [2, 4, 6, 8],     3,   2,    [2, 4, 3, 8] ),
    ('k_eq_dim',        [5, 10],         10,  -1,    [5, 10]      ),
    ('k_eq_1',          [3, 7],           1,  -1,    [3, 1]       ),
    ])
def test_topk_shape_preserved(tname, x_shape, k, axis, expected):
    op = _make_topk_op(tname, axis=axis)
    _, oT = _run(op, x_shape, k)
    assert oT[0].shape == expected
    assert oT[1].shape == expected


@pytest.mark.unit
def test_topk_dtype_propagation():
    op = _make_topk_op('Topk_dtype', axis=-1)
    _, oT = _run(op, x_shape=[4, 6], k=2, dtype='bfloat16')
    assert oT[0].dtype == DataType.BFLOAT16
    assert oT[1].dtype == DataType.INT64

@pytest.mark.unit
@pytest.mark.parametrize("k", [-1, -5])
def test_topk_rejects_bad_k(k):
    op = _make_topk_op(f'TopK_bad_value_{k}', axis=-1)
    with pytest.raises(ValueError):
        _run(op, [4, 8], k=k)

@pytest.mark.unit
def test_topk_rejects_k_larger_than_axis_dim():
    op = _make_topk_op('TopK_k_too_big', axis=-1)
    with pytest.raises(ValueError):
        _run(op, [4, 8], k=20)

@pytest.mark.unit
@pytest.mark.parametrize("axis", [2, 3, -3])
def test_topk_rejects_out_of_range_axis(axis):
    op = _make_topk_op('TopK_k_too_big', axis=axis)
    with pytest.raises(ValueError):
        _run(op, [4, 8], k=2)

@pytest.mark.unit
def test_topk_rejects_non_int64_k():
    op = _make_topk_op('TopK_bad_k_type', axis=-1)
    with pytest.raises(ValueError):
        _run(op, [4, 8], k=3, dtype='float32', kdtype='int32')

@pytest.mark.unit
def test_topk_rejects_scalar_input():
    op = _make_topk_op('TopK_scalar_x', axis=0)
    with pytest.raises(ValueError):
        _run(op, [], k=1)

