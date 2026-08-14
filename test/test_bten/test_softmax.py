
import pytest

from src.bten.op import make_op
from src.bten.tensor import make_tensor
from src.utils.data_types import DataType


def _make_softmax_op(opname, axis=None):
    attrs = {}
    if axis is not None:
        attrs['axis'] = axis
    return make_op(
            name = opname,
            optype='Softmax',
            inList=['X'],
            outList=['Y'],
            attrs=attrs
            )

def _run(op, x_shape, dtype='float32'):
    iT = [make_tensor(name='X', shape=list(x_shape), dtype=dtype)]
    oT = [make_tensor(name='Y')]
    iT[0].op_in  = [op.name]
    oT[0].op_out = [op.name]
    op(iT, oT)
    return iT, oT


@pytest.mark.unit
@pytest.mark.parametrize("tname, x_shape, axis_in, expected_norm_axis", [
    ('4d_axis1',       [2, 4, 6, 8],     1,  1, ),
    ('3d_neg_axis',    [2, 3, 16],      -1,  2, ),
    ('3d_axis0',       [2, 3, 16],       0,  0, ),
    ('4d_neg_axis2',   [2, 4, 6, 8],    -2,  2, ),
    ('2d_axis_minus1', [4, 8],          -1,  1, ),
    ('1d_axis0',       [10],             0,  0, ),
    ('1d_neg_axis',    [10],            -1,  0, ),
    ])
def test_softmax_shape_preserved(tname, x_shape, axis_in, expected_norm_axis):
    op = _make_softmax_op(tname, axis=axis_in)
    iT, oT = _run(op, x_shape)
    assert oT[0].shape == iT[0].shape
    stored_axis = op.attrs.get('axis')
    assert isinstance(stored_axis, int)
    assert stored_axis >= 0
    assert stored_axis == expected_norm_axis

@pytest.mark.unit
@pytest.mark.parametrize("x_shape, expected_norm_axis", [
    ([10],         0, ),
    ([4, 8],       1, ),
    ([2, 3, 16],   2, ),
    ([2, 4, 6, 8], 3, ),
    ])
def test_softmax_default_axis_is_last(x_shape, expected_norm_axis):
    op = _make_softmax_op(f'Softmax_default_{len(x_shape)}d') #no axis
    iT, oT = _run(op, x_shape)
    assert oT[0].shape == iT[0].shape
    assert op.attrs.get('axis') == expected_norm_axis

@pytest.mark.unit
def test_softmax_dtype_propagation():
    op = _make_softmax_op('Softmax_dtype', axis=-1)
    _, oT = _run(op, [4, 6], dtype='bfloat16')
    assert oT[0].dtype == DataType.BFLOAT16

@pytest.mark.unit
@pytest.mark.parametrize("axis", [3, -4, 100, -100])
def test_softmax_rejects_out_of_range_axis(axis):
    op = _make_softmax_op(f'Softmax_oor_{axis}', axis=axis)
    with pytest.raises(ValueError):
        _run(op, [2, 3, 8]) #rank=3, valid range [-3, 2]

@pytest.mark.unit
def test_softmax_rejects_non_int_axis():
    op = _make_softmax_op('Softmax_bad_type', axis=1.5)
    with pytest.raises(TypeError):
        _run(op, [2, 3, 8])

