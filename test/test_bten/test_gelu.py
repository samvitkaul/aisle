
import pytest

from src.bten.op import make_op
from src.bten.tensor import make_tensor
from src.utils.data_types import DataType


def _make_gelu_op(opname, approx=None):
    attrs = {}
    if approx is not None:
        attrs['approximate'] = approx
    return make_op(
            name = opname,
            optype='Gelu',
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
@pytest.mark.parametrize("tname, x_shape", [
    ('rank0', []),
    ('rank1', [10]),
    ('rank2', [4, 8]),
    ('rank3', [2, 3, 5]),
    ('rank4', [2, 3, 6, 12]),
    ])
def test_gelu_shape_preserved(tname, x_shape):
    op = _make_gelu_op(tname)
    iT, oT = _run(op, x_shape)
    assert oT[0].shape == iT[0].shape

@pytest.mark.unit
def test_gelu_default_approximate_is_none():
    op = _make_gelu_op('Gelu_default')
    _run(op, [4, 8])
    assert op.attrs.get('approximate') == 'none'

@pytest.mark.unit
@pytest.mark.parametrize("approx", ['none', 'tanh'])
def test_gelu_approximate_check(approx):
    op = _make_gelu_op(f'Gelu_approx_{approx}', approx=approx)
    _run(op, [2, 3, 8])
    assert op.attrs.get('approximate') == approx

@pytest.mark.unit
def test_gelu_dtype_propagation():
    op = _make_gelu_op('Gelu_dtype', approx='tanh')
    _, oT = _run(op, [4, 6], dtype='bfloat16')
    assert oT[0].dtype == DataType.BFLOAT16
    assert op.attrs.get('approximate') == 'tanh'

@pytest.mark.unit
@pytest.mark.parametrize("approx", ['gelu', 'exact', 'TANH', 'None', ''])
def test_gelu_rejects_bad_approximate_value(approx):
    op = _make_gelu_op(f'Gelu_bad_value_{approx}', approx=approx)
    with pytest.raises(ValueError):
        _run(op, [2, 3, 8])

@pytest.mark.unit
@pytest.mark.parametrize("approx", [1, 1.0, True, ['tanh'], {'mode': 'tanh'}])
def test_gelu_rejects_bad_approximate_type(approx):
    op = _make_gelu_op(f'Gelu_bad_type_{approx}', approx=approx)
    with pytest.raises(TypeError):
        _run(op, [2, 3, 8])

